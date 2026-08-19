"""
logger.py — 结构化日志

【为什么用结构化日志而不是 print / 默认 format（面试高频）】
1. 生产环境的日志要被采集进 ELK/Loki 之类的系统，JSON 一行一条是通用格式；
2. Agent 系统的日志比普通 API 复杂：一次请求内要串联
   "意图识别 → 规划 → SQL 查询 → RAG 检索 → 路由判断 → 生成回答" 多段日志，
   结构化字段（request_id / session_id / node / iteration）让一次 Agent
   请求的完整调用链可以被 grep/jq 一条龙串起来——这是排查 LLM 行为问题的关键。
   纯文本日志在多行堆栈和字段粘连下基本没法自动分析。

【面试点：Agent 的可观测性三件套】
- 日志（logger.py）：单次请求内的行为细节——"Agent 为什么这么想"
- 指标（middleware.py / Prometheus）：聚合趋势——QPS、P95、错误率、LLM 调用次数
- 链路（生产可用 OpenTelemetry）：跨服务分布式追踪
三者回答不同粒度的问题，不能互相替代。
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """把日志记录格式化为一行 JSON。

    只序列化我们白名单内的字段（ts/level/logger/msg + extra），
    避免意外把敏感对象 dump 进日志。
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # extra 里允许带结构化上下文（request_id / session_id / node 等）
        # 注意：error 字段必须保留——排障时异常详情是最重要的信息
        for key in ("request_id", "session_id", "node", "iteration", "tool", "latency_ms", "error", "from", "to", "chunks"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """进程启动时调用一次，全局日志输出为单行 JSON。

    【为什么日志走 stdout 而不是写文件？】
    容器化部署的标准约定：应用只负责写 stdout，日志收集交给
    docker log driver / K8s 的采集器。应用内写日志文件在容器里
    反而带来轮转、多副本合并等一系列问题。
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # 降噪：uvicorn 的访问日志保留，但第三方库的 DEBUG 不输出
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
