"""
config.py — 全局配置中心（pydantic-settings）

【为什么用 pydantic-settings 而不是 os.getenv 散落在各处？】
1. 类型校验：DB_POOL_SIZE 传了字符串 "abc" 会在启动时就报错，而不是运行时才炸；
2. 单点定义：所有环境变量名集中在一处，.env.example 与代码不会脱节；
3. 面试点：生产环境配置必须"fail fast"——配置错误应该在进程启动时暴露，
   而不是等第一个用户请求打进来才 500。

【面试常问：为什么这些参数有默认值？】
Demo 一键启动（docker compose up 什么都不配就能跑）。
生产环境应显式注入所有敏感配置（API key 无默认值，缺失即启动失败）。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 查找路径：优先当前工作目录，其次项目根目录（backend/ 的上一级）。
# 本地开发从 backend/ 启动 uvicorn 时 cwd 下没有 .env，必须回退到项目根——
# 一个 .env 放根目录，两种启动方式都能读到（docker 用环境变量注入，不依赖文件）
_ENV_CANDIDATES = (Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env")


class Settings(BaseSettings):
    """全局配置。字段名与大写环境变量一一对应。"""

    model_config = SettingsConfigDict(
        env_file=_ENV_CANDIDATES,  # cwd 优先，回退项目根目录
        env_file_encoding="utf-8",
        extra="ignore",            # 容忍多余的环境变量（容器里常有）
    )

    # ---------------- 应用 ----------------
    app_name: str = "Enterprise AI Procurement Agent"
    debug: bool = False
    log_level: str = "INFO"

    # ---------------- LLM（DeepSeek，OpenAI 兼容协议） ----------------
    # API key 故意没有默认值——Agent 没有 LLM 就无意义，缺失应启动即报错
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    # 单次 LLM 调用超时。Agent 一次请求串行调用 8~12 次 LLM，
    # 单次超时若设为 120s，最坏情况总延迟 20+ 分钟——所以这里必须收紧
    llm_timeout: float = 60.0
    llm_max_retries: int = 1        # 网络抖动重试 1 次（指数退避，见 llm_client）

    # ---------------- Embedding / Rerank（硅基流动） ----------------
    # Embedding key 允许为空：为空时 RAG 降级为关键词 mock 检索（见 tools.py），
    # Agent 流程依然可跑通——Phase 2 的 mock 不是临时胶带，是正式降级路径
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---------------- PostgreSQL ----------------
    # 主连接（app_role）：建表、种子数据、文档写入
    database_url: str = (
        "postgresql+asyncpg://app_role:app_password@localhost:5432/enterprise_agent"
    )
    # 只读连接（agent_readonly）：execute_sql tool 专用
    # ——三层只读防御的第三层：即使 prompt 与正则都被绕过，数据库直接拒绝写操作
    database_readonly_url: str = (
        "postgresql+asyncpg://agent_readonly:readonly_password@localhost:5432/enterprise_agent"
    )
    # ---- 连接池参数（面试高频，为什么是这些数见 connection.py 注释）----
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # ---------------- Redis（Agent 会话状态） ----------------
    redis_url: str = "redis://localhost:6379/0"
    # 会话 30 分钟不活动过期。为什么 30 分钟：Agent 分析任务本身可能耗时数分钟，
    # TTL 太短会导致用户看一份报告的时间就把会话丢了；太长则 Redis 内存被死会话占用
    session_ttl_seconds: int = 1800
    # 注入 Planner 的历史摘要最多保留几轮——token 控制的关键
    max_history_turns: int = 5

    # ---------------- Langfuse（LLM 可观测性，增强依赖） ----------------
    # 默认关闭：不配置 key 就能跑——tracing 挂了绝不影响 Agent 主链路
    # （与 Redis 同级容错哲学，见 monitoring/langfuse.py 注释）
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---------------- Agent 图 ----------------
    # 最大迭代次数：LLM 路由 + 代码硬兜底中的"代码兜底"
    max_iterations: int = 6
    # 整个 Agent 请求超时：超时强制终止，返回已有结果并在 errors 里注明
    agent_timeout_seconds: int = 120
    # Supervisor 多 Agent 模式：最多派发几轮子 Agent（防 LLM 来回空转）
    max_supervisor_rounds: int = 4

    # ---------------- RAG ----------------
    chunk_size: int = 500          # 中文字符数。500 字 ≈ 700+ token，BGE 中文场景常用区间
    chunk_overlap: int = 50        # 重叠 = chunk_size 的 1/10，递归分割防语义截断
    retrieval_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """进程级单例。lru_cache 保证 .env 只解析一次。"""
    return Settings()
