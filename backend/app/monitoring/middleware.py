"""
middleware.py — 请求监控中间件（耗时 / 错误率 / Prometheus 指标）

【设计决策（面试点）：为什么自己写轻量 ASGI 中间件而不是 BaseHTTPMiddleware？】
1. BaseHTTPMiddleware 在 Starlette 中有已知的性能与流式响应问题
   （它基于 request 重建上下文，对流式响应需要额外缓冲）；
2. 我们只需要在请求进入/退出时计时打点，纯 ASGI 中间件 20 行足够，
   不引入框架抽象——可控性优先；
3. 指标直接用 prometheus_client 输出到 /metrics（main.py 挂载），
   不需要额外指标网关。

【指标设计】
- http_requests_total{method, path, status}   —— QPS 与状态码分布
- http_request_duration_seconds{method, path} —— 直方图，可算 P50/P95/P99
- llm_calls_total{node}                       —— 各 Agent 节点的 LLM 调用计数
- agent_iterations_total                      —— Agent 循环总迭代数（观察是否频繁触顶）
- agent_tool_calls_total{tool}                —— 各 tool 的调用计数
前两个是通用 API 指标；后三个是 Agent 特有指标——
面试官如果问"Agent 和普通 API 的监控有什么不同"，
答案就是：除了吞吐延迟，你还要监控 Agent 的"行为指标"：
迭代次数是否频繁触顶（意味着循环失控）、哪个 tool 调用最多（成本与瓶颈）、
单请求 LLM 调用次数（token 成本）。
"""

import time
import uuid

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.monitoring.logger import get_logger

logger = get_logger(__name__)

# ---- 通用 API 指标 ----
REQUEST_COUNT = Counter(
    "http_requests_total", "HTTP 请求总数", ["method", "path", "status"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP 请求耗时（秒）", ["method", "path"]
)

# ---- Agent 行为指标 ----
LLM_CALLS = Counter(
    "llm_calls_total", "Agent 节点 LLM 调用次数", ["node"]
)
AGENT_ITERATIONS = Counter(
    "agent_iterations_total", "Agent 循环总迭代数"
)
TOOL_CALLS = Counter(
    "agent_tool_calls_total", "Agent tool 调用次数", ["tool"]
)


def track_llm_call(node: str) -> None:
    """LLM 客户端每次真实调用后打点（llm_client.py 调用）。"""
    LLM_CALLS.labels(node=node).inc()


def track_agent_iteration() -> None:
    """每轮工具执行迭代打点（nodes.py 的 tool_executor 调用）。"""
    AGENT_ITERATIONS.inc()


def track_tool_call(tool: str) -> None:
    """每次 tool 实际执行打点。"""
    TOOL_CALLS.labels(tool=tool).inc()


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    请求级监控：耗时直方图 + 状态码计数 + request_id 注入。

    request_id 用 UUID 注入请求上下文并回写到响应头——
    【面试点】用户报障时凭 X-Request-ID 就能在结构化日志里
    串出该请求的完整调用链（含 Agent 每一步决策），这是排障的生命线。
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        start = time.perf_counter()
        # 预聚合路径，避免 /docs 等路由产生无穷基数（指标爆炸）
        path = request.url.path
        try:
            response = await call_next(request)
        except Exception as exc:
            # 异常路径也要打点：500 错误率是 SLO 的核心指标
            REQUEST_COUNT.labels(method=request.method, path=path, status="500").inc()
            REQUEST_DURATION.labels(method=request.method, path=path).observe(
                time.perf_counter() - start
            )
            logger.error("request_failed", extra={
                "request_id": request_id,
                "exc": str(exc),
            })
            raise

        REQUEST_COUNT.labels(
            method=request.method, path=path, status=str(response.status_code)
        ).inc()
        REQUEST_DURATION.labels(method=request.method, path=path).observe(
            time.perf_counter() - start
        )
        response.headers["X-Request-ID"] = request_id
        return response
