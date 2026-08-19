"""
langfuse.py — Langfuse LLM 可观测性（tracing）接入

【为什么是"增强依赖"（设计决策 16 同款哲学）】
Langfuse 是观测工具，不是 Agent 主链路的一部分：
- 不配置（langfuse_enabled=False 默认）→ 零成本关闭，行为与未接入前完全一致；
- 配置了但服务不可用 → 丢弃 trace，记一条 warning，绝不影响 LLM 调用。
判断标准一句话："这个依赖挂了，Agent 还能不能给出有用的回答？"
能——所以本模块的对外契约是：任何函数都不抛出，失败只降级。
（注：SDK 的 span 导出是后台线程异步进行的，服务不可用时只在日志出现
  "Failed to export span batch"，连 warning 都不必每次都记——不干扰主链路。）

【为什么手动 observation 而不是 @observe 装饰器？（面试点）】
1. 装饰器是静态的，而开关/密钥在进程启动后才知道（测试还要 mock）；
2. 我们只关心"LLM 调用"这一层（model / 输入 / 输出 / token 用量 / 耗时），
   手动 start_observation → update → end 三行就能表达，不引入装饰器耦合；
3. 与 monitoring/middleware.py 同哲学：可控性优先，每个埋点可解释。

【为什么用 context manager（async with）而不是事后补记？】
generation span 的时长 = start 到 end 的时间差。事后补记时长≈0，
"这个调用花了多久"是最有用的排障信息（延迟/成本/重试）。
把 await create() 包进 async with，异常时自动记 ERROR 级 span——
trace 里能看到"哪一步尝试失败过"，这是事后补记做不到的。

【trace 与 generation 的关系（面试点）】
Langfuse 对象模型：trace（一次请求）> span（一个阶段）> generation（一次
LLM 调用）。当前只打 generation 层（无 trace 的 generation 会被 SDK 自动
归入独立 trace）——粒度足够回答"为什么慢/贵"。升级到请求级 trace 树
（串联所有节点调用）是量变不是质变：在 API 层包一个 trace，
把 session_id 放 metadata 即可。
"""

import contextlib
import logging
from typing import Any, AsyncIterator

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: Any = None        # langfuse.Langfuse | None
_init_attempted = False    # 初始化失败后不再重试（避免每次 LLM 调用都重试构造）


def get_langfuse() -> Any:
    """懒加载单例。未启用 / 初始化失败 → None（调用方直接跳过埋点）。"""
    global _client, _init_attempted
    if _client is not None:
        return _client
    settings = get_settings()
    if not settings.langfuse_enabled or _init_attempted:
        return None
    _init_attempted = True
    try:
        from langfuse import Langfuse  # 延迟导入：未安装该包时也不影响主链路

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            debug=settings.debug,
        )
    except Exception as exc:  # noqa: BLE001 —— 初始化失败降级为不追踪
        logger.warning("langfuse_init_failed", extra={"error": str(exc)})
        _client = None
    return _client


@contextlib.asynccontextmanager
async def llm_call_observation(
    *,
    node: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    attempt: int,
) -> AsyncIterator[Any]:
    """
    一次真实 LLM 调用的 generation 观测（async context manager）。

    用法（llm_client.py）：
        async with llm_call_observation(node=..., model=..., messages=messages,
                                        tools=tools, attempt=attempt) as observation:
            resp = await client.chat.completions.create(**kwargs)
            ...  # 成功后用 update_llm_call_observation 补 output/usage

    语义：
    - 未启用 → yield None（调用方跳过更新即可，零开销）；
    - 调用抛异常 → 出块前自动记 ERROR 级 span 再原样抛给上层；
    - 时长 = 进入块到出块（真实调用耗时）。
    """
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        observation = client.start_observation(
            name=f"chat.{node}",
            as_type="generation",
            model=model,
            model_parameters={"temperature": get_settings().llm_temperature},
            input={"messages": messages, "tools": tools},
            metadata={"node": node, "attempt": attempt},
        )
    except Exception as exc:  # noqa: BLE001 —— 观测创建失败也绝不影响主链路
        logger.warning("langfuse_trace_failed", extra={"error": str(exc)})
        yield None
        return
    try:
        yield observation
    except Exception as exc:  # noqa: BLE001 —— 失败尝试也要留痕（重试排障）
        try:
            observation.update(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        except Exception:  # noqa: BLE001 —— 更新失败不再掩盖原始异常
            pass
        raise
    finally:
        try:
            observation.end()
        except Exception:  # noqa: BLE001
            pass


def update_llm_call_observation(
    observation: Any,
    *,
    output: str | None,
    usage: dict[str, int],
) -> None:
    """
    调用成功后补记输出与 token 用量（契约：永不抛出）。

    usage 是 llm_client 的 {prompt, completion, total} 语义，
    这里转换成 Langfuse 的 {input, output, total} 语义。
    """
    if observation is None:
        return
    try:
        observation.update(
            output={"content": output},
            usage_details={
                "input": usage.get("prompt", 0),
                "output": usage.get("completion", 0),
                "total": usage.get("total", 0),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_trace_failed", extra={"error": str(exc)})


def flush_langfuse() -> None:
    """冲刷未上报的 trace（uvicorn shutdown 钩子可调用；失败静默）。"""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception:  # noqa: BLE001
        pass


def reset_langfuse() -> None:
    """测试辅助：重置单例（config 变化后重建 / 测试隔离）。"""
    global _client, _init_attempted
    _client = None
    _init_attempted = False
