"""
llm_client.py — LLM 调用客户端（DeepSeek，OpenAI 兼容协议）

【为什么不用 langchain 的 ChatOpenAI 而是直接 openai SDK（面试高频）】
LangGraph 只管图编排，LLM 调用是普通函数——直接 openai SDK 的好处：
1. 依赖更薄：langchain-openai 会拉进一整套抽象；
2. 行为可控：重试、超时、token 统计、结构化输出解析全部自己掌握，
   面试时能讲清每一行；
3. DeepSeek/硅基流动都兼容 OpenAI 协议，一套代码两个供应商通用
   （LLM 走 DeepSeek，Embedding/Rerank 走硅基流动，只是 base_url 不同）。

【重试策略（面试点：指数退避）】
LLM 调用失败分两类：
- 网络抖动/5xx：可重试。1 次重试 + 指数退避（0.5s → 1s）。
  为什么只重试 1 次？Agent 一次请求串行 8~12 次 LLM 调用，
  每次重试都会放大总延迟；且重试次数再多，模型服务宕机时也没用——
  不如快速失败，让上层节点记录 error 并降级。
- 4xx（如 401 key 错、400 参数错）：不重试，重试必然失败，直接抛。

【token 统计为什么重要（面试点）】
Agent 的成本不可预测——同样一个问题，模型可能查 1 次 SQL 也可能查 5 次。
每次调用累加 usage 到 state.token_usage，最终 API 响应透出。
企业场景里这就是成本中心要的账单粒度：单次请求多少钱。
"""

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.monitoring.langfuse import llm_call_observation, update_llm_call_observation
from app.monitoring.middleware import track_llm_call

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """LLM 调用失败（网络/服务端错误，或结构化输出解析失败）。"""
    pass


class LLMClient:
    """DeepSeek Chat 客户端封装。

    【为什么是单例？】
    AsyncOpenAI 内部维护连接池，每次请求 new 一个会反复建连。
    lru_cache 保证进程内只有一个实例（FastAPI 单进程多协程天然复用）。
    """

    def __init__(self, settings: Settings):
        # max_retries=0：关闭 SDK 内置重试——重试策略我们自己掌控，
        # 避免 SDK 静默重试导致重复计费且行为不可观测
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            max_retries=0,
        )
        self._settings = settings

    async def chat(
        self,
        messages: list[dict],
        *,
        node: str = "unknown",
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """
        一次 LLM 调用（带 1 次指数退避重试）。

        返回：
        {
            "content": str | None,      # 文本输出（tool 模式下可能为 None）
            "tool_calls": list | None,  # OpenAI 格式 tool_calls
            "usage": {"prompt": n, "completion": n, "total": n},
        }
        """
        settings = self._settings
        last_error: Exception | None = None

        for attempt in range(settings.llm_max_retries + 1):
            if attempt > 0:
                # 指数退避：0.5s → 1s。轻量抖动不必加 jitter（单进程 demo）
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))
            try:
                kwargs: dict[str, Any] = {
                    "model": settings.llm_model,
                    "messages": messages,
                    "temperature": settings.llm_temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                if response_format:
                    kwargs["response_format"] = response_format

                # Langfuse tracing（增强依赖）：把真实调用包进 generation 观测，
                # 时长=真实耗时；失败尝试自动记 ERROR 级 span；未启用时 yield None
                # 零开销。观测层的任何异常都不会影响主链路（见 monitoring/langfuse.py）
                async with llm_call_observation(
                    node=node,
                    model=settings.llm_model,
                    messages=messages,
                    tools=tools,
                    attempt=attempt,
                ) as observation:
                    resp = await self._client.chat.completions.create(**kwargs)
                    choice = resp.choices[0]
                    message = choice.message

                    usage = {
                        "prompt": resp.usage.prompt_tokens if resp.usage else 0,
                        "completion": resp.usage.completion_tokens if resp.usage else 0,
                        "total": resp.usage.total_tokens if resp.usage else 0,
                    }
                    track_llm_call(node)  # Prometheus 指标：各节点的 LLM 调用数
                    update_llm_call_observation(observation, output=message.content, usage=usage)
                return {
                    "content": message.content,
                    "tool_calls": (
                        [tc.model_dump() for tc in message.tool_calls]
                        if message.tool_calls
                        else None
                    ),
                    "usage": usage,
                }
            except Exception as exc:  # noqa: BLE001 —— 网络错误/超时/5xx 统一处理
                # 4xx 客户端错误不重试：key 错/参数错，重试必然失败。
                # 记录响应体：排查请求格式问题（如 tool 消息协议不合规）的关键线索
                if getattr(exc, "status_code", None) and 400 <= exc.status_code < 500:
                    body = ""
                    try:
                        body = str(getattr(exc, "response", None) and exc.response.text or "")[:500]
                    except Exception:  # noqa: BLE001
                        pass
                    raise LLMClientError(
                        f"LLM 客户端错误({exc.status_code}): {body}"
                    ) from exc
                last_error = exc
                logger.warning(
                    "llm_call_retry",
                    extra={"node": node, "attempt": attempt + 1, "error": str(exc)},
                )

        raise LLMClientError(f"LLM 调用失败（已重试 {settings.llm_max_retries} 次）: {last_error}") from last_error

    async def chat_json(
        self,
        messages: list[dict],
        *,
        node: str = "unknown",
    ) -> dict[str, Any]:
        """
        要求 LLM 输出 JSON 并可靠解析。

        【为什么需要 _parse_json 而不是直接 json.loads（面试点）】
        LLM 输出不是编译器输出：常带 ```json 围栏、前后缀说明文字、
        偶尔还有格式小错误。解析必须容忍这些噪声——
        先剥围栏，再从第一个 { 到最后一个 } 截取，最后才 json.loads。
        解析失败向上抛 LLMClientError，由节点层记录到 state.errors。
        """
        result = await self.chat(
            messages,
            node=node,
            response_format={"type": "json_object"},
        )
        content = result["content"] or "{}"
        result["content"] = _parse_json(content)
        return result


def _parse_json(text: str) -> dict:
    """容忍 LLM 输出噪声的 JSON 解析。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # 剥除 markdown 代码围栏
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    # 截取第一个 { 到最后一个 }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMClientError(f"LLM 输出中没有 JSON: {text[:200]}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"JSON 解析失败: {exc}") from exc


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """全局单例（进程级缓存）。"""
    global _client
    if _client is None:
        _client = LLMClient(get_settings())
    return _client


def reset_llm_client() -> None:
    """测试辅助：重置单例（config 变化后重建）。"""
    global _client
    _client = None
