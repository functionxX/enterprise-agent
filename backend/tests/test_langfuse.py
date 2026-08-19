"""
test_langfuse.py — Langfuse tracing 接入测试（全 mock，不碰真实 SDK 网络）

【测试策略】
验证 Langfuse 接入（设计决策 16"增强依赖"哲学的落地）的三个核心性质
+ API include_contexts 契约：
1. 默认关闭：未启用时观测层零开销（yield None），chat() 行为与
   未接入前完全一致——40 项既有测试就是这点的回归保障
2. 启用后：真实 LLM 调用被包进 generation 观测（name=chat.{node}、
   input 含 messages/tools、output 与 usage_details 成功回填、调用
   end 收尾）——mock 掉 start_observation 的对象层，不碰真实 SDK
3. 增强依赖容错：start_observation 抛异常 → 调用照常成功；
   调用失败 → 观测记 ERROR 级 span 后原样抛出（trace 里能看到重试）
4. API 契约：include_contexts=true 时响应携带 Agent 实际依据的
   上下文原文（SQL 查询 + 返回行）；默认 false 不携带（响应轻量）

【mock 手法】
- settings：monkeypatch app.monitoring.langfuse.get_settings → 启用
  的 Settings（不建真实 Langfuse 客户端）
- 观测对象：FakeLangfuse / FakeObservation 记录调用参数
- OpenAI 客户端：FakeCompletions 直接返回伪响应（无网络）
"""

import pytest

from app.agent.llm_client import LLMClient, LLMClientError
from app.agent.tools import TOOL_IMPLEMENTATIONS
from app.config import Settings, get_settings
from app.monitoring import langfuse as lf_mod
from app.monitoring.langfuse import (
    flush_langfuse,
    get_langfuse,
    llm_call_observation,
    reset_langfuse,
    update_llm_call_observation,
)


# =============================================================
# 观测层 fake
# =============================================================

class FakeObservation:
    """记录 update/end 调用的伪观测对象。"""

    def __init__(self, name: str):
        self.name = name
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


class FakeLangfuse:
    """记录 start_observation 调用的伪 Langfuse 客户端。"""

    def __init__(self, *, raising: bool = False):
        self.started: list[tuple[dict, FakeObservation]] = []
        self.raising = raising

    def start_observation(self, **kwargs):
        if self.raising:
            raise RuntimeError("langfuse service down")
        obs = FakeObservation(kwargs["name"])
        self.started.append((kwargs, obs))
        return obs

    def flush(self):
        pass


# =============================================================
# OpenAI 客户端层 fake
# =============================================================

class _FakeMessage:
    content = "供应商 A 风险等级为 high。"
    tool_calls = None


class _FakeChoice:
    message = _FakeMessage()


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50
    total_tokens = 150


class _FakeResp:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, *, exc: Exception | None = None):
        self.exc = exc

    async def create(self, **kwargs):
        if self.exc:
            raise self.exc
        return _FakeResp()


def _fake_openai(client: LLMClient, *, exc: Exception | None = None) -> None:
    from types import SimpleNamespace
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(exc=exc))
    )


def _enable_langfuse(monkeypatch, fake: FakeLangfuse) -> None:
    """启用 Langfuse 并把单例预置为 fake（不触发真实 SDK 导入/构造）。"""
    monkeypatch.setattr(
        lf_mod, "get_settings",
        lambda: Settings(langfuse_enabled=True,
                         langfuse_public_key="pk-test",
                         langfuse_secret_key="sk-test"),
    )
    monkeypatch.setattr(lf_mod, "_client", fake)
    monkeypatch.setattr(lf_mod, "_init_attempted", True)


@pytest.fixture(autouse=True)
def _clean_langfuse():
    reset_langfuse()
    yield
    reset_langfuse()


# =============================================================
# 核心性质 1：默认关闭 → 零开销
# =============================================================

def test_disabled_by_default_returns_none():
    """默认配置（langfuse_enabled=False）→ 观测层直接 None，无任何副作用。"""
    assert get_settings().langfuse_enabled is False
    assert get_langfuse() is None
    # update 对 None 是无操作
    update_llm_call_observation(None, output="x", usage={"prompt": 1})
    assert lf_mod._client is None  # 未启用时不应构造任何客户端


async def test_disabled_observation_yields_none():
    """未启用时 llm_call_observation 直接 yield None，调用主链路不受扰。"""
    async with llm_call_observation(
        node="planner", model="m", messages=[], tools=None, attempt=0
    ) as observation:
        assert observation is None


# =============================================================
# 核心性质 2：启用后真实调用被完整观测
# =============================================================

async def test_enabled_records_generation_with_usage(monkeypatch):
    fake = FakeLangfuse()
    _enable_langfuse(monkeypatch, fake)

    client = LLMClient(get_settings())
    _fake_openai(client)

    result = await client.chat(
        [{"role": "user", "content": "供应商 A 风险等级？"}],
        node="planner",
    )

    # 观测层记录了完整调用链：start（name/model/input/metadata）→ update（output/usage）→ end
    assert len(fake.started) == 1
    start_kwargs, obs = fake.started[0]
    assert start_kwargs["name"] == "chat.planner"
    assert start_kwargs["as_type"] == "generation"
    assert start_kwargs["model"] == get_settings().llm_model
    assert start_kwargs["metadata"] == {"node": "planner", "attempt": 0}
    assert obs.updates[0]["output"] == {"content": "供应商 A 风险等级为 high。"}
    # llm_client 的 {prompt, completion, total} 被转换为 Langfuse 的 {input, output, total}
    assert obs.updates[0]["usage_details"] == {"input": 100, "output": 50, "total": 150}
    assert obs.ended

    # 主链路返回值不受观测影响（token 统计与之前完全一致）
    assert result["content"] == "供应商 A 风险等级为 high。"
    assert result["usage"] == {"prompt": 100, "completion": 50, "total": 150}


# =============================================================
# 核心性质 3：增强依赖容错
# =============================================================

async def test_trace_failure_does_not_break_call(monkeypatch):
    """观测层自身挂掉（start_observation 抛异常）→ 调用照常成功。"""
    fake = FakeLangfuse(raising=True)
    _enable_langfuse(monkeypatch, fake)

    client = LLMClient(get_settings())
    _fake_openai(client)

    result = await client.chat([{"role": "user", "content": "hi"}], node="planner")
    assert result["content"] == "供应商 A 风险等级为 high。"


async def test_failed_call_marks_error_span_then_rethrows(monkeypatch):
    """调用失败 → 观测记 ERROR 级 span（含异常信息），再原样抛给上层。"""
    fake = FakeLangfuse()
    _enable_langfuse(monkeypatch, fake)

    client = LLMClient(get_settings())
    _fake_openai(client, exc=RuntimeError("connection reset"))

    with pytest.raises(LLMClientError):
        await client.chat([{"role": "user", "content": "hi"}], node="planner")

    _, obs = fake.started[0]
    error_updates = [u for u in obs.updates if u.get("level") == "ERROR"]
    assert error_updates
    assert "connection reset" in error_updates[0]["status_message"]
    assert obs.ended


# =============================================================
# API 契约：include_contexts
# =============================================================

class ScriptedLLMClient(LLMClient):
    """按 node 名脚本化响应（与 test_hitl 同款，队列耗尽收敛）。"""

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = {k: list(v) for k, v in responses.items()}

    def _pop(self, node: str):
        queue = self.responses.get(node)
        if not queue:
            return {"content": None, "tool_calls": None, "usage": _usage()}
        script = queue.pop(0) if len(queue) > 1 else queue[0]
        return dict(script)

    async def chat(self, messages, *, node="unknown", tools=None, response_format=None):
        return self._pop(node)

    async def chat_json(self, messages, *, node="unknown"):
        import json
        result = self._pop(node)
        if isinstance(result.get("content"), str):
            result["content"] = json.loads(result["content"])
        return result


def _usage(prompt=100, completion=50):
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}


def _sql_script() -> dict[str, list[dict]]:
    return {
        "intent_analyzer": [
            {"content": '{"intent_type": "data_query", "entities": {}, "core_question": "q"}',
             "usage": _usage()},
        ],
        "planner": [
            {"content": '{"task_plan": [{"step": 1, "goal": "查风险", "tool": "execute_sql", "hint": "h"}]}',
             "usage": _usage()},
        ],
        "tool_executor": [
            {"content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "execute_sql",
                                          "arguments": '{"query": "SELECT risk_level FROM suppliers WHERE name = \'A\'"}'}}],
             "usage": _usage()},
        ],
        "router": [
            {"content": '{"decision": "finish", "reason": "够了", "next_action": ""}',
             "usage": _usage()},
        ],
        "response_generator": [
            {"content": "供应商 A 风险等级为 high。", "usage": _usage()},
        ],
    }


def _patch_tools(monkeypatch):
    async def fake_execute_sql(query):
        return {"columns": ["risk_level"], "rows": [["high"]], "total_rows": 1,
                "truncated": False, "truncation_note": None}
    monkeypatch.setitem(TOOL_IMPLEMENTATIONS, "execute_sql", fake_execute_sql)


async def test_api_include_contexts_contract(monkeypatch):
    """include_contexts=true → 响应带 SQL 上下文原文；默认 false → 不带。"""
    from app.agent.graph import build_graph
    from app.api.agent import chat as chat_route
    from app.database.schemas import ChatRequest

    fake = ScriptedLLMClient(_sql_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)
    _patch_tools(monkeypatch)
    monkeypatch.setattr("app.api.agent.get_graph", lambda enable_hitl=False: build_graph())

    response = await chat_route(ChatRequest(
        query="供应商 A 的风险等级？", include_contexts=True
    ))
    assert response.answer
    assert response.contexts, "include_contexts=true 应返回上下文原文"
    assert response.contexts[0].startswith("[SQL]")
    assert "SELECT risk_level" in response.contexts[0]
    assert "high" in response.contexts[0]  # 返回行进了上下文（faithfulness 评分依据）

    # 默认关闭：响应保持轻量
    response2 = await chat_route(ChatRequest(query="供应商 A 的风险等级？"))
    assert response2.contexts == []
