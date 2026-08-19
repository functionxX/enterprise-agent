"""
test_hitl.py — HITL 人工审批测试（全 mock，进程内 checkpointer）

【测试策略】
验证设计决策 19 的三个核心性质 + API 全链路：
1. 挂起：enable_hitl=True 的图在 execute_sql 执行前停下，返回的 state
   带 __interrupt__ 标记，pending 载荷含 SQL 全文（前端据此展示审批卡片）
   ——注意：langgraph 0.6+ 的 ainvoke 不抛 GraphInterrupt，改为在返回
   状态里带 __interrupt__ 键（0.6 的契约变更，API 层据此判断挂起）
2. 批准续跑：Command(resume={"approved": True}) 从挂起点恢复，
   SQL 正常执行、结果回填、图走完（工具链不因 HITL 断掉）
3. 拒绝续跑：Command(resume={"approved": False}) 不执行 SQL，
   拒绝理由进 errors，图正常收尾（有兜底回答）
4. API 全链路：chat 端点 hitl=true → pending_approval 响应 →
   带 approval 续跑 → 最终回答（含 400 校验：approval 必须带 hitl）

【mock 手法】
与 test_agent / test_supervisor 相同的脚本化 FakeLLMClient +
TOOL_IMPLEMENTATIONS 替换；checkpointer 用真实 InMemorySaver
（HITL 的核心机制就是要测它——挂起-恢复依赖 checkpoint）。
"""

import pytest
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.llm_client import LLMClient
from app.agent.state import AgentState
from app.database.schemas import ChatRequest


class ScriptedLLMClient(LLMClient):
    """按 node 名脚本化响应；队列耗尽后返回"无工具调用"（收敛语义）。"""

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses: dict[str, list[dict]] = {k: list(v) for k, v in responses.items()}

    def _pop(self, node: str):
        queue = self.responses.get(node)
        if not queue:
            return {"content": None, "tool_calls": None, "usage": _usage()}
        script = queue.pop(0) if len(queue) > 1 else queue[0]
        return dict(script)

    async def chat(self, messages, *, node="unknown", tools=None, response_format=None):
        return self._pop(node)

    async def chat_json(self, messages, *, node="unknown"):
        result = self._pop(node)
        if isinstance(result.get("content"), str):
            import json
            result["content"] = json.loads(result["content"])
        return result


def _usage(prompt=100, completion=50):
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}


def _initial_state() -> AgentState:
    import time
    return AgentState(
        user_query="供应商 A 的风险等级？",
        conversation_history=[],
        session_id="hitl-test",
        intent={}, task_plan=[], current_step=0,
        tool_results=[], retrieved_documents=[], errors=[], retry_count=0,
        iteration_count=0, tools_used=[], router_decision=None,
        llm_messages=[], last_tool_requested=None, pending_tool_call=None,
        executor_says_done=False,
        token_usage={"prompt": 0, "completion": 0, "total": 0},
        started_at=time.time(), final_answer="", warnings=[],
        allowed_tools=None, current_worker=None,
        supervisor_trace=[], worker_reports=[],
        hitl_enabled=True,
    )


def _sql_script() -> dict[str, list[dict]]:
    """意图 → 规划 → 请求 execute_sql → router finish → 生成回答。"""
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
    """替换 execute_sql 实现与 LLM 客户端（HITL 之外的一切走真实代码）。"""
    async def fake_execute_sql(query):
        return {"columns": ["risk_level"], "rows": [["high"]], "total_rows": 1,
                "truncated": False, "truncation_note": None}
    import app.agent.tools as tools_mod
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "execute_sql", fake_execute_sql)


# =============================================================
# 测试
# =============================================================

async def test_hitl_interrupts_with_pending_payload(monkeypatch):
    """
    核心性质 1：挂起。enable_hitl=True 的图在 execute_sql 执行前停下，
    ainvoke 返回的 state 带 __interrupt__ 标记，pending 载荷包含 SQL
    全文——前端据此渲染审批卡片。
    """
    fake = ScriptedLLMClient(_sql_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)
    _patch_tools(monkeypatch)

    graph = build_graph(enable_hitl=True)
    config = {"recursion_limit": 50, "configurable": {"thread_id": "hitl-test"}}

    result = await graph.ainvoke(_initial_state(), config=config)

    interrupts = list(result.get("__interrupt__", []))
    assert len(interrupts) == 1
    pending = interrupts[0].value
    assert pending["kind"] == "tool_approval"
    assert pending["tool"] == "execute_sql"
    assert "SELECT risk_level" in pending["query"]
    assert pending["message"]
    # 挂起时 tool_executor 的更新尚未合并（SQL 没执行、迭代未计数）
    assert result.get("iteration_count", 0) == 0
    assert result.get("tools_used", []) == []


async def test_hitl_approve_resumes_and_completes(monkeypatch):
    """
    核心性质 2：批准续跑。resume={"approved": true} 从挂起点恢复，
    SQL 正常执行、结果回填、图走完——工具链不因 HITL 断掉。
    """
    fake = ScriptedLLMClient(_sql_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)
    _patch_tools(monkeypatch)

    graph = build_graph(enable_hitl=True)
    config = {"recursion_limit": 50, "configurable": {"thread_id": "hitl-test"}}

    result = await graph.ainvoke(_initial_state(), config=config)
    assert result.get("__interrupt__")

    final = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert final["tools_used"] == ["execute_sql"]
    assert final["tool_results"][0]["status"] == "success"
    assert final["final_answer"]
    # 被批准的 SQL 没有进 errors
    assert not any("拒绝" in e["message"] for e in final["errors"])


async def test_hitl_reject_skips_execution_and_completes(monkeypatch):
    """
    核心性质 3：拒绝续跑。resume={"approved": false} 不执行 SQL，
    拒绝理由进 errors，图正常收尾（Router finish → 兜底回答）。
    """
    executed: list[str] = []

    async def fake_execute_sql(query):
        executed.append(query)
        return {"columns": [], "rows": [], "total_rows": 0,
                "truncated": False, "truncation_note": None}
    import app.agent.tools as tools_mod
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "execute_sql", fake_execute_sql)
    fake = ScriptedLLMClient(_sql_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)

    graph = build_graph(enable_hitl=True)
    config = {"recursion_limit": 50, "configurable": {"thread_id": "hitl-test"}}

    result = await graph.ainvoke(_initial_state(), config=config)
    assert result.get("__interrupt__")

    final = await graph.ainvoke(
        Command(resume={"approved": False, "reason": "查询范围过大，需先缩小"}),
        config=config,
    )

    # SQL 从未真正执行（HITL 阻断生效）
    assert executed == []
    assert "execute_sql" not in final["tools_used"]
    assert any("拒绝" in e["message"] for e in final["errors"])
    # 图仍然收敛：Router finish → 有兜底回答
    assert final["final_answer"]


# =============================================================
# API 全链路
# =============================================================

async def test_api_hitl_full_roundtrip(monkeypatch):
    """
    API 全链路：chat 端点 hitl=true → 返回 pending_approval（HTTP 语义：
    挂起不是故障）→ 带 approval + session_id 续跑 → 最终回答。
    同时验证 400 校验：approval 必须配 hitl。
    """
    from app.api.agent import chat as chat_route

    fake = ScriptedLLMClient(_sql_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)
    _patch_tools(monkeypatch)

    # 固定复用同一个 HITL 图实例（checkpoint 跨请求保留）
    hitl_graph = build_graph(enable_hitl=True)
    monkeypatch.setattr("app.api.agent.get_graph", lambda enable_hitl=False: hitl_graph)
    monkeypatch.setattr("app.api.agent.get_supervisor_graph", lambda enable_hitl=False: hitl_graph)

    # ---- 第 1 次：挂起 ----
    response = await chat_route(ChatRequest(query="供应商 A 的风险等级？", hitl=True))
    assert response.pending_approval is not None
    assert response.pending_approval["tool"] == "execute_sql"
    assert response.session_id  # 前端必须保存这个 id 用于续跑

    # ---- 校验：approval 不配 hitl → 400 ----
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as excinfo:
        await chat_route(ChatRequest(
            query="x", session_id=response.session_id, approval={"approved": True}
        ))
    assert excinfo.value.status_code == 400

    # ---- 第 2 次：批准续跑 ----
    final = await chat_route(ChatRequest(
        query="供应商 A 的风险等级？",
        session_id=response.session_id,
        hitl=True,
        approval={"approved": True},
    ))
    assert final.pending_approval is None
    assert final.answer
    assert final.tools_used == ["execute_sql"]
