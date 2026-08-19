"""
test_supervisor.py — Supervisor 多 Agent 模式测试（全 mock，不依赖外部服务）

【测试策略】
验证 Supervisor 架构的三个核心性质（与单 Agent 测试互补）：
1. 调度控制流：supervisor 依次派 sql_agent → rag_agent → finish，
   子 Agent 执行后回到 supervisor（不是线性走完）。
2. 工具白名单隔离：sql_agent 的 executor 只收到 execute_sql 定义，
   rag_agent 只收到 search_knowledge——专业化隔离是拆 Agent 的意义。
3. 硬兜底：supervisor 重复派发同一 worker 时收敛到 finish。

【mock 手法】
与 test_agent.py 相同的 FakeLLMClient，但额外记录每次 chat 收到的
tools 参数（RecorderFakeLLMClient）——白名单断言依赖它。
"""

import pytest

from app.agent.llm_client import LLMClient
from app.agent.state import AgentState
from app.agent.supervisor import build_supervisor_graph


class RecorderFakeLLMClient(LLMClient):
    """脚本化 LLM + 记录每次调用收到的 tools 参数（白名单断言依据）。"""

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses: dict[str, list[dict]] = {k: list(v) for k, v in responses.items()}
        self.calls: list[str] = []
        self.tools_seen: dict[str, list[list[str] | None]] = {}  # node -> 每次调用的工具名列表

    def _pop(self, node: str, tools=None):
        self.calls.append(node)
        self.tools_seen.setdefault(node, []).append(
            [t["function"]["name"] for t in tools] if tools else None
        )
        queue = self.responses.get(node)
        if not queue:
            return {"content": None, "tool_calls": None, "usage": _usage()}
        script = queue.pop(0) if len(queue) > 1 else queue[0]
        return dict(script)

    async def chat(self, messages, *, node="unknown", tools=None, response_format=None):
        return self._pop(node, tools)

    async def chat_json(self, messages, *, node="unknown"):
        result = self._pop(node)
        if isinstance(result.get("content"), str):
            import json
            result["content"] = json.loads(result["content"])
        return result


def _usage(prompt=100, completion=50):
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}


def _initial_state(query: str) -> AgentState:
    import time
    return AgentState(
        user_query=query,
        conversation_history=[],
        session_id="test",
        intent={}, task_plan=[], current_step=0,
        tool_results=[], retrieved_documents=[], errors=[], retry_count=0,
        iteration_count=0, tools_used=[], router_decision=None,
        llm_messages=[], last_tool_requested=None, executor_says_done=False,
        token_usage={"prompt": 0, "completion": 0, "total": 0},
        started_at=time.time(), final_answer="", warnings=[],
        allowed_tools=None, current_worker=None,
        supervisor_trace=[], worker_reports=[],
    )


# 完整脚本：supervisor 派 sql → sql worker 一轮循环后汇报 →
# supervisor 派 rag → rag worker 一轮循环后汇报 → supervisor finish → 汇总
def _two_worker_script() -> dict[str, list[dict]]:
    return {
        "supervisor": [
            {"content": '{"next": "sql_agent", "reason": "需要结构化数据", "handoff_note": ""}',
             "usage": _usage()},
            {"content": '{"next": "rag_agent", "reason": "需要对照标准", "handoff_note": ""}',
             "usage": _usage()},
            {"content": '{"next": "finish", "reason": "信息已足够", "handoff_note": ""}',
             "usage": _usage()},
        ],
        "intent_analyzer": [
            {"content": '{"intent_type": "data_query", "entities": {}, "core_question": "q"}',
             "usage": _usage()},
            {"content": '{"intent_type": "knowledge_query", "entities": {}, "core_question": "q"}',
             "usage": _usage()},
        ],
        "planner": [
            {"content": '{"task_plan": [{"step": 1, "goal": "查数据", "tool": "execute_sql", "hint": "h"}]}',
             "usage": _usage()},
            {"content": '{"task_plan": [{"step": 1, "goal": "查标准", "tool": "search_knowledge", "hint": "h"}]}',
             "usage": _usage()},
        ],
        "tool_executor": [
            # sql_agent 的 executor：请求 execute_sql（白名单内唯一工具）
            {"content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "execute_sql",
                                          "arguments": '{"query": "SELECT 1"}'}}],
             "usage": _usage()},
            # rag_agent 的 executor：请求 search_knowledge
            {"content": None,
             "tool_calls": [{"id": "c2", "type": "function",
                             "function": {"name": "search_knowledge",
                                          "arguments": '{"query": "评估标准", "top_k": 3}'}}],
             "usage": _usage()},
        ],
        "router": [
            {"content": '{"decision": "finish", "reason": "够了", "next_action": ""}',
             "usage": _usage()},
            {"content": '{"decision": "finish", "reason": "够了", "next_action": ""}',
             "usage": _usage()},
        ],
        "worker_report": [
            {"content": '{"summary": "SQL 查到 50 家供应商", "gaps": ""}', "usage": _usage()},
            {"content": '{"summary": "检索到评估标准条文", "gaps": ""}', "usage": _usage()},
        ],
        "response_generator": [
            {"content": "汇总：50 家供应商，对照标准给出结论。", "usage": _usage()},
        ],
    }


# =============================================================
# 测试：调度控制流
# =============================================================

@pytest.mark.asyncio
async def test_supervisor_dispatches_both_workers_then_finishes(monkeypatch):
    """
    核心测试：supervisor 依次派发两个子 Agent，最后收尾汇总。
    断言：
    - supervisor_trace 记录了 sql → rag → finish 三次决策
    - 两个 worker 都产生了报告（worker_reports 两条）
    - 两个工具的调用都被记录（跨 worker 的审计轨迹共享同一 state）
    """
    fake = RecorderFakeLLMClient(_two_worker_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)

    async def fake_execute_sql(query):
        return {"columns": ["n"], "rows": [[50]], "total_rows": 1,
                "truncated": False, "truncation_note": None}
    async def fake_search(query, top_k=5):
        return [{"text": "标准条文", "source": "quality_standard.txt",
                 "chunk_index": 0, "score": 0.9, "mode": "mock"}]
    import app.agent.tools as tools_mod
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "execute_sql", fake_execute_sql)
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "search_knowledge", fake_search)
    # rag_retriever 里的延迟导入也要替换
    import app.agent.nodes as nodes_mod
    monkeypatch.setattr("app.rag.retriever.search", fake_search)

    graph = build_supervisor_graph()
    state = await graph.ainvoke(_initial_state("供应商风险如何，对照标准给结论"),
                                config={"recursion_limit": 200})

    assert [t["next"] for t in state["supervisor_trace"]] == [
        "sql_agent", "rag_agent", "finish"
    ]
    assert [r["agent"] for r in state["worker_reports"]] == ["sql_agent", "rag_agent"]
    assert state["tools_used"] == ["execute_sql", "search_knowledge"]
    assert state["final_answer"]


@pytest.mark.asyncio
async def test_worker_tool_allowlist_isolation(monkeypatch):
    """
    工具白名单隔离：sql_agent 的 executor 只收到 execute_sql 定义，
    rag_agent 的 executor 只收到 search_knowledge 定义。
    （专业化隔离是拆 Agent 的意义——sql_agent 的上下文里不该有知识库工具的噪音。）
    """
    fake = RecorderFakeLLMClient(_two_worker_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)

    async def fake_execute_sql(query):
        return {"columns": [], "rows": [], "total_rows": 0,
                "truncated": False, "truncation_note": None}
    async def fake_search(query, top_k=5):
        return [{"text": "x", "source": "s.txt", "chunk_index": 0,
                 "score": 0.9, "mode": "mock"}]
    import app.agent.tools as tools_mod
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "execute_sql", fake_execute_sql)
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "search_knowledge", fake_search)
    monkeypatch.setattr("app.rag.retriever.search", fake_search)

    graph = build_supervisor_graph()
    await graph.ainvoke(_initial_state("q"), config={"recursion_limit": 200})

    executor_calls = fake.tools_seen.get("tool_executor", [])
    assert len(executor_calls) == 2
    # 第 1 次是 sql_agent（白名单只有 execute_sql），第 2 次是 rag_agent
    assert executor_calls[0] == ["execute_sql"]
    assert executor_calls[1] == ["search_knowledge"]


@pytest.mark.asyncio
async def test_supervisor_does_not_redispatch_completed_worker(monkeypatch):
    """
    硬兜底：LLM 尝试重复派发已完成 worker 时收敛到 finish
    （防"子 Agent 无限互相调用"——面试必问的可靠性问题）。
    """
    script = _two_worker_script()
    # 第二个 supervisor 决策改为重复派 sql_agent（已完成）——应被收敛为 finish，
    # 之后的 rag_agent 派发决策永远不会发生（fake 脚本被跳过也无妨）
    script["supervisor"][1] = {"content": '{"next": "sql_agent", "reason": "再查一次"}',
                               "usage": _usage()}
    fake = RecorderFakeLLMClient(script)
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)

    async def fake_execute_sql(query):
        return {"columns": [], "rows": [], "total_rows": 0,
                "truncated": False, "truncation_note": None}
    async def fake_search(query, top_k=5):
        return [{"text": "x", "source": "s.txt", "chunk_index": 0,
                 "score": 0.9, "mode": "mock"}]
    import app.agent.tools as tools_mod
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "execute_sql", fake_execute_sql)
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "search_knowledge", fake_search)
    monkeypatch.setattr("app.rag.retriever.search", fake_search)

    graph = build_supervisor_graph()
    state = await graph.ainvoke(_initial_state("q"), config={"recursion_limit": 200})

    # 第二次决策被收敛为 finish：trace 里没有第二个 sql_agent
    nexts = [t["next"] for t in state["supervisor_trace"]]
    assert nexts == ["sql_agent", "finish"]
    # 只有一个 worker 执行过
    assert [r["agent"] for r in state["worker_reports"]] == ["sql_agent"]
    # 图正常收尾并生成回答
    assert state["final_answer"]
