"""
test_agent.py — Agent 流程测试（全 mock，不依赖外部服务）

【测试策略（面试点：Agent 系统怎么测？）】
Agent 测试的分层：
1. 单元测试（本文件）：mock LLM 客户端，验证图的控制流——
   循环是否发生、Router 是否按决策路由、迭代上限是否生效、
   SQL 防御是否拦截写操作。零外部依赖，CI 秒级跑完。
2. 集成测试（evaluation/）：真实 LLM + 真实数据库 + 标注用例，
   验证端到端任务完成率——evaluate.py。
单元测试验证"流程对不对"，集成测试验证"效果好不好"，缺一不可。

【mock 手法】
monkeypatch 替换 llm_client.get_llm_client 返回 FakeLLMClient，
后者按脚本队列依次返回预设响应——LLM 行为完全确定，
测试可以精确断言"第 N 次调用后图走到哪个节点"。
"""

import pytest

from app.agent.graph import build_graph
from app.agent.llm_client import LLMClient
from app.agent.state import AgentState
from app.agent.tools import ForbiddenSQLException, _validate_readonly_sql


# =============================================================
# Fake LLM 客户端：按脚本队列返回预设响应
# =============================================================

class FakeLLMClient(LLMClient):
    """
    脚本化 LLM：按节点名分队列，每个节点依次弹出自己的预设响应。

    【为什么按 node 分队列而不是全局单队列？】
    图的控制流里，某些节点可能因代码硬兜底而跳过 LLM 调用
    （如 Router 达到 max_iterations 直接 finish、不调 LLM）。
    全局单队列会导致"下一个调用者消费了上一个节点没消费的响应"，
    测试脚本错位。按 node 分队列后，每个节点消费自己的脚本，
    与真实控制流解耦，测试更稳。
    """

    def __init__(self, responses: dict[str, list[dict]]):
        # 每个 node 的响应队列；消费到最后一条时复用（模拟稳态 LLM）
        self.responses: dict[str, list[dict]] = {k: list(v) for k, v in responses.items()}
        self.calls: list[str] = []  # 记录每次调用的 node，供断言

    def _pop(self, node: str) -> dict:
        self.calls.append(node)
        queue = self.responses.get(node)
        if not queue:
            # 未脚本化的节点：返回保守的空响应
            return {"content": None, "tool_calls": None, "usage": _usage()}
        script = queue.pop(0) if len(queue) > 1 else queue[0]
        return dict(script)

    async def chat(self, messages, *, node="unknown", tools=None, response_format=None):
        return self._pop(node)

    async def chat_json(self, messages, *, node="unknown"):
        # chat_json 复用 chat，但把 content 当 JSON 解析——Fake 直接返回 dict
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
    )


# 完整的"一次 tool 循环"脚本（按节点分队列）：
# intent → planner → executor(调 execute_sql) → router(continue)
# → executor(无 tool，完成) → router(finish) → generator
def _single_loop_script() -> dict[str, list[dict]]:
    return {
        "intent_analyzer": [
            {"content": '{"intent_type": "data_query", "entities": {}, "core_question": "查采购数据"}',
             "usage": _usage()},
        ],
        "planner": [
            {"content": '{"task_plan": [{"step": 1, "goal": "查采购数据", "tool": "execute_sql", "hint": "查表"}]}',
             "usage": _usage()},
        ],
        "tool_executor": [
            # executor #1：请求 execute_sql
            {"content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "execute_sql",
                                          "arguments": '{"query": "SELECT * FROM suppliers"}'}}],
             "usage": _usage()},
            # executor #2：不再调工具
            {"content": "信息收集完成", "tool_calls": None, "usage": _usage()},
        ],
        "router": [
            # router #1：继续
            {"content": '{"decision": "continue", "reason": "数据不够", "next_action": "再查"}',
             "usage": _usage()},
            # router #2：结束
            {"content": '{"decision": "finish", "reason": "够了", "next_action": ""}',
             "usage": _usage()},
        ],
        "response_generator": [
            {"content": "根据数据，供应商共 N 家，高风险 5 家。", "usage": _usage()},
        ],
    }


# =============================================================
# 测试：图控制流
# =============================================================

@pytest.mark.asyncio
async def test_graph_loops_through_tool_executor_twice(monkeypatch):
    """
    核心测试：验证条件循环真的发生——
    ToolExecutor 被访问 2 次（Router 判 continue 后回到执行器）。
    这验证了"条件循环图而非线性链"（设计决策 1）。
    """
    fake = FakeLLMClient(_single_loop_script())
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)
    monkeypatch.setattr("app.agent.graph.build_graph", build_graph)
    # 替换 tool 实现：避免真实数据库
    async def fake_execute_sql(query):
        return {"columns": ["name", "amount"], "rows": [["华芯半导体", 120000]],
                "total_rows": 1, "truncated": False, "truncation_note": None}
    monkeypatch.setitem(__import__("app.agent.tools", fromlist=["TOOL_IMPLEMENTATIONS"]).TOOL_IMPLEMENTATIONS,
                        "execute_sql", fake_execute_sql)

    graph = build_graph()
    state = await graph.ainvoke(_initial_state("有多少供应商"), config={"recursion_limit": 50})

    # ToolExecutor 进入 2 次 → iteration_count == 2
    assert state["iteration_count"] == 2
    # 调用链：execute_sql 一次
    assert state["tools_used"] == ["execute_sql"]
    # 最终回答生成
    assert state["final_answer"]
    # node 调用序列：intent → planner → executor → router → executor → router → generator
    assert fake.calls == [
        "intent_analyzer", "planner",
        "tool_executor", "router", "tool_executor", "router",
        "response_generator",
    ]


@pytest.mark.asyncio
async def test_max_iterations_guard_forces_finish(monkeypatch):
    """
    验证硬兜底：LLM 永远判 continue 时，max_iterations 强制终止。
    （设计决策 4：LLM 路由 + 代码兜底）
    """
    settings = __import__("app.config", fromlist=["get_settings"]).get_settings()
    # 构造"永远 continue"的脚本：executor 永远请求 execute_sql，router 永远 continue。
    # 注意：Router 在 iteration >= max_iterations 时走代码硬兜底、不调 LLM，
    # 所以 router 队列不需要与 executor 严格等长——按节点分队列天然免疫错位。
    executor_call = {"content": None,
                     "tool_calls": [{"id": "c", "type": "function",
                                     "function": {"name": "execute_sql",
                                                  "arguments": '{"query": "SELECT 1"}'}}],
                     "usage": _usage()}
    router_continue = {"content": '{"decision": "continue", "reason": "不够", "next_action": ""}',
                       "usage": _usage()}
    fake = FakeLLMClient({
        "intent_analyzer": [
            {"content": '{"intent_type": "data_query", "entities": {}, "core_question": "q"}', "usage": _usage()},
        ],
        "planner": [
            {"content": '{"task_plan": [{"step": 1, "goal": "g", "tool": "execute_sql", "hint": "h"}]}', "usage": _usage()},
        ],
        "tool_executor": [dict(executor_call) for _ in range(settings.max_iterations + 2)],
        "router": [dict(router_continue) for _ in range(settings.max_iterations)],
        "response_generator": [{"content": "最终回答（基于已有数据）", "usage": _usage()}],
    })
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)
    async def fake_execute_sql(query):
        return {"columns": [], "rows": [], "total_rows": 0, "truncated": False, "truncation_note": None}
    monkeypatch.setitem(__import__("app.agent.tools", fromlist=["TOOL_IMPLEMENTATIONS"]).TOOL_IMPLEMENTATIONS,
                        "execute_sql", fake_execute_sql)

    graph = build_graph()
    state = await graph.ainvoke(_initial_state("q"), config={"recursion_limit": 100})

    # 迭代数不超过 max_iterations（Router 在触顶时强制 finish）
    assert state["iteration_count"] <= settings.max_iterations
    # 图正常终止并生成了回答
    assert state["final_answer"]
    # warnings 里包含强制终止说明
    assert any("最大迭代次数" in w for w in state["warnings"])


# =============================================================
# 测试：SQL 只读防御（第 2 层，设计决策 3）
# =============================================================

@pytest.mark.parametrize("bad_sql", [
    "INSERT INTO suppliers (name) VALUES ('x')",
    "UPDATE suppliers SET rating = 5",
    "DELETE FROM purchase_orders",
    "DROP TABLE suppliers",
    "SELECT 1; DROP TABLE suppliers;",   # 分号后的写操作
    "SELECT * FROM suppliers; DELETE FROM invoices",  # 多语句
    "TRUNCATE suppliers",
    "GRANT SELECT ON suppliers TO x",
])
def test_validate_rejects_write_sql(bad_sql):
    """写操作 SQL 必须被第 2 层正则防御拦截。"""
    with pytest.raises(ForbiddenSQLException):
        _validate_readonly_sql(bad_sql)


@pytest.mark.parametrize("good_sql", [
    "SELECT * FROM suppliers",
    "SELECT name, rating FROM suppliers WHERE risk_level = 'high'",
    "SELECT s.name, COUNT(o.id) FROM suppliers s JOIN purchase_orders o ON o.supplier_id = s.id GROUP BY s.name",
    "WITH recent AS (SELECT * FROM purchase_orders WHERE order_date > CURRENT_DATE - 90) SELECT COUNT(*) FROM recent",
    "SELECT * FROM suppliers WHERE name LIKE '%华%'",
])
def test_validate_accepts_readonly_sql(good_sql):
    """合法只读查询（含 JOIN / CTE / 子查询）必须放行。"""
    _validate_readonly_sql(good_sql)  # 不抛异常即通过


def test_validate_ignores_keywords_in_strings_and_comments():
    """危险词出现在字符串字面量或注释里不应误杀（剥离后再扫描）。"""
    _validate_readonly_sql("SELECT 'INSERT 是危险词' AS note FROM suppliers")
    _validate_readonly_sql("SELECT * FROM suppliers -- UPDATE 说明文档\nWHERE id = 1")


# =============================================================
# 测试：ToolExecutor 的 SQL 自愈回填
# =============================================================

@pytest.mark.asyncio
async def test_sql_error_is_fed_back_to_llm(monkeypatch):
    """
    验证自愈循环的关键机制：SQL 失败后，错误信息回填进 llm_messages，
    LLM 下一轮能看到错误并重写 SQL（循环图 vs 线性链的分水岭）。
    直接调用 tool_executor_node，脚本第一项即 executor 的 tool_call 响应。
    """
    executor_tool_call = {
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "execute_sql",
                                     "arguments": '{"query": "SELECT * FROM suppliers"}'}}],
        "usage": _usage(),
    }
    fake = FakeLLMClient({"tool_executor": [executor_tool_call]})
    monkeypatch.setattr("app.agent.nodes.get_llm_client", lambda: fake)

    # execute_sql 抛错——模拟 LLM 生成了一条非法 SQL
    async def broken_execute_sql(query):
        raise Exception('syntax error at or near "FROM"')
    monkeypatch.setitem(__import__("app.agent.tools", fromlist=["TOOL_IMPLEMENTATIONS"]).TOOL_IMPLEMENTATIONS,
                        "execute_sql", broken_execute_sql)

    from app.agent.nodes import tool_executor_node
    state = _initial_state("q")
    state["task_plan"] = [{"step": 1, "goal": "g", "tool": "execute_sql", "hint": "h"}]

    result = await tool_executor_node(state)
    # 错误被记入 errors（带 source_node）
    assert any(e["source_node"] == "tool_executor" for e in result["errors"])
    # 错误信息回填进对话序列（LLM 能看到）
    feedback = [m for m in result["llm_messages"] if m.get("role") == "tool"]
    assert feedback and "SQL 执行失败" in feedback[-1]["content"]
