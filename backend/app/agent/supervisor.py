"""
supervisor.py — Supervisor 多 Agent 协作架构（设计决策：多 Agent 版本）

【为什么需要 Supervisor 模式？（面试核心考点：单 Agent vs 多 Agent）】
单 Agent 条件循环图（graph.py）在"任务边界清晰、工具数量少"的场景已经够用，
但有两个结构性局限：
1. 上下文串味：SQL 查询与知识库检索共用一条 llm_messages 链，检索到的长文
   会挤占后续 SQL 推理的注意力，反之亦然——专业化信息不该共享一个"脑子"。
2. 责任不分层：单 Agent 里 Router 既管"要不要继续查"（任务内循环），
   又隐式承担"查数据还是查标准"（任务间调度），两种决策耦合在一起。
Supervisor 模式把系统拆成三个角色：
- supervisor：LLM 调度器——根据用户问题与子 Agent 报告，决定派谁/收尾
- sql_agent：结构化数据专家（只有 execute_sql，独立上下文）
- rag_agent：非结构化知识专家（只有 search_knowledge，独立上下文）
- 汇总（response_generator）：合并两路证据生成最终回答

【拓扑】
                 ┌──────────────────────────────────────┐
                 ▼                                      │
START → supervisor ──(next=sql_agent)──→ sql_worker ────┤
                 │   (next=rag_agent)──→ rag_worker ────┤
                 │   (next=finish)    ─→ response_generator → END
每个 worker 内部是完整的条件循环图（意图→规划→执行⇄路由→汇报），
跑完回到 supervisor——supervisor 基于 worker 的发现摘要决定再派谁或收尾。

【面试追问："子 Agent 会不会无限互相调用？"】
不会：1) supervisor 硬兜底——派发轮数 >= max_supervisor_rounds 强制收尾；
2) 每个 worker 最多派一次（完成过的直接排除）；3) LLM 输出非法值收敛到 finish。
与单 Agent 的 max_iterations 兜底同哲学：LLM 调度 + 代码兜底。

【面试追问："worker 与主图是什么关系？"】
worker 是编译好的 LangGraph 子图，作为节点嵌进 supervisor 图——
supergraph 的 state 与子图共享同一 schema（AgentState），
工具白名单（allowed_tools）由 worker 入口节点写入 state，
tool_executor 读取过滤——同一份节点代码服务两种模式，零 fork。
"""

from typing import Any

from langgraph.graph import END, StateGraph

from app.agent.graph import _hitl_saver, _route_after_tool_executor
from app.agent.nodes import (
    intent_analyzer_node,
    planner_node,
    rag_retriever_node,
    response_generator_node,
    router_node,
    supervisor_node,
    tool_executor_node,
    worker_report_node,
)
from app.agent.state import AgentState


async def _init_worker(state: AgentState) -> dict[str, Any]:
    """
    worker 子图入口：写入工具白名单与当前 worker 名。

    【为什么白名单走 state 而不是编译期闭包？】
    编译期闭包也能做到（每个 worker 编译一份带不同 tools 的节点），
    但那样 tool_executor 的过滤逻辑会分裂成两份，白名单与 worker 的
    对应关系藏在 lambda 里不可见。state 通道让"谁有什么工具"成为
    运行时显式数据（supervisor_trace 同级的审计信息），测试也更好断言。
    """
    return {
        "allowed_tools": state.get("allowed_tools"),
        "current_worker": state.get("current_worker"),
    }


def _route_worker_after_router(state: AgentState) -> str:
    """worker 内部循环：Router 判 continue 回到执行器，finish 进汇报。"""
    if state.get("router_decision") == "continue":
        return "tool_executor"
    return "worker_report"


def build_worker_graph(worker_name: str, allowed_tools: list[str]) -> Any:
    """构建一个专业子 Agent 子图。

    拓扑：init → intent → planner → tool_executor ⇄ router → worker_report → END
    （与单 Agent 图同构，只是循环出口从 response_generator 换成 worker_report，
    并且工具被白名单限制为 allowed_tools。）

    【为什么要 init 节点？】
    LangGraph 子图不接收"入口参数"——白名单只能通过 state 注入。
    init 是子图里第一个 superstep，把本 worker 的配置写入共享 state，
    后续节点（tool_executor / worker_report）从 state 读取。
    """
    workflow = StateGraph(AgentState)

    async def _init(state: AgentState) -> dict[str, Any]:
        return {"allowed_tools": allowed_tools, "current_worker": worker_name}

    workflow.add_node("init", _init)
    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("router", router_node)
    workflow.add_node("worker_report", worker_report_node)

    workflow.set_entry_point("init")
    workflow.add_edge("init", "intent_analyzer")
    workflow.add_edge("intent_analyzer", "planner")
    workflow.add_edge("planner", "tool_executor")

    # 条件边 1：search_knowledge → rag_retriever（与单 Agent 图一致的分流）
    workflow.add_conditional_edges(
        "tool_executor",
        _route_after_tool_executor,
        {"router": "router", "rag_retriever": "rag_retriever"},
    )
    workflow.add_edge("rag_retriever", "router")

    # 条件边 2：worker 内部循环（finish → 汇报，而不是生成回答）
    workflow.add_conditional_edges(
        "router",
        _route_worker_after_router,
        {"tool_executor": "tool_executor", "worker_report": "worker_report"},
    )
    workflow.add_edge("worker_report", END)

    return workflow.compile()


def _route_after_supervisor(state: AgentState) -> str:
    """supervisor 决策 → 下一个节点：派 sql / 派 rag / 进汇总。"""
    next_agent = state.get("supervisor_trace", [{}])[-1].get("next", "finish")
    if next_agent == "sql_agent":
        return "sql_agent"
    if next_agent == "rag_agent":
        return "rag_agent"
    return "response_generator"


def build_supervisor_graph(enable_hitl: bool = False) -> Any:
    """构建 Supervisor 多 Agent 图。

    拓扑：supervisor → sql_agent / rag_agent（子图，执行完回到 supervisor）
    → response_generator（汇总）→ END。

    enable_hitl：与单 Agent 图同一约定——开启时挂 checkpointer，
    interrupt() 在 worker 子图内同样生效（子图共享 supergraph 的
    checkpoint，按 thread_id 恢复；见 graph.py 设计决策 19 注释）。
    """
    workflow = StateGraph(AgentState)

    # 两个专业 worker 子图（编译一次，全局复用）
    sql_worker = build_worker_graph("sql_agent", ["execute_sql"])
    rag_worker = build_worker_graph("rag_agent", ["search_knowledge"])

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("sql_agent", sql_worker)
    workflow.add_node("rag_agent", rag_worker)
    workflow.add_node("response_generator", response_generator_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "response_generator": "response_generator",
        },
    )
    # 子 Agent 执行完回到 supervisor，由它决定是否再派
    workflow.add_edge("sql_agent", "supervisor")
    workflow.add_edge("rag_agent", "supervisor")

    workflow.add_edge("response_generator", END)

    # 注意：recursion_limit 由调用方传入（api/agent.py 的 supervisor 模式用
    # 更大值：2 个 worker × 内部循环 + supervisor 往返，superstep 数远超单 Agent）
    return workflow.compile(checkpointer=_hitl_saver if enable_hitl else None)


# 应用级单例（与 graph.py 的 _graph 同一模式：编译一次，全局复用）
_supervisor_graphs: dict[bool, Any] = {}


def get_supervisor_graph(enable_hitl: bool = False):
    if enable_hitl not in _supervisor_graphs:
        _supervisor_graphs[enable_hitl] = build_supervisor_graph(enable_hitl)
    return _supervisor_graphs[enable_hitl]
