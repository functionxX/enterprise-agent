"""
graph.py — LangGraph 条件循环图定义（设计决策 1：本项目最重要的架构点）

拓扑（与 README 的 ASCII 图一致）：

                        ┌──────────────────────────────────┐
                        │         循环体（条件边）           │
                        ▼                                  │
START → IntentAnalyzer → Planner → ToolExecutor → Router ──┘
                              │            │
                              │            ▼ (tool = search_knowledge)
                              │        RAGRetriever → Router
                              ▼ (没有更多 tool 要调 / 硬兜底触发)
                        ResponseGenerator → END

【为什么必须是条件循环图而不是线性链（面试核心考点）】
线性链（Intent→Plan→Execute→Answer）的问题：
执行一次 tool 后无论结果如何都直接生成回答——数据不够、SQL 报错、
检索结果不相关，全都无能为力。这是"带工具的聊天机器人"，不是 Agent。

条件循环图的核心价值：
1. 动态决策：Router 每轮基于"新获得的数据"重新判断是否继续——
   第一轮查 SQL 发现供应商 A 有 3 张逾期发票 → 决定第二轮查该供应商
   近 3 个月订单明细 → 发现交付率骤降 → 第三轮搜知识库对照评估标准。
   这个"发现→追查"的链路在线性链里不可能存在。
2. 自愈能力：SQL 报错信息回填对话 → Router 判 continue → LLM 修正 SQL 重试。
3. 受控循环：max_iterations 硬兜底保证循环必然终止（面试追问
   "怎么保证不无限循环？"的标准答案：LLM 路由 + 代码双重兜底）。

【LangGraph 实现要点（面试追问）】
- add_conditional_edges：Router 的返回值字符串 → 下一节点名的映射。
  Router 返回 "continue" 时回到 tool_executor，返回 "finish" 时进 generator。
- 节点间共享 state：LangGraph 把 state 传给每个节点，节点返回 partial state
  合并回主状态——tool_results 的"追加"语义天然支持完整调用链审计。
- compile() 产出的图对象是无状态的：并发安全由"每请求独立 state"保证
  （设计决策 17：graph 是全局单例，state 是每请求新建）。
- recursion_limit：LangGraph 自身的防死循环机制（默认 25 个 superstep），
  我们设 50 留足余量；业务层的 max_iterations 是真正的循环控制。
"""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    intent_analyzer_node,
    planner_node,
    rag_retriever_node,
    response_generator_node,
    router_node,
    tool_executor_node,
)
from app.agent.state import AgentState


def _route_after_tool_executor(state: AgentState) -> str:
    """
    条件边 1：ToolExecutor 之后的去向。
    - 本次请求的 tool 是 search_knowledge → RAGRetriever（完整检索 pipeline）
    - 其他情况（execute_sql 已执行完 / 无 tool 可调）→ Router
    """
    if state.get("last_tool_requested") == "search_knowledge":
        return "rag_retriever"
    return "router"


def _route_after_router(state: AgentState) -> str:
    """
    条件边 2（核心循环）：Router 之后的去向。
    - continue → 回到 ToolExecutor，开始下一轮工具迭代
    - finish   → ResponseGenerator 生成最终回答
    """
    return "tool_executor" if state.get("router_decision") == "continue" else "response_generator"


# HITL 用 checkpointer（设计决策 19 的前提）：interrupt() 挂起时把运行中
# 状态存进 checkpoint，恢复时按 thread_id（= session_id）找回现场。
# InMemorySaver 是进程内线程安全实现——demo / 单机部署够用；
# 生产环境换 PostgresSaver（langgraph-checkpoint-postgres，存储层在
# 开源贡献 #8620 中验证过 PostgreSQL / CockroachDB 双库兼容）。
# 【面试追问："为什么 checkpointer 是 HITL 的前提？"】
# interrupt() 会中止整次运行并抛 GraphInterrupt——没有 checkpoint 保存
# 现场，"恢复"就无从谈起。挂起-恢复依赖持久化的运行状态。
_hitl_saver = InMemorySaver()


def build_graph(enable_hitl: bool = False):
    """构建并编译 Agent 图。

    【为什么用函数而不是模块级常量？】
    图对象应在应用启动时构建一次（开销在 compile），
    函数式构建让单测可以自由替换节点实现（mock LLM）。

    【enable_hitl 为什么是编译期参数而不是运行期参数？】
    开启 HITL 的图必须挂 checkpointer（interrupt 的前提），而挂上
    checkpointer 后每次调用都要求 thread_id——默认图没有这个约束。
    两个编译变体让"开 HITL"成为显式的 opt-in 模式：
    不开的图行为与原来逐字节一致（默认零影响，见 api/agent.py）。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("router", router_node)
    workflow.add_node("response_generator", response_generator_node)

    workflow.set_entry_point("intent_analyzer")

    # ---- 线性段：意图 → 规划 → 执行 ----
    workflow.add_edge("intent_analyzer", "planner")
    workflow.add_edge("planner", "tool_executor")

    # ---- 条件边 1：执行后按 tool 类型分流 ----
    workflow.add_conditional_edges(
        "tool_executor",
        _route_after_tool_executor,
        {"router": "router", "rag_retriever": "rag_retriever"},
    )
    workflow.add_edge("rag_retriever", "router")

    # ---- 条件边 2：核心循环（Router 决定 continue / finish）----
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {"tool_executor": "tool_executor", "response_generator": "response_generator"},
    )

    workflow.add_edge("response_generator", END)

    # recursion_limit=50：循环图每个 superstep 都会计数，
    # 6 次迭代 × 2~3 个 superstep + 线性段 ≈ 25，50 留足余量。
    # 真正的循环上限由 router_node 的 max_iterations 控制（业务语义），
    # recursion_limit 只是框架层的最后保险。
    return workflow.compile(checkpointer=_hitl_saver if enable_hitl else None)


# 应用级单例：编译一次，全局复用（图无状态，并发安全）。
# 按 enable_hitl 缓存两个变体——HITL 版挂 checkpointer，默认版保持原行为。
_graph: dict[bool, Any] = {}


def get_graph(enable_hitl: bool = False):
    if enable_hitl not in _graph:
        _graph[enable_hitl] = build_graph(enable_hitl)
    return _graph[enable_hitl]
