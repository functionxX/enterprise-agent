"""
state.py — Agent State Schema（LangGraph 图的"数据面"）

【为什么 state 用 TypedDict 而不是 dataclass（面试点）】
LangGraph 的节点签名是 node(state) -> partial_state，
框架负责把返回的部分状态合并回主状态。TypedDict 给出字段级
类型检查，同时保持 dict 的合并语义——这是 LangGraph 的官方惯例。

【State 设计的核心思想：审计轨迹（面试高频）】
State 不只是节点间传数据的容器，它是 Agent 一次运行完的"黑匣子"：
- tool_results   —— 每次 tool 调用的输入输出，完整调用链
- errors         —— 每个节点的失败记录（带 source_node，不中断流程）
- iteration_count—— 真实循环了几轮（Router 兜底的依据）
- token_usage    —— 成本累计（API 直接透出，成本可观测）
API 响应里的 sources / tools_used / iterations / token_usage 全部从 state 派生——
【面试话术】"Agent 的回答可追溯，因为 state 就是这次推理的完整审计日志。"

【errors 为什么是 list[dict] 而不是 list[str]（设计决策 14）】
错误必须带 source_node（哪个节点出的错），Router 才能做选择性回退：
IntentAnalyzer 出错回 IntentAnalyzer，Planner 出错回 Planner。
如果 errors 只是字符串列表，Router 拿到错误也不知道该回到哪个节点。
"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Agent 图状态。total=False：节点只返回自己更新的字段。"""

    # ---- 输入 ----
    user_query: str                    # 本轮用户问题
    conversation_history: list[dict]   # 历史对话摘要（Redis 会话，见 api/agent.py）
    session_id: str | None             # 会话 ID

    # ---- IntentAnalyzer 产出 ----
    intent: dict[str, Any]             # {"intent_type": ..., "entities": ..., "core_question": ...}

    # ---- Planner 产出 ----
    task_plan: list[dict[str, Any]]    # [{"step": 1, "goal": ..., "tool": ..., "hint": ...}]
    current_step: int                  # 当前执行到第几步（从 0 起）

    # ---- 执行轨迹（审计日志）----
    tool_results: list[dict[str, Any]]       # 每次 tool 调用的输入/输出（含来源与时间）
    retrieved_documents: list[dict[str, Any]]  # RAG 检索到的 chunk（text/source/score）
    errors: list[dict[str, Any]]             # [{"source_node": ..., "message": ..., "attempt": n}]
    retry_count: int                         # 工具执行失败重试计数
    iteration_count: int                     # 工具执行迭代轮数（Router 硬兜底依据）
    tools_used: list[str]                    # tool 名称调用序列（允许重复，如实反映调用链）

    # ---- Router / 循环控制 ----
    router_decision: str | None        # "continue" | "finish"（Router 写入）
    llm_messages: list[dict[str, Any]] # ToolExecutor 的多轮 tool-calling 消息序列
    last_tool_requested: str | None    # 本次请求的 tool 名（图路由依据）
    executor_says_done: bool           # 执行者认为信息收集完成（无 tool 可调）
    # ToolExecutor 写给 rag_retriever 的待执行 tool_call。
    # 【真实踩坑】必须声明在 schema 里——LangGraph 会静默丢弃
    # TypedDict schema 之外的键，未声明时 rag_retriever 拿不到
    # tool_call_id，回填 tool 消息用了兜底 id "unknown"，
    # 与 assistant 消息的 tool_calls 不匹配 → LLM API 400。
    pending_tool_call: dict[str, Any] | None

    # ---- 成本与时间 ----
    token_usage: dict[str, int]        # {"prompt": n, "completion": n, "total": n}
    started_at: float                  # 请求开始时间戳（超时兜底依据）

    # ---- 输出 ----
    final_answer: str                  # ResponseGenerator 产出
    warnings: list[str]                # 非致命告警（降级/超时/截断提示）

    # ---- HITL 人工审批（设计决策 19） ----
    # 开启时，敏感工具（nodes.HITL_APPROVAL_TOOLS）执行前 interrupt() 挂起，
    # 人工批准后由 API 用 Command(resume=...) 续跑。API 从请求参数注入。
    hitl_enabled: bool

    # ---- Supervisor 多 Agent 模式（supervisor.py） ----    # 子 Agent 工具白名单：worker 子图入口写入，tool_executor 据此过滤工具
    # 【设计决策：为什么是 state 字段而不是编译期参数？】
    # worker 子图是编译产物（无状态单例），工具限制是"本次运行"的属性，
    # 走 state 通道既能让同一份节点代码服务不同 worker，又能保持子图可复用。
    allowed_tools: list[str] | None
    # 当前正在执行的 worker 名（worker_report / supervisor 依据）
    current_worker: str | None
    # Supervisor 派发审计日志：[{"round": n, "next": "sql_agent", "reason": ...}]
    supervisor_trace: list[dict[str, Any]]
    # 各 worker 的发现摘要：[{"agent": "sql_agent", "summary": "..."}]
    # supervisor 依据摘要决定是否还要派另一个 worker，synthesizer 依据它补充回答
    worker_reports: list[dict[str, Any]]
