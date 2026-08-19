"""
nodes.py — LangGraph 图节点实现（5 个节点 + 2 个路由函数）

节点清单（与 graph.py 的拓扑对应）：
1. intent_analyzer_node   —— 意图分析（LLM，JSON 输出）
2. planner_node           —— 任务规划（LLM，JSON 输出）
3. tool_executor_node     —— 工具执行（LLM tool-calling + 实际执行 execute_sql）
4. rag_retriever_node     —— RAG 检索（Query Rewrite → 混合检索 → Rerank）
5. router_node            —— 路由判断（代码硬兜底 + LLM 主判断）
6. response_generator_node—— 最终回答生成（LLM，基于全部证据）

【为什么 IntentAnalyzer 和 Planner 拆成两个节点（设计决策 2，面试必考）】
两者是顺序依赖：意图错了，计划必然错。拆开的意义是"选择性重试"：
Intent 阶段出错 → Router 只回退 IntentAnalyzer；Planner 出错 → 只回退 Planner。
合并成一个节点时，LLM 失败你根本不知道要回退到哪一步——
【面试话术】"企业系统追求稳定可恢复，不会为了省一次 LLM 调用
牺牲故障定位能力。节点边界 = 故障回退边界。"

【错误处理哲学：错误收集，不中断流程（设计决策 4）】
任何节点失败都不会让整个请求 500：
- LLM 调用失败 → 记入 state.errors（带 source_node）→ 降级继续
- SQL 执行失败 → 错误信息写回 llm_messages，LLM 下一轮自愈重写 SQL
- 最终确实无法恢复 → Router 判 finish，Generator 基于已有数据回答并说明缺失
这样设计的理由：Agent 请求本身就是长链路（8~12 次 LLM 调用），
任何单点失败直接抛 500 会让可用性趋近于零；"部分结果 + 明确说明"
比"没有结果"有价值得多。

【自愈循环（面试亮点）】
SQL 写错了怎么办？—— 这是循环图 vs 线性链的分水岭：
线性链：执行节点报错 → 整个流程失败。
循环图：报错信息 append 回 llm_messages → Router 判 continue
→ 回到 ToolExecutor → LLM 看到错误信息，重写 SQL 再试。
这就是"执行 tool 后发现信息不够/有错，循环回去再查"的具体形态。
"""

import json
import logging
import time
from typing import Any

from langgraph.types import interrupt

from app.agent.llm_client import LLMClientError, get_llm_client
from app.agent.state import AgentState
from app.agent.tools import (
    ForbiddenSQLException,
    TOOL_DEFINITIONS,
    TOOL_IMPLEMENTATIONS,
)
from app.config import get_settings
from app.monitoring.middleware import track_agent_iteration, track_tool_call

logger = logging.getLogger(__name__)

# =============================================================
# 各节点的 System Prompt
# =============================================================

INTENT_SYSTEM_PROMPT = """你是企业采购智能 Agent 的意图分析模块。
分析用户问题，输出 JSON（不要输出其他内容）：
{
  "intent_type": "data_query | knowledge_query | hybrid | chitchat",
  "entities": {"suppliers": ["涉及的供应商名"], "time_range": "涉及的时间范围", "categories": ["涉及的产品类别"]},
  "core_question": "用户真正想问什么（一句话）"
}
分类规则：
- data_query：需要查结构化业务数据（供应商/订单/发票）才能回答
- knowledge_query：需要查知识库（政策/规则/标准）才能回答
- hybrid：数据与知识两者都需要
- chitchat：与采购业务无关的闲聊"""

PLANNER_SYSTEM_PROMPT = """你是企业采购智能 Agent 的规划模块。
根据意图分析结果制定任务执行计划。输出 JSON（不要输出其他内容）：
{
  "task_plan": [
    {"step": 1, "goal": "本步骤目标（做什么）", "tool": "execute_sql 或 search_knowledge", "hint": "执行提示（查询条件/检索关键词）"}
  ],
  "note": "总体思路（一句话）"
}
规划规则：
- 简单问题 1~2 步，复杂分析 3~4 步，不要超过 4 步
- 事实类数据（数字、名单、金额）→ execute_sql
- 标准/政策/规则对照 → search_knowledge
- 先查数据后查标准，或先查标准后查数据，按问题逻辑排顺序
- hint 要具体：SQL 步骤给出过滤条件（表名、日期范围、状态），检索步骤给出关键词"""

TOOL_EXECUTOR_SYSTEM_PROMPT = """你是企业采购智能 Agent 的执行模块。
根据任务计划调用工具收集信息，工具说明见工具定义。
执行规则：
1. 每次只调用一个工具
2. SQL 执行失败时，根据错误信息修正 SQL 后重试
3. 查询结果被截断（truncated=true）时，加 WHERE 条件缩小范围重查
4. 已收集足够信息后，停止调用工具，直接说明"信息收集完成"
5. 禁止编造任何数据"""

ROUTER_SYSTEM_PROMPT = """你是 Agent 流程的路由判断模块。
根据任务计划与已收集信息，判断是否继续调用工具。输出 JSON（不要输出其他内容）：
{"decision": "continue 或 finish", "reason": "判断理由", "next_action": "若 continue，下一步做什么"}
判断规则：
- 任务计划中还有步骤未完成 → continue
- 关键数据缺失、且还有工具可查 → continue
- 已收集信息足以回答用户问题 → finish
- 工具连续失败、重试无望 → finish（用已有信息生成回答，回答中说明数据缺口）
- SQL 结果被截断且缩小范围也无效时 → finish（说明数据量限制）"""

GENERATOR_SYSTEM_PROMPT = """你是企业采购智能 Agent 的回答生成模块。
基于工具执行结果与知识库检索结果生成最终回答。
要求：
1. 只能基于提供的数据回答，禁止编造不存在的供应商、订单或数字
2. 引用数据时注明来源（表名 / 文档名）
3. 数据被截断或工具失败导致信息不完整时，在回答中明确说明
4. 用中文回答，结构化输出（标题/列表/表格均可）
5. 对照知识库标准分析时：先引用标准原文要点，再给出数据对比与结论
6. 有数据异常（如交付率骤降、发票逾期、高风险供应商订单激增）时主动指出"""


# =============================================================
# HITL 人工审批（设计决策 19）
# =============================================================

# 需要人工审批的工具集合。当前只读 Agent 里"敏感操作"就是执行 SQL——
# 即使是只读 SQL，查什么表、用什么条件也值得人先看一眼（企业数据合规：
# 数据团队对生产库的查询普遍有人工审批流，Agent 只是把审批流从
# "邮件/工单"变成"对话内确认"）。
# 【为什么是代码常量而不是 config？】
# 审批范围是业务规则不是部署配置：换环境不会改审批策略；
# 且它服务于安全边界——配置项容易被随手关掉，代码常量不会。
HITL_APPROVAL_TOOLS: frozenset[str] = frozenset({"execute_sql"})


# =============================================================
# 工具函数
# =============================================================

def _add_usage(state: AgentState, usage: dict | None) -> dict[str, int]:
    """把一次 LLM 调用的 usage 累加到 state.token_usage。

    【为什么 token_usage 要进 state？】
    单次请求的 LLM 成本 = 各节点调用的 token 之和。
    state 是唯一贯穿全流程的对象，放这里最终由 API 透出，
    用户能看到"这次查询花了多少 token"——成本透明是 Agent 系统的必修课。
    """
    if not usage:
        return state.get("token_usage", {"prompt": 0, "completion": 0, "total": 0})
    current = state.get("token_usage", {"prompt": 0, "completion": 0, "total": 0})
    return {
        "prompt": current["prompt"] + usage.get("prompt", 0),
        "completion": current["completion"] + usage.get("completion", 0),
        "total": current["total"] + usage.get("total", 0),
    }


def _append_error(state: AgentState, source_node: str, message: str) -> list[dict]:
    """追加一条带 source_node 的错误记录（设计决策 14：Router 选择性回退的依据）。"""
    return state.get("errors", []) + [{
        "source_node": source_node,
        "message": message[:500],
        "attempt": state.get("retry_count", 0),
    }]


def _format_tool_results_for_llm(
    tool_results: list[dict],
    max_chars: int = 6000,
    sample_rows: int = 3,
) -> str:
    """
    把 tool_results 序列化为 LLM 可读文本（给 Router / Generator 用）。

    【为什么需要这个函数（设计决策 9 的延伸）】
    tool_results 里存的是结构化 dict，直接塞进 prompt 会有 JSON 语法
    干扰和 token 浪费。这里转成紧凑文本，并设总长度上限，
    防止多轮循环后上下文无限膨胀。

    sample_rows 的取值差异（面试点：同一份数据，不同节点不同视角）：
    - Router 只要"数据够不够、质量好不好" → 3 行样例足够判断；
    - Generator 要基于数据作答 → 需要完整行（sample_rows=15），
      否则答案会基于样例而非全量数据（E2E 测试实际踩过的坑：
      "仅获取到 3 笔订单明细，其余 10 笔未返回"——数据明明查全了，
      是格式化函数只给 LLM 看了 3 行）。
    """
    lines: list[str] = []
    total = 0
    for i, tr in enumerate(tool_results, 1):
        head = f"[{i}] tool={tr.get('tool')} status={tr.get('status')}"
        if tr.get("tool") == "execute_sql":
            detail = (
                f"query={tr.get('query', '')[:200]}"
                f" rows={tr.get('total_rows', 0)}"
                f" truncated={tr.get('truncated', False)}"
            )
            lines.append(f"{head}\n    {detail}")
            if tr.get("error"):
                lines.append(f"    error={tr['error'][:200]}")
            elif tr.get("rows"):
                # 样例行让 LLM 感知数据结构；sample_rows 控制给多少行
                lines.append(
                    f"    sample_rows={json.dumps(tr['rows'][:sample_rows], ensure_ascii=False)[:2000]}"
                )
        else:  # search_knowledge
            docs = tr.get("documents", [])
            lines.append(f"{head} mode={tr.get('retrieval_mode', 'mock')} docs={len(docs)}")
            for d in docs[:3]:
                lines.append(f"    [{d.get('source')}#{d.get('chunk_index')}] {d.get('text', '')[:200]}")
        total = len("\n".join(lines))
        if total > max_chars:
            lines.append("...（tool 结果过长，已截断展示）")
            break
    return "\n".join(lines) if lines else "（尚无工具执行结果）"


# =============================================================
# 节点 1：IntentAnalyzer —— 意图分析
# =============================================================

async def intent_analyzer_node(state: AgentState) -> dict[str, Any]:
    """分析用户意图。LLM 失败时降级为 hybrid 意图（尽量尝试回答问题）。"""
    llm = get_llm_client()
    history = state.get("conversation_history", [])
    history_text = "\n".join(
        f"- 历史问题: {h.get('query', '')}（结论摘要: {h.get('summary', '')}）"
        for h in history[-3:]  # 意图分析只需最近 3 轮上下文
    ) or "（无历史对话）"

    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"历史对话：\n{history_text}\n\n当前用户问题：{state['user_query']}"},
    ]
    try:
        result = await llm.chat_json(messages, node="intent_analyzer")
        intent = result["content"]
        # 防御：LLM 可能返回缺字段的 JSON，补默认值
        intent.setdefault("intent_type", "hybrid")
        intent.setdefault("entities", {})
        intent.setdefault("core_question", state["user_query"])
        return {"intent": intent, "token_usage": _add_usage(state, result["usage"])}
    except Exception as exc:  # noqa: BLE001 —— LLM 失败/JSON 解析失败统一降级
        # 降级策略：意图识别失败不致命——按 hybrid 处理，让 Planner 直接规划
        logger.warning("intent_analyzer_failed", extra={"error": str(exc)})
        return {
            "intent": {
                "intent_type": "hybrid",
                "entities": {},
                "core_question": state["user_query"],
            },
            "errors": _append_error(state, "intent_analyzer", f"意图分析失败，降级为 hybrid：{exc}"),
        }


# =============================================================
# 节点 2：Planner —— 任务规划
# =============================================================

async def planner_node(state: AgentState) -> dict[str, Any]:
    """基于意图产出任务计划。失败时降级为"直接执行用户查询"的单步计划。"""
    llm = get_llm_client()
    intent = state.get("intent", {})
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"用户问题：{state['user_query']}\n"
            f"意图分析：{json.dumps(intent, ensure_ascii=False)}\n"
            "请制定任务执行计划。"
        )},
    ]
    try:
        result = await llm.chat_json(messages, node="planner")
        plan = result["content"].get("task_plan", [])
        if not plan:
            raise LLMClientError("Planner 返回空计划")
        return {
            "task_plan": plan,
            "current_step": 0,
            "token_usage": _add_usage(state, result["usage"]),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("planner_failed", extra={"error": str(exc)})
        return {
            "task_plan": [{
                "step": 1,
                "goal": "直接回答用户问题",
                "tool": "execute_sql",
                "hint": "根据用户问题选择合适的数据表查询",
            }],
            "current_step": 0,
            "errors": _append_error(state, "planner", f"规划失败，降级为单步计划：{exc}"),
        }


# =============================================================
# 节点 3：ToolExecutor —— 工具执行（LLM tool-calling）
# =============================================================

async def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """
    一次工具执行迭代：LLM 决定调哪个 tool → 执行（execute_sql 在本节点执行；
    search_knowledge 路由到 rag_retriever 节点执行完整检索 pipeline）。

    【为什么 search_knowledge 要单独走 rag_retriever 节点？】
    execute_sql 是单步操作（校验→执行→截断），一个节点内完成即可；
    search_knowledge 背后是完整 RAG pipeline（Query Rewrite → 混合检索 → Rerank），
    有自己的失败模式与重试语义，拆成独立节点让图结构如实反映系统结构——
    【面试话术】"图节点划分的标准是职责边界，不是代码行数。"

    【为什么一次只执行一个 tool？】
    1 次迭代 = 1 次 tool 调用，循环语义清晰（iteration_count 有意义），
    Router 每轮都能基于"新数据"重新决策——如果一次塞 3 个 tool，
    Router 就退化成摆设，循环图的价值（动态决策）就没了。
    """
    llm = get_llm_client()
    settings = get_settings()
    updates: dict[str, Any] = {}

    # 首次进入时初始化 tool-calling 消息序列
    llm_messages = state.get("llm_messages") or [
        {"role": "system", "content": TOOL_EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"用户问题：{state['user_query']}\n\n任务计划：\n"
            + "\n".join(
                f"步骤{s.get('step')}：{s.get('goal')}（tool: {s.get('tool')}，提示: {s.get('hint')}）"
                for s in state.get("task_plan", [])
            )
        )},
    ]

    # 迭代计数 + 指标（每次进入本节点 = 一轮工具执行迭代）
    iteration = state.get("iteration_count", 0) + 1
    track_agent_iteration()
    updates["iteration_count"] = iteration
    updates["current_step"] = state.get("current_step", 0) + 1

    # 硬兜底前哨：迭代数触顶时 Router 会终止，这里不再发起新的 LLM 调用
    if iteration > settings.max_iterations:
        updates["llm_messages"] = llm_messages
        updates["executor_says_done"] = True
        updates["errors"] = _append_error(
            state, "tool_executor", f"已达最大迭代次数 {settings.max_iterations}"
        )
        return updates

    # 【Supervisor 模式】worker 子图通过 state["allowed_tools"] 限制本 worker 的
    # 工具白名单（sql_agent 只见 execute_sql，rag_agent 只见 search_knowledge）。
    # 单 Agent 模式下 allowed_tools 为 None → 全部工具，行为与原来完全一致。
    allowed_tools = state.get("allowed_tools")
    available_tools = (
        [t for t in TOOL_DEFINITIONS if t["function"]["name"] in allowed_tools]
        if allowed_tools
        else TOOL_DEFINITIONS
    )

    try:
        result = await llm.chat(llm_messages, node="tool_executor", tools=available_tools)
    except LLMClientError as exc:
        # LLM 调用失败：无法继续取工具，置 done 让 Router 决定收尾
        logger.warning("tool_executor_llm_failed", extra={"error": str(exc)})
        updates["llm_messages"] = llm_messages
        updates["executor_says_done"] = True
        updates["errors"] = _append_error(state, "tool_executor", f"工具选择失败：{exc}")
        return updates

    updates["token_usage"] = _add_usage(state, result["usage"])
    tool_calls = result.get("tool_calls") or []

    if not tool_calls:
        # LLM 没有请求工具 → 认为信息收集完成
        updates["llm_messages"] = llm_messages + [
            {"role": "assistant", "content": result.get("content") or "信息收集完成"}
        ]
        updates["executor_says_done"] = True
        return updates

    # 只取第一个 tool_call（每次迭代一个工具，见 docstring 说明）
    call = tool_calls[0]
    fn_name = call["function"]["name"]
    try:
        fn_args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        fn_args = {}

    # assistant 消息（带 tool_calls）追加进对话序列——OpenAI tool-calling 协议要求
    assistant_msg = {"role": "assistant", "content": None, "tool_calls": tool_calls[:1]}
    messages_after = llm_messages + [assistant_msg]

    if fn_name not in TOOL_IMPLEMENTATIONS:
        # LLM 幻觉出未知 tool：记错 + 回填错误信息，让它下一轮纠正
        err_text = f"未知工具 {fn_name}，可用工具：{list(TOOL_IMPLEMENTATIONS)}"
        messages_after.append({"role": "tool", "tool_call_id": call["id"], "content": err_text})
        updates["llm_messages"] = messages_after
        updates["errors"] = _append_error(state, "tool_executor", err_text)
        return updates

    if fn_name == "execute_sql":
        # SQL 类工具在本节点内执行（含自愈回填）
        query = fn_args.get("query", "")
        tool_result: dict[str, Any] = {
            "tool": "execute_sql",
            "query": query,
            "status": "success",
        }

        # 【HITL 人工审批（设计决策 19，面试高频）】
        # 启用时，敏感工具执行前 interrupt() 挂起：整次图运行中止并把
        # 待审批信息随 GraphInterrupt 抛给调用方（API），checkpoint 保存
        # 现场；人工批准后 API 用 Command(resume=决定) 续跑，interrupt()
        # 的返回值就是人工的决定——节点从挂起点继续执行。
        # 【面试追问："interrupt 和普通 return 有什么区别？"】
        # return 之后图继续往下走；interrupt() 会中止运行、持久化状态，
        # 恢复时从挂起点接着跑——"Agent 停下来等人工"是 LangGraph
        # 原生的控制流原语，不是用轮询/消息队列模拟出来的。
        # 【为什么内联在本节点而不是单独审批节点？】
        # 审批通过后要走"执行 SQL → 回填结果 → Router 决策"的完整链路，
        # 内联可以复用同一段执行代码；单独节点反而要把执行逻辑拆两份。
        if state.get("hitl_enabled") and fn_name in HITL_APPROVAL_TOOLS:
            decision = interrupt({
                "kind": "tool_approval",
                "tool": "execute_sql",
                "tool_call_id": call["id"],
                "query": query,
                "message": f"Agent 请求执行只读 SQL：{query[:200]}",
            })
            if not (isinstance(decision, dict) and decision.get("approved")):
                # 拒绝：不执行 SQL，拒绝理由回填给 LLM——Router 判 continue 后
                # LLM 看到"被拒绝"会换查询思路或收尾，而不是硬重试同一条 SQL
                reason = (
                    decision.get("reason", "未说明")
                    if isinstance(decision, dict)
                    else "未说明"
                )
                messages_after.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": f"SQL 查询被人工拒绝。理由：{reason}。请更换查询思路或结束。",
                })
                updates["llm_messages"] = messages_after
                updates["errors"] = _append_error(
                    state, "tool_executor", f"SQL 被人工拒绝：{reason}"
                )
                return updates
        try:
            executed = await TOOL_IMPLEMENTATIONS["execute_sql"](query)
            track_tool_call("execute_sql")
            tool_result.update(executed)
            # 结果回填进对话（OpenAI 协议：tool 角色消息对应 tool_call_id）
            messages_after.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps({
                    "columns": executed.get("columns"),
                    "rows": executed.get("rows"),
                    "truncated": executed.get("truncated", False),
                    "truncation_note": executed.get("truncation_note"),
                }, ensure_ascii=False, default=str),
            })
        except (ForbiddenSQLException, Exception) as exc:  # noqa: BLE001
            # 【自愈循环关键】SQL 失败 → 错误回填进对话 + 记入 errors，
            # Router 判 continue 后，LLM 下一轮看到错误重写 SQL
            tool_result["status"] = "error"
            tool_result["error"] = str(exc)
            messages_after.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": f"SQL 执行失败：{exc}\n请修正 SQL 后重试。",
            })
            updates["errors"] = _append_error(state, "tool_executor", f"SQL 执行失败：{exc}")
            updates["retry_count"] = state.get("retry_count", 0) + 1
        updates["llm_messages"] = messages_after
        updates["tool_results"] = state.get("tool_results", []) + [tool_result]
        updates["tools_used"] = state.get("tools_used", []) + ["execute_sql"]
        return updates

    if fn_name == "search_knowledge":
        # 路由到 rag_retriever 节点执行完整检索 pipeline
        updates["llm_messages"] = messages_after
        updates["last_tool_requested"] = "search_knowledge"
        updates["pending_tool_call"] = {
            "id": call["id"],
            "name": "search_knowledge",
            "arguments": fn_args,
        }
        return updates

    return updates  # 理论不可达


# =============================================================
# 节点 4：RAGRetriever —— 知识库检索（完整 RAG pipeline）
# =============================================================

async def rag_retriever_node(state: AgentState) -> dict[str, Any]:
    """
    执行 search_knowledge tool：Query Rewrite → 混合检索 → Rerank。

    输入：state["pending_tool_call"]（ToolExecutor 写入的待执行调用）
    输出：检索结果写入 tool_results / retrieved_documents，并回填 llm_messages。
    """
    from app.rag.retriever import search  # 延迟导入：rag 包依赖 pgvector 等

    pending = state.get("pending_tool_call") or {}
    fn_args = pending.get("arguments", {})
    query = fn_args.get("query", state["user_query"])
    top_k = int(fn_args.get("top_k", 5))
    call_id = pending.get("id", "unknown")

    tool_result: dict[str, Any] = {
        "tool": "search_knowledge",
        "query": query,
        "top_k": top_k,
        "status": "success",
    }
    updates: dict[str, Any] = {}

    try:
        docs = await search(query, top_k=top_k)
        track_tool_call("search_knowledge")
        tool_result["documents"] = docs
        tool_result["retrieval_mode"] = docs[0].get("mode", "mock") if docs else "mock"
        updates["retrieved_documents"] = state.get("retrieved_documents", []) + docs
        # 回填对话序列：LLM 下一轮能看到检索结果。
        # 【真实踩坑】chunk 文本不能截断——chunk_size=500 本身已是上限，
        # 再截 300 会把关键规则条文（如"5 个工作日出具风险说明"）切掉，
        # 实测导致 Agent 反复重搜直到 max_iterations 触顶。
        messages = state.get("llm_messages", []) + [{
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({
                "documents": [
                    {"source": d.get("source"), "chunk_index": d.get("chunk_index"),
                     "text": d.get("text") or "", "score": d.get("score")}
                    for d in docs
                ]
            }, ensure_ascii=False),
        }]
        updates["llm_messages"] = messages
    except Exception as exc:  # noqa: BLE001
        logger.warning("rag_retrieve_failed", extra={"error": str(exc)})
        tool_result["status"] = "error"
        tool_result["error"] = str(exc)
        updates["errors"] = _append_error(state, "rag_retriever", f"知识库检索失败：{exc}")
        updates["llm_messages"] = state.get("llm_messages", []) + [{
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"知识库检索失败：{exc}",
        }]

    updates["tool_results"] = state.get("tool_results", []) + [tool_result]
    updates["tools_used"] = state.get("tools_used", []) + ["search_knowledge"]
    return updates


# =============================================================
# 节点 5：Router —— 路由判断（代码硬兜底 + LLM 主判断）
# =============================================================

async def router_node(state: AgentState) -> dict[str, Any]:
    """
    判断继续循环还是进入回答生成。

    【设计决策 4：代码硬兜底 + LLM 主判断（面试高频）】
    为什么不纯用 LLM？——LLM 可能"幻觉"：明明数据够了还说"需要再查"，
    循环到死。为什么不纯用代码？——代码只能判断"还有步骤没做完"，
    无法判断"已有数据质量够不够回答"。
    所以：代码负责"什么时候必须停"（硬兜底），LLM 负责"什么时候可以停"（主判断）。

    硬兜底（先执行，无条件优先）：
    1. iteration_count >= max_iterations → finish（防无限循环）
    2. elapsed >= agent_timeout_seconds → finish（防超时）
    3. executor_says_done（执行者认为信息够了）→ 大概率 finish，仍过 LLM 确认
    """
    llm = get_llm_client()
    settings = get_settings()
    updates: dict[str, Any] = {}

    elapsed = time.time() - state.get("started_at", time.time())
    iteration = state.get("iteration_count", 0)
    errors = state.get("errors", [])

    # ---- 硬兜底 1：迭代上限 ----
    if iteration >= settings.max_iterations:
        warnings = state.get("warnings", []) + [
            f"达到最大迭代次数 {settings.max_iterations}，强制结束（已执行 {iteration} 轮）"
        ]
        return {"router_decision": "finish", "warnings": warnings}

    # ---- 硬兜底 2：总超时 ----
    if elapsed >= settings.agent_timeout_seconds:
        warnings = state.get("warnings", []) + [
            f"请求超时（{elapsed:.0f}s），强制结束并返回已有结果"
        ]
        return {"router_decision": "finish", "warnings": warnings}

    # ---- LLM 主判断 ----
    plan_text = "\n".join(
        f"步骤{s.get('step')}：{s.get('goal')}（tool: {s.get('tool')}）"
        for s in state.get("task_plan", [])
    )
    error_text = "\n".join(
        f"- [{e.get('source_node')}] {e.get('message')}" for e in errors[-3:]
    ) or "（无错误）"
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"用户问题：{state['user_query']}\n\n"
            f"任务计划：\n{plan_text}\n\n"
            f"已执行迭代：{iteration} 轮\n"
            f"已用工具：{state.get('tools_used', [])}\n"
            f"执行者状态：{'信息收集完成' if state.get('executor_says_done') else '还在收集中'}\n"
            f"错误记录：\n{error_text}\n\n"
            f"工具执行结果摘要：\n{_format_tool_results_for_llm(state.get('tool_results', []))}"
        )},
    ]
    try:
        result = await llm.chat_json(messages, node="router")
        decision = result["content"].get("decision", "finish")
        decision = decision if decision in ("continue", "finish") else "finish"
        reason = result["content"].get("reason", "")
        next_action = result["content"].get("next_action", "")

        # 执行者说"够了"但 Router 要 continue？允许——LLM 可能发现数据被截断需要补查
        updates["router_decision"] = decision
        updates["token_usage"] = _add_usage(state, result["usage"])

        # continue 时把路由建议回填进对话，引导下一轮 ToolExecutor
        if decision == "continue":
            updates["llm_messages"] = state.get("llm_messages", []) + [{
                "role": "user",
                "content": f"【路由建议】{reason} 下一步：{next_action}",
            }]
        else:
            logger.info("router_finish", extra={"reason": reason, "iterations": iteration})
        return updates
    except LLMClientError as exc:
        # Router LLM 失败：保守收尾（finish）——已有数据生成回答优于无限等待
        logger.warning("router_llm_failed", extra={"error": str(exc)})
        return {
            "router_decision": "finish",
            "errors": _append_error(state, "router", f"路由判断失败，保守收尾：{exc}"),
        }


# =============================================================
# 节点 6：ResponseGenerator —— 最终回答生成
# =============================================================

async def response_generator_node(state: AgentState) -> dict[str, Any]:
    """基于全部证据生成最终回答（含来源与数据缺口说明）。"""
    llm = get_llm_client()
    # Generator 需要全量数据作答：sample_rows=15（MAX_ROWS=20 的截断上限内）
    tool_text = _format_tool_results_for_llm(state.get("tool_results", []), sample_rows=15)
    # chunk 全文（500 字上限）——截断会切掉关键条文，与 rag_retriever 回填同理
    doc_text = "\n".join(
        f"[{d.get('source')}#{d.get('chunk_index')}] {d.get('text', '')}"
        for d in state.get("retrieved_documents", [])[:6]
    ) or "（无知识库检索结果）"
    warning_text = "\n".join(f"- {w}" for w in state.get("warnings", [])) or "（无）"
    error_text = "\n".join(
        f"- [{e.get('source_node')}] {e.get('message')}" for e in state.get("errors", [])
    ) or "（无）"

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"用户问题：{state['user_query']}\n\n"
            f"工具执行结果：\n{tool_text}\n\n"
            f"知识库检索结果：\n{doc_text}\n\n"
            f"执行告警：\n{warning_text}\n"
            f"执行错误：\n{error_text}\n\n"
            "请生成最终回答。"
        )},
    ]
    try:
        result = await llm.chat(messages, node="response_generator")
        return {
            "final_answer": result["content"] or "（回答生成失败：LLM 返回空内容）",
            "token_usage": _add_usage(state, result["usage"]),
        }
    except LLMClientError as exc:
        # 生成失败：返回已有数据摘要 + 明确错误说明（决策 16：失败也有产出）
        logger.error("generator_failed", extra={"error": str(exc)})
        fallback = (
            f"回答生成失败（{exc}）。以下是本次已收集到的数据摘要：\n\n{tool_text[:2000]}"
        )
        return {
            "final_answer": fallback,
            "errors": _append_error(state, "response_generator", f"回答生成失败：{exc}"),
        }


# =============================================================
# Supervisor 多 Agent 模式的节点（supervisor.py 的图使用）
# =============================================================
#
# 【为什么 Supervisor 模式要拆成独立节点而不是复用 Router？】
# 职责不同：Router 回答的是"要不要再查一次工具"（任务内循环），
# Supervisor 回答的是"该派哪个子 Agent / 信息是否够收尾"（任务间调度）。
# 两者都是 LLM 路由，但决策依据、失败语义、兜底策略完全不同，
# 拆开才能各自独立演进（例如 Supervisor 未来加"并行派发"不影响单 Agent 图）。

SUPERVISOR_SYSTEM_PROMPT = """你是企业采购多 Agent 系统的 Supervisor（调度器）。
你手下有两个专业子 Agent：
- sql_agent：查结构化业务数据（供应商/采购订单/发票的数字、名单、统计）
- rag_agent：查知识库（供应商准入政策、采购流程规则、质量标准等非结构化知识）
根据用户问题与已完成子 Agent 的发现，决定下一步。输出 JSON（不要输出其他内容）：
{"next": "sql_agent | rag_agent | finish", "reason": "决策理由（一句话）", "handoff_note": "给子 Agent 的执行提示（若派发）"}
调度规则：
- 需要结构化数据（数字、名单、金额、趋势）→ sql_agent
- 需要政策/规则/标准对照 → rag_agent
- 两者都需要 → 先派数据后派标准，或按问题逻辑顺序
- 每个子 Agent 最多派一次；都完成后或信息已足够 → finish
- 子 Agent 返回的发现已能完整回答用户问题 → finish（不要重复派发）"""


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Supervisor：多 Agent 调度器。

    【设计决策：LLM 调度 + 代码硬兜底（与 Router 同一哲学）】
    硬兜底（先执行）：
    1. 派发轮数 >= max_supervisor_rounds → finish（防 LLM 来回空转）
    2. 两个 worker 都已执行过 → finish（每个 worker 最多派一次，防重复劳动）
    LLM 主判断：根据用户问题 + worker_reports 决定下一个 worker 或收尾。
    LLM 失败：保守 finish（已有发现生成回答优于无限等待）。
    """
    llm = get_llm_client()
    settings = get_settings()
    trace = state.get("supervisor_trace", [])
    reports = state.get("worker_reports", [])

    # ---- 硬兜底 1：派发轮数上限 ----
    if len(trace) >= settings.max_supervisor_rounds:
        return {
            "supervisor_trace": trace + [{
                "round": len(trace) + 1,
                "next": "finish",
                "reason": f"达到最大派发轮数 {settings.max_supervisor_rounds}，强制收尾",
            }],
            "warnings": state.get("warnings", []) + [
                f"Supervisor 派发轮数达到上限 {settings.max_supervisor_rounds}，强制收尾"
            ],
        }

    # ---- 硬兜底 2：两个 worker 都已完成 → 直接收尾（不浪费 LLM 调用）----
    done_workers = {r.get("agent") for r in reports}
    if {"sql_agent", "rag_agent"} <= done_workers:
        return {
            "supervisor_trace": trace + [{
                "round": len(trace) + 1,
                "next": "finish",
                "reason": "两个子 Agent 均已完成，进入汇总",
            }],
        }

    # ---- LLM 主判断 ----
    report_text = "\n".join(
        f"- [{r.get('agent')}] {r.get('summary')}" for r in reports
    ) or "（尚无子 Agent 报告）"
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"用户问题：{state['user_query']}\n\n"
            f"已完成子 Agent 的发现：\n{report_text}\n\n"
            "请决定下一步。"
        )},
    ]
    try:
        result = await llm.chat_json(messages, node="supervisor")
        next_agent = result["content"].get("next", "finish")
        # 防御：LLM 输出非法值 / 重复派发已完成 worker 时收敛到 finish
        if next_agent not in ("sql_agent", "rag_agent", "finish"):
            next_agent = "finish"
        if next_agent in done_workers:
            next_agent = "finish"
        reason = result["content"].get("reason", "")
        return {
            "supervisor_trace": trace + [{
                "round": len(trace) + 1,
                "next": next_agent,
                "reason": reason,
            }],
            "token_usage": _add_usage(state, result["usage"]),
        }
    except LLMClientError as exc:
        # Supervisor LLM 失败：保守收尾
        logger.warning("supervisor_llm_failed", extra={"error": str(exc)})
        return {
            "supervisor_trace": trace + [{
                "round": len(trace) + 1,
                "next": "finish",
                "reason": f"调度判断失败，保守收尾：{exc}",
            }],
            "errors": _append_error(state, "supervisor", f"调度判断失败，保守收尾：{exc}"),
        }


WORKER_REPORT_SYSTEM_PROMPT = """你是企业采购多 Agent 系统中某个专业子 Agent 的汇报模块。
基于本子 Agent 收集到的工具执行结果，输出 JSON（不要输出其他内容）：
{"summary": "本次发现的核心事实（数字/结论/来源，150 字内）", "gaps": "信息缺口（若有）"}
要求：只陈述基于工具结果的事实，禁止编造；数据不全时在 gaps 中说明。"""


async def worker_report_node(state: AgentState) -> dict[str, Any]:
    """
    子 Agent 收尾汇报：把本 worker 的工具执行结果压缩成 supervisor 可读的摘要。
    worker 子图的最后一个节点，产出写入 worker_reports。
    LLM 失败时降级为机械摘要（工具调用序列 + 行数统计），不中断调度。
    """
    llm = get_llm_client()
    worker = state.get("current_worker") or "unknown"
    messages = [
        {"role": "system", "content": WORKER_REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"用户问题：{state['user_query']}\n\n"
            f"本子 Agent 的工具执行结果：\n"
            f"{_format_tool_results_for_llm(state.get('tool_results', []), sample_rows=5)}"
        )},
    ]
    try:
        result = await llm.chat_json(messages, node="worker_report")
        summary = result["content"].get("summary", "")
        gaps = result["content"].get("gaps", "")
        return {
            "worker_reports": state.get("worker_reports", []) + [{
                "agent": worker,
                "summary": summary or f"子 Agent {worker} 完成，无摘要",
                "gaps": gaps,
            }],
            "token_usage": _add_usage(state, result["usage"]),
        }
    except LLMClientError as exc:
        # 降级：机械摘要（不依赖 LLM），保证 supervisor 仍有决策依据
        logger.warning("worker_report_failed", extra={"error": str(exc), "worker": worker})
        fallback_summary = (
            f"{worker} 执行了 {len(state.get('tools_used', []))} 次工具调用"
            f"（{state.get('tools_used', [])}），详见汇总阶段。"
        )
        return {
            "worker_reports": state.get("worker_reports", []) + [{
                "agent": worker,
                "summary": fallback_summary,
                "gaps": f"汇报生成失败：{exc}",
            }],
            "errors": _append_error(state, "worker_report", f"汇报生成失败：{exc}"),
        }
