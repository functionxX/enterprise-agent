"""
agent.py — Agent Chat API（POST /api/v1/agent/chat）

【会话管理（设计决策 7，面试高频）】
存什么：每轮对话的结构化摘要（query + 一句话 summary + tools_used），
    不是完整 tool_results 原文——token 控制的关键。
存哪：Redis。会话是临时状态，丢了不可惜；PostgreSQL 存会话历史是过度设计。
TTL：30 分钟，每次请求续期（EXPIRE 刷新）。
   为什么 30 分钟：Agent 分析任务本身可能耗时数分钟，TTL 太短会导致
   用户看一份报告的时间就把会话丢了；太长则 Redis 内存被死会话占用。
注入策略：只把最近 5 轮摘要给意图分析/规划用（MAX_HISTORY_TURNS），
    防止 prompt 无限膨胀。
过期处理：静默重建新会话（session_renewed=true 告知前端），绝不报错——
    企业 Agent 系统的会话管理跟 Web session 一样，用户无感知。

【降级策略（设计决策 16，面试高频）】
Redis 是"增强依赖"：挂了就无状态运行（没有历史上下文），
功能降级但可用——所以 Redis 异常全部捕获，返回 200 + warnings。
数据库/LLM 是"核心依赖"：挂了会向上抛（LLM 错误 → 502，数据库错误 → 503），
由 main.py 的异常处理器统一返回。
判断标准一句话："这个依赖挂了，Agent 还能不能给出有用的回答？"

【并发安全（设计决策 17）】
graph 对象全局单例（无状态编译产物），state 每请求新建——
两个并发请求互不干扰。共享的是连接池（LLM/DB/Redis 客户端），
这些组件本身就是为并发复用设计的。

【HITL 人工审批（设计决策 19）】
hitl=true 时：图在敏感工具执行前 interrupt() 挂起 → langgraph 0.6+
的 ainvoke 不抛异常，返回的 state 里带 __interrupt__ 标记——本端点
据此把待审批信息放 pending_approval 返回（HTTP 200，挂起是业务流程
不是故障）→ 前端展示批准/拒绝 → 携带 approval 参数续跑同一
session_id：graph.ainvoke(Command(resume=approval)) 从 checkpoint
（thread_id=session_id）恢复挂起点继续执行。
注意：开启 HITL 的图挂 checkpointer，同 session 并发续跑会冲突
（同一 thread 的 checkpoint 竞写）——HITL 会话应串行使用。
"""

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from redis.asyncio import Redis

from app.agent.graph import get_graph
from app.agent.llm_client import LLMClientError
from app.agent.supervisor import get_supervisor_graph
from app.agent.state import AgentState
from app.config import get_settings
from app.database.schemas import (
    ChatRequest,
    ChatResponse,
    RagSource,
    SqlSource,
    TokenUsage,
)
from app.monitoring.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# =============================================================
# Redis 会话存取（全部容错——Redis 是增强依赖）
# =============================================================

def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def _redis_client() -> Redis:
    """
    创建 Redis 客户端。protocol=2 强制 RESP2 协议——
    redis-py 5.x 默认 RESP3（HELLO 命令），老版本 Redis（<6）不支持会超时。
    RESP2 在 Redis 5/6/7 全系列可用，兼容性最好的选择（demo 常见坑）。
    """
    settings = get_settings()
    return Redis.from_url(settings.redis_url, socket_connect_timeout=2, protocol=2)


async def _load_session(session_id: str) -> list[dict] | None:
    """读取会话历史摘要。Redis 不可用或 key 不存在都返回 None（调用方静默重建）。"""
    settings = get_settings()
    try:
        client = _redis_client()
        try:
            raw = await client.get(_session_key(session_id))
            if raw is None:
                return None
            data = json.loads(raw)
            turns = data.get("turns", [])
            # 续期：每次访问刷新 TTL（30 分钟不活动才过期）
            await client.expire(_session_key(session_id), settings.session_ttl_seconds)
            return turns[-settings.max_history_turns :]  # 只保留最近 N 轮注入
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001 —— Redis 挂了就降级，绝不因为缓存抛 500
        logger.warning("redis_load_failed", extra={"error": str(exc)})
        return None


async def _save_turn(session_id: str, query: str, summary: str, tools_used: list[str]) -> None:
    """保存本轮对话摘要（query + 一句话 summary + tools_used），刷新 TTL。"""
    settings = get_settings()
    try:
        client = _redis_client()
        try:
            key = _session_key(session_id)
            raw = await client.get(key)
            data = json.loads(raw) if raw else {"turns": []}
            data["turns"].append({
                "query": query,
                "summary": summary[:300],  # 摘要截断——存的是摘要不是全文
                "tools_used": tools_used,
                "ts": time.time(),
            })
            # 只保留最近 max_history_turns 轮，防单会话无限膨胀
            data["turns"] = data["turns"][-settings.max_history_turns :]
            await client.set(key, json.dumps(data, ensure_ascii=False))
            await client.expire(key, settings.session_ttl_seconds)
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis_save_failed", extra={"error": str(exc)})


# =============================================================
# Chat 端点
# =============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Agent 对话端点。

    流程：会话加载（容错）→ 构建 state → 跑图（超时兜底）→ 存会话 → 组装响应。
    """
    settings = get_settings()
    session_id = request.session_id or str(uuid.uuid4())
    warnings: list[str] = []

    # ---- 1. 会话加载（静默重建语义）----
    history: list[dict] = []
    session_renewed = False
    if request.session_id:
        loaded = await _load_session(request.session_id)
        if loaded is None:
            # 传入的 session_id 不存在或已过期 → 静默重建，不报错
            session_renewed = True
            logger.info("session_renewed", extra={"session_id": request.session_id})
        else:
            history = loaded

    # ---- 2. 构建初始 state（每请求独立，并发安全）----
    initial_state: AgentState = {
        "user_query": request.query,
        "conversation_history": history,
        "session_id": session_id,
        "intent": {},
        "task_plan": [],
        "current_step": 0,
        "tool_results": [],
        "retrieved_documents": [],
        "errors": [],
        "retry_count": 0,
        "iteration_count": 0,
        "tools_used": [],
        "router_decision": None,
        "llm_messages": [],
        "last_tool_requested": None,
        "pending_tool_call": None,
        "executor_says_done": False,
        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        "started_at": time.time(),
        "final_answer": "",
        "warnings": [],
        "allowed_tools": None,
        "current_worker": None,
        "supervisor_trace": [],
        "worker_reports": [],
        "hitl_enabled": request.hitl,
    }

    # ---- 2.5 HITL 续跑校验（防误用）----
    if request.approval is not None:
        if not request.hitl:
            raise HTTPException(status_code=400, detail="approval 续跑仅支持 hitl=true 的请求")
        if not request.session_id:
            raise HTTPException(
                status_code=400,
                detail="approval 续跑必须携带原 session_id（thread_id 定位挂起点）",
            )

    # ---- 3. 跑图 ----
    # agent_mode 选择图：单 Agent 条件循环图（默认）或 Supervisor 多 Agent 图。
    # Supervisor 模式 superstep 更多（2 个 worker × 内部循环 + 调度往返），
    # recursion_limit 相应放宽；真正的循环上限仍由业务兜底控制。
    supervisor_mode = request.agent_mode == "supervisor"
    graph = (
        get_supervisor_graph(enable_hitl=request.hitl)
        if supervisor_mode
        else get_graph(enable_hitl=request.hitl)
    )
    recursion_limit = 200 if supervisor_mode else 50
    # HITL：thread_id = session_id——interrupt 挂起时 checkpoint 按线程存状态，
    # 续跑请求用同一个 thread_id 找回现场（设计决策 19）。
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if request.hitl:
        config["configurable"] = {"thread_id": session_id}
    try:
        # 超时兜底：asyncio.wait_for 是 Router 内软超时之外的最后防线
        # （Router 在超时前会优雅收尾；这里防的是 LLM 调用挂死等极端情况）
        if request.approval is not None:
            # HITL 续跑：把人工决定（{"approved": true/false, "reason": ...}）
            # 传给 interrupt() 挂起点——Command(resume=...) 从 checkpoint
            # 恢复运行，interrupt() 的返回值就是该决定
            final_state = await asyncio.wait_for(
                graph.ainvoke(Command(resume=request.approval), config=config),
                timeout=settings.agent_timeout_seconds + 30,
            )
        else:
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state, config=config),
                timeout=settings.agent_timeout_seconds + 30,
            )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Agent 执行超时。请简化问题或稍后重试。",
        )
    except LLMClientError as exc:
        # LLM 是核心依赖：失败明确返回 502（降级策略见 docstring）
        raise HTTPException(status_code=502, detail=f"LLM 服务不可用：{exc}")

    # ---- 3.5 HITL 挂起判断 ----
    # langgraph 0.6+ 的 ainvoke 遇到 interrupt() 不抛异常，而是在返回的
    # state 里带 __interrupt__ 标记（挂起前状态快照 + 待审批载荷）。
    # 【核心：挂起不是故障，是业务流程】把待审批信息返回给前端
    # （展示 SQL + 批准/拒绝按钮），前端带 approval 参数续跑同一条会话。
    if request.hitl and final_state.get("__interrupt__"):
        interrupts = list(final_state["__interrupt__"])
        pending = interrupts[0].value if interrupts else None
        logger.info("hitl_approval_pending", extra={"session_id": session_id})
        return ChatResponse(
            answer="",
            sources=[],
            tools_used=[],
            iterations=0,
            token_usage=TokenUsage(),
            session_id=session_id,
            pending_approval=pending,
            warnings=[],
        )

    warnings.extend(final_state.get("warnings", []))
    if session_renewed:
        warnings.append("传入的 session_id 已过期，已自动重建新会话")

    # ---- 4. 保存本轮摘要（Redis 挂了也不影响响应）----
    answer = final_state.get("final_answer", "")
    await _save_turn(session_id, request.query, answer, final_state.get("tools_used", []))

    # ---- 5. 组装 sources（从 tool_results / retrieved_documents 派生）----
    sources: list[SqlSource | RagSource] = []
    for tr in final_state.get("tool_results", []):
        if tr.get("tool") == "execute_sql":
            summary = (
                f"返回 {tr.get('total_rows', 0)} 行"
                + ("（已截断）" if tr.get("truncated") else "")
                if tr.get("status") == "success"
                else f"执行失败：{tr.get('error', '')[:100]}"
            )
            sources.append(SqlSource(query=tr.get("query", ""), result_summary=summary))
    for doc in final_state.get("retrieved_documents", []):
        sources.append(RagSource(
            document=doc.get("source", "unknown"),
            chunk_index=doc.get("chunk_index", 0),
            score=doc.get("score"),
        ))

    # ---- 5.5 评估上下文（include_contexts=true 时）----
    # Agent 实际依据了什么，评估就该对着什么打分——faithfulness 检查的是
    # "回答有没有忠实于依据"，不是"回答有没有忠实于理想答案"。
    # 从 tool_results 取原文：SQL 查询 + 返回行、检索 chunk 全文。
    # 默认 false：生产响应保持轻量（这些原文可能很长，只有评估/调试才需要）。
    contexts: list[str] = []
    if request.include_contexts:
        for tr in final_state.get("tool_results", []):
            if tr.get("tool") == "execute_sql" and tr.get("status") == "success":
                contexts.append(
                    f"[SQL] {tr.get('query', '')}\n"
                    f"返回 {tr.get('total_rows', 0)} 行: "
                    f"{json.dumps({'columns': tr.get('columns', []), 'rows': tr.get('rows', [])}, ensure_ascii=False)}"
                )
            elif tr.get("tool") == "search_knowledge":
                for d in tr.get("documents", []):
                    contexts.append(
                        f"[文档 {d.get('source')}#{d.get('chunk_index')}] {d.get('text', '')}"
                    )

    usage = final_state.get("token_usage", {})
    return ChatResponse(
        answer=answer,
        sources=sources,
        tools_used=final_state.get("tools_used", []),
        iterations=final_state.get("iteration_count", 0),
        token_usage=TokenUsage(
            prompt=usage.get("prompt", 0),
            completion=usage.get("completion", 0),
            total=usage.get("total", 0),
        ),
        session_id=session_id,
        session_renewed=session_renewed,
        warnings=warnings,
        contexts=contexts,
    )
