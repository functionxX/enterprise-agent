"""
schemas.py — Pydantic v2 请求/响应模型（API 契约层）

【为什么 ORM 模型和 Pydantic 模型分开（面试高频）】
- ORM 模型表达"数据库里怎么存"，Pydantic 模型表达"API 对外暴露什么"；
- 两者生命周期不同：数据库表演进 ≠ API 契约演进。
  直接把 ORM 对象序列化返回，意味着改表结构就改 API 响应——耦合；
- Pydantic v2 负责请求校验（query 非空、长度限制）与响应序列化，
  API 契约在 schemas.py 一处定义，前端/文档/测试都对齐这里。

【ChatResponse 的字段与 AgentState 的映射关系（面试点）】
- answer      ← state["final_answer"]
- sources     ← state["tool_results"] + state["retrieved_documents"] 加工而来
- tools_used  ← state["tools_used"]（按调用顺序，允许重复——如实反映调用链）
- iterations  ← state["iteration_count"]（真实循环了几轮）
- token_usage ← state["token_usage"]（累计消耗，成本可观测）
回答的"可追溯性"是 Agent 系统与聊天机器人的本质区别：
用户拿到的不只是一段文字，还有数据来源与调用链。
"""

from typing import Literal

from pydantic import BaseModel, Field


# ================= 请求模型 =================

class ChatRequest(BaseModel):
    """Agent 对话请求。"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(
        default=None,
        description="会话 ID。不传则新建会话；传了但已过期则静默重建（见 api/agent.py）",
    )
    agent_mode: str = Field(
        default="single",
        description='Agent 运行模式："single"（单 Agent 条件循环图，默认）或 "supervisor"（Supervisor 多 Agent 协作）',
    )
    hitl: bool = Field(
        default=False,
        description="HITL 人工审批：开启后敏感工具（SQL 执行）执行前挂起，等待人工批准后续跑",
    )
    approval: dict | None = Field(
        default=None,
        description='HITL 续跑决定：{"approved": true/false, "reason": "..."}。仅在上一次响应返回 pending_approval 后使用，且必须携带原 session_id',
    )
    include_contexts: bool = Field(
        default=False,
        description="评估/调试用：为 true 时响应额外返回 Agent 实际依据的上下文原文（检索 chunk 全文 + SQL 查询与返回行），供 RAGAS 等自动化评估消费；默认 false 保持响应轻量",
    )


# ================= 响应模型 =================

class SqlSource(BaseModel):
    """SQL 查询来源（Agent 查过哪些表、用了什么语句）。"""
    type: Literal["sql"] = "sql"
    query: str = Field(description="Agent 生成并执行的 SQL（只读）")
    result_summary: str = Field(description="返回行数摘要，如 '返回 12 行（截断前 45 行）'")


class RagSource(BaseModel):
    """RAG 检索来源（引用了知识库的哪篇文档哪个 chunk）。"""
    type: Literal["rag"] = "rag"
    document: str = Field(description="知识库文档名")
    chunk_index: int = Field(description="chunk 序号")
    score: float | None = Field(default=None, description="检索相关度分数（0~1）")


class TokenUsage(BaseModel):
    """单次 Agent 请求的 token 消耗。"""
    prompt: int = 0
    completion: int = 0
    total: int = 0


class ChatResponse(BaseModel):
    """Agent 对话响应。"""
    answer: str
    sources: list[SqlSource | RagSource] = Field(default_factory=list)
    tools_used: list[str] = Field(
        default_factory=list,
        description="按调用顺序排列（含重复），如实反映 Agent 的调用链",
    )
    iterations: int = Field(description="工具执行迭代轮数")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    session_id: str | None = None
    session_renewed: bool = Field(
        default=False,
        description="传入的 session_id 已过期、自动重建时为 true（前端据此更新本地 ID）",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="降级/超时/数据截断等非致命告警（Agent 正常返回但信息可能不完整）",
    )
    pending_approval: dict | None = Field(
        default=None,
        description="HITL：有待人工审批的操作时返回（{kind, tool, query, message}），批准后带 approval 参数续跑；无则 None",
    )
    contexts: list[str] = Field(
        default_factory=list,
        description="Agent 回答所依据的上下文原文（SQL 查询+返回行、检索 chunk 全文）——仅 include_contexts=true 时填充；RAGAS 评估的 faithfulness 评分依据",
    )


class UploadResponse(BaseModel):
    """文档上传响应。"""
    filename: str
    chunks_created: int
    status: Literal["indexed"] = "indexed"


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: Literal["healthy", "degraded"]
    database: Literal["connected", "disconnected"]
    vector_store: Literal["connected", "disconnected"]
    redis: Literal["connected", "disconnected"]
