"""
tools.py — Agent 通用 Tool 定义与执行（设计决策 3）

【为什么 Tools 是通用能力，不是写死的业务函数（面试核心考点）】
❌ 错误示范（面试时最容易被看穿的写法）：
    async def query_supplier_risk(supplier_name: str): ...
    —— 这是业务函数。Agent 变成了"填参机器人"，毫无自主性。
✅ 正确做法：给 Agent 两个通用能力：
    1. execute_sql      —— 在 PostgreSQL 上执行只读 SQL（Agent 自己写 SQL）
    2. search_knowledge —— 从企业知识库检索（Agent 自己定检索词）

面试话术：
> "Agent 自己写 SQL 并执行——不是我提前写好几个查询函数。
>  它根据用户意图自主决定查什么表、用什么条件、分几步查。
>  换一个业务场景（比如换成 HR 系统），我的 Agent 图和 tool 定义
>  一行不改，只换数据库里的表——这就是通用 tool 的价值。"

【execute_sql 的三层只读防御（设计决策 3 追问，面试必考）】
第 1 层：Prompt 约束 —— tool description 明确"仅限只读查询"。
    LLM 是概率系统，这层只是意图引导，不保证。
第 2 层：代码层关键词校验 —— _validate_readonly_sql()。
    执行前用正则拒绝 INSERT/UPDATE/DELETE/DROP 等关键词。
    【面试追问】"为什么用正则而不是 SQL 解析器？"
    答：sqlparse 对非法 SQL 不报错、对合法 SQL 也不做权限判断；
    正则轻量、明确拒绝已知危险词、零依赖。对只读校验这个需求，
    正则是工程上最合适的工具。LLM 输出不是编译器输出，
    防御要的是"任何危险词出现就拒绝"，而不是"正确解析 SQL"。
第 3 层：数据库权限 —— agent_readonly 角色只有 SELECT 权限
    （docker/init.sql 创建；connection.py 的只读引擎使用）。
    前两层被绕过时，PostgreSQL 内核直接拒绝写操作——
    最后防线在 DBMS，不在应用代码。

【为什么不让 Agent 执行写操作 SQL（面试常问"既然是 Agent 为什么不放开"）】
1. 责任边界：Agent 的 LLM 输出不可完全信任，写操作造成的损失不可逆
   （删了订单表，重试一万次也回不来）；
2. 审计边界：只读 Agent 的所有行为可回放、可审计；
3. 权限最小化原则：Agent 的职责是"分析"，写操作属于人工审批流程。
   企业采购场景里，Agent 给出的结论最终由人拍板执行——
   这也是"人机协同"的落地形态。

【结果截断策略（设计决策 9，面试高频）】
Agent 合法执行 SELECT 也可能查出 50 行 × 10 列的宽数据：
- 直接把 raw 结果塞给 LLM → token 爆炸、成本失控、噪声淹没关键信息
- 截断 + 明确告知：MAX_ROWS=20 / MAX_CELL=200，超出部分标注
  "truncated": true 和 "实际 N 行，仅返回前 20 行"
  这个 flag 是触发下一轮循环的重要信号——LLM 读到"数据被截断"，
  会自主决定"加 WHERE 条件缩小范围再查一次"，这就是循环图的价值。
"""

import logging
import re
from typing import Any

from sqlalchemy import text

from app.database.connection import get_readonly_session_factory

logger = logging.getLogger(__name__)

# =============================================================
# Tool 定义（OpenAI function-calling 格式，DeepSeek 兼容）
# =============================================================

# 数据库 schema 摘要：Agent 写 SQL 必须知道有哪些表、哪些列。
# 【面试点】schema 为什么写在 tool description 里而不是 system prompt 里？
# 答：工具描述是 LLM 决定"调不调这个 tool"时读到的上下文，
# 把 schema 放在这里保证"只要它想查数据，就一定能看到表结构"；
# 放 system prompt 里会在 token 统计上浪费（非数据查询时也占位）。
# 【面试点：枚举值必须写进 schema hint】
# 评估实测踩坑：Agent 用 WHERE risk_level = '高风险'（中文）查库，
# 而库里存的是 'high'，答出"0 家高风险"。LLM 会凭常识猜枚举值，
# schema hint 必须把每个枚举字段的合法取值写死——这是
# "text-to-SQL 可靠性的第一课：永远不要让模型猜 schema"。
_DB_SCHEMA_HINT = (
    "可用表结构（只读）：\n"
    "- suppliers(id, name, industry, country, rating, risk_level) 供应商表，"
    "risk_level 仅取 low/medium/high（低/中/高风险）\n"
    "- purchase_orders(id, supplier_id, product_name, category, amount, quantity, status, order_date) 采购订单表，"
    "status 仅取 pending/completed/cancelled\n"
    "- invoices(id, order_id, invoice_amount, payment_status, due_date) 发票表，"
    "payment_status 仅取 paid/unpaid/overdue\n"
    "关联方式：purchase_orders.supplier_id = suppliers.id；invoices.order_id = purchase_orders.id。"
    "日期比较用 order_date >= CURRENT_DATE - INTERVAL '90 days' 这类写法。"
)

# 知识库目录：Agent 决定"查哪篇文档"的依据（设计决策 11）
# 【面试点】Agent 预先不知道知识库有什么，目录写在 tool description 里，
# 让 Agent 在 Planner 阶段就能决策检索范围，而不是盲目 top_k。
_KNOWLEDGE_CATALOG = (
    "企业采购知识库包含以下文档：\n"
    "1. supplier_policy.txt —— 供应商准入条件、评估标准（评分/风险分级/复审周期）\n"
    "2. procurement_rules.txt —— 采购流程、审批权限、合规要求\n"
    "3. quality_standard.txt —— 产品质量标准、交付验收规范、环保合规要求\n"
    "查询时使用与目标文档内容相关的关键词；问题涉及多个方面时请分次检索。"
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "在 PostgreSQL 上执行只读 SQL 查询（仅限 SELECT / WITH...SELECT）。\n"
                "用于查询供应商、采购订单、发票等结构化业务数据。\n"
                f"{_DB_SCHEMA_HINT}\n"
                "禁止任何写操作（INSERT/UPDATE/DELETE/DROP 等）。\n"
                "结果最多返回 20 行；若结果被截断会明确标注，此时应加 WHERE 条件缩小范围后重查。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "只读 SQL 查询语句（SELECT 开头）",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "从企业知识库中检索相关文档内容。用于查政策、规则、标准等非结构化知识。\n"
                f"{_KNOWLEDGE_CATALOG}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询（用与目标文档内容相关的关键词）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回 chunk 数量，建议 3~5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# =============================================================
# 只读 SQL 防御（第 2 层：代码关键词校验）
# =============================================================

# 危险关键词（大写匹配）。COpy 特例：PostgreSQL 的 COPY 可以从文件读写，
# 属于高危操作；WITH 后面的 SELECT 允许（递归 CTE 是合法查询写法）。
FORBIDDEN_KEYWORDS: list[str] = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "COPY", "MERGE", "CALL", "DO ",
]


class ForbiddenSQLException(Exception):
    """SQL 校验失败——包含 INSERT/UPDATE 等写操作，或非 SELECT 开头。"""


def _validate_readonly_sql(sql: str) -> None:
    """
    第 2 层只读防御：代码层关键词校验。

    【面试追问】"如果 LLM 写 SELECT 1; DROP TABLE x; 你的正则拦得住吗？"
    答：拦得住——FORBIDDEN_KEYWORDS 对整条 SQL 全文扫描，
    分号后的 DROP 同样命中。这题考的是"校验必须覆盖整个字符串，不能只看开头"。

    【面试追问】"正则只匹配单词边界，写成 dr/*x*/op 能绕过吗？"
    答：先剥离 SQL 注释再扫描（见下方注释剥离），剥离后的文本
    不含注释内容。防御目标是拦住"意外生成"的写操作——
    真要被恶意对抗，最终兜底是第 3 层数据库权限（agent_readonly 角色）。
    """
    # 剥离字符串字面量与注释，防止危险词藏在字符串/注释里造成误判或漏判
    cleaned = re.sub(r"'[^']*'", "''", sql)          # 字符串字面量
    cleaned = re.sub(r"--[^\n]*", " ", cleaned)      # 行注释
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)  # 块注释

    upper = cleaned.upper().strip()
    # 必须以 SELECT 或 WITH 开头（WITH 递归 CTE 是合法查询）
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ForbiddenSQLException(
            f"只允许 SELECT/WITH 开头的只读查询，收到: {sql[:80]}..."
        )
    # 全文扫描危险关键词（含分号后的语句）
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise ForbiddenSQLException(
                f"检测到写操作关键词 {keyword}，Agent 仅允许只读查询"
            )


# =============================================================
# Tool 实现
# =============================================================

# 截断参数（设计决策 9）
MAX_ROWS = 20       # 最多返回行数
MAX_CELL = 200      # 单个单元格最大字符数（超出截断加 "..."）


async def execute_sql(query: str) -> dict[str, Any]:
    """
    Tool 1：执行只读 SQL。

    流程：正则校验（第2层）→ 只读引擎执行（第3层）→ 截断标注。
    执行失败时抛异常（含 SQL 错误信息），由 ToolExecutor 节点捕获后
    记录到 state.errors 并反馈给 LLM 重写 SQL——这是自愈循环的关键：
    LLM 看到错误信息，下一轮用修正后的 SQL 重试。
    """
    _validate_readonly_sql(query)

    # 用只读引擎（agent_readonly 角色）执行——数据库层兜底
    factory = get_readonly_session_factory()
    async with factory() as session:
        result = await session.execute(text(query))
        rows = result.fetchall()
        columns = list(result.keys())

    # 截断 + 标注（设计决策 9）
    total_rows = len(rows)
    truncated = total_rows > MAX_ROWS
    rows = rows[:MAX_ROWS]

    def _format_cell(value: Any) -> str:
        text = str(value)
        return text[:MAX_CELL] + "..." if len(text) > MAX_CELL else text

    return {
        "columns": columns,
        "rows": [[_format_cell(c) for c in row] for row in rows],
        "total_rows": total_rows,
        "truncated": truncated,
        "truncation_note": (
            f"结果被截断：实际 {total_rows} 行，仅返回前 {MAX_ROWS} 行。"
            if truncated
            else None
        ),
    }


async def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Tool 2：知识库检索。真实实现委托给 rag.retriever（Phase 3），
    rag 未就绪（无 embedding key / 向量库无数据）时自动降级为
    关键词 mock 检索——Phase 2 的 mock 不是临时胶带，是正式的降级路径。
    """
    from app.rag.retriever import search  # 延迟导入：避免循环依赖

    docs = await search(query, top_k=top_k)
    return {
        "documents": docs,
        "retrieval_mode": docs[0].get("mode", "mock") if docs else "mock",
        "top_k": top_k,
    }


# tool 名称 → 实现函数映射（ToolExecutor 节点的分发表）
TOOL_IMPLEMENTATIONS: dict[str, Any] = {
    "execute_sql": execute_sql,
    "search_knowledge": search_knowledge,
}
