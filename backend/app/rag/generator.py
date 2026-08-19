"""
generator.py — RAG Prompt 构建 + LLM 生成（standalone RAG 使用）

【这个模块在 Agent 架构中的定位】
Agent 流程里的回答生成由 response_generator_node（nodes.py）完成，
它基于 tool_results（SQL 数据）+ retrieved_documents（RAG 知识）综合生成。
本模块提供的是"纯 RAG 场景"（只问知识库、不查数据库）的独立生成能力：
- Document Upload 之外独立验证 RAG pipeline 的检索→生成闭环
- 测试与评估时单点调用，隔离 Agent 图的其他环节
- 面试演示 RAG-demo 能力时直接复用（设计决策 5：复用已有 RAG 逻辑）

【Prompt 构建的防幻觉设计（面试高频）】
1. 明确指令"只根据提供的文档回答"，并为"文档中没有答案"给出模板：
   LLM 的幻觉大多源于"文档没有、模型硬编"——给模型一条体面的退路，
   它才敢说"我不知道"；
2. 每条引用 chunk 标注 [来源名 #chunk序号]，回答中要求带引用标记——
   可追溯是 RAG 与"LLM 瞎编"的分界线；
3. temperature 用全局配置的 0.1（低温度减少编造倾向）。
"""

import logging
from typing import Any

from app.agent.llm_client import LLMClientError, get_llm_client

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是企业采购知识库问答助手。
规则：
1. 只能根据提供的文档片段回答，禁止使用文档外的知识编造内容
2. 回答中引用文档内容时，用 [来源名 #片段号] 标注出处
3. 文档片段中没有相关信息时，明确回答"知识库中没有相关信息"，
   不要猜测或编造
4. 用中文回答，结构化输出"""


def build_rag_prompt(
    query: str,
    documents: list[dict[str, Any]],
    history: list[dict] | None = None,
) -> list[dict]:
    """
    构建 RAG 生成消息序列。

    【为什么检索到的 chunk 要编号注入（面试点）】
    编号让 LLM 的回答可以引用具体位置，也让下游能做"引用校验"
    （评估集检查答案是否真的来自检索内容）。无编号的拼接
    无法验证"答案有没有依据"——防幻觉要可验证，不是靠 prompt 祈祷。
    """
    context_lines = []
    for i, doc in enumerate(documents, 1):
        context_lines.append(
            f"[{i}] 来源: {doc.get('source', 'unknown')} "
            f"片段#{doc.get('chunk_index', '-')}\n{doc.get('text', '')}"
        )
    context = "\n\n".join(context_lines) if context_lines else "（无检索结果）"

    messages: list[dict] = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"知识库检索到的相关文档片段：\n\n{context}\n\n"
                f"用户问题：{query}\n\n请根据文档片段回答。"
            ),
        },
    ]
    if history:
        # 多轮上下文（standalone RAG 的会话支持）
        messages.insert(1, {"role": "user", "content": "历史对话摘要：" + " | ".join(
            f"{h.get('query', '')}" for h in history[-3:]
        )})
    return messages


async def generate(
    query: str,
    documents: list[dict[str, Any]],
) -> str:
    """独立 RAG 生成（检索 → 生成闭环的收尾）。"""
    llm = get_llm_client()
    messages = build_rag_prompt(query, documents)
    try:
        result = await llm.chat(messages, node="rag_generator")
        return result["content"] or ""
    except LLMClientError as exc:
        logger.error("rag_generate_failed", extra={"error": str(exc)})
        return f"生成失败：{exc}"
