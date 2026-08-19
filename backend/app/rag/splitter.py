"""
splitter.py — 递归文本分割（手写实现）

【为什么要"递归"分割（面试高频）】
一次性按固定长度硬切会把语义单元拦腰截断：
"供应商年度评估分为四个等级：优秀（4.5 分以上）、良好（4.0~4.5）……"
可能被切成 "...四个等级：优秀（4.5 分以" 和 "上）、良好..."。
递归分割的策略是"先按大分隔符、再按小分隔符、最后才按字符硬切"：
\n\n（段落）→ \n（换行）→ 。（中文句子）→ 字符兜底。
每一层都尽量在语义边界上切，切不动的才降级到下一层——
这和 langchain 的 RecursiveCharacterTextSplitter 原理一致，
手写出来面试时能逐行解释。

【chunk_size 与 overlap 的取值依据（设计决策 5）】
- chunk_size=500：中文约 500 字/chunk。BGE 中文 embedding 在这个
  粒度上语义完整性最好（太短信息不足，太长检索后噪声大）。
- overlap=50 = chunk_size 的 1/10：重叠窗口保证跨 chunk 边界的
  句子在相邻 chunk 中都有完整上下文，防止"关键句恰好在切缝上"导致漏检。
  10% 是经验值——太小防不住语义截断，太大造成存储和 token 浪费。

【面试追问："overlap 会不会导致检索出重复内容？"】
会，但可接受：检索阶段对重复 chunk 做去重（retriever 里处理），
成本是存储多了 10%，收益是召回率显著提升。这是 RAG 工程的经典权衡。
"""

import re
from typing import Any

# 分隔符优先级：从大到小（先按段落切，再按行，再按句子，最后字符兜底）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    递归分割文本为 chunk 列表，每个 chunk 带 chunk_index 元数据。

    返回：[{"text": ..., "chunk_index": 0}, ...]
    source 元数据由上层（loader → vector_store 写入时）补充。
    """
    if not text.strip():
        return []

    pieces = _recursive_split(text, chunk_size)
    # 用 overlap 缝合相邻片段，保持语义跨 chunk 连续性
    chunks: list[dict[str, Any]] = []
    for i, piece in enumerate(pieces):
        chunk_text = piece
        if i > 0 and overlap > 0:
            # 取上一段的尾部 overlap 字符作为本段开头（有重叠）
            chunk_text = pieces[i - 1][-overlap:] + piece
        chunks.append({"text": chunk_text, "chunk_index": i})
    return chunks


def _recursive_split(text: str, chunk_size: int) -> list[str]:
    """
    递归分割核心：按分隔符优先级逐层尝试，
    每层都优先在分隔符处切；单段仍超长时降到下一层。
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in _SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            # 保留分隔符：句子边界信息不丢失（chunk 尾部以"。"等结尾，
            # LLM 能感知句子完整性；这也是 langchain splitter 的标准做法）
            if len(parts) > 1:
                parts = [p + sep for p in parts[:-1]] + [parts[-1]]
            merged: list[str] = []
            buffer = ""
            for part in parts:
                candidate = buffer + part
                if len(candidate) <= chunk_size:
                    buffer = candidate
                else:
                    if buffer.strip():
                        merged.append(buffer)
                    # 单个 part 本身超长：递归用下一级分隔符处理
                    if len(part) > chunk_size:
                        merged.extend(_recursive_split(part, chunk_size))
                        buffer = ""
                    else:
                        buffer = part
            if buffer.strip():
                merged.append(buffer)
            return merged

    # 所有分隔符都不存在（如连续无标点长串）：字符级硬切兜底
    return [
        text[i : i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def normalize_whitespace(text: str) -> str:
    """清洗 PDF 提取文本常见的断行与多余空白（loader → splitter 之间调用）。"""
    text = re.sub(r"[　\t]+", " ", text)      # 全角空格/制表符
    text = re.sub(r"(?<![。！？；\n])\n(?!\n)", "", text)  # 合并句中断行
    return text.strip()
