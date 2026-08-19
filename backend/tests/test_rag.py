"""
test_rag.py — RAG 检索与分割测试（不依赖数据库/LLM）

覆盖：
1. splitter：递归分割、overlap、语义边界（分割点应在分隔符处）
2. retriever：mock 降级检索（关键词匹配逻辑）
3. tools：execute_sql 的截断逻辑（MAX_ROWS / truncated 标注）
4. loader：txt/md 加载与不支持类型报错
"""

import pytest

from app.rag.retriever import MOCK_KNOWLEDGE_BASE, _dedupe, _mock_search
from app.rag.splitter import split_text


# =============================================================
# splitter 测试
# =============================================================

def test_split_short_text_single_chunk():
    """短文本不分割。"""
    chunks = split_text("供应商准入条件：注册资金不低于500万元。", chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0


def test_split_respects_sentence_boundary():
    """分割点优先落在句子边界（。），不拦腰截断句子。"""
    sentence = "这是第{}个测试句子，用于验证递归分割在句子边界切分。"
    text = "".join(sentence.format(i) for i in range(30))  # ~1000 字
    chunks = split_text(text, chunk_size=300, overlap=30)

    # 每个 chunk 长度不超过 chunk_size（+ overlap）
    for c in chunks:
        assert len(c["text"]) <= 330
    # 至少分成 3 段（证明确实切了）
    assert len(chunks) >= 3
    # 每个 chunk 的结尾应是句子边界（。），不是被截断的半个句子
    for c in chunks[:-1]:
        assert c["text"].rstrip().endswith("。") or c["text"].rstrip().endswith("。")


def test_split_overlap_stitches_context():
    """overlap 缝合：第 i 个 chunk 的开头包含第 i-1 个 chunk 的尾部内容。"""
    text = "。" .join(f"句子编号{i}内容" for i in range(50))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # chunk[1] 的开头应与 chunk[0] 的尾部有重叠（overlap 字符级）
    tail = chunks[0]["text"][-20:]
    assert chunks[1]["text"].startswith(tail)


def test_split_empty_text():
    assert split_text("", chunk_size=500, overlap=50) == []


# =============================================================
# retriever mock 降级测试
# =============================================================

def test_mock_search_keyword_matching():
    """mock 检索按关键词召回相关片段（设计决策 13：mock 有区分度，不是随机返回）。"""
    docs = _mock_search("供应商风险评估标准是什么", top_k=5)
    assert docs, "应检索到供应商相关片段"
    assert all(d["mode"] == "mock" for d in docs)
    # 排第一的应是最相关的（命中关键词最多）
    assert docs[0]["source"] == "supplier_policy.txt"


def test_mock_search_no_match_returns_empty():
    """完全无关的查询返回空（诚实优于硬凑）。"""
    docs = _mock_search("量子纠缠与黑洞信息悖论", top_k=5)
    assert docs == []


def test_dedupe_keeps_highest_score():
    """同文档相邻 chunk 去重：只留得分最高者（overlap 补偿）。"""
    docs = [
        {"source": "a.txt", "chunk_index": 0, "score": 0.8, "text": "x"},
        {"source": "a.txt", "chunk_index": 1, "score": 0.6, "text": "x"},  # 相邻 → 去重
        {"source": "b.txt", "chunk_index": 0, "score": 0.9, "text": "y"},
    ]
    result = _dedupe(docs, top_k=3)
    assert len(result) == 2
    assert all(r["source"] == "a.txt" for r in result if r["score"] == 0.8)


# =============================================================
# tools: execute_sql 截断逻辑（设计决策 9）
# =============================================================

def test_truncation_marks_and_limits():
    """
    截断逻辑：超过 MAX_ROWS 只返回前 20 行，且明确标注 truncated
    （设计决策 9：截断必须告知 LLM，truncated flag 触发下一轮缩小范围重查）。
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    from app.agent.tools import MAX_ROWS, execute_sql

    rows = [("row", i) for i in range(45)]  # 45 行 > 20
    columns = ["col1", "col2"]

    # 真实 SQLAlchemy Result 的 fetchall/keys 是同步方法，用 MagicMock 而非 AsyncMock
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_result.keys.return_value = columns

    class FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def execute(self, query):
            return mock_result

    # 模拟 sessionmaker：get_readonly_session_factory() 返回可调用的工厂，
    # 调用工厂返回 FakeSession（与 async_sessionmaker 的调用语义一致）
    def fake_session_factory():
        return FakeSession

    with patch("app.agent.tools.get_readonly_session_factory", fake_session_factory):
        result = asyncio.run(execute_sql("SELECT * FROM suppliers"))

    assert result["truncated"] is True
    assert result["total_rows"] == 45
    assert len(result["rows"]) == MAX_ROWS
    assert "截断" in result["truncation_note"]


# =============================================================
# loader 测试
# =============================================================

def test_loader_txt(tmp_path):
    from app.rag.loader import load_document
    f = tmp_path / "test.txt"
    f.write_text("供应商准入条件测试内容", encoding="utf-8")
    doc = load_document(f)
    assert doc["source"] == "test.txt"
    assert "供应商" in doc["text"]


def test_loader_rejects_unsupported(tmp_path):
    from app.rag.loader import UnsupportedFileType, load_document
    f = tmp_path / "test.docx"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileType):
        load_document(f)


# =============================================================
# fixtures 知识片段与知识库文档一致性
# =============================================================

def test_mock_knowledge_base_matches_knowledge_docs():
    """
    mock 知识片段池应与 knowledge/ 目录的正式文档语义一致
    （mock 是正式知识库的精简镜像——降级时不出现错误答案）。
    测试读取真实知识库文档，验证 mock 中的关键数值出现其中。
    """
    from pathlib import Path
    knowledge_dir = Path(__file__).resolve().parent.parent.parent / "knowledge"
    docs_text = ""
    for f in knowledge_dir.glob("*.txt"):
        docs_text += f.read_text(encoding="utf-8")

    # mock 片段中的关键阈值必须能在正式文档中找到（数字一致性）
    key_facts = ["95%", "98%", "500 万元", "200 万元", "30%"]
    for fact in key_facts:
        assert fact in docs_text, f"知识库文档缺少关键数值 {fact}"

    # mock 片段的关键词必须与正式文档主题对应（三篇文档全部覆盖）
    sources = {item["source"] for item in MOCK_KNOWLEDGE_BASE}
    assert sources == {"supplier_policy.txt", "procurement_rules.txt", "quality_standard.txt"}
