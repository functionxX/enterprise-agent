"""
retriever.py — 检索器：三种检索模式 + Query Rewrite + Rerank + mock 降级

【三种检索模式（面试高频：分别解决什么问题）】
1. vector（纯向量）——语义匹配强，但精确关键词（型号、编号、专有名词）弱；
2. hybrid（混合）——向量 + PostgreSQL 全文检索（tsvector），两者互补：
   向量管"意思相近"，全文管"词面命中"。融合用 RRF（倒数排名融合），
   无需调权重，两个榜单各取排名倒数相加——简单且对分数尺度不敏感；
3. rerank（重排）——先向量粗排召回 top_k*2，再用 bge-reranker 精排。
   为什么需要：向量模型是"双塔"，query 与 doc 编码时互不可见，
   精度上限低；reranker 是"交叉塔"，query-doc 拼接后共同编码，
   精度高但贵——所以工程模式永远是"粗排召回 + 精排重排"两段式。

【Query Rewrite 为什么存在（面试点）】
用户口语 query 与知识库书面语有表达差异：
"供应商不合格怎么办" → 书面表述是"供应商评估不达标处理流程"。
改写（用 LLM）把口语转成检索友好的书面查询词。
注意：rewrite 只在检索前做一次，不改变最终回答的语义——这是
RAG 工程里"检索增强"与"回答忠实"的边界：可以改写 query 提升召回，
不能改写答案偏离原文。

【mock 降级（设计决策 13 的落地）】
embedding key 未配置 / 向量库为空 / API 故障时，降级为关键词匹配：
从预置知识片段池中按关键词召回。接口签名与真实检索完全一致——
Phase 2 的 mock 不是临时胶带，是正式降级路径。
这样"没有 embedding key 也能跑通 Agent 全流程 demo"，
面试演示不依赖第三方 API 可用性。

【去重（splitter 的 overlap 追问在这里兑现）】
相邻 chunk 因 overlap 存在重复文本，检索 top_k 中同一文档
相邻 chunk_index 只保留得分最高者——防止重复内容挤占 top_k 名额。
"""

import logging
import re

from sqlalchemy import text

from app.config import get_settings
from app.database.connection import get_session_factory
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)

# =============================================================
# mock 知识片段池（embedding 不可用时的降级检索源）
# 内容与 knowledge/ 目录的三篇文档一致（精简版），保证 demo 语义正确
# =============================================================

MOCK_KNOWLEDGE_BASE: list[dict] = [
    {"text": "供应商准入条件：注册资金不低于 500 万元，具备 ISO9001 质量体系认证，"
             "近三年无重大质量事故。新供应商须通过准入评审后方可进入合格供应商名录。",
     "source": "supplier_policy.txt", "chunk_index": 0,
     "keywords": ["供应商", "准入", "认证", "注册资金"]},
    {"text": "供应商年度评估标准：评分 1~5 分。4 分以上为优秀供应商可加大采购份额；"
             "3.5~4 分为良好；3~3.5 分为需关注；3 分以下启动淘汰流程。"
             "评估维度包括交付率（30%）、质量合格率（30%）、价格竞争力（20%）、服务响应（20%）。",
     "source": "supplier_policy.txt", "chunk_index": 1,
     "keywords": ["评估", "评分", "交付率", "淘汰", "等级"]},
    {"text": "供应商风险分级：低风险供应商年度评估一次；中风险供应商半年度评估一次；"
             "高风险供应商月度复审，且须提交改善计划。连续两次复审不达标的高风险供应商"
             "暂停采购资格并启动替代供应商开发。",
     "source": "supplier_policy.txt", "chunk_index": 2,
     "keywords": ["风险", "分级", "复审", "低风险", "中风险", "高风险"]},
    {"text": "采购订单审批权限：单笔金额 50 万以下由采购经理审批；"
             "50 万至 200 万由采购总监审批；200 万以上须经采购委员会集体决策并报总经理批准。",
     "source": "procurement_rules.txt", "chunk_index": 0,
     "keywords": ["审批", "权限", "金额", "50万", "200万"]},
    {"text": "采购流程要求：采购订单须经三家以上供应商比价（独家供应商除外）；"
             "合同须明确交付周期、验收标准与违约责任；付款条件原则上为货到验收后 30 天。",
     "source": "procurement_rules.txt", "chunk_index": 1,
     "keywords": ["流程", "比价", "合同", "验收", "付款"]},
    {"text": "合规要求：禁止与列入失信名单的供应商交易；采购人员利益关联供应商须回避；"
             "超过 50 万元的单一来源采购须提供书面论证并留存审计记录。",
     "source": "procurement_rules.txt", "chunk_index": 2,
     "keywords": ["合规", "失信", "回避", "审计", "单一来源"]},
    {"text": "交付验收标准：按期交付率不低于 95%；来料质量合格率不低于 98%。"
             "连续三个批次交付延迟或质量不合格的供应商，暂停新订单并启动质量专项审核。",
     "source": "quality_standard.txt", "chunk_index": 0,
     "keywords": ["交付", "验收", "交付率", "95", "合格率", "质量"]},
    {"text": "产品质量标准：关键原材料须提供批次检测报告；电子元器件须符合 RoHS 环保标准；"
             "进口物料须提供原产地证明与报关单据。环保合规材料占比不低于年度采购额的 80%。",
     "source": "quality_standard.txt", "chunk_index": 1,
     "keywords": ["质量", "检测", "环保", "RoHS", "进口", "原产地"]},
]


# =============================================================
# 主入口
# =============================================================

async def search(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    rewrite: bool = True,
) -> list[dict]:
    """
    检索主入口（Agent 的 search_knowledge tool 调用此函数）。

    流程：Query Rewrite（可选）→ 按 mode 检索 → 去重 → 返回。
    mode ∈ {vector, hybrid, rerank}，默认 hybrid（演示时效果最稳）。
    """
    embedder = get_embedder()

    # ---- 降级判断：embedding 未配置 → mock 关键词检索 ----
    if not embedder.is_configured:
        return _mock_search(query, top_k)

    # ---- Query Rewrite：口语 → 检索友好书面语（LLM）----
    search_query = query
    if rewrite:
        try:
            search_query = await _rewrite_query(query)
        except Exception as exc:  # noqa: BLE001 —— rewrite 失败不阻塞检索
            logger.warning("query_rewrite_failed", extra={"error": str(exc)})

    try:
        query_vector = await embedder.embed_query(search_query)
    except Exception as exc:  # noqa: BLE001 —— embedding API 故障 → mock 降级
        logger.warning("embedding_failed_fallback_mock", extra={"error": str(exc)})
        return _mock_search(query, top_k)

    if mode == "vector":
        docs = await _vector_search(query_vector, top_k, search_query)
    elif mode == "rerank":
        docs = await _rerank_search(query_vector, search_query, top_k)
    else:  # hybrid（默认）
        docs = await _hybrid_search(query_vector, search_query, top_k)

    # ---- 去重：同文档相邻 chunk 因 overlap 重复，只留最高分 ----
    return _dedupe(docs, top_k)


# =============================================================
# Query Rewrite（LLM）
# =============================================================

async def _rewrite_query(query: str) -> str:
    """
    用 LLM 把用户口语 query 改写为检索友好的书面查询词。

    【为什么改写要"轻"？】
    rewrite 的目标是提升召回，不是改语义。Prompt 强制
    "只输出改写后的查询词，保持原意"——改写过重会引入语义漂移，
    检索回来的文档和用户问题对不上，比不改写更糟。
    """
    from app.agent.llm_client import get_llm_client

    llm = get_llm_client()
    messages = [
        {"role": "system", "content": (
            "你是检索查询改写器。把用户的口语化问题改写为适合"
            "在企业采购知识库中做语义检索的书面查询词。"
            "要求：保持原意、使用知识库中可能的专业表述、只输出改写后的查询词本身，"
            "不要任何解释或标点包裹。"
        )},
        {"role": "user", "content": query},
    ]
    result = await llm.chat(messages, node="query_rewrite")
    rewritten = (result.get("content") or "").strip().strip('"').strip("“”")
    if not rewritten or len(rewritten) < 2:
        return query
    logger.info("query_rewritten", extra={"from": query[:60], "to": rewritten[:60]})
    return rewritten


# =============================================================
# 三种检索模式
# =============================================================

async def _vector_search(query_vector: list[float], top_k: int, query: str) -> list[dict]:
    """模式 1：纯向量检索（语义匹配）。"""
    from app.rag.vector_store import search_by_vector

    docs = await search_by_vector(query_vector, top_k=top_k)
    for d in docs:
        d["mode"] = "vector"
    return docs


async def _hybrid_search(query_vector: list[float], query: str, top_k: int) -> list[dict]:
    """
    模式 2：混合检索 = 向量相似度 + PostgreSQL 全文检索，RRF 融合。

    【为什么用 RRF（Reciprocal Rank Fusion）融合（面试高频）】
    向量分数是余弦相似度（~0.5~0.9），全文分数是 ts_rank（尺度不定），
    两个分数不可比，加权求和需要反复调权重。
    RRF 只关心排名：score = Σ 1/(k + rank_i)，k=60 是常用平滑常数。
    好处：无需归一化、对分数尺度不敏感、两榜融合稳定。
    """
    settings = get_settings()
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    # 双榜召回：向量榜 top_k*2 + 全文榜 top_k*2，融合后取 top_k
    fetch_k = top_k * 2
    factory = get_session_factory()
    async with factory() as session:
        # 向量榜（CAST 显式类型转换——`:vec::vector` 简写会让
        # SQLAlchemy 绑定参数解析出错，见 vector_store.py 注释）
        vec_result = await session.execute(text(
            "SELECT id, text, source, chunk_index, 1 - (embedding <#> CAST(:vec AS vector)) AS score "
            "FROM document_chunks WHERE embedding IS NOT NULL "
            "ORDER BY embedding <#> CAST(:vec AS vector) LIMIT :k"
        ), {"vec": vector_literal, "k": fetch_k})
        vec_rows = vec_result.fetchall()

        # 全文榜：tsvector 英文分词 + simple 配置（中文按字切分也能命中专有名词）
        # 【面试点】PG 内置全文对中文分词弱（无 jieba/zhparser），
        # 但 simple 配置把中文按连续字串切，专有名词（"供应商"）仍可命中，
        # 对知识库检索足够——生产中文场景可挂 zhparser 扩展，架构不变。
        fts_result = await session.execute(text(
            "SELECT id, text, source, chunk_index, ts_rank(to_tsvector('simple', text), plainto_tsquery('simple', :q)) AS score "
            "FROM document_chunks WHERE to_tsvector('simple', text) @@ plainto_tsquery('simple', :q) "
            "ORDER BY 5 DESC LIMIT :k"
        ), {"q": query, "k": fetch_k})
        fts_rows = fts_result.fetchall()

    # RRF 融合
    scores: dict[int, dict] = {}
    for rank, row in enumerate(vec_rows, 1):
        scores[row.id] = {
            "id": row.id, "text": row.text, "source": row.source,
            "chunk_index": row.chunk_index, "rrf": 1.0 / (60 + rank),
        }
    for rank, row in enumerate(fts_rows, 1):
        if row.id in scores:
            scores[row.id]["rrf"] += 1.0 / (60 + rank)
        else:
            scores[row.id] = {
                "id": row.id, "text": row.text, "source": row.source,
                "chunk_index": row.chunk_index, "rrf": 1.0 / (60 + rank),
            }

    merged = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
    return [
        {
            "text": m["text"],
            "source": m["source"],
            "chunk_index": m["chunk_index"],
            "score": round(m["rrf"], 5),
            "mode": "hybrid",
        }
        for m in merged
    ]


async def _rerank_search(query_vector: list[float], query: str, top_k: int) -> list[dict]:
    """
    模式 3：粗排（向量 top_k*2）+ 精排（bge-reranker API）两段式。

    【为什么两段式是 RAG 检索的标准答案（面试高频）】
    双塔模型（embedding）把 query 和 doc 分别编码，速度快但
    query-doc 交互信息丢失，精度天花板低；交叉编码器（reranker）
    query-doc 拼接共同编码，精度高但每个对都要过模型，慢。
    所以：向量粗排召回候选（便宜）→ reranker 精排（贵但只算少量对）。
    这是"用便宜的模型缩小范围，用贵的模型做精细判断"的通用工程模式。
    """
    from app.rag.vector_store import search_by_vector
    from app.agent.llm_client import get_llm_client

    settings = get_settings()
    # 1. 向量粗排
    candidates = await search_by_vector(query_vector, top_k=top_k * 2)

    # 2. reranker 精排（硅基流动 bge-reranker-v2-m3）
    try:
        import httpx

        payload = {
            "model": settings.rerank_model,
            "query": query,
            "documents": [c["text"] for c in candidates],
            "return_documents": False,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.embedding_base_url}/rerank",
                json=payload,
                headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        ranked = sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
        docs = []
        for r in ranked[:top_k]:
            cand = candidates[r.get("index", 0)]
            docs.append({
                "text": cand["text"],
                "source": cand["source"],
                "chunk_index": cand["chunk_index"],
                "score": round(r.get("relevance_score", 0.0), 4),
                "mode": "rerank",
            })
        return docs
    except Exception as exc:  # noqa: BLE001 —— reranker 故障降级为向量结果
        logger.warning("rerank_failed_fallback_vector", extra={"error": str(exc)})
        return candidates[:top_k]


# =============================================================
# 去重 + mock 降级
# =============================================================

def _dedupe(docs: list[dict], top_k: int) -> list[dict]:
    """
    同文档相邻 chunk 去重（splitter overlap 的补偿）。

    保留得分最高者：overlap 会导致相邻 chunk 文本部分重复，
    同时返回只会浪费 top_k 名额、稀释信息密度。
    """
    seen: dict[tuple[str, int], dict] = {}
    for doc in docs:
        key = (doc.get("source", ""), int(doc.get("chunk_index", -1)) // 2)
        if key not in seen or (doc.get("score") or 0) > (seen[key].get("score") or 0):
            seen[key] = doc
    return sorted(seen.values(), key=lambda d: d.get("score") or 0, reverse=True)[:top_k]


def _mock_search(query: str, top_k: int) -> list[dict]:
    """
    mock 降级检索：关键词匹配（设计决策 13——接口与真实检索一致）。

    【面试话术】
    "Phase 2 的 mock 不是写死返回，而是关键词匹配 + 与真实检索
    完全一致的接口签名。embedding API 不可用时（无 key、超时、宕机），
    Agent 流程依然完整可跑——降级路径是设计出来的，不是临时糊的。"
    """
    q = query.lower()
    scored: list[tuple[int, dict]] = []
    for item in MOCK_KNOWLEDGE_BASE:
        hits = sum(1 for kw in item["keywords"] if kw.lower() in q)
        if hits > 0:
            scored.append((hits, {
                "text": item["text"],
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "score": round(min(hits / 3.0, 1.0), 3),
                "mode": "mock",
            }))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
