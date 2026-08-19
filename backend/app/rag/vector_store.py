"""
vector_store.py — pgvector 向量存储 + 相似度检索（手写 SQL）

【为什么手写 SQL 而不是用 vectorstore 框架抽象（面试点）】
langchain 的 PGVector 把 SQL 藏在抽象后面，面试官一问
"你的相似度是怎么算的"就露馅。手写之后每个环节都可解释：
- 建表（document_chunks，embedding vector(1024)）
- 写入（INSERT 批量，向量转 pgvector 字符串字面量）
- 检索（ORDER BY embedding <#> query_vec LIMIT k —— 负内积升序 = 相似度降序）
- 索引（ivfflat，灌数据后创建）

【为什么只用 pgvector 不用 OpenSearch（设计决策 2，面试必考）】
1. 规模现实：采购知识库几百~几千 chunk，单机 PostgreSQL 的
   ivfflat/hnsw 索引毫秒级响应，OpenSearch 的分布式检索是过度设计；
2. 架构统一：业务数据（供应商/订单）和向量数据同库，
   一个事务里可以同时写订单 + 查文档（如合规校验场景）；
3. 运维成本：docker compose 一个数据库容器 vs 一个 OpenSearch 集群
   （后者要堆内存、调 JVM、管分片）；
4. 面试话术："选型的第一原则是规模匹配。能解释为什么不要 OpenSearch，
   比会堆 OpenSearch 更值钱。"

【ivfflat 索引的两个工程细节（面试高频）】
1. 建索引时机：ivfflat 需要真实数据训练聚类中心（lists 个），
   所以建表时不建、灌完数据后建（ensure_index）；
2. lists 取值：经验值 ≈ sqrt(数据行数) 或数据行数/1000，
   1000 行内 ≈ 10 个。数据量级变化时要重建索引——
   这也是为什么 ensure_index 是幂等函数。
"""

import logging
from typing import Any

from sqlalchemy import text

from app.database.connection import get_session_factory
from app.database.models import DocumentChunk

logger = logging.getLogger(__name__)


async def add_chunks(
    source: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """
    批量写入文档 chunk（text + embedding + source + chunk_index）。

    【为什么用 bulk insert 而不是逐条 add？】
    一份文档几十个 chunk，逐条 INSERT 意味着几十次网络往返。
    批量 insert 一次事务完成，快一个数量级且原子（失败不残留半截文档）。
    """
    assert len(chunks) == len(embeddings), "chunks 与 embeddings 数量必须一致"
    factory = get_session_factory()
    async with factory() as session:
        for chunk, vector in zip(chunks, embeddings):
            session.add(DocumentChunk(
                text=chunk["text"],
                embedding=vector,
                source=source,
                chunk_index=chunk["chunk_index"],
            ))
        await session.commit()
    logger.info("chunks_indexed", extra={"source": source, "count": len(chunks)})
    return len(chunks)


async def search_by_vector(
    query_vector: list[float],
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    纯向量检索：cosine 相似度 top_k。

    <#> 是 pgvector 的负内积操作符——向量已 L2 归一化（embedder.py），
    负内积升序 = 相似度降序。这是 pgvector 官方推荐的高性能写法，
    比显式 cosine 距离（<=>）在归一化场景下更快。

    【面试追问："为什么用 <#> 而不是 <=>？"】
    答：归一化后两者等价，但内积的计算量小于余弦距离
    （余弦要除两个模长，归一化后模长恒为 1，内积就是余弦）。
    这是"在数学等价的基础上选计算更便宜的"。
    """
    # 向量列表转 pgvector 字面量 '[0.1,0.2,...]'
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    filter_sql = ""
    params: dict[str, Any] = {"vec": vector_literal, "k": top_k}
    if source_filter:
        filter_sql = "AND source = :src"
        params["src"] = source_filter

    # 注意：不能用 `:vec::vector` 的简写 cast——SQLAlchemy text() 的
    # 绑定参数解析会把 `:vec::` 吃进去导致绑定失败（真实踩坑：
    # PostgresSyntaxError "语法错误 在 ':' 附近"）。
    # 必须用 CAST(:vec AS vector) 显式 cast，绑定参数与类型转换明确分离。
    sql = f"""
        SELECT text, source, chunk_index,
               1 - (embedding <#> CAST(:vec AS vector)) AS score
        FROM document_chunks
        WHERE embedding IS NOT NULL {filter_sql}
        ORDER BY embedding <#> CAST(:vec AS vector)
        LIMIT :k
    """
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    return [
        {
            "text": row.text,
            "source": row.source,
            "chunk_index": row.chunk_index,
            "score": round(float(row.score or 0.0), 4),
            "mode": "vector",
        }
        for row in rows
    ]


async def ensure_index() -> None:
    """
    幂等创建 ivfflat 索引（灌完数据后调用——ivfflat 需要数据训练）。

    lists=10：适用几千行以内的小库（经验值：数据行数/1000，最少 10）。
    面试追问"数据到百万行怎么办"：答——重建索引调大 lists，
    或换 HNSW（查询快、构建慢、内存占用大），两者都是 pgvector
    内置能力，无需换存储系统。
    """
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding "
            "ON document_chunks USING ivfflat (embedding vector_cosine_ops) "
            "WITH (lists = 10)"
        ))
        await session.commit()


async def count_chunks() -> int:
    """知识库 chunk 总数（health check / 文档状态展示用）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM document_chunks"))
        return int(result.scalar() or 0)
