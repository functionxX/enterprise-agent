"""
embedder.py — BGE Embedding（硅基流动 API，OpenAI 兼容协议）

【为什么选 BAAI/bge-large-zh-v1.5（面试点）】
1. 中文检索的强基线：BGE 系列在 C-MTEB 中文检索榜长期领先，
   采购知识库是纯中文场景，用英文系模型（如 text-embedding-ada）
   会因分词粒度错配掉召回；
2. 1024 维在精度/存储/检索速度之间平衡得当（维度越高越贵越慢）；
3. 通过硅基流动 API 调用——与 LLM 一样走 OpenAI 兼容协议，
   一个 openai SDK 通吃（base_url 不同而已），零额外依赖。
   生产自托管时换成本地 vLLM 部署，只改一个 URL。

【归一化为什么重要（面试高频）】
BGE 官方建议 query 和 passage 向量都做 L2 归一化后再算相似度。
归一化后余弦相似度 = 向量点积——pgvector 的 <#> 操作符（负内积）
可以当余弦相似度用，且内积比余弦距离计算更快。
"""

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """BGE 文本向量化客户端。

    【为什么是类而不是函数？】
    AsyncOpenAI 客户端内部维护连接池，应在进程内复用。
    应用启动时构建单例，与 LLM 客户端同构。
    """

    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            timeout=30.0,   # embedding 比 chat 快，30s 足够
            max_retries=1,  # embedding 失败由上层降级路径兜底（mock 检索）
        )
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

    @property
    def is_configured(self) -> bool:
        """API key 是否存在——未配置时 RAG 走 mock 降级（见 retriever.py）。"""
        return bool(get_settings().embedding_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量向量化文本，返回归一化后的向量列表。

        【为什么批量？】
        文档切出几十个 chunk 逐个调用 API 会打爆速率限制。
        一次请求带全部 chunk，API 侧并行处理，成本与延迟都低。
        """
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )
        vectors = [item.embedding for item in resp.data]
        return [_normalize(v) for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        """单条 query 向量化（检索时用）。"""
        vectors = await self.embed([text])
        return vectors[0]


def _normalize(vector: list[float]) -> list[float]:
    """L2 归一化：使余弦相似度可用点积（pgvector <#>）计算。"""
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return vector
    return [x / norm for x in vector]


# 进程级单例（与 LLM 客户端同构）
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(get_settings())
    return _embedder
