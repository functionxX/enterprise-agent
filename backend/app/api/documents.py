"""
documents.py — Document Upload API（POST /api/v1/documents/upload）

流程：上传文件 → loader 加载 → splitter 分割 → embedder 向量化 → vector_store 写入 → 建索引

【设计点（面试追问）】
1. 为什么是同步上传而不是后台任务？
   Demo 规模：一篇知识库文档几十个 chunk，embedding 批量一次调用，
   整个 pipeline 秒级完成。生产环境大文档/大批量应换 Celery/RQ 队列，
   上传接口只返回"已受理"——但 demo 阶段同步响应让 API 语义简单，
   面试能讲清"什么规模做什么选择"即可。
2. 文件大小限制：8MB。LLM/embedding API 按 token 计费，
   不限制上传大小等于开放了一个烧钱入口。
3. 覆盖式索引：同名文档重新上传不产生重复 chunk——先删旧 chunks
   再写入（DELETE + INSERT 同事务）。
4. 依赖语义：embedding 是硬依赖（没有它无法索引），返回 503
   而不是假装成功——与 Agent chat 里 Redis 的软降级形成对比，
   面试时可以主动讲这组对比："索引必须真实，会话历史可以没有"。
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import delete

from app.config import get_settings
from app.database.schemas import UploadResponse
from app.rag.embedder import get_embedder
from app.rag.loader import UnsupportedFileType, load_document
from app.rag.splitter import normalize_whitespace, split_text
from app.rag.vector_store import add_chunks, ensure_index

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8MB


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """上传采购知识库文档（txt/md/pdf），完成切片、向量化、入库、建索引。"""
    settings = get_settings()

    # ---- 依赖检查：embedding 是硬依赖（与 Redis 软降级对比，见 docstring）----
    if not get_embedder().is_configured:
        raise HTTPException(
            status_code=503,
            detail="EMBEDDING_API_KEY 未配置，文档索引不可用。请配置硅基流动 API Key。",
        )

    # ---- 读取文件（带大小限制）----
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_SIZE // 1024 // 1024}MB 限制")
    if not content:
        raise HTTPException(status_code=400, detail="空文件")

    # ---- 落盘临时文件（loader 按路径读取，支持 pdf 等二进制格式）----
    suffix = Path(file.filename or "doc.txt").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ---- loader → splitter → embedder → vector_store ----
        try:
            doc = load_document(tmp_path)
            doc["source"] = file.filename or doc["source"]  # 用原始上传名
        except UnsupportedFileType as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        text = normalize_whitespace(doc["text"])
        chunks = split_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档没有可索引的文本内容")

        embedder = get_embedder()
        embeddings = await embedder.embed([c["text"] for c in chunks])

        # 覆盖式索引：同名文档先删旧 chunks（幂等重传）
        from app.database.connection import get_session_factory
        from app.database.models import DocumentChunk

        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.source == doc["source"])
            )
            await session.commit()

        chunks_created = await add_chunks(doc["source"], chunks, embeddings)
        await ensure_index()  # ivfflat 需数据训练，灌完再建（幂等）

        logger.info("document_indexed", extra={
            "doc_source": doc["source"], "chunks": chunks_created,
        })
        return UploadResponse(filename=doc["source"], chunks_created=chunks_created)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 —— embedding API 等外部故障
        logger.error("upload_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=502, detail=f"文档索引失败：{exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # 清理临时文件
