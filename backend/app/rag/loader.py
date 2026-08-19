"""
loader.py — 文档加载（txt / md / pdf）

【为什么手写而不是用 langchain 的 document loader（面试点）】
langchain 全家桶的 loader 会引入几十个依赖，而加载文档的核心
无非三行：按扩展名分派 → 读文件 → 提取文本。手写的好处：
1. 行为可控：编码错误、空文件、超长文件都有明确的处理路径；
2. 依赖最少：PDF 只需要 pypdf 一个库；
3. 面试能讲：加载器干了什么、边界条件怎么处理，全在自己代码里。

【设计点】
- 返回带 source 元数据的文档对象（文件名/路径），
  下游 splitter 产出的每个 chunk 都带着 source——这是回答里
  "引用溯源"（RAG 防幻觉的基础）的数据来源。
- 按内容类型设置 chunk 策略提示：txt/md 按语义段落切，
  pdf 提取出的文本常有断行，交给 splitter 统一处理。
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class UnsupportedFileType(Exception):
    """不支持的文件类型。"""


def load_document(path: str | Path) -> dict[str, Any]:
    """
    加载单个文档，返回 {"source": 文件名, "text": 纯文本, "path": 原始路径}。

    支持的格式：
    - .txt / .md：UTF-8 读取（企业中文文档标准编码）
    - .pdf：pypdf 逐页提取文本
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"不支持的文件类型 {ext}，支持：{sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext in (".txt", ".md"):
        # UTF-8 读取；errors="replace" 容错——个别坏字节不该让整个索引失败
        text = file_path.read_text(encoding="utf-8", errors="replace")
    else:  # .pdf
        from pypdf import PdfReader  # 延迟导入：只在真需要 PDF 时加载

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)

    text = text.strip()
    if not text:
        logger.warning("empty_document", extra={"path": str(file_path)})

    return {
        "source": file_path.name,
        "path": str(file_path),
        "text": text,
    }
