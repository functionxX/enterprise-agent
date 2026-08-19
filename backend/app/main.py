"""
main.py — FastAPI 应用入口

【启动流程（lifespan）】
1. 初始化结构化日志
2. 校验 LLM_API_KEY 存在（Agent 没 LLM 就无意义——fail fast）
3. 初始化双数据库引擎（主 + 只读）
4. 挂载路由 + 监控中间件 + /metrics
5. 关闭时优雅释放连接池

【面试点：FastAPI lifespan 与启动顺序】
依赖初始化放在 lifespan 而不是模块 import 时——
模块导入期做 I/O（连数据库）会让单测、CLI 脚本（seed.py 等）
全部被迫拉起连接，也违反"可导入即无副作用"的工程约定。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.agent import router as agent_router
from app.api.documents import router as documents_router
from app.api.health import health_check
from app.config import get_settings
from app.database.connection import dispose_engines, init_engines
from app.monitoring.logger import get_logger, setup_logging
from app.monitoring.middleware import MonitoringMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动初始化 / 关闭清理。"""
    settings = get_settings()
    setup_logging(settings.log_level)

    # LLM key 缺失直接启动失败：Agent 应用没有 LLM 等于空壳，
    # 与其让用户请求时才发现，不如启动时明确报错（fail fast）
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY 未配置。请复制 .env.example 为 .env 并填写 DeepSeek API Key。"
        )

    init_engines(settings)

    # 建表（幂等 create_all）：ORM 模型是表结构的唯一事实来源，
    # init.sql 只负责环境（扩展/角色/权限）——两处定义必然漂移，一处为准
    from app.database.connection import get_session_factory
    from app.database.models import Base

    factory = get_session_factory()
    async with factory() as session:
        # create_all 必须跑在 Connection 上（session.run_sync 传的是 Session 会报错），
        # session.connection() 取底层连接后 run_sync 执行同步 DDL
        conn = await session.connection()
        await conn.run_sync(Base.metadata.create_all)

    logger.info("app_started", extra={"request_id": "-"})
    yield
    # Langfuse 冲刷：进程退出前把未上报的 trace 送出（增强依赖，失败静默）
    from app.monitoring.langfuse import flush_langfuse
    flush_langfuse()
    await dispose_engines()
    logger.info("app_stopped", extra={"request_id": "-"})


app = FastAPI(
    title="Enterprise AI Procurement Agent",
    description=(
        "企业采购智能 Agent：LangGraph 条件循环图 + 通用 Tool Calling + 手写 RAG。\n\n"
        "核心端点：POST /api/v1/agent/chat（Agent 对话）｜"
        "POST /api/v1/documents/upload（知识库文档上传）｜GET /health（健康检查）"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---- 监控中间件（请求耗时/错误率 + request_id 注入）----
app.add_middleware(MonitoringMiddleware)

# ---- 路由挂载 ----
app.include_router(agent_router, prefix="/api/v1/agent")
app.include_router(documents_router, prefix="/api/v1/documents")


@app.get("/health")
async def health() -> JSONResponse:
    """健康检查（docker healthcheck / K8s probe 使用）。"""
    return JSONResponse(await health_check())


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus 抓取端点。指标定义见 monitoring/middleware.py。"""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root() -> dict:
    """根路径：跳转提示。"""
    return {
        "app": "Enterprise AI Procurement Agent",
        "docs": "/docs",
        "health": "/health",
        "agent_chat": "/api/v1/agent/chat",
    }
