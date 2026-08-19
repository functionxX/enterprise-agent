"""
health.py — 健康检查 API（GET /health）

【为什么要区分 healthy / degraded 两个状态（面试点）】
K8s 的 liveness/readiness probe 需要的不只是"进程活着"，
而是"依赖是否可用"：
- liveness：进程是否活着 → 只要 app 起来就 200
- readiness：依赖是否就绪 → DB/Redis 断了一个就应该 degraded
生产实践里这两者分离（K8s 配两个 probe 指向不同路径），
demo 合并为一个端点、用 status 字段表达。所有检查失败都吞掉异常——
健康检查自己不能因为一个依赖超时就把请求挂死（它要快速返回）。
"""

from redis.asyncio import Redis
from sqlalchemy import text

from app.database.connection import get_session_factory
from app.config import get_settings


async def _check_database() -> bool:
    """SELECT 1 探测业务库（pgvector 扩展与业务表同库）。"""
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _check_vector_store() -> bool:
    """探测 pgvector：查询向量表 + 确认扩展已安装。"""
    try:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            return result.scalar() == 1
    except Exception:
        return False


async def _check_redis() -> bool:
    """PING 探测 Redis（会话状态存储）。

    protocol=2：RESP2 在 Redis 5/6/7 全系列可用（RESP3 的 HELLO 命令
    老版本不支持会超时）——与 api/agent.py 的 _redis_client 保持一致。
    """
    try:
        settings = get_settings()
        client = Redis.from_url(settings.redis_url, socket_connect_timeout=2, protocol=2)
        try:
            return bool(await client.ping())
        finally:
            await client.aclose()
    except Exception:
        return False


async def health_check() -> dict:
    """汇总三项依赖检查，任一项失败 → degraded。

    返回体符合 docker-compose 的 healthcheck 需求：
    docker 容器内 wget/curl 这个端点即可做健康探测。
    """
    db_ok = await _check_database()
    vec_ok = await _check_vector_store()
    redis_ok = await _check_redis()

    return {
        "status": "healthy" if (db_ok and vec_ok and redis_ok) else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "vector_store": "connected" if vec_ok else "disconnected",
        "redis": "connected" if redis_ok else "disconnected",
    }
