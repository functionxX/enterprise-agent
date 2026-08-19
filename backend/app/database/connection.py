"""
connection.py — 数据库连接管理（双引擎设计）

【为什么有两个引擎？】
1. main_engine（app_role）    —— 应用本体：建表、种子数据、文档 chunk 写入
2. readonly_engine（agent_readonly）—— execute_sql tool 专用：数据库层面只有 SELECT 权限

这是"三层只读防御"（prompt 约束 → 正则拦截 → 数据库权限）的第三层。
前两层都是应用内逻辑，存在被绕过/写错的可能；数据库角色权限是 PostgreSQL
内核强制的，Agent 生成的 SQL 即使写进 INSERT 也会被数据库拒绝——
【面试追问】"如果 LLM 写出了 INSERT 而你的正则没拦住怎么办？"
答："数据库用 agent_readonly 角色拒绝它，最后防线在 DBMS 而不是应用代码。"

【为什么本地开发时 readonly_url 可以等于主连接？】
Docker 部署时 init.sql 会创建两个角色并收紧权限（见 docker/init.sql）。
本地裸跑 PostgreSQL 时如果没有单独创建 agent_readonly 角色，
readonly_url 会 fallback 到主连接——功能可用，但第三层防御缺失。
代码里保留显式 fallback 而不是直接报错，是为了 demo 易跑；
生产部署（docker compose）一定有独立只读角色。

【连接池参数为什么是这些数（面试高频）】
- pool_size=10：经验公式 ≈ 2×CPU核数+1（4核容器→9，取整 10）。
  更深层原因：asyncpg 单连接是单路复用的——一个连接同一时刻只能跑一条 SQL，
  Agent 图内节点串行执行，单请求实际只用 1~2 个连接，10 是给并发请求的余量。
- max_overflow=10：流量尖峰时允许临时多开 10 个（总上限 20），
  防止连接数爆炸把 PostgreSQL 拖垮。
- pool_timeout=30：拿不到连接时 30 秒抛异常而不是无限等，上层捕获返回 503。
- pool_recycle=1800：Docker 场景特有——容器间 overlay 网络里空闲连接更容易
  被 NAT 表淘汰而"静默死亡"。闲置 30 分钟主动回收重建，避免拿到死连接才报错。
  本地直连 localhost 基本不会断，但参数统一设置，行为一致。
- pool_pre_ping=True：每次取连接先发轻量 ping 探测。与 recycle 双保险。

【并发安全（面试高频）】
引擎与连接池是全局共享的，但它们是线程/协程安全的资源池，
设计目标就是被多个请求并发复用。真正的并发风险不在"串数据"——
每个请求的 AgentState 是独立对象——而在"资源耗尽"：
50 个并发 Agent 请求 × 每请求 8~12 次 LLM 调用 + 多次 DB 查询，
连接池会先被打爆。所以 pool 参数与 LLM 调用频率才是瓶颈，state 隔离是 LangGraph 白送的。
"""

import logging
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 模块级单例引擎。FastAPI 是单进程多协程模型，全局共享是标准做法；
# 多 worker 部署时每个进程各自建池（uvicorn --workers N），不存在跨进程共享。
_main_engine: AsyncEngine | None = None
_readonly_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_readonly_session_factory: async_sessionmaker[AsyncSession] | None = None


def _create_engine(settings: Settings, url: str) -> AsyncEngine:
    """按统一参数创建异步引擎。两个引擎共用同一套池参数。"""
    return create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
        echo=settings.debug,
    )


def init_engines(settings: Settings) -> None:
    """应用启动时初始化双引擎（main.py lifespan 调用）。"""
    global _main_engine, _readonly_engine, _session_factory, _readonly_session_factory

    _main_engine = _create_engine(settings, settings.database_url)
    _session_factory = async_sessionmaker(_main_engine, expire_on_commit=False)

    # 只读引擎：优先使用独立只读连接；未单独配置时 fallback 到主引擎
    # （fallback 时第三层数据库权限防御缺失，仅本地 demo 可接受）
    _readonly_url = settings.database_readonly_url or settings.database_url
    if _readonly_url == settings.database_url:
        logger.warning(
            "database_readonly_url 与主连接相同：execute_sql 的数据库层只读防御未生效。"
            "生产部署请使用 docker/init.sql 创建的 agent_readonly 角色。"
        )
    _readonly_engine = _create_engine(settings, _readonly_url)
    _readonly_session_factory = async_sessionmaker(_readonly_engine, expire_on_commit=False)


async def dispose_engines() -> None:
    """应用关闭时释放连接池（优雅退出，防连接泄漏）。"""
    if _main_engine is not None:
        await _main_engine.dispose()
    if _readonly_engine is not None:
        await _readonly_engine.dispose()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """主会话工厂（业务读写）。"""
    assert _session_factory is not None, "引擎未初始化——请先调用 init_engines()"
    return _session_factory


def get_readonly_session_factory() -> async_sessionmaker[AsyncSession]:
    """只读会话工厂（execute_sql tool 专用）。"""
    assert _readonly_session_factory is not None, "引擎未初始化——请先调用 init_engines()"
    return _readonly_session_factory


@asynccontextmanager
async def session_scope():
    """
    主会话上下文管理器：自动 commit / rollback / close。

    【为什么用上下文管理器而不是裸 session？】
    裸 session 最典型的 bug 是连接泄漏——异常路径忘了 close，
    连接池被慢慢掏空，最后表现为"偶发 pool timeout"。
    上下文管理器保证任何路径（包括异常）都会释放连接回池。
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
