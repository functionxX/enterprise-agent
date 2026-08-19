-- =============================================================
-- PostgreSQL 初始化 SQL（docker-entrypoint-initdb.d 自动执行）
--
-- 职责：
-- 1. 安装 pgvector 扩展
-- 2. 创建两个数据库角色（三层只读防御的第 3 层，见 tools.py 注释）
--    - app_role         ：应用本体（建表/种子数据/文档 chunk 写入）
--    - agent_readonly   ：execute_sql tool 专用，仅 SELECT 权限
-- 3. ALTER DEFAULT PRIVILEGES：app_role 未来建的任何表自动给
--    agent_readonly 授予 SELECT——权限配置一次到位，不需要
--    每张新表手工 GRANT。
--
-- 【面试追问："为什么不用 superuser 跑应用？"】
-- 答：最小权限原则。应用只有建表需求，不需要 DROP DATABASE、
-- 创建用户等 superuser 能力。Agent 的 SQL tool 权限更窄——
-- 只读。多角色设计让"谁可以写、谁只能读"在数据库层就分得清清楚楚，
-- 即使应用代码被注入，损失也被角色权限限制住。
-- =============================================================

-- pgvector 扩展（向量检索）
CREATE EXTENSION IF NOT EXISTS vector;

-- 应用主角色：业务读写 + 向量写入
CREATE ROLE app_role WITH LOGIN PASSWORD 'app_password';

-- Agent 只读角色：execute_sql tool 专用（第 3 层只读防御）
CREATE ROLE agent_readonly WITH LOGIN PASSWORD 'readonly_password';

-- 两个角色均可连接本库
GRANT CONNECT ON DATABASE enterprise_agent TO app_role, agent_readonly;

-- 应用角色：schema 完整权限（建表/写入/建索引）
GRANT ALL ON SCHEMA public TO app_role;

-- 只读角色：schema 使用权 + 所有表现存数据只读
GRANT USAGE ON SCHEMA public TO agent_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_readonly;

-- 关键：app_role 未来创建的任何表，自动授权 agent_readonly 只读
ALTER DEFAULT PRIVILEGES FOR ROLE app_role IN SCHEMA public
    GRANT SELECT ON TABLES TO agent_readonly;

-- 业务表建在 init.sql 还是由 SQLAlchemy 建？
-- 由 SQLAlchemy（models.py Base.metadata）在应用启动时建：
-- ORM 模型是表结构的唯一事实来源（single source of truth），
-- SQL 脚本重复定义会导致两处漂移。init.sql 只负责"环境"（扩展+角色+权限）。
