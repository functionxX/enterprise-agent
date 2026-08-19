# Enterprise AI Procurement Agent Platform — 构建 Prompt


---

## 项目定位

**企业采购智能 Agent 平台** — 面向 SAP AI Agent Backend Developer 岗位的 portfolio 项目。

核心展示能力：
1. **Agent 架构** — LangGraph 条件循环图，Tool Calling，自主规划
2. **RAG 集成** — 复用已有的纯手写 RAG pipeline 逻辑，嵌入为 Agent Tool
3. **Backend 工程** — FastAPI + PostgreSQL + Redis，生产级 API 设计
4. **评估体系** — Ground truth 测试集 + Agent 任务完成率评估
5. **DevOps** — Docker Compose 一键启动（K8s manifests 备查）

---

## 项目结构

```
enterprise-ai-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 入口
│   │   ├── config.py                  # 环境变量配置
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py               # POST /api/v1/agent/chat
│   │   │   ├── documents.py           # POST /api/v1/documents/upload
│   │   │   └── health.py              # GET /health
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py               # LangGraph 循环图定义
│   │   │   ├── state.py               # Agent State Schema
│   │   │   ├── nodes.py               # 图节点实现（Intent/Planner/Executor/Generator）
│   │   │   └── tools.py               # Agent Tools 定义（通用 tool，不写死业务逻辑）
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py              # 文档加载（txt/pdf/md）
│   │   │   ├── splitter.py            # 文本分割（递归分割）
│   │   │   ├── embedder.py            # BGE Embedding
│   │   │   ├── vector_store.py        # pgvector 存储 + 检索
│   │   │   ├── retriever.py           # 三种检索模式 + Rerank + Query Rewrite
│   │   │   └── generator.py           # RAG Prompt 构建 + LLM 生成
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py          # 数据库连接管理
│   │   │   ├── models.py              # SQLAlchemy ORM 模型
│   │   │   ├── schemas.py             # Pydantic 请求/响应模型
│   │   │   └── seed.py                # 种子数据生成脚本
│   │   └── monitoring/
│   │       ├── __init__.py
│   │       ├── logger.py              # 结构化日志
│   │       └── middleware.py          # 请求耗时/错误率中间件
│   └── tests/
│       ├── test_agent.py              # Agent 流程测试
│       ├── test_rag.py                # RAG 检索测试
│       └── fixtures/                  # 测试数据 + Ground Truth 评估集
├── knowledge/                         # 采购知识库文档（txt/md/pdf）
│   ├── supplier_policy.txt
│   ├── procurement_rules.txt
│   ├── quality_standard.txt
│   └── README.md                      # 知识库说明
├── evaluation/                        # 评估体系
│   ├── ground_truth.json              # 标注测试用例（query + expected_tools + expected_answer）
│   └── evaluate.py                    # 评估脚本
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── init.sql                       # 数据库初始化 SQL
├── kubernetes/                        # 备查，不做主要 demo
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 关键设计决策（面试时会被追问，代码里必须体现）

### 1. Agent 图必须是条件循环图，不是线性链 ❗最重要

线性流程是错的。Agent 的核心在于"执行 tool 后发现信息不够，循环回去再查"。

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
START → IntentAnalyzer → Planner → ToolExecutor → Router
                                      │               │
                                      ▼               │
                                  RAGRetriever ───────┘
                                      │
                                      ▼ (没有更多 tool 要调)
                              ResponseGenerator → END
```

LangGraph 实现要点：
- 用 `add_conditional_edges` 做路由——Router 节点判断"还有 tool 需要调吗？"→ 有则回到 ToolExecutor，无则进入 Generator
- 每个 tool 执行后追加到 `tool_results`，保留完整调用链
- 设置 `max_iterations`（如 6）防止无限循环

### 2. 只用一个向量存储：pgvector

不要 OpenSearch。理由：
- pgvector 已覆盖向量检索 + PostgreSQL 全文检索（`tsvector`）
- 一个数据库同时服务 OLTP（采购数据）+ 向量检索（RAG）
- 运维简单，Docker Compose 一键启动
- 面试时能解释比能堆砌更重要

### 3. Agent Tools 是通用能力，不是写死的 SQL

❌ 错误示范：`query_supplier_risk(supplier_name)` — 这是业务函数
✅ 正确示范：

```python
# Agent 能用的三个通用 tool
tools = [
    {
        "name": "execute_sql",
        "description": "在 PostgreSQL 上执行只读 SQL 查询。返回查询结果。",
        "parameters": {"query": "SELECT 语句"}
    },
    {
        "name": "search_knowledge",
        "description": "从企业知识库中检索相关文档。用于查政策、规则、标准等。",
        "parameters": {"query": "检索查询", "top_k": "返回数量"}
    },
    {
        "name": "generate_report",
        "description": "将分析结果格式化为结构化报告。",
        "parameters": {"content": "分析内容", "format": "markdown|json"}
    }
]
```

面试时这样讲：
> "Agent 自己写 SQL 并执行——不是我提前写好几个查询函数。它根据用户意图自主决定查什么表、用什么条件。"

### 4. Agent State 要包含错误处理和重试

```python
class AgentState(TypedDict):
    user_query: str
    conversation_history: list[dict]   # 多轮对话
    task_plan: list[str]               # Planner 产出的步骤列表
    current_step: int                  # 当前执行到第几步
    tool_results: list[dict]           # 每次 tool 调用的输入输出
    retrieved_documents: list[dict]    # RAG 检索结果
    errors: list[str]                  # 错误收集（不中断流程）
    retry_count: int                   # 重试计数
    final_answer: str
    tools_used: list[str]              # 最终用了哪些 tool（供 API 返回）
```

### 5. RAG 模块复用已有逻辑

不需要从零设计 RAG——直接在 `rag/` 目录下实现和 RAG-demo 相同的 pipeline：
- `loader.py` → `splitter.py`（递归分割）→ `embedder.py`（BGE-large-zh-v1.5）→ `vector_store.py`（pgvector）→ `retriever.py`（向量/混合/Rerank）→ `generator.py`（Prompt + LLM）

和 RAG-demo 的唯一区别：知识库文档换成采购相关的。其他代码逻辑直接复用。

### 6. 必须有种子数据脚本

`database/seed.py` 生成：
- 50 个供应商（不同行业、风险等级、国家）
- 200 条采购订单（过去 6 个月）
- 30 条发票

没有数据，Agent Tool 跑不了，整个 demo 演示不了。

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | Python 3.11+ |
| ORM | SQLAlchemy 2.0 | 异步 |
| 数据校验 | Pydantic v2 | |
| 数据库 | PostgreSQL 16 + pgvector | 同时存业务数据 + 向量 |
| 缓存/状态 | Redis | Agent 会话状态 |
| Agent 框架 | LangGraph | 条件循环图 |
| LLM | DeepSeek (OpenAI 兼容) | deepseek-chat |
| Embedding | BAAI/bge-large-zh-v1.5 | 硅基流动 API |
| 部署 | Docker Compose | |
| 监控 | Prometheus + Grafana | 备查 |

---

## 数据库表设计（只读业务数据 + 向量数据放在一起）

```sql
-- 供应商
CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    country VARCHAR(100),
    rating DECIMAL(2,1),              -- 1.0-5.0
    risk_level VARCHAR(20),           -- low/medium/high
    created_at TIMESTAMP DEFAULT NOW()
);

-- 采购订单
CREATE TABLE purchase_orders (
    id SERIAL PRIMARY KEY,
    supplier_id INT REFERENCES suppliers(id),
    product_name VARCHAR(255),
    category VARCHAR(100),
    amount DECIMAL(12,2),
    quantity INT,
    status VARCHAR(50),               -- pending/completed/cancelled
    order_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 发票
CREATE TABLE invoices (
    id SERIAL PRIMARY KEY,
    order_id INT REFERENCES purchase_orders(id),
    invoice_amount DECIMAL(12,2),
    payment_status VARCHAR(50),       -- paid/unpaid/overdue
    due_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- RAG 文档 chunk（pgvector）
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1024),
    source VARCHAR(255),
    chunk_index INT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
```

---

## API 设计

### POST /api/v1/agent/chat

```json
// Request
{
  "query": "分析供应商A过去三个月的采购风险和交付表现",
  "session_id": "uuid-optional"
}

// Response
{
  "answer": "供应商A过去三个月...",
  "sources": [
    {"type": "sql", "query": "SELECT ...", "result_summary": "返回 12 行"},
    {"type": "rag", "document": "supplier_policy.txt", "chunk_index": 3}
  ],
  "tools_used": ["execute_sql", "search_knowledge", "execute_sql"],
  "iterations": 4,
  "token_usage": {"prompt": 1200, "completion": 800}
}
```

### POST /api/v1/documents/upload

```json
// Request: multipart/form-data, file field
// Response
{
  "filename": "supplier_policy.pdf",
  "chunks_created": 15,
  "status": "indexed"
}
```

### GET /health

```json
{
  "status": "healthy",
  "database": "connected",
  "vector_store": "connected",
  "redis": "connected"
}
```

---

## 评估体系（evaluation/）

`evaluation/ground_truth.json`：

```json
[
  {
    "id": "eval_001",
    "query": "过去三个月采购金额最高的供应商是谁？",
    "expected_tools": ["execute_sql"],
    "expected_tables": ["purchase_orders", "suppliers"],
    "expected_answer_contains": ["供应商名称", "金额"],
    "difficulty": "easy"
  },
  {
    "id": "eval_002",
    "query": "供应商A的交货表现如何？对照我们的供应商评估标准分析",
    "expected_tools": ["execute_sql", "search_knowledge"],
    "expected_tables": ["purchase_orders"],
    "expected_answer_contains": ["交付率", "评估标准", "风险"],
    "difficulty": "medium"
  }
  // ... 至少 15 个标注用例
]
```

`evaluation/evaluate.py` 评估维度：
1. **Tool 调用准确率** — Agent 调了预期的 tool 吗？
2. **答案包含关键信息** — 预期关键词是否出现在回答中？
3. **无幻觉** — 没有编造不存在的供应商/数据

---

## Phase 执行计划

### Phase 1：Backend 骨架 + 数据库 + 种子数据（P0）
- FastAPI 项目初始化
- PostgreSQL + SQLAlchemy 模型 + pgvector 初始化
- 种子数据脚本 `seed.py`（50 供应商 + 200 订单 + 30 发票）
- 健康检查 API
- `.env.example` + `config.py`

### Phase 2：Agent 核心（P0）
- LangGraph 循环图（`graph.py` + `nodes.py` + `state.py`）
- Tool 定义（`tools.py`）— `execute_sql` / `search_knowledge` / `generate_report`
- Agent Chat API（`POST /api/v1/agent/chat`）
- 错误处理 + 重试 + `max_iterations` 限制
- 先 mock RAG tool（返回假检索结果），确保 Agent 流程跑通

### Phase 3：RAG Pipeline（P0）
- 复用已有 RAG 逻辑实现 `rag/` 模块
- 三种检索模式（向量/混合/Rerank）+ Query Rewrite
- 采购知识库文档准备（`knowledge/` 目录至少 3 篇 txt）
- Document Upload API
- 把 RAG tool 接入 Agent

### Phase 4：评估体系（P1）
- `evaluation/ground_truth.json`（至少 15 个标注用例）
- `evaluation/evaluate.py` 评估脚本
- `tests/` 下的 Agent 和 RAG 单元测试

### Phase 5：Docker Compose（P1）
- Dockerfile（FastAPI 应用）
- `docker-compose.yml`（app + PostgreSQL/pgvector + Redis + Prometheus + Grafana）
- `docker/init.sql`

### Phase 6：K8s + CI/CD + 监控（P2 — 备查）
- `kubernetes/deployment.yaml` + `service.yaml`
- `Jenkinsfile`
- `monitoring/prometheus.yml` + Grafana Dashboard JSON
- 结构化日志中间件

---

## 要求

1. **所有代码用 Python 3.11 + FastAPI + 异步（async/await）**
2. **Agent 图必须是条件循环图，不是线性链**（见设计决策 1）
3. **Tools 是通用能力，不写死业务 SQL**（见设计决策 3）
4. **每个 `.py` 文件有详细中文注释**——解释"为什么这么设计"，面试时能直接讲
5. **一次性完成所有 Phase**，不需要中途停下来确认——完整建好整个项目
6. **README.md 包含 Demo vs Production 对比表**（参考 RAG-demo 的风格）
7. **面试常见追问写在代码注释里**——例如 `tools.py` 注释里解释"为什么不让 Agent 执行写操作 SQL 而是限制为只读查询"
