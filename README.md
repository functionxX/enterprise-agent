# Enterprise AI Procurement Agent Platform

**企业采购智能 Agent 平台** — 面向 AI Agent 应用工程师 / SAP AI Agent Backend Developer 岗位的 portfolio 项目。

展示六大能力：**Agent 架构**（LangGraph 条件循环图 + Supervisor 多 Agent 协作 + Tool Calling）｜**人机协同**（HITL 人工审批挂起/恢复）｜**RAG 集成**（纯手写 pipeline 嵌入为 Agent Tool）｜**工具服务化**（MCP Server 双协议出口）｜**评估与可观测**（规则化 + LLM-as-judge 双轨评估，Langfuse tracing）｜**DevOps**（Docker Compose 一键启动，K8s 备查）。

---

## 架构总览

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
START → IntentAnalyzer → Planner → ToolExecutor → Router
                                      │               │
                                      │               ▼
                                      │           RAGRetriever
                                      │               │
                                      │               ▼
                                      └───────────→ Router
                                                      │
                                                      ▼ (没有更多 tool 要调 / 硬兜底)
                                              ResponseGenerator → END
```

**核心：条件循环图，不是线性链。** Router 每轮基于新获得的数据判断"继续查还是回答"——
执行 tool 后发现信息不够、SQL 报错、检索不相关，都会循环回去再查，直到信息充分或触发 `max_iterations=6` 硬兜底。

### 模式二：Supervisor 多 Agent 协作（`agent_mode="supervisor"`）

```
                 ┌──────────────────────────────────────┐
                 ▼                                      │
START → supervisor ──(next=sql_agent)──→ sql_worker ────┤
                 │   (next=rag_agent)──→ rag_worker ────┤
                 │   (next=finish)    ─→ response_generator → END
```

- **职责分层**：supervisor 只做调度（派谁/收尾）；sql_agent 与 rag_agent 是独立编译的子图，
  各自持有完整条件循环与**独立上下文**——结构化数据推理与知识库检索不再串味；
- **白名单隔离**：工具白名单经 state 通道按 worker 注入（sql_agent 只见 `execute_sql`）；
- **收敛保证**：派发轮数上限 + 已完成 worker 不重复派发 + LLM 非法输出收敛到 finish，
  与单 Agent 的 `max_iterations` 同哲学——LLM 调度 + 代码硬兜底。

### 通用工具与双协议出口

Agent 只有两个**通用 tool**（不是写死的业务函数）：

| Tool | 能力 | Agent 自主决定什么 |
|------|------|-------------------|
| `execute_sql` | 只读 SQL 查询（三层只读防御） | 查什么表、什么条件、怎么 JOIN |
| `search_knowledge` | 知识库检索（RAG pipeline） | 检索什么、检索几次、对照哪篇文档 |

这两个工具同时暴露为**标准 MCP Server**（`python -m app.mcp_server`，stdio 传输）：
复用同一份工具实现与描述（单一事实源，LangGraph 与 MCP 双协议出口零复制），
三层只读防御随协议原样生效——外部 Agent 生态可以直接消费本平台的采购分析能力。


### 人机协同：HITL 人工审批（`hitl=true`）

敏感操作（SQL 执行）前图挂起，把决定权交给人：

```
LLM 请求执行 SQL ─→ interrupt() 挂起 ─→ API 返回 pending_approval（HTTP 200，挂起是业务流程不是故障）
        ▲                                        │
        │            Command(resume=...)         ▼
        └──── 人工批准/拒绝 ──（checkpoint 按 thread_id 恢复现场）
```

- 批准 → SQL 正常执行；拒绝 → 理由回填 LLM，换策略重来（不是硬重试）；
- 开启 HITL 的图挂 checkpointer，挂起-恢复全链路 4 项测试覆盖，实测跨请求恢复成功；
- 编译期双变体：`enable_hitl` 默认关，不开 HITL 的图零改动、零开销。

---

## 快速开始

### Docker Compose 一键启动

```bash
cp .env.example .env        # 填写 LLM_API_KEY（必须）、EMBEDDING_API_KEY（可选）
docker compose -f docker/docker-compose.yml up -d --build

# 初始化种子数据（50 供应商 + 200 订单 + 30 发票，含埋点异常）
docker compose -f docker/docker-compose.yml exec app python -m app.database.seed
```

### 本地开发

```bash
cd backend
pip install -r ../requirements.txt
python -m app.database.seed                      # 种子数据（需要本地 PostgreSQL）
uvicorn app.main:app --reload                    # http://localhost:8000
```

### 验证

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "找出近三个月交付表现明显恶化的供应商，对照质量标准分析风险"}'
```

---

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/agent/chat` | POST | Agent 对话（`query` + 可选 `session_id` / `agent_mode` / `hitl` / `approval` / `include_contexts`） |
| `/api/v1/documents/upload` | POST | 知识库文档上传（txt/md/pdf） |
| `/health` | GET | 健康检查（database / vector_store / redis） |
| `/metrics` | GET | Prometheus 指标 |
| `/docs` | GET | Swagger 文档 |

Chat 请求参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `agent_mode` | `"single"` | `"single"` 单 Agent 条件循环图 / `"supervisor"` Supervisor 多 Agent 协作 |
| `hitl` | `false` | 开启 HITL：敏感工具（SQL）执行前挂起，响应返回 `pending_approval` |
| `approval` | `null` | HITL 续跑决定 `{"approved": true/false, "reason": "..."}`，需携带原 `session_id` |
| `include_contexts` | `false` | 评估/调试用：响应额外返回 Agent 实际依据的上下文原文（RAGAS 评分依据） |

Agent Chat 响应示例：

```json
{
  "answer": "恒达精密制造近三个月……",
  "sources": [
    {"type": "sql", "query": "SELECT ...", "result_summary": "返回 12 行"},
    {"type": "rag", "document": "quality_standard.txt", "chunk_index": 0, "score": 0.812}
  ],
  "tools_used": ["execute_sql", "search_knowledge", "execute_sql"],
  "iterations": 3,
  "token_usage": {"prompt": 4800, "completion": 2100, "total": 6900},
  "session_id": "uuid",
  "session_renewed": false,
  "warnings": [],
  "pending_approval": null,
  "contexts": []
}
```

`tools_used` 按调用顺序排列（含重复）——如实反映调用链。`sources` 提供数据溯源，`token_usage` 提供成本观测。
`pending_approval`：`hitl=true` 时敏感操作挂起返回待审批载荷（含 SQL 全文）；`contexts`：`include_contexts=true` 时返回 Agent 实际依据的上下文原文。

---

## 关键设计决策（17 项）

| # | 决策 | 要点 |
|---|------|------|
| 1 | **条件循环图** | Router 每轮动态决策；`add_conditional_edges` 实现循环；`max_iterations` 硬兜底 |
| 2 | **单一向量存储 pgvector** | 业务数据+向量同库；规模匹配原则：能解释为什么不要 OpenSearch |
| 3 | **通用 Tool + 三层只读防御** | prompt 约束 → 正则拦截 → `agent_readonly` 数据库角色 |
| 4 | **State 含错误与重试** | errors 带 `source_node`（选择性回退）；错误不中断流程 |
| 5 | **RAG 复用既有 pipeline** | 手写 loader→splitter→embedder→vector_store→retriever→generator |
| 6 | **种子数据非随机白噪音** | 风险 65/25/10 金字塔；订单帕累托分布；埋 3 组异常供 Agent 发现 |
| 7 | **多轮会话** | Redis 存摘要（非全文），30min TTL，5 轮注入，静默重建 |
| 8 | **LLM 调用成本意识** | 每轮 4 次调用，典型请求 8~12 次；token_usage 全程累计透出 |
| 9 | **SQL 结果截断** | MAX_ROWS=20 + `truncated` flag——触发下一轮"缩小范围重查" |
| 10 | **只保留两个 tool** | 砍掉伪需求 `generate_report`（LLM 原生会格式化）；两个 tool 各有不可替代性 |
| 11 | **知识库目录写入 tool description** | Agent 无需猜知识库结构；描述即索引 |
| 12 | **埋点异常联动评估** | 交付率骤降/发票逾期/高风险激增——评估集反向测试标准对照能力 |
| 13 | **mock 是降级路径** | 无 embedding key 时关键词检索，接口签名一致，drop-in 替换 |
| 14 | **错误带 source_node** | Router 选择性回退的依据；意图错→回意图节点，规划错→回规划节点 |
| 15 | **连接池参数** | pool_size=10、overflow=10、recycle=1800——每个数的理由在 connection.py |
| 16 | **降级策略分层** | Redis=增强依赖（挂了降级继续）；DB/LLM=核心依赖（挂了明确报错） |
| 17 | **并发安全** | graph 全局单例（无状态）+ state 每请求独立；瓶颈在资源池不在隔离 |
| 18 | **MCP 工具服务化** | 单一事实源：复用 tools.py 的工具描述与实现，LangGraph 与 MCP 双协议出口零复制；三层只读防御随协议原样生效；5 项适配层测试 |
| 19 | **HITL 人工审批** | `interrupt()` 在敏感工具执行前挂起 + checkpointer 按 `thread_id` 保存现场；`Command(resume=...)` 从挂起点恢复，批准执行 / 拒绝回填理由换策略；编译期双变体默认零改动 |
| 20 | **Supervisor 多 Agent** | worker 子图独立上下文 + 工具白名单经 state 注入隔离 + 派发轮数硬兜底（代码注释记为"多 Agent 版本"）；API `agent_mode` 双模式切换 |

---

## 评估体系

### 轨 1：规则化指标（可复现，骨架）

```bash
# 前置：服务运行中 + 种子数据已灌 + 知识库已上传（或 mock 降级）
python evaluation/evaluate.py --api-url http://localhost:8000
```

**17 条标注用例**（`evaluation/ground_truth.json`），三维度评分：

1. **Tool 调用准确率** — 调了预期的 tool 吗？（不匹配 SQL 字符串——JOIN 还是子查询是合法实现自由）
2. **答案关键信息覆盖** — 预期关键词命中率
3. **幻觉检测** — 回答中出现种子数据外的供应商名（名单用与 seed 相同的算法重建，永远一致）

最终指标：**任务完成率** = 三维度全部达标的用例占比。结果写入 `evaluation/evaluation_report.json`。
确定性 100%——跑 10 次结果一样，这是评估体系的第一性要求。

### 轨 2：LLM-as-judge（RAGAS，衡量真实质量）

```bash
# 独立 venv（评估栈与运行时隔离：ragas 0.2.x 配套 langchain 0.3.x，见 evaluation/requirements.txt）
python -m venv evaluation/.venv
PYTHONUTF8=1 evaluation/.venv/Scripts/python -m pip install -r evaluation/requirements.txt
PYTHONUTF8=1 evaluation/.venv/Scripts/python evaluation/ragas_eval.py --limit 3   # 试跑 3 条
PYTHONUTF8=1 evaluation/.venv/Scripts/python evaluation/ragas_eval.py              # 全量 17 条
```

- **指标**：`faithfulness`（回答对依据的忠实度）+ `answer_relevancy`（回答切题度）——
  规则测不到的主观质量维度交给评分 LLM（DeepSeek，temperature=0 保证评委确定性）；
- **为什么不用 context_precision/recall**：它们需要完整人工参考答案，而 ground_truth.json
  只标注了关键信息点——诚实边界：只跑有真实数据支撑的指标；
- **评的是真实调用链**：`include_contexts=true` 返回 Agent 实际依据的上下文原文
  （SQL + 返回行、检索 chunk 全文），faithfulness 对着真实依据打分，不是理想答案；
- 结果写入 `evaluation/ragas_report.json`。

### Langfuse tracing（LLM 调用级可观测）

配置 `.env` 后启动（不配置默认关闭，零影响）：
```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...      # cloud.langfuse.com 或自托管
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

每次 LLM 调用记一个 generation 观测：模型 / 输入输出 / token 用量 / 耗时，
失败尝试自动记 ERROR 级 span（重试排障可见）。**增强依赖容错**——tracing 挂了
只丢 trace 记 warning，绝不影响 Agent 主链路（6 项测试覆盖开关/埋点/容错边界）。

单测（全 mock，秒级）：`cd backend && python -m pytest tests/ -v`（46 项）

---

## Demo vs Production

| 维度 | Demo（本仓库） | Production |
|------|---------------|------------|
| **Agent 编排** | LangGraph 条件循环图，单 agent | 同架构；+ 多 agent 协作 / 子图（可平滑演进） |
| **LLM** | DeepSeek API（OpenAI 兼容） | 换 base_url 即换供应商；+ 私有化部署 + fallback 多模型路由 |
| **Embedding** | 硅基流动 BGE API | 本地 vLLM 自托管（只改 URL）；敏感文档不出内网 |
| **数据库** | PostgreSQL 16 + pgvector 单库 | 主从复制；chunk 千万级换 HNSW 索引或分库 |
| **会话状态** | Redis 单实例，30min TTL | Redis Sentinel/Cluster；会话跨实例共享 |
| **SQL 执行** | 只读角色 + 正则防御 | + 行级安全（RLS）+ SQL 预算/配额 + 审计日志 |
| **评估** | 规则化三维度 + RAGAS LLM-as-judge 双轨，本地跑 | CI 定时跑 + 评估集回归门禁（指标下降阻断合入） |
| **任务队列** | 同步请求 | 长任务 Celery/ASGI background + 进度回调 |
| **可观测性** | 结构化日志 + Prometheus + Grafana + Langfuse（LLM 调用级 tracing） | + OpenTelemetry 全链路（跨服务分布式追踪，Langfuse 埋点天然可升级） |
| **部署** | Docker Compose 一键起 | K8s（manifests 备查）+ ArgoCD + 金丝雀 |
| **安全** | 密钥走 .env，只读角色 | Secret Manager + Vault + IAM + 网络隔离 |

**每个"同架构/仅换 X"都是面试时的演进叙事**：demo 的架构骨架不需要推翻，只在规模和安全上加固。

---

## 项目结构

```
enterprise-ai-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI 入口（lifespan / 路由 / metrics）
│   │   ├── config.py              # pydantic-settings 配置中心（含 Langfuse）
│   │   ├── api/                   # agent.py / documents.py / health.py
│   │   ├── agent/                 # LangGraph：state / tools / nodes / graph / supervisor / llm_client
│   │   ├── mcp_server/            # MCP Server：单一事实源双协议出口（server.py + __main__.py）
│   │   ├── rag/                   # 手写 RAG：loader/splitter/embedder/vector_store/retriever/generator
│   │   ├── database/              # connection / models / schemas / seed
│   │   └── monitoring/            # 结构化日志 + Prometheus 中间件 + langfuse.py（LLM tracing）
│   └── tests/                     # test_agent / test_supervisor / test_mcp_server / test_hitl / test_langfuse / test_rag（46 项）
├── knowledge/                     # 采购知识库（3 篇文档，标准阈值与评估集联动）
├── evaluation/                    # ground_truth.json（17 用例）+ evaluate.py + ragas_eval.py（+ 独立 .venv）
├── docker/                        # Dockerfile / docker-compose.yml / init.sql（角色+权限）
├── kubernetes/                    # deployment + service（备查）
├── monitoring/                    # prometheus.yml + grafana dashboard
└── Jenkinsfile                    # CI 示意（评估阶段手动触发）
```

---
