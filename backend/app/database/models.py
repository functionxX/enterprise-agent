"""
models.py — SQLAlchemy 2.0 ORM 模型（业务数据 + 向量数据同库）

【为什么业务表和向量表放在同一个 PostgreSQL 里（设计决策 2）？】
- pgvector 是 PostgreSQL 扩展，向量检索和 OLTP 事务天然同库；
- 一个数据库服务两类查询：采购业务 SQL + 文档向量检索；
- 运维面最小：docker compose 只起一个数据库容器；
- 面试点：比"堆 OpenSearch"更重要的是能解释"为什么不需要"——
  数据量 < 千万 chunk 时，pgvector + ivfflat/hnsw 完全够用，
  OpenSearch 的分布式检索能力在这个规模是过度设计。

【ORM 风格说明】
用 SQLAlchemy 2.0 声明式风格（Mapped / mapped_column），
不是旧式 Column 风格——2.0 的类型标注让 IDE 和类型检查可用，
也是"生产级代码"的观感。
"""

from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""
    pass


class Supplier(Base):
    """供应商。

    【字段设计说明】
    - rating DECIMAL(2,1)：1.0~5.0，精确到 0.1——用 DECIMAL 不用 FLOAT，
      因为评分参与比较运算，浮点误差会导致 3.5 分边界判断出错（面试细节）。
    - risk_level：low/medium/high，种子数据按 65/25/10 金字塔分布（见 seed.py），
      现实企业中被保留的 high-risk 供应商通常是不可替代的战略供应商。
    """

    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    industry: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))
    rating: Mapped[Decimal | None] = mapped_column(Numeric(2, 1))
    risk_level: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class PurchaseOrder(Base):
    """采购订单。

    【为什么订单和发票拆成两张表？】
    现实采购系统里一单一票、一单多票（分批发货）都常见；
    拆表才能让 Agent 演示"JOIN 查询 + 发现逾期发票"这类真实分析路径。
    """

    __tablename__ = "purchase_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"))
    product_name: Mapped[str | None] = mapped_column(String(255))
    category: Mapped[str | None] = mapped_column(String(100))
    amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    quantity: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(50))  # pending/completed/cancelled
    order_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Invoice(Base):
    """发票。

    payment_status 的 overdue 状态是种子数据故意埋的异常点——
    评估集里的"发现逾期发票"题目依赖它（见 seed.py 与 ground_truth.json）。
    """

    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("purchase_orders.id"))
    invoice_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    payment_status: Mapped[str | None] = mapped_column(String(50))  # paid/unpaid/overdue
    due_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DocumentChunk(Base):
    """RAG 文档 chunk（pgvector 向量存储）。

    【为什么维度是 1024？】
    BAAI/bge-large-zh-v1.5 的输出维度固定为 1024。
    vector(1024) 在表创建时就定死——若换 embedding 模型（如 bge-m3 的 1024 之外），
    需要迁移表结构。这是 pgvector 的一个工程约束，面试时常被问到。

    【为什么建表时不直接建 ivfflat 索引？】
    ivfflat 需要先用真实数据训练（lists 聚类中心），空表建索引没有意义。
    索引在批量写入 chunk 之后由 vector_store.ensure_index() 创建——
    这也是"先灌数据、后建索引"的工程顺序。
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    source: Mapped[str | None] = mapped_column(String(255))
    chunk_index: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
