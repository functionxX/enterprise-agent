# =============================================================
# FastAPI 应用镜像
#
# 构建（项目根目录）：
#   docker build -f docker/Dockerfile -t enterprise-agent:latest .
#
# 【多阶段构建为什么（面试点）】
# 第一阶段装依赖、第二阶段只拷贝运行所需——最终镜像没有
# gcc/构建工具链，体积小 30~50%，攻击面也更小。
# =============================================================

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
# 安装到独立目录，方便第二阶段整体拷贝（不需要重跑 pip）
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- 运行阶段 ----
FROM python:3.11-slim

# 非 root 用户：容器安全基线（面试点：镜像安全）
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY backend/ ./backend/
COPY knowledge/ ./knowledge/

ENV PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1

USER appuser

# 健康检查：复用应用自带的 /health（与 docker-compose 的 healthcheck 一致）
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
