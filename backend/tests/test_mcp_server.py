"""
test_mcp_server.py — MCP Server 封装测试（全 mock，进程内内存传输）

【测试策略】
MCP 适配层测试（与 test_agent / test_supervisor 互补）：
1. 工具发现：客户端能发现 execute_sql / search_knowledge，名称、描述、
   参数 schema 与 tools.py 的 TOOL_DEFINITIONS 完全一致（单一事实源断言）
2. execute_sql 端到端：只 mock 数据库会话工厂，其余走真实代码——
   证明 MCP 复用同一份实现，不是第二份胶水代码
3. 只读防御保留：DROP 语句经 MCP 调用同样被拒（isError）——协议无旁路
4. search_knowledge 端到端：只 mock retriever，走真实 tools.search_knowledge
5. 生产入口：python -m app.mcp_server 以 stdio 传输启动

【为什么用内存传输而不是 stdio 子进程？】
create_connected_server_and_client_session 在进程内跑完整 MCP 握手与
工具调用——零外部进程、零环境依赖（Windows 下 stdio 子进程还有事件
循环兼容性问题）；stdio 传输只在 __main__.py 里作为生产入口单独验证。
"""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from app.agent.tools import TOOL_DEFINITIONS
from app.mcp_server.server import mcp


def _result_payload(result: CallToolResult) -> str:
    """归一化工具返回：text 内容直接取，structured 内容 JSON 序列化。"""
    for content in result.content:
        if content.type == "text":
            return content.text
        if content.type == "structured":
            return json.dumps(content.structured, ensure_ascii=False)
    return ""


# =============================================================
# 假数据库层（只替换 execute_sql 的会话工厂，其余走真实代码）
# =============================================================

class _FakeDBResult:
    def __init__(self, columns: list[str], rows: list[list]):
        self._columns = columns
        self._rows = rows

    def fetchall(self) -> list[list]:
        return self._rows

    def keys(self) -> list[str]:
        return self._columns


class _FakeDBSession:
    """async with 上下文 + 一次 execute 返回固定结果的假会话。"""

    def __init__(self, columns: list[str], rows: list[list]):
        self._result = _FakeDBResult(columns, rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def execute(self, query):
        return self._result


@pytest.fixture
def fake_sql_session(monkeypatch):
    """把 execute_sql 的数据库层替换为固定结果（MCP 之外的一切都走真实代码）。

    真实调用链：get_readonly_session_factory() → sessionmaker → session
    （见 connection.py），所以 mock 也要返回一个"工厂的工厂"。
    """
    import app.agent.tools as tools_mod

    def make_session():
        return _FakeDBSession(["n"], [(1,)])

    monkeypatch.setattr(tools_mod, "get_readonly_session_factory", lambda: make_session)


# =============================================================
# 测试
# =============================================================

async def test_tools_discoverable_matching_source_of_truth():
    """
    工具发现：MCP 客户端能看到两个工具，且名称、描述、参数 schema
    与 tools.py 的 TOOL_DEFINITIONS 完全一致（单一事实源断言——
    两个协议出口必须共享同一份工具定义）。
    """
    async with create_connected_server_and_client_session(mcp) as client:
        listed = await client.list_tools()

    names = [t.name for t in listed.tools]
    assert names == ["execute_sql", "search_knowledge"]

    by_name = {t["function"]["name"]: t["function"] for t in TOOL_DEFINITIONS}
    for tool in listed.tools:
        definition = by_name[tool.name]
        # 描述：与 LangGraph 工具共享同一份文案
        assert tool.description == definition["description"]
        # 参数 schema：类型标注生成的 properties 与定义一致
        expected_props = set(definition["parameters"]["properties"].keys())
        assert set(tool.inputSchema.get("properties", {}).keys()) == expected_props


async def test_execute_sql_roundtrip_reuses_real_implementation(fake_sql_session):
    """
    execute_sql 端到端：只 mock 数据库层，走真实 tools.execute_sql
    （含截断标注、单元格格式化逻辑——返回结构与 LangGraph 内完全一致）。
    """
    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool("execute_sql", {"query": "SELECT 1"})

    assert not result.isError
    payload = json.loads(_result_payload(result))
    assert payload["columns"] == ["n"]
    assert payload["rows"] == [["1"]]
    assert payload["total_rows"] == 1
    assert payload["truncated"] is False


async def test_readonly_defense_not_bypassed_by_mcp(fake_sql_session):
    """
    只读防御随工具原样暴露：DROP 语句经 MCP 调用同样被拒——
    协议是新的，三层防御还是那三层，MCP 不新增旁路。
    """
    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool("execute_sql", {"query": "DROP TABLE suppliers"})

    assert result.isError
    payload = _result_payload(result)
    assert "DROP" in payload or "只允许" in payload


async def test_search_knowledge_roundtrip_reuses_real_implementation(monkeypatch):
    """
    search_knowledge 端到端：只 mock retriever，走真实 tools.search_knowledge
    （含降级标注逻辑——retrieval_mode 字段与 LangGraph 内一致）。
    """
    async def fake_search(query, top_k=5):
        return [{"text": "标准条文", "source": "quality_standard.txt",
                 "chunk_index": 0, "score": 0.9, "mode": "mock"}]
    monkeypatch.setattr("app.rag.retriever.search", fake_search)

    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool(
            "search_knowledge", {"query": "质量标准", "top_k": 3}
        )

    assert not result.isError
    payload = json.loads(_result_payload(result))
    assert payload["retrieval_mode"] == "mock"
    assert payload["top_k"] == 3
    assert payload["documents"][0]["source"] == "quality_standard.txt"


async def test_stdio_entrypoint(monkeypatch):
    """生产入口：python -m app.mcp_server 以 stdio 传输启动。"""
    from app.mcp_server import __main__ as main

    captured: dict[str, str] = {}
    monkeypatch.setattr(
        main.mcp, "run",
        lambda transport="stdio": captured.setdefault("transport", transport),
    )
    main.main()

    assert captured["transport"] == "stdio"
