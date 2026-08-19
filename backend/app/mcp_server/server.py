"""
server.py — 工具即服务：MCP Server 封装（设计决策 18，面试核心考点）

【为什么把工具封装成 MCP Server？（Agent 内部能力 → 生态可复用能力）】
LangGraph 里的 execute_sql / search_knowledge 是 OpenAI function-calling
格式的工具——消费方只有本 Agent 一个，绑定在 graph 内部。
MCP（Model Context Protocol）把工具变成标准协议服务：任何支持 MCP 的
客户端（Claude Desktop、IDE、自研 Agent 平台）都能"发现 → 调用"，
工具从"Agent 内部函数"升级为"可以被整个生态复用的能力"。

【设计原则：单一事实源，零复制（面试追问："新加一个工具改几个文件？"）】
工具实现（TOOL_IMPLEMENTATIONS）与工具描述（TOOL_DEFINITIONS）全部
来自 tools.py——本模块只做协议适配（FastMCP 装饰器 + 参数类型标注），
不维护第二份工具代码。LangGraph 与 MCP 是同一份工具的两个协议出口：
改 tools.py 一处，两个出口自动生效。

【安全边界不随协议消失】
MCP 入口暴露的是同一个 execute_sql：三层只读防御（prompt 约束 /
正则校验 / 数据库 agent_readonly 角色）随工具原样暴露。
协议是新的，防御层还是那三层——MCP 不新增任何旁路。

【传输选型：stdio】
MCP 的标准传输是 stdio（子进程 stdin/stdout）——进程即服务，
零网络配置、零鉴权面，最适合"一个 Agent 平台挂载一组工具服务"。
sse / streamable-http 传输适合跨主机服务化（见 mcp.run 参数）。
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.agent.tools import (
    TOOL_DEFINITIONS,
    execute_sql as _execute_sql,        # 真实实现（含三层只读防御）
    search_knowledge as _search_knowledge,
)

# 服务实例：`python -m app.mcp_server` 即启动（__main__.py）。
# instructions 是给 MCP 客户端的服务级介绍（Client 侧 LLM 会读到）。
mcp = FastMCP(
    "procurement-agent",
    instructions=(
        "企业采购分析工具集：提供只读 SQL 查询（结构化数据）与知识库检索"
        "（政策/规则/标准等非结构化知识）两个能力。所有查询均为只读。"
    ),
)


def _tool_description(name: str) -> str:
    """从 tools.py 的 TOOL_DEFINITIONS 取描述——LangGraph 与 MCP 共享同一份文案。"""
    for definition in TOOL_DEFINITIONS:
        if definition["function"]["name"] == name:
            return definition["function"]["description"]
    raise KeyError(f"工具未定义: {name}")


# 参数 schema 由类型标注自动生成（FastMCP 约定），与 TOOL_DEFINITIONS
# 的参数形状保持一致：execute_sql(query) / search_knowledge(query, top_k=5)。


@mcp.tool(description=_tool_description("execute_sql"))
async def execute_sql(query: str) -> dict[str, Any]:
    """在 PostgreSQL 上执行只读 SQL 查询（复用 tools.execute_sql，三层只读防御原样生效）。"""
    return await _execute_sql(query)


@mcp.tool(description=_tool_description("search_knowledge"))
async def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """从企业知识库检索（复用 tools.search_knowledge，无 embedding key 时自动降级关键词检索）。"""
    return await _search_knowledge(query, top_k=top_k)
