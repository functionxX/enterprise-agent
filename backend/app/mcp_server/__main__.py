"""MCP Server 入口：`python -m app.mcp_server` 以 stdio 传输启动。

启动后等待 MCP 客户端（Claude Desktop / IDE / 自研 Agent 平台）通过
标准输入输出连接——stdio 是 MCP 默认传输，进程即服务，零网络配置。
"""

from app.mcp_server.server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
