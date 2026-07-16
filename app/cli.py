# laminate-mcp 콘솔 진입점 — stdio(기본) / http 전송 선택 (계획서 D6, §10)
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="laminate-mcp",
        description="적층 복합재 중립면·ABD 평가 MCP 서버 (stdio 기본, --transport http로 streamable HTTP)")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0이면 OS 자동 할당 (stderr 기동 로그로 확인)")
    args = parser.parse_args()

    if args.transport == "stdio":
        # stdio 모드에서 stdout은 JSON-RPC 전용 — 로그는 전부 stderr (계획서 §6.1)
        from app.mcp_server import mcp
        mcp.run()
    else:
        import uvicorn

        from app.main import app
        print(f"[laminate-mcp] http transport: {args.host}:{args.port or '(auto)'} /mcp", file=sys.stderr)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
