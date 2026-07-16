# HEAXHub fastapi 스택 진입점 — /health + MCP streamable HTTP(/mcp)를 한 앱에 마운트 (계획서 §16.6)
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import config
from app.mcp_server import mcp

_mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # 마운트된 서브앱의 lifespan은 자동 실행되지 않으므로 세션 매니저를 직접 구동한다.
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="laminate-analyzer-mcp", version=config.SERVER_VERSION, lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    """HEAXHub 스택 헬스체크(health_path: /health) + MCP 엔드포인트 안내."""
    return {"status": "ok", "service": "laminate-analyzer-mcp",
            "version": config.SERVER_VERSION, "mcp_endpoint": "/mcp"}


app.mount("/", _mcp_app)
