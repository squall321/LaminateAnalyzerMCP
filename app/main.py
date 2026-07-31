# HEAXHub fastapi 스택 진입점 — /health + MCP streamable HTTP(/mcp)를 한 앱에 마운트 (계획서 §16.6)
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import config
from app.mcp_server import mcp
from app.rest import router as rest_router

_mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # 마운트된 서브앱의 lifespan은 자동 실행되지 않으므로 세션 매니저를 직접 구동한다.
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="Laminate Analyzer",
    version=config.SERVER_VERSION,
    lifespan=_lifespan,
    description=(
        "적층 복합재의 중립면·ABD·평가지표·파손·열/흡습 휨·좌굴·진동·층간·피로를 "
        "결정론적으로 계산하는 서버. 두 표면을 제공한다 — "
        "**MCP**(`POST /mcp`, streamable HTTP)와 **REST**(`/api/v1/tools`). "
        "두 표면은 같은 계산·검증·응답 envelope를 쓴다. "
        "도구 목록·스키마는 `GET /api/v1/tools`, 사용 규약은 `GET /api/v1/guide`."),
)


app.include_router(rest_router)


@app.get("/health")
def health() -> dict:
    """HEAXHub 스택 헬스체크(health_path: /health) + MCP 엔드포인트 안내."""
    return {"status": "ok", "service": "laminate-analyzer-mcp",
            "version": config.SERVER_VERSION,
            "mcp_endpoint": "/mcp",
            "rest_base": "/api/v1",
            "docs": "/docs"}


app.mount("/", _mcp_app)
