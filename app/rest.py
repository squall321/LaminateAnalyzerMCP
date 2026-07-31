# REST 표면 — MCP 도구 레지스트리를 그대로 HTTP로 재노출 (MCP 클라이언트 없는 소비자용)
"""도구를 두 번 정의하지 않는다. FastMCP의 ToolManager를 단일 진실 공급원으로 삼아
목록·스키마·실행을 REST로 얇게 중계하므로, 새 도구를 MCP에 추가하면 REST에 자동 반영된다.

- GET  /api/v1/tools            도구 목록 + 입력 JSON Schema (자기설명 — 에이전트/사람 공용)
- GET  /api/v1/tools/{name}     한 도구의 상세(설명·스키마)
- POST /api/v1/tools/{name}     실행 (본문 = 도구 인자 객체) → MCP와 동일한 응답 envelope
- GET  /api/v1/guide            laminate://guide 리소스와 동일한 사용 가이드
- GET  /api/v1/info             서버·엔진 버전, 단위계, 한계값, 규약

MCP 경로(/mcp)와 완전히 같은 계산·검증·envelope를 쓴다 — 표면만 다르고 결과는 동일하다.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app import config
from app.mcp_server import guide as _guide
from app.mcp_server import mcp

router = APIRouter(prefix="/api/v1", tags=["laminate"])


def _tool_summary(t) -> dict:
    return {"name": t.name,
            "description": (t.description or "").strip(),
            "input_schema": t.parameters}


@router.get("/tools", summary="도구 목록과 입력 스키마")
def list_tools() -> dict:
    """개발된 계산 도구 전체와 각 도구의 입력 JSON Schema.

    MCP `tools/list`와 동일한 레지스트리다. 응답의 input_schema를 그대로 POST 본문으로 쓴다.
    """
    tools = sorted(mcp._tool_manager.list_tools(), key=lambda t: t.name)
    return {"count": len(tools), "tools": [_tool_summary(t) for t in tools],
            "usage": "POST /api/v1/tools/{name} 에 input_schema 형태의 JSON을 보내면 실행됩니다. "
                     "사용 규약은 GET /api/v1/guide 참조"}


@router.get("/tools/{name}", summary="도구 상세")
def get_tool(name: str) -> dict:
    for t in mcp._tool_manager.list_tools():
        if t.name == name:
            return _tool_summary(t)
    raise HTTPException(status_code=404,
                        detail=f"도구 '{name}'가 없습니다. 목록은 GET /api/v1/tools")


@router.post("/tools/{name}", summary="도구 실행")
async def run_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """도구를 실행하고 MCP와 동일한 응답 envelope를 반환한다.

    본문은 인자 객체(예: {"laminate": {...}, "loads": {...}}). 계산 오류는 HTTP 200에
    envelope의 status="error"·errors[]로 담겨 온다(MCP 관례와 동일 — suggestion으로 자가 복구).
    """
    if not any(t.name == name for t in mcp._tool_manager.list_tools()):
        raise HTTPException(status_code=404,
                            detail=f"도구 '{name}'가 없습니다. 목록은 GET /api/v1/tools")
    try:
        return await mcp._tool_manager.call_tool(name, arguments or {})
    except Exception as e:  # noqa: BLE001 — 인자 스키마 위반 등
        raise HTTPException(status_code=422,
                            detail=f"인자가 도구 스키마와 맞지 않습니다: {type(e).__name__}: {e}") from e


@router.get("/guide", response_class=PlainTextResponse, summary="사용 가이드")
def guide() -> str:
    """규약·워크플로·연계 지침 (MCP 리소스 laminate://guide와 동일 내용)."""
    return _guide()


@router.get("/info", summary="서버 정보")
def info() -> dict:
    from app.mcp_server import get_server_info
    return get_server_info()
