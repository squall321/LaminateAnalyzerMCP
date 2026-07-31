# HTTP 모드 스모크 — /health와 streamable HTTP /mcp initialize (계획서 §16.6 HEAXHub 패키징)
# 주의: StreamableHTTPSessionManager는 프로세스당 1회만 run() 가능 → TestClient(lifespan)를 모듈 공유.
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mcp_endpoint"] == "/mcp"


def test_mcp_streamable_http_initialize(client):
    r = client.post(
        "/mcp",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-03-26",
                         "capabilities": {},
                         "clientInfo": {"name": "smoke", "version": "0"}}},
    )
    assert r.status_code == 200
    assert "mcp-session-id" in {k.lower() for k in r.headers.keys()}
    assert "laminate-analyzer" in r.text


# ── REST 표면 (MCP와 동일 계산·envelope, 도구 레지스트리 단일 소스) ────────

def test_rest_tool_registry_mirrors_mcp(client):
    """REST 도구 목록이 MCP 레지스트리와 완전히 같다 (도구 추가 시 자동 반영)."""
    from app.mcp_server import mcp
    rest = client.get("/api/v1/tools").json()
    mcp_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert rest["count"] == len(mcp_names)
    assert {t["name"] for t in rest["tools"]} == mcp_names
    # 자기설명: 각 도구에 설명과 입력 스키마가 있다
    for t in rest["tools"]:
        assert t["description"] and "properties" in t["input_schema"]


def test_rest_execution_matches_mcp(client):
    """같은 입력에 REST와 MCP(파이썬 직접 호출)가 동일 결과를 준다."""
    import app.mcp_server as srv
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.125, "angle_deg": a,
                        "material": {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0,
                                     "G12": 7170.0, "nu12": 0.28}} for a in (0, 90, 90, 0)]}
    rest = client.post("/api/v1/tools/analyze_laminate", json={"laminate": lam}).json()
    direct = srv.analyze_laminate(lam)
    assert rest["status"] == direct["status"] == "ok"
    assert rest["data"]["abd"]["A"] == direct["data"]["abd"]["A"]
    assert rest["metadata"]["payload_hash"] == direct["metadata"]["payload_hash"]


def test_rest_error_envelope_and_http_codes(client):
    """계산 오류는 200+envelope, 없는 도구는 404, 스키마 위반은 422."""
    bad = client.post("/api/v1/tools/analyze_laminate", json={"laminate": {"laminae": []}})
    assert bad.status_code == 200 and bad.json()["errors"][0]["code"] == "E101"
    assert bad.json()["errors"][0]["suggestion"]
    assert client.get("/api/v1/tools/nope").status_code == 404
    assert client.post("/api/v1/tools/nope", json={}).status_code == 404
    assert client.post("/api/v1/tools/analyze_laminate", json={"wrong_arg": 1}).status_code == 422


def test_rest_guide_info_and_openapi(client):
    """가이드·서버정보·OpenAPI 문서가 REST로도 발견 가능하다."""
    g = client.get("/api/v1/guide")
    assert g.status_code == 200 and "unit_system" in g.text and "laminae[0]" in g.text
    info = client.get("/api/v1/info").json()
    assert info["server"] == "laminate-analyzer" and "error_codes" in info
    spec = client.get("/openapi.json").json()
    assert {"/api/v1/tools", "/api/v1/tools/{name}", "/api/v1/guide", "/api/v1/info",
            "/health"} <= set(spec["paths"])
    assert client.get("/health").json()["rest_base"] == "/api/v1"
