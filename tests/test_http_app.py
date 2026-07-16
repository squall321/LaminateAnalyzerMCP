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
