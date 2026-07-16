# MCP 프로토콜 왕복 테스트 — 인메모리 세션으로 tools/resources 노출 검증 (materialtwin 패턴)
from __future__ import annotations

import json

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp_server import mcp

EXPECTED_TOOLS = {"analyze_laminate", "validate_laminate_input", "compute_abd_matrix",
                  "compute_neutral_axis", "evaluate_laminate", "get_reference_cases",
                  "get_server_info",
                  # V1
                  "solve_load_response", "run_sensitivity_analysis",
                  "batch_evaluate_laminates", "generate_design_report"}


def test_tools_and_resources_exposed():
    async def main():
        async with create_connected_server_and_client_session(mcp._mcp_server) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert EXPECTED_TOOLS <= names
            # description이 곧 문서 — 규약 핵심이 박혀 있어야 한다 (§6.6)
            analyze = next(t for t in tools.tools if t.name == "analyze_laminate")
            assert "laminae[0]" in analyze.description

            res = await client.list_resources()
            assert "laminate://guide" in {str(r.uri) for r in res.resources}
            guide = await client.read_resource("laminate://guide")
            text = guide.contents[0].text
            assert "materialtwin" in text and "unit_system" in text

    anyio.run(main)


def test_call_tool_roundtrip():
    async def main():
        async with create_connected_server_and_client_session(mcp._mcp_server) as client:
            ref = await client.call_tool("get_reference_cases", {"case_id": "cross_ply_symmetric"})
            case = json.loads(ref.content[0].text)
            result = await client.call_tool("analyze_laminate", {"laminate": case["input"]["laminate"]})
            env = json.loads(result.content[0].text)
            assert env["status"] == "ok"
            assert env["data"]["laminate_summary"]["is_symmetric_stack"] is True
            assert abs(env["data"]["abd"]["B"][0][0]) < 1e-9 * abs(env["data"]["abd"]["A"][0][0])

    anyio.run(main)


def test_call_tool_error_envelope_over_protocol():
    async def main():
        async with create_connected_server_and_client_session(mcp._mcp_server) as client:
            result = await client.call_tool("analyze_laminate", {"laminate": {"laminae": []}})
            env = json.loads(result.content[0].text)
            assert env["status"] == "error"
            assert env["errors"][0]["code"] == "E101"
            assert env["errors"][0]["suggestion"]  # 자가 복구용 문장 존재

    anyio.run(main)
