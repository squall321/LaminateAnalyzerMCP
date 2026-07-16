# 결정론 테스트 P8 — 동일 payload → 바이트 동일 응답, debug 블록만 예외 (계획서 §6.5, D8)
from __future__ import annotations

import app.mcp_server as srv
from app.services.envelope import canonical_json, payload_hash

T300_MM = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}


def payload():
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": 0.125, "angle_deg": a, "material": dict(T300_MM)}
                        for a in (0.0, 45.0, -45.0, 90.0)]}


def test_byte_identical_responses():
    e1 = srv.analyze_laminate(payload())
    e2 = srv.analyze_laminate(payload())
    assert canonical_json(e1) == canonical_json(e2)
    assert "debug" not in e1  # 기본 응답에 비결정 요소 없음


def test_debug_block_is_the_only_nondeterminism():
    e1 = srv.analyze_laminate(payload(), include_debug=True)
    e2 = srv.analyze_laminate(payload(), include_debug=True)
    assert "debug" in e1 and "timestamp" in e1["debug"]
    d1, d2 = dict(e1), dict(e2)
    d1.pop("debug"), d2.pop("debug")
    assert canonical_json(d1) == canonical_json(d2)


def test_payload_hash_stable_and_order_insensitive():
    h1 = payload_hash(payload())
    h2 = payload_hash(payload())
    assert h1 == h2 and h1.startswith("sha256:") and len(h1) == 71
    # 키 순서가 달라도 canonical 정렬로 동일 해시
    p = payload()
    reordered = {"laminae": p["laminae"], "unit_system": p["unit_system"]}
    assert payload_hash(reordered) == h1


def test_error_envelope_also_deterministic():
    bad = {"unit_system": "SI_mm", "laminae": [{"thickness": -1, "angle_deg": 0, "material": dict(T300_MM)}]}
    assert canonical_json(srv.analyze_laminate(bad)) == canonical_json(srv.analyze_laminate(bad))
