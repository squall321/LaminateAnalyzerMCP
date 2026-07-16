# 강건성 테스트 — 512 ply, 극단 물성, 경계 각도 (계획서 §7.2 Robustness, Phase 4)
from __future__ import annotations

import pytest

import app.mcp_server as srv
from app import config

T300_MM = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}


def test_max_plies_512_analyzes_cleanly():
    angles = [(0, 45, -45, 90)[i % 4] for i in range(config.MAX_PLIES)]
    payload = {"unit_system": "SI_mm",
               "laminae": [{"thickness": 0.125, "angle_deg": a, "material": dict(T300_MM)}
                           for a in angles]}
    env = srv.analyze_laminate(payload)
    assert env["status"] in ("ok", "warning")
    d = env["data"]
    assert d["laminate_summary"]["n_plies"] == config.MAX_PLIES
    assert d["indices"]["positive_definite"]["value"] is True
    # 준등방 4각도 반복 → 준등방성에 근접해야 함 (수치 폭주 없음의 물리적 방증)
    assert d["indices"]["quasi_isotropy_score"]["value"] > 0.9


def test_extreme_modulus_ratio_warns_but_computes():
    mat = {"type": "orthotropic_2d", "E1": 300000.0, "E2": 0.03, "G12": 0.03, "nu12": 0.001}
    payload = {"unit_system": "SI_mm",
               "laminae": [{"thickness": 0.1, "angle_deg": a, "material": dict(mat)}
                           for a in (0, 90)]}
    env = srv.analyze_laminate(payload)
    assert env["status"] == "warning"
    codes = {w["code"] for w in env["warnings"]}
    assert "W112" in codes
    assert env["data"]["indices"]["positive_definite"]["value"] is True


def test_ultrathin_ply_warns_but_computes():
    payload = {"unit_system": "SI_mm",
               "laminae": [
                   {"thickness": 10.0, "angle_deg": 0, "material": dict(T300_MM)},
                   {"thickness": 1.0e-6, "angle_deg": 90, "material": dict(T300_MM)},
               ]}
    env = srv.analyze_laminate(payload)
    assert env["status"] == "warning"
    assert "W111" in {w["code"] for w in env["warnings"]}
    assert env["data"] is not None


@pytest.mark.parametrize("angle,ok", [(359.9, True), (-359.9, True), (360.0, False), (-360.0, False)])
def test_angle_boundaries(angle, ok):
    payload = {"unit_system": "SI_mm",
               "laminae": [{"thickness": 0.1, "angle_deg": angle, "material": dict(T300_MM)}]}
    env = srv.analyze_laminate(payload)
    if ok:
        assert env["status"] in ("ok", "warning")
    else:
        assert [e["code"] for e in env["errors"]] == ["E301"]
