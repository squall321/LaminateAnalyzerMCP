# V1 Tool 테스트 — solve/sensitivity/batch/report + E400/E500 실트리거 (계획서 §6.2 Tool 8~11)
from __future__ import annotations

import pytest

import app.mcp_server as srv
from app import config
from app.services import pipeline as PIPE

T300_MM = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}


def lam(angles, t=0.125, name=None):
    d = {"unit_system": "SI_mm",
         "laminae": [{"thickness": t, "angle_deg": a, "material": dict(T300_MM)} for a in angles]}
    if name:
        d["name"] = name
    return d


# ── solve_load_response ─────────────────────────────────────────────────────

def test_solve_symmetric_membrane_only():
    env = srv.solve_load_response(lam([0, 90, 90, 0]), loads={"N": [100.0, 0.0, 0.0]})
    assert env["status"] == "ok"
    d = env["data"]
    assert d["response"]["epsilon0"][0] > 0
    # 대칭 → 막하중이 곡률을 만들지 않음
    assert all(abs(k) < 1e-12 for k in d["response"]["kappa"])
    assert d["v1_indices"]["membrane_bending_leakage"]["value"] == pytest.approx(0.0, abs=1e-12)
    # [0/90]s 크로스플라이: 유효 Ex는 E2와 E1 사이
    ex = d["effective_constants"]["membrane"]["Ex"]
    assert 10300.0 < ex < 181000.0  # MPa 표시 단위


def test_solve_unsymmetric_couples_and_principal_scan():
    env = srv.solve_load_response(lam([0, 90]), loads={"M": [1.0, 0.0, 0.0]})
    assert env["status"] in ("ok", "warning")
    d = env["data"]
    assert any(abs(e) > 0 for e in d["response"]["epsilon0"])  # B 커플링으로 막변형 유발
    assert d["v1_indices"]["membrane_bending_leakage"]["value"] > 0.01
    pdir = d["v1_indices"]["in_plane_principal_direction"]
    assert pdir["Ex_max"] >= pdir["Ex_min"] > 0


def test_solve_loads_validation():
    base = lam([0, 90])
    assert srv.solve_load_response(base, loads={})["errors"][0]["code"] == "E100"          # 모두 0
    assert srv.solve_load_response(base, loads={"N": [1, 2]})["errors"][0]["code"] == "E100"  # 길이
    assert srv.solve_load_response(base, loads=[1, 2, 3])["errors"][0]["code"] == "E100"      # 타입


def test_solve_e400_singular(monkeypatch):
    def boom(*a, **k):
        raise PIPE.RESP.SingularSystemError("test")
    monkeypatch.setattr(PIPE.RESP, "solve_response", boom)
    env = srv.solve_load_response(lam([0, 90]), loads={"N": [1.0, 0, 0]})
    assert [e["code"] for e in env["errors"]] == ["E400"]


def test_solve_units_bridge():
    """N/mm 하중과 N/m 하중의 물리 동등성 — ε0 동일, κ는 1/mm↔1/m 변환 관계."""
    si_mm = lam([0, 90])
    si = {"unit_system": "SI",
          "laminae": [{"thickness": 0.125e-3, "angle_deg": a,
                       "material": {"type": "orthotropic_2d", "E1": 181e9, "E2": 10.3e9,
                                    "G12": 7.17e9, "nu12": 0.28}} for a in (0, 90)]}
    e_mm = srv.solve_load_response(si_mm, loads={"N": [100.0, 0, 0]})["data"]      # 100 N/mm
    e_si = srv.solve_load_response(si, loads={"N": [100.0e3, 0, 0]})["data"]       # = 1e5 N/m
    assert e_mm["response"]["epsilon0"][0] == pytest.approx(e_si["response"]["epsilon0"][0], rel=1e-12)
    assert e_mm["response"]["kappa"][0] == pytest.approx(e_si["response"]["kappa"][0] * 1e-3, rel=1e-12)
    assert e_mm["effective_constants"]["membrane"]["Ex"] == pytest.approx(
        e_si["effective_constants"]["membrane"]["Ex"] * 1e-6, rel=1e-12)


# ── run_sensitivity_analysis ────────────────────────────────────────────────

def test_sensitivity_structure_and_symmetry_break():
    env = srv.run_sensitivity_analysis(lam([30, -45]))
    assert env["status"] in ("ok", "warning")
    d = env["data"]
    assert set(d["base"].keys()) == {"D_hat_11", "coupling_ratio", "zeta_x"}
    assert len(d["rows"]) == 2 * 3  # ply 2 × (각도/두께/E1)
    # 두께 섭동은 항상 중립면을 움직인다 (비대칭 케이스)
    thick_rows = [r for r in d["rows"] if "thickness" in r["parameter"]]
    assert any(abs(r["response_half_diff"]["zeta_x"]) > 0 for r in thick_rows)


def test_sensitivity_param_validation_and_timeout(monkeypatch):
    assert srv.run_sensitivity_analysis(lam([0]), angle_delta_deg=-1)["errors"][0]["code"] == "E100"
    monkeypatch.setattr(config, "COMPUTE_TIMEOUT_S", -1.0)
    env = srv.run_sensitivity_analysis(lam([0, 90]))
    assert [e["code"] for e in env["errors"]] == ["E500"]


# ── batch_evaluate_laminates ────────────────────────────────────────────────

def test_batch_mixed_results_with_criteria():
    cases = [lam([0, 90, 90, 0], name="sym"),
             lam([0, 90], name="unsym"),
             {"unit_system": "SI_mm", "laminae": []}]  # E102
    env = srv.batch_evaluate_laminates(cases, criteria={"max_coupling_ratio": 0.05})
    d = env["data"]
    assert d["n_total"] == 3 and d["n_ok"] == 2 and d["n_error"] == 1
    rows = {r["name"]: r for r in d["results"] if r["name"]}
    assert rows["sym"]["pass_all"] is True
    assert rows["unsym"]["pass_all"] is False
    assert d["results"][2]["error_codes"] == ["E102"]


def test_batch_limits_and_timeout(monkeypatch):
    assert srv.batch_evaluate_laminates([])["errors"][0]["code"] == "E100"
    too_many = [lam([0]) for _ in range(config.MAX_BATCH + 1)]
    assert srv.batch_evaluate_laminates(too_many)["errors"][0]["code"] == "E103"
    monkeypatch.setattr(config, "COMPUTE_TIMEOUT_S", -1.0)
    assert srv.batch_evaluate_laminates([lam([0])])["errors"][0]["code"] == "E500"


# ── generate_design_report ──────────────────────────────────────────────────

def test_report_korean_default():
    env = srv.generate_design_report(lam([0, 90, 90, 0], name="검증용"),
                                     criteria={"max_coupling_ratio": 0.05})
    assert env["status"] == "ok"
    md = env["data"]["report_markdown"]
    assert "# 적층 평가 리포트 — 검증용" in md
    assert "payload_hash" in md and "| criterion |" in md
    assert env["data"]["summary"]["is_symmetric_stack"] is True


def test_report_english_and_bad_language():
    md = srv.generate_design_report(lam([0, 90]), language="en")["data"]["report_markdown"]
    assert "# Laminate Evaluation Report" in md
    assert srv.generate_design_report(lam([0]), language="jp")["errors"][0]["code"] == "E100"


def test_report_propagates_input_errors():
    env = srv.generate_design_report({"laminae": [{"thickness": 1, "angle_deg": 0,
                                                   "material": dict(T300_MM)}]})
    assert env["errors"][0]["code"] == "E101"


# ── dominant_coupling_terms (V1 지표) ───────────────────────────────────────

def test_dominant_coupling_terms():
    idx = srv.evaluate_laminate(lam([0, 90]))["data"]["indices"]
    terms = idx["dominant_coupling_terms"]["value"]
    assert terms and {t["term"] for t in terms} <= {"B11", "B12", "B16", "B22", "B26", "B66"}
    assert {"B11", "B22"} <= {t["term"] for t in terms}  # 크로스플라이의 지배 항
    sym_terms = srv.evaluate_laminate(lam([0, 90, 90, 0]))["data"]["indices"]["dominant_coupling_terms"]["value"]
    assert sym_terms == []
