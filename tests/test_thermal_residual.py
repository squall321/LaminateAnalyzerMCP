# §19.10 열잔류 — 진행성 파손·피로에 잔류응력 반영 (2순위 2번)
from __future__ import annotations

import numpy as np
import pytest

import app.mcp_server as srv
from app.services import validation as VAL
from app.solver import failure as FAIL
from app.solver import progressive as PROG
from app.solver import thermal as TH

STR = {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}
MAT = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28,
       "alpha1": 0.02e-6, "alpha2": 22.5e-6, "strength": STR}
MAT_F = dict(MAT, fatigue={"model_type": "log_linear", "k": 0.1})


def lam(angles=(0.0, 90.0, 90.0, 0.0), mat=MAT):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": mat, "angle_deg": a, "thickness": 0.125} for a in angles]}


# ── 오프셋 Tsai-Wu ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("sig", [[200.0, 15.0, 30.0], [-300.0, -20.0, 10.0], [50.0, 35.0, 5.0]])
def test_offset_reduces_to_plain_tsai_wu(sig):
    """잔류가 0 이면 기존 비례하중 식과 정확히 일치한다."""
    a = FAIL.tsai_wu(np.array(sig), *STR.values())["strength_ratio"]
    b = FAIL.tsai_wu_with_offset(np.array(sig), np.zeros(3), *STR.values())["strength_ratio"]
    assert a == pytest.approx(b, rel=1e-15)


def test_offset_root_lies_on_failure_surface():
    """구한 R 에서 R·σ_mech + σ_res 가 정확히 파손면 위에 있어야 한다 (검산)."""
    sm, sr = np.array([200.0, 15.0, 30.0]), np.array([-20.0, 25.0, 0.0])
    R = FAIL.tsai_wu_with_offset(sm, sr, *STR.values())["strength_ratio"]
    assert FAIL.tsai_wu(R * sm + sr, *STR.values())["failure_index"] == pytest.approx(1.0, rel=1e-12)


def test_offset_detects_failure_without_load():
    """잔류만으로 파손면에 도달하면 R = 0 이다 — 이걸 놓치면 조용히 비보수다."""
    out = FAIL.tsai_wu_with_offset(np.array([200.0, 15.0, 30.0]), np.array([0.0, 60.0, 0.0]),
                                   *STR.values())
    assert out["strength_ratio"] == 0.0
    assert "이미 파손" in out["note"]


def test_residual_lowers_strength_ratio():
    """냉각 잔류(90° 횡인장)는 R 을 낮춘다 — 비례하중 가정은 이걸 놓친다."""
    sm = np.array([200.0, 15.0, 30.0])
    r0 = FAIL.tsai_wu_with_offset(sm, np.zeros(3), *STR.values())["strength_ratio"]
    r1 = FAIL.tsai_wu_with_offset(sm, np.array([-20.0, 25.0, 0.0]), *STR.values())["strength_ratio"]
    assert r1 < r0 * 0.7


# ── 진행성 파손 ─────────────────────────────────────────────────────────────

def test_progressive_zero_free_strain_is_identical():
    """ε_free = 0 이면 사건·모드·ultimate 가 잔류 없음과 완전히 같다."""
    si, _, _ = VAL.validate_and_convert(lam())
    N, M = np.array([1e5, 0.0, 0.0]), np.zeros(3)
    a = PROG.run(si.plies, N, M, 0.1)
    b = PROG.run(si.plies, N, M, 0.1, eps_free=[np.zeros(3) for _ in si.plies])
    assert [e["R"] for e in a["events"]] == [e["R"] for e in b["events"]]
    assert [e["mode"] for e in a["events"]] == [e["mode"] for e in b["events"]]
    assert a["ultimate_R"] == b["ultimate_R"]


def test_progressive_residual_lowers_fpf_monotonically():
    """냉각이 깊을수록 FPF 하중이 단조 감소한다."""
    si, _, _ = VAL.validate_and_convert(lam())
    N, M = np.array([1e5, 0.0, 0.0]), np.zeros(3)
    prev = PROG.run(si.plies, N, M, 0.1)["events"][0]["R"]
    for dT in (-50.0, -100.0, -150.0):
        ef = [TH.alpha_vector(p.alpha1, p.alpha2, p.angle_deg) * dT for p in si.plies]
        r = PROG.run(si.plies, N, M, 0.1, eps_free=ef)["events"][0]["R"]
        assert r < prev
        prev = r


def test_progressive_tool_reports_residual_failure():
    """ΔT 가 충분히 크면 하중 없이 잔류만으로 파손(R=0)하고 경고한다."""
    env = srv.run_progressive_failure(lam(), loads={"N": [100.0, 0.0, 0.0]}, delta_T=-200.0)
    assert env["data"]["first_ply_failure_R"] == 0.0
    assert env["data"]["delta_T"] == -200.0
    assert any(w["code"] == "W130" and "잔류응력만으로 이미" in w["message"] for w in env["warnings"])


def test_progressive_warns_when_cte_present_but_dT_omitted():
    """CTE 가 있는데 ΔT 를 안 주면 잔류가 빠졌다고 알린다."""
    env = srv.run_progressive_failure(lam(), loads={"N": [100.0, 0.0, 0.0]})
    assert env["data"]["delta_T"] is None
    assert any(w["code"] == "W130" and "경화 냉각 잔류응력이 빠져" in w["message"]
               for w in env["warnings"])
    # CTE 가 없으면 경고하지 않는다
    no_cte = dict(MAT)
    del no_cte["alpha1"], no_cte["alpha2"]
    clean = srv.run_progressive_failure(lam(mat=no_cte), loads={"N": [100.0, 0.0, 0.0]})
    assert not any("경화 냉각 잔류응력이 빠져" in w["message"] for w in clean["warnings"])
    assert srv.run_progressive_failure(lam(mat=no_cte), loads={"N": [100.0, 0.0, 0.0]},
                                       delta_T=-150.0)["errors"][0]["code"] == "E203"


# ── 피로 ────────────────────────────────────────────────────────────────────

def test_fatigue_zero_dT_is_identical():
    """delta_T=0 은 미지정과 같은 답을 준다."""
    a = srv.estimate_fatigue_life(lam(mat=MAT_F), loads_max={"N": [60.0, 0.0, 0.0]})
    b = srv.estimate_fatigue_life(lam(mat=MAT_F), loads_max={"N": [60.0, 0.0, 0.0]}, delta_T=0.0)
    assert a["data"]["life_cycles"] == b["data"]["life_cycles"]


def test_fatigue_residual_shifts_mean_and_cuts_life():
    """잔류는 진폭이 아니라 평균을 이동시킨다 — Goodman 인장 평균 벌점으로 수명이 준다."""
    base = srv.estimate_fatigue_life(lam(mat=MAT_F),
                                     loads_max={"N": [60.0, 0.0, 0.0]})["data"]["life_cycles"]
    prev = base
    for dT in (-50.0, -100.0, -150.0):
        life = srv.estimate_fatigue_life(lam(mat=MAT_F), loads_max={"N": [60.0, 0.0, 0.0]},
                                         delta_T=dT)["data"]["life_cycles"]
        assert life < prev
        prev = life
    assert prev < base * 1e-3          # 실측으로 자릿수가 통째로 갈린다


def test_fatigue_amplitude_unchanged_by_residual():
    """잔류가 진폭을 바꾸지 않는지 확인 — 지배 성분의 진폭이 그대로여야 한다."""
    kw = dict(loads_max={"N": [60.0, 0.0, 0.0]}, loads_min={"N": [-60.0, 0.0, 0.0]}, detail="full")
    a = srv.estimate_fatigue_life(lam(mat=MAT_F), **kw)["data"]["critical_ply"]
    b = srv.estimate_fatigue_life(lam(mat=MAT_F), delta_T=-100.0, **kw)["data"]["critical_ply"]
    assert a["sigma_amplitude"] == pytest.approx(b["sigma_amplitude"], rel=1e-9)
    assert a["sigma_mean"] == pytest.approx(0.0, abs=1e-9)      # 완전반복이라 평균 0
    assert abs(b["sigma_mean"]) > 1.0                            # 잔류가 평균을 이동시킨다


def test_fatigue_warns_when_cte_present_but_dT_omitted():
    env = srv.estimate_fatigue_life(lam(mat=MAT_F), loads_max={"N": [60.0, 0.0, 0.0]})
    assert any(w["code"] == "W130" and "평균응력을 이동" in w["message"] for w in env["warnings"])
    assert srv.estimate_fatigue_life(lam(mat=MAT_F), loads_max={"N": [60.0, 0.0, 0.0]},
                                     delta_T=float("nan"))["errors"][0]["code"] == "E100"


def test_thermal_residual_tools_are_deterministic():
    kw = dict(loads={"N": [100.0, 0.0, 0.0]}, delta_T=-150.0)
    a, b = srv.run_progressive_failure(lam(), **kw), srv.run_progressive_failure(lam(), **kw)
    assert a["data"] == b["data"] and a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]
