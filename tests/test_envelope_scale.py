# §19.15~19.17 항복 게이트·파손 포락선·필요 두께 배율 (2순위 15·8·9번)
from __future__ import annotations

import json

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import failure as FAIL

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28,
        "strength": {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}}
T300C = dict(T300, alpha1=0.02e-6, alpha2=22.5e-6)
CU = {"type": "isotropic", "E": 117000.0, "nu": 0.34, "sigma_y": 250.0}
FR4 = {"type": "isotropic", "E": 22000.0, "nu": 0.15}
PANEL = {"Lx": 300.0, "Ly": 300.0}


def lam(mat=T300, angles=(0.0, 90.0, 90.0, 0.0)):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": mat, "angle_deg": a, "thickness": 0.125} for a in angles]}


def pcb():
    return {"unit_system": "SI_mm", "laminae": [
        {"material": CU, "angle_deg": 0.0, "thickness": 0.035},
        {"material": FR4, "angle_deg": 0.0, "thickness": 0.2},
        {"material": CU, "angle_deg": 0.0, "thickness": 0.035}]}


# ── §19.15 항복 게이트 ──────────────────────────────────────────────────────

def test_von_mises_landmarks():
    """단축 = σ, 등2축 = σ, 순수전단 = √3·τ."""
    assert FAIL.von_mises_plane_stress(np.array([100.0, 0.0, 0.0])) == pytest.approx(100.0)
    assert FAIL.von_mises_plane_stress(np.array([100.0, 100.0, 0.0])) == pytest.approx(100.0)
    assert FAIL.von_mises_plane_stress(np.array([0.0, 0.0, 100.0])) == pytest.approx(
        100.0 * 3 ** 0.5, rel=1e-12)
    assert FAIL.von_mises_plane_stress(np.zeros(3)) == 0.0


def test_yield_gate_fires_and_scales():
    """항복하면 W130 과 함께 배수를 보고한다. 하중에 선형 비례한다."""
    low = srv.recover_ply_stresses(pcb(), loads={"N": [10.0, 0.0, 0.0]})
    assert low["data"]["yielding"]["plies"] == []
    hi = srv.recover_ply_stresses(pcb(), loads={"N": [60.0, 0.0, 0.0]})
    ys = hi["data"]["yielding"]["plies"]
    assert ys and ys[0]["ratio"] > 2.0
    assert any(w["code"] == "W130" and "완전탄성 가정이 깨졌다" in w["message"]
               for w in hi["warnings"])
    mid = srv.recover_ply_stresses(pcb(), loads={"N": [30.0, 0.0, 0.0]})["data"]["yielding"]
    assert ys[0]["ratio"] == pytest.approx(2.0 * mid["plies"][0]["ratio"], rel=1e-9)


def test_yield_gate_silent_without_sigma_y():
    """항복강도가 없으면 yielding 블록 자체가 없다 (경고 피로 방지)."""
    env = srv.recover_ply_stresses(lam(), loads={"N": [100.0, 0.0, 0.0]})
    assert "yielding" not in env["data"]
    assert not any("완전탄성" in w["message"] for w in env["warnings"])


# ── §19.16 파손 포락선 ──────────────────────────────────────────────────────

def test_envelope_is_deterministic_and_complete():
    env = srv.compute_failure_envelope(lam(), magnitude=100.0)
    d = env["data"]
    assert env["errors"] == []
    assert d["n_directions"] == 72 and len(d["points"]) == 72
    angles = [p["angle_deg"] for p in d["points"]]
    assert angles == sorted(angles) and angles[0] == 0.0
    b = srv.compute_failure_envelope(lam(), magnitude=100.0)
    assert env["data"] == b["data"]
    json.dumps(env)          # numpy 스칼라 누출 없음


def test_envelope_weakest_is_transverse_for_cross_ply():
    """[0/90]s 의 최약 방향은 90° 층 횡인장이 지배한다."""
    d = srv.compute_failure_envelope(lam(), magnitude=100.0)["data"]
    assert d["weakest"]["mode"] == "transverse_tension"
    assert d["anisotropy_ratio"] > 3.0
    # 최약 방향의 R 이 실제 전 방향 최소인지
    rmin = min(p["strength_ratio"] for p in d["points"])
    w = [p for p in d["points"] if p["angle_deg"] == d["weakest"]["angle_deg"]][0]
    assert w["strength_ratio"] == pytest.approx(rmin, rel=1e-12)


def test_envelope_failure_load_is_on_failure_surface():
    """failure_load 를 그대로 걸면 R = 1 이어야 한다 (검산)."""
    d = srv.compute_failure_envelope(lam(), magnitude=100.0)["data"]
    p = d["points"][7]
    env = srv.recover_ply_stresses(lam(), loads={"N": [p["failure_load"][0],
                                                      p["failure_load"][1], 0.0]})
    assert env["data"]["first_ply_failure"]["tsai_wu_R"] == pytest.approx(1.0, rel=1e-6)


def test_envelope_magnitude_does_not_change_result():
    """탐침 하중 크기는 결과를 바꾸지 않는다 (R 이 비례해 흡수한다)."""
    a = srv.compute_failure_envelope(lam(), magnitude=10.0)["data"]
    b = srv.compute_failure_envelope(lam(), magnitude=1000.0)["data"]
    assert a["weakest"]["failure_load"] == pytest.approx(b["weakest"]["failure_load"], rel=1e-9)


def test_envelope_with_thermal_residual_shrinks():
    """열잔류를 넣으면 포락선이 줄어든다 (§19.10 과 정합)."""
    a = srv.compute_failure_envelope(lam(mat=T300C), magnitude=100.0)["data"]
    b = srv.compute_failure_envelope(lam(mat=T300C), magnitude=100.0, delta_T=-150.0)["data"]
    assert abs(b["weakest"]["failure_load"][1]) < abs(a["weakest"]["failure_load"][1])
    assert b["delta_T"] == -150.0


def test_envelope_planes_and_errors():
    for pl in ("Nx-Ny", "Nx-Nxy", "Mx-My"):
        assert srv.compute_failure_envelope(lam(), plane=pl)["errors"] == []
    assert srv.compute_failure_envelope(lam(), plane="x")["errors"][0]["code"] == "E100"
    assert srv.compute_failure_envelope(lam(), magnitude=-1.0)["errors"][0]["code"] == "E100"
    no_str = {"unit_system": "SI_mm", "laminae": [
        {"material": {"type": "orthotropic_2d", "E1": 1e5, "E2": 1e4, "G12": 5e3, "nu12": 0.3},
         "angle_deg": 0.0, "thickness": 0.125}]}
    assert srv.compute_failure_envelope(no_str)["errors"][0]["code"] == "E100"


# ── §19.17 필요 두께 배율 ───────────────────────────────────────────────────

def test_required_scale_is_exact_closed_form():
    """s = (target/current)^(1/3) 이 정확하다 — 배율 적용 후 실제 여유로 검산."""
    for target in (0.5, 1.0, 2.0, 4.0):
        r = srv.solve_required_thickness_scale(lam(), panel=PANEL, applied_Nx=1.0,
                                               target_margin=target)["data"]
        s = r["required_scale"]
        scaled = {"unit_system": "SI_mm",
                  "laminae": [{"material": T300, "angle_deg": a, "thickness": 0.125 * s}
                              for a in (0.0, 90.0, 90.0, 0.0)]}
        actual = srv.compute_buckling(scaled, panel=PANEL,
                                      applied_Nx=1.0)["data"]["margin"]["factor"]
        assert actual == pytest.approx(target, rel=1e-9)


def test_required_scale_cube_law():
    """N_cr ∝ s³ — 배율이 2배면 여유가 8배."""
    r = srv.solve_required_thickness_scale(lam(), panel=PANEL, applied_Nx=1.0,
                                           target_margin=8.0)["data"]
    base = srv.solve_required_thickness_scale(lam(), panel=PANEL, applied_Nx=1.0,
                                              target_margin=1.0)["data"]
    assert r["required_scale"] / base["required_scale"] == pytest.approx(2.0, rel=1e-12)
    assert r["N_cr_scaled"] / r["N_cr_current"] == pytest.approx(r["required_scale"] ** 3,
                                                                 rel=1e-12)


def test_required_scale_respects_boundary():
    """고정단이면 이미 뻣뻣해 필요 배율이 작다."""
    ss = srv.solve_required_thickness_scale(lam(), panel=PANEL, applied_Nx=1.0,
                                            target_margin=2.0)["data"]
    cc = srv.solve_required_thickness_scale(lam(), panel=PANEL, applied_Nx=1.0,
                                            target_margin=2.0, boundary="CCCC")["data"]
    assert cc["required_scale"] < ss["required_scale"]


def test_required_scale_warns_and_errors():
    env = srv.solve_required_thickness_scale(lam(), panel=PANEL, applied_Nx=1.0,
                                             target_margin=1.0)
    assert any(w["code"] == "W130" and "균일 배율만" in w["message"] for w in env["warnings"])
    big = srv.solve_required_thickness_scale(lam(), panel=PANEL, applied_Nx=1.0,
                                             target_margin=100.0)
    assert any("필요 배율" in w["message"] for w in big["warnings"])
    for kw in ({"applied_Nx": -1.0}, {"target_margin": 0.0}, {"boundary": "SSSF"}):
        base = dict(panel=PANEL, applied_Nx=1.0, target_margin=1.0)
        base.update(kw)
        assert srv.solve_required_thickness_scale(lam(), **base)["errors"][0]["code"] == "E100"
    assert srv.solve_required_thickness_scale(lam(), panel={"Lx": 1.0},
                                              applied_Nx=1.0)["errors"][0]["code"] == "E100"
