# 층별 응력 복원·파손 판정 테스트 — 막/굽힘 폐형해, 변환, Tsai-Wu 환원, [0/90]s FPF (계획서 §17.4)
from __future__ import annotations

import math

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import failure as FAIL

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
CFRP_STRENGTH = {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}  # 대표 입력값


def iso_lam(E=70000.0, nu=0.3, t=1.0, n=1, strength=None):
    m = {"type": "isotropic", "E": E, "nu": nu}
    if strength:
        m["strength"] = dict(strength)
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": t, "angle_deg": 0, "material": dict(m)} for _ in range(n)]}


# ── 응력 복원 폐형해 ─────────────────────────────────────────────────────────

def test_membrane_stress_nx_over_h():
    """단일 등방판 Nx → σx = Nx/h 균일, σy≈0."""
    env = srv.recover_ply_stresses(iso_lam(t=2.0), loads={"N": [100.0, 0, 0]})
    st = env["data"]["plies"][0]["stresses"]
    for loc in ("bottom", "mid", "top"):
        assert st[loc]["sigma_xyz"][0] == pytest.approx(100.0 / 2.0, rel=1e-10)  # 50 MPa
        assert abs(st[loc]["sigma_xyz"][1]) < 1e-9


def test_bending_stress_6m_over_h2():
    """단일 등방판 Mx → 표면 σx = ±6Mx/h² (정역학 항등 — ν 무관)."""
    h, Mx = 2.0, 10.0
    env = srv.recover_ply_stresses(iso_lam(t=h), loads={"M": [Mx, 0, 0]})
    st = env["data"]["plies"][0]["stresses"]
    assert st["top"]["sigma_xyz"][0] == pytest.approx(6 * Mx / h**2, rel=1e-9)
    assert st["bottom"]["sigma_xyz"][0] == pytest.approx(-6 * Mx / h**2, rel=1e-9)
    assert abs(st["mid"]["sigma_xyz"][0]) < 1e-12


def test_material_axis_transform_45deg():
    """순수 σx를 45° 재료축으로: σ1=σ2=σx/2, τ12=−σx/2."""
    s = FAIL.stress_to_material_axes(np.array([100.0, 0.0, 0.0]), 45.0)
    assert s[0] == pytest.approx(50.0, rel=1e-12)
    assert s[1] == pytest.approx(50.0, rel=1e-12)
    assert s[2] == pytest.approx(-50.0, rel=1e-12)


def test_membrane_force_equilibrium_cross_ply():
    """[0/90] Nx: ply 중앙 σx 가중합 = Nx (ply 내 σ z-선형 → 중앙값 적분 정확)."""
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.5, "angle_deg": a, "material": dict(T300)} for a in (0, 90)]}
    env = srv.recover_ply_stresses(lam, loads={"N": [200.0, 0, 0]})
    tot = sum(p["stresses"]["mid"]["sigma_xyz"][0] * 0.5 for p in env["data"]["plies"])
    assert tot == pytest.approx(200.0, rel=1e-9)


# ── 파손 기준 ────────────────────────────────────────────────────────────────

def test_tsai_wu_reduces_to_uniaxial():
    st = (1500.0, 1200.0, 40.0, 246.0, 68.0)
    r_t = FAIL.tsai_wu(np.array([750.0, 0, 0]), *st)["strength_ratio"]
    assert r_t == pytest.approx(1500.0 / 750.0, rel=1e-9)          # 인장: R=Xt/σ1
    r_c = FAIL.tsai_wu(np.array([-600.0, 0, 0]), *st)["strength_ratio"]
    assert r_c == pytest.approx(1200.0 / 600.0, rel=1e-9)          # 압축: R=Xc/|σ1|
    r_s = FAIL.tsai_wu(np.array([0, 0, 34.0]), *st)["strength_ratio"]
    assert r_s == pytest.approx(2.0, rel=1e-9)                     # 전단: R=S/|τ|


def test_max_stress_mode_identification():
    st = (1500.0, 1200.0, 40.0, 246.0, 68.0)
    assert FAIL.max_stress(np.array([100.0, 30.0, 0]), *st)["mode"] == "transverse_tension"
    assert FAIL.max_stress(np.array([-1300.0, 0, 0]), *st)["mode"] == "fiber_compression"
    assert FAIL.max_stress(np.array([0, 0, 70.0]), *st)["failure_index"] > 1.0


def test_cross_ply_fpf_is_90_ply_transverse_tension():
    """[0/90]s Nx 인장 — 고전 결과: 90° ply가 횡인장(Yt)으로 먼저 파손."""
    mat = {**T300, "strength": dict(CFRP_STRENGTH)}
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.125, "angle_deg": a, "material": dict(mat)}
                       for a in (0, 90, 90, 0)]}
    env = srv.recover_ply_stresses(lam, loads={"N": [100.0, 0, 0]})
    fpf = env["data"]["first_ply_failure"]
    assert fpf["ply"] in (1, 2)                       # 90° ply
    assert fpf["governing_mode"] == "transverse_tension"
    assert fpf["tsai_wu_R"] > 0
    # 하중을 R배로 올리면 R≈1 (강도비의 의미 검증)
    scaled = srv.recover_ply_stresses(lam, loads={"N": [100.0 * fpf["tsai_wu_R"], 0, 0]})
    assert scaled["data"]["first_ply_failure"]["tsai_wu_R"] == pytest.approx(1.0, rel=1e-6)
    assert scaled["data"]["first_ply_failure"]["fails_at_current_load"] is True


def test_strength_optional_and_partial_notes():
    mat_s = {"type": "isotropic", "E": 70000.0, "nu": 0.3, "strength":
             {"Xt": 300.0, "Xc": 300.0, "Yt": 300.0, "Yc": 300.0, "S": 180.0}}
    mat_ns = {"type": "isotropic", "E": 70000.0, "nu": 0.3}
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 1.0, "angle_deg": 0, "material": mat_s},
                       {"thickness": 1.0, "angle_deg": 0, "material": mat_ns}]}
    env = srv.recover_ply_stresses(lam, loads={"N": [100.0, 0, 0]})
    d = env["data"]
    assert "failure" in d["plies"][0]["stresses"]["mid"]
    assert "failure" not in d["plies"][1]["stresses"]["mid"]
    assert "laminae[1]" in d["note"]
    assert d["first_ply_failure"]["ply"] == 0


def test_thermal_superposition_and_errors():
    cu = {"type": "isotropic", "E": 110000.0, "nu": 0.34, "alpha": 17e-6}
    inv = {"type": "isotropic", "E": 140000.0, "nu": 0.29, "alpha": 1.5e-6}
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.5, "angle_deg": 0, "material": cu},
                       {"thickness": 0.5, "angle_deg": 0, "material": inv}]}
    env = srv.recover_ply_stresses(lam, delta_T=100.0)          # 열 단독 (loads 생략 허용)
    assert env["status"] == "ok"
    assert abs(env["data"]["plies"][0]["stresses"]["mid"]["sigma_xyz"][0]) > 1.0  # 잔류응력 발생

    no_cte = iso_lam()
    assert srv.recover_ply_stresses(no_cte, delta_T=50.0)["errors"][0]["code"] == "E203"
    assert srv.recover_ply_stresses(no_cte)["errors"][0]["code"] == "E100"        # 하중도 ΔT도 없음
    bad_strength = iso_lam(strength={"Xt": 300.0, "Xc": -1.0, "Yt": 300.0, "Yc": 300.0, "S": 180.0})
    assert srv.recover_ply_stresses(bad_strength, loads={"N": [1, 0, 0]})["errors"][0]["code"] == "E100"


def test_units_bridge_stress():
    lam_mm = iso_lam(t=2.0)
    lam_si = {"unit_system": "SI",
              "laminae": [{"thickness": 2.0e-3, "angle_deg": 0,
                           "material": {"type": "isotropic", "E": 70e9, "nu": 0.3}}]}
    s_mm = srv.recover_ply_stresses(lam_mm, loads={"N": [100.0, 0, 0]})["data"]["plies"][0]["stresses"]["mid"]["sigma_xyz"][0]
    s_si = srv.recover_ply_stresses(lam_si, loads={"N": [100.0e3, 0, 0]})["data"]["plies"][0]["stresses"]["mid"]["sigma_xyz"][0]
    assert s_mm == pytest.approx(s_si * 1e-6, rel=1e-12)   # MPa vs Pa
