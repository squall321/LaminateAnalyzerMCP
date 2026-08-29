# 적대 검증에서 확정된 게이트·체인 결함 회귀 시험 (GATE/NC/DIF 계열)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import diffusion as DF

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28,
        "strength": {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}}
PANEL = {"Lx": 150.0, "Ly": 150.0}


def _lam(angles, mat=T300, t=0.125):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": mat, "angle_deg": a, "thickness": t} for a in angles]}


# --- GATE-01: panel 형식 오류를 조용히 넘기지 않는다 ---------------------------

def test_bad_panel_keys_do_not_silently_disable_the_gate():
    r = srv.recover_ply_stresses(_lam([0, 90, 90, 0]), loads={"N": [-60.0, 0.0, 0.0]},
                                 panel={"Lx": 150.0, "Ly": 150.0, "typo": 1})
    assert any("panel 형식이 유효하지" in w["message"] for w in r["warnings"])


def test_valid_panel_still_runs_the_gate():
    r = srv.recover_ply_stresses(_lam([0, 90, 90, 0]), loads={"N": [-60.0, 0.0, 0.0]},
                                 panel=PANEL)
    assert r["data"]["governing_mode"]["buckling"]["N_cr"] > 0
    assert not any("panel 형식이 유효하지" in w["message"] for w in r["warnings"])


# --- NC-02: 순수 전단도 압축이다(주응력) ---------------------------------------

def test_pure_shear_is_not_silent():
    r = srv.recover_ply_stresses(_lam([45, -45, -45, 45]), loads={"N": [0.0, 0.0, 20.0]},
                                 panel=PANEL)
    assert any("Nxy" in w["message"] and "주응력" in w["message"] for w in r["warnings"])


def test_pure_tension_stays_silent():
    r = srv.recover_ply_stresses(_lam([0, 90, 90, 0]), loads={"N": [60.0, 0.0, 0.0]},
                                 panel=PANEL)
    assert not any("좌굴" in w["message"] or "Nxy" in w["message"] for w in r["warnings"])


# --- GATE-06: 수치 잔여를 압축으로 오인하지 않는다 -----------------------------

def test_numerical_residual_is_not_read_as_compression():
    """변위 제어 체인이 내놓는 N ≈ −1e-33 은 압축이 아니다."""
    r = srv.recover_ply_stresses(_lam([0, 90, 90, 0]),
                                 loads={"N": [-7e-33, 0.0, 0.0], "M": [0.5, 0.0, 0.0]},
                                 panel=PANEL)
    assert not any("좌굴" in w["message"] for w in r["warnings"])


# --- GATE-02: 지배모드 게이트가 횡전단 보정을 건다 -----------------------------

def test_governing_gate_applies_shear_correction():
    d = srv.recover_ply_stresses(_lam([0, 90, 90, 0]), loads={"N": [-60.0, 0.0, 0.0]},
                                 panel=PANEL)["data"]["governing_mode"]["buckling"]
    assert d["shear_flexibility_applied"] is True
    assert d["N_cr"] <= d["N_cr_uncorrected"]


# --- GATE-03: 열 구속 좌굴이 y축 압축도 본다 -----------------------------------

_T300C = dict(T300, alpha1=0.02e-6, alpha2=22.5e-6)

_NEG_A = {"type": "orthotropic_2d", "E1": 390000.0, "E2": 6000.0, "G12": 4800.0, "nu12": 0.35,
          "alpha1": -1.2e-6, "alpha2": 30.0e-6}


def test_thermal_gate_catches_y_only_compression():
    """α1<0·α2>0 이면 가열 시 x 는 인장이고 y 만 압축이다 — 전에는 완전히 침묵했다."""
    r = srv.compute_thermal_response(_lam([0] * 8, mat=_NEG_A), delta_T=120.0,
                                     panel={"Lx": 200.0, "Ly": 200.0})
    b = r["data"]["restrained_buckling"]
    assert b["compressed_axis"] == "y"
    assert b["restrained_Nx"] > 0 > b["restrained_Ny"]
    assert b["load_factor_to_buckling"] < 1.0
    assert any("이미 좌굴한다" in w["message"] for w in r["warnings"])


def test_thermal_gate_y_axis_matches_direct_buckling():
    """x↔y 교환 해가 90도 회전 적층의 compute_buckling 과 정확히 일치한다."""
    b = srv.compute_thermal_response(_lam([0] * 8, mat=_NEG_A), delta_T=120.0,
                                     panel={"Lx": 200.0, "Ly": 200.0})["data"]["restrained_buckling"]
    ratio = b["restrained_Nx"] / b["restrained_Ny"]
    direct = srv.compute_buckling(_lam([90] * 8, mat=_NEG_A),
                                  panel={"Lx": 200.0, "Ly": 200.0},
                                  load_ratio=ratio)["data"]["N_cr"]
    assert b["N_cr"] == pytest.approx(direct, rel=1e-12)


def test_thermal_gate_silent_on_tension():
    """냉각한 통상 CFRP 는 구속 반력이 인장이라 좌굴 게이트가 돌지 않는다."""
    r = srv.compute_thermal_response(_lam([0, 90, 90, 0], mat=_T300C), delta_T=-100.0,
                                     panel={"Lx": 200.0, "Ly": 200.0})
    assert all(v < 0 for v in r["data"]["thermal_loads"]["N_thermal"][:2])   # 구속 시 인장
    assert "restrained_buckling" not in r["data"]


# --- F1: 포락선 이방비 0 나눗셈 -------------------------------------------------

def test_envelope_survives_zero_strength_ratio():
    """열잔류만으로 파손면 위인 방향이 있어도 E501 이 아니라 답이 나온다."""
    r = srv.compute_failure_envelope(_lam([0, 90, 90, 0], mat=_T300C), delta_T=-600.0)
    assert r["status"] in ("ok", "warning")
    assert r["errors"] == []
    if any(p["strength_ratio"] <= 0.0 for p in r["data"]["points"]):
        assert r["data"]["anisotropy_ratio"] is None
        assert any("이미 파손면 위" in w["message"] for w in r["warnings"])


def test_envelope_normal_case_keeps_anisotropy():
    d = srv.compute_failure_envelope(_lam([0, 90, 90, 0]))["data"]
    assert d["anisotropy_ratio"] > 1.0


# --- GATE-07: surface_strain → assess_crack_shielding 체인 ---------------------

_CU = {"type": "isotropic", "E": 110000.0, "nu": 0.34, "sigma_y": 200.0}
_PI = {"type": "isotropic", "E": 4000.0, "nu": 0.35}
_FLEX = {"unit_system": "SI_mm", "laminae": [
    {"material": _PI, "angle_deg": 0.0, "thickness": 0.05},
    {"material": _CU, "angle_deg": 0.0, "thickness": 0.018},
    {"material": _PI, "angle_deg": 0.0, "thickness": 0.05}]}


def test_prescribed_gives_scalar_strain_for_the_chain():
    """문서에 적힌 체인이 실제로 통해야 한다 — 3벡터를 넘기면 E100 이었다."""
    d = srv.solve_prescribed_curvature(_FLEX, bend_radius=30.0)["data"]
    eps = d["surface_strain"]["applied_strain_x"]["top"]
    assert isinstance(eps, float)
    assert eps == pytest.approx(d["surface_strain"]["top"][0], rel=1e-12)
    r = srv.assess_crack_shielding(_FLEX, target_ply=1,
                                   fracture={"applied_strain": eps, "gamma_target": 50.0})
    assert r["errors"] == []


# --- GATE-08: 항복 게이트가 변위 제어에도 있다 ---------------------------------

def test_prescribed_curvature_has_the_yield_gate():
    r = srv.solve_prescribed_curvature(_FLEX, bend_radius=1.0)
    assert r["data"]["yielding"]["plies"][0]["ply"] == 1
    assert r["data"]["yielding"]["plies"][0]["ratio"] > 1.0
    assert any("완전탄성 가정이 깨졌다" in w["message"] for w in r["warnings"])


def test_prescribed_curvature_yield_gate_stays_quiet_when_elastic():
    r = srv.solve_prescribed_curvature(_FLEX, bend_radius=200.0)
    assert r["data"]["yielding"]["plies"] == []
    assert not any("완전탄성 가정이 깨졌다" in w["message"] for w in r["warnings"])


# --- DIF-01: 작은 τ 에서 급수 대신 폐형해 --------------------------------------

def _converged_uptake(tau, terms=5_000_000):
    s = 0.0
    for n in range(terms):
        m = 2 * n + 1
        e = m * m * math.pi * math.pi * tau
        if e > 700.0:
            break
        s += math.exp(-e) / (m * m)
    return 1.0 - (8.0 / math.pi ** 2) * s


def test_uptake_is_accurate_at_tiny_tau():
    """500항 급수는 τ=1e-8 에서 97% 틀렸다 — 폐형해 분기로 바로잡는다."""
    for tau in (1e-9, 1e-8, 3e-7, 1e-6, 1e-4, 0.005):
        assert DF.uptake_fraction(tau) == pytest.approx(_converged_uptake(tau), rel=1e-9)


def test_uptake_is_continuous_across_the_branch():
    lo = DF.uptake_fraction(DF.TAU_SERIES_MIN * (1 - 1e-7))
    hi = DF.uptake_fraction(DF.TAU_SERIES_MIN * (1 + 1e-7))
    assert lo == pytest.approx(hi, rel=1e-6)
    assert lo < hi                                  # 여전히 단조 증가


def test_uptake_large_tau_unchanged():
    for tau in (0.05, 0.2, 1.0, 5.0):
        assert DF.uptake_fraction(tau) == pytest.approx(_converged_uptake(tau), rel=1e-12)


def test_profile_uses_erfc_at_tiny_tau():
    """τ=1e-7 에서 중앙은 완전히 말라 있어야 한다(급수는 0.0015 를 냈다)."""
    assert DF.concentration_profile(1e-7, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert DF.concentration_profile(1e-7, 0.05) == pytest.approx(0.0, abs=1e-12)


# --- GATE-09: 균일 delta_C 체인의 한계 -----------------------------------------

_WET = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0,
        "nu12": 0.28, "beta1": 0.0, "beta2": 0.6}
_DIFF = {"D": 3e-7, "M_inf": 0.015}                 # mm²/s


def _time_for_tau(tau, h=0.5, d=3e-7):
    return tau * h * h / d


@pytest.mark.parametrize("tau,should_warn", [(0.01, True), (0.05, True), (0.1, True),
                                             (0.2, False), (0.5, False), (2.0, False)])
def test_moisture_chain_warns_only_while_non_uniform(tau, should_warn):
    r = srv.compute_moisture_uptake(_lam([0, 90, 90, 0], mat=_WET), diffusion=_DIFF,
                                    time_s=_time_for_tau(tau))
    fired = any("불균일" in w["message"] for w in r["warnings"])
    assert fired is should_warn
    assert 0.0 <= r["data"]["state"]["profile_uniformity"] <= 1.0


def test_moisture_chain_note_states_the_uniform_assumption():
    r = srv.compute_moisture_uptake(_lam([0, 90, 90, 0], mat=_WET), diffusion=_DIFF,
                                    time_s=_time_for_tau(0.05))
    assert "균일하게" in r["data"]["state"]["chain"]
