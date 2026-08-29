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
    dd = srv.compute_buckling(_lam([90] * 8, mat=_NEG_A), panel={"Lx": 200.0, "Ly": 200.0},
                              load_ratio=ratio)["data"]
    assert b["N_cr_clt"] == pytest.approx(dd["N_cr"], rel=1e-12)
    # 보정 후 값도 같은 경로를 탄다
    fs = dd.get("transverse_shear_fsdt")
    assert b["N_cr"] == pytest.approx(min(dd["N_cr"], fs["N_cr"]) if fs else dd["N_cr"],
                                      rel=1e-12)


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


# --- PC2-01 계열: 횡전단 보정을 전 모드에서 최소화한다 -------------------------

_SW_F = {"type": "orthotropic_2d", "E1": 70000.0, "E2": 70000.0, "G12": 26000.0, "nu12": 0.34}
_SW_C = {"type": "orthotropic_2d", "E1": 200.0, "E2": 200.0, "G12": 77.0, "nu12": 0.3,
         "G13": 60.0, "G23": 40.0}
_SW_P = {"Lx": 300.0, "Ly": 300.0}


def _sandwich(scale=1.0):
    return {"unit_system": "SI_mm", "laminae": [
        {"material": _SW_F, "angle_deg": 0.0, "thickness": 0.5 * scale},
        {"material": _SW_C, "angle_deg": 0.0, "thickness": 10.0 * scale},
        {"material": _SW_F, "angle_deg": 0.0, "thickness": 0.5 * scale}]}


def test_fsdt_minimum_is_below_clt_mode_correction():
    """CLT 임계모드에서만 보정하면 임계 모드 이동을 놓친다."""
    d = srv.compute_buckling(_sandwich(), panel=_SW_P, applied_Nx=200.0)["data"]
    fs = d["transverse_shear_fsdt"]
    assert fs["N_cr"] <= d["transverse_shear"]["corrected_N_cr"]
    assert (fs["mode"]["m"], fs["mode"]["n"]) != (d["mode"]["m"], d["mode"]["n"])
    assert d["margin"]["governing_factor"] == pytest.approx(fs["N_cr"] / 200.0, rel=1e-9)


def test_fsdt_saturates_at_the_crimping_limit():
    """m→∞ 에서 N_fsdt → A55 — 코어 전단 크림핑 폐형해 Gc·d²/tc 와 일치."""
    fs = srv.compute_buckling(_sandwich(), panel=_SW_P)["data"]["transverse_shear_fsdt"]
    closed_form = 60.0 * 10.5 ** 2 / 10.0            # G_c·d²/t_c (Allen)
    assert fs["A55"] == pytest.approx(closed_form, rel=2e-3)
    assert fs["N_cr"] < fs["A55"]                    # 임계하중은 크림핑 한계 아래다


@pytest.mark.parametrize("nx,target", [(200.0, 1.5), (1000.0, 1.5), (2000.0, 1.0)])
def test_required_scale_hits_target_under_crimping(nx, target):
    """크림핑이 지배해도 배율이 목표를 정확히 맞춘다 — s=1 모드 고정은 한계를 넘겼다."""
    d = srv.solve_required_thickness_scale(
        _sandwich(), panel=_SW_P, applied_Nx=nx, target_margin=target)["data"]
    b = srv.compute_buckling(_sandwich(d["required_scale"]), panel=_SW_P,
                             applied_Nx=nx)["data"]
    assert b["margin"]["governing_factor"] == pytest.approx(target, rel=1e-5)


def test_required_scale_never_promises_more_than_crimping():
    """약속한 N_cr 이 크림핑 물리 상한(A55·s)을 넘으면 안 된다."""
    d = srv.solve_required_thickness_scale(
        _sandwich(), panel=_SW_P, applied_Nx=2000.0, target_margin=1.0)["data"]
    limit = d["shear_flexibility"]["crimping_limit_A55"] * d["required_scale"]
    # FSDT 최소는 m→∞ 에서 A55·s 로 **위에서** 수렴한다 — 유한 격자라 한 톨 위다.
    # 전에는 이 상한을 1.36배 넘겨 약속했다.
    assert d["N_cr_scaled"] <= limit * 1.001
    assert d["N_cr_scaled"] >= limit * 0.999      # 이 하중대는 크림핑이 지배한다


# --- 열 구속 좌굴도 같은 보정을 받는다 (PC2-01/contract) -----------------------

_TF = dict(_SW_F, alpha1=23e-6, alpha2=23e-6)
_TC = dict(_SW_C, G13=40.0, G23=40.0, alpha1=40e-6, alpha2=40e-6)
_T_SW = {"unit_system": "SI_mm", "laminae": [
    {"material": _TF, "angle_deg": 0.0, "thickness": 0.5},
    {"material": _TC, "angle_deg": 0.0, "thickness": 10.0},
    {"material": _TF, "angle_deg": 0.0, "thickness": 0.5}]}


def test_thermal_restrained_buckling_gets_shear_correction():
    """열 구속 경로만 원시 CLT 를 쓰던 것 — compute_buckling 과 일치해야 한다."""
    panel = {"Lx": 400.0, "Ly": 400.0}
    b = srv.compute_thermal_response(_T_SW, delta_T=100.0,
                                     panel=panel)["data"]["restrained_buckling"]
    assert b["shear_flexibility_applied"] is True
    assert b["N_cr"] < b["N_cr_clt"]
    ref = srv.compute_buckling(_T_SW, panel=panel, applied_Nx=abs(b["restrained_Nx"]),
                               load_ratio=b["restrained_Ny"] / b["restrained_Nx"])["data"]
    assert b["load_factor_to_buckling"] == pytest.approx(
        ref["margin"]["governing_factor"], rel=1e-9)
    assert b["load_factor_to_buckling"] < 1.0        # 실제로는 이미 좌굴이다


# --- 전단 지배 구속 반력에 침묵하지 않는다 (PC2-02) ----------------------------

_NEG_SHEAR = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0,
              "nu12": 0.28, "alpha1": -4.0e-6, "alpha2": 22.5e-6}
_UNBAL = {"unit_system": "SI_mm", "laminae": [
    {"material": _NEG_SHEAR, "angle_deg": a, "thickness": 0.125} for a in (30.0, 60.0, 60.0, 30.0)]}


def test_thermal_shear_dominated_principal_compression_warns():
    """축 반력은 둘 다 인장인데 Nxy 주응력이 압축인 경우 — 전에는 경고 0건이었다."""
    r = srv.compute_thermal_response(_UNBAL, delta_T=100.0, panel={"Lx": 300.0, "Ly": 300.0})
    nr = [-v for v in r["data"]["thermal_loads"]["N_thermal"]]
    assert nr[0] > 0 and nr[1] > 0 and abs(nr[2]) > 0
    assert "restrained_buckling" not in r["data"]
    assert any("주응력이 압축" in w["message"] for w in r["warnings"])


def test_gate_does_not_claim_y_compression_when_both_are_tension():
    """순수 전단 압축에 'y방향만 압축'이라 안내하면 실행 불가능한 지시가 된다."""
    lam = _lam([45, -45, -45, 45])
    r = srv.recover_ply_stresses(lam, loads={"N": [10.0, 10.0, 60.0]}, panel=PANEL)
    msgs = " ".join(w["message"] for w in r["warnings"])
    assert "y방향만 압축" not in msgs
    assert "둘 다 인장" in msgs and "전단 좌굴을 풀지" in msgs


# --- DIF-D1: 탈습 프로파일은 흡습의 여집합이다 --------------------------------

@pytest.mark.parametrize("tau", [0.05, 0.1, 0.3, 1.0])
def test_desorption_profile_is_the_complement(tau):
    r = srv.compute_moisture_uptake(_lam([0, 90, 90, 0], mat=_WET), diffusion=_DIFF,
                                    time_s=_time_for_tau(tau), mode="desorption")["data"]
    a = srv.compute_moisture_uptake(_lam([0, 90, 90, 0], mat=_WET), diffusion=_DIFF,
                                    time_s=_time_for_tau(tau))["data"]
    for pd, pa in zip(r["profile"], a["profile"]):
        assert pd["c_over_cinf"] == pytest.approx(1.0 - pa["c_over_cinf"], abs=1e-12)
    # 노출면이 먼저 마르고 중앙이 가장 늦게 마른다
    assert r["profile"][0]["c_over_cinf"] < r["profile"][4]["c_over_cinf"]


def test_desorption_profile_average_matches_remaining_fraction():
    """두께평균이 remaining_fraction 과 맞아야 한다 — 반전 상태에선 22배 어긋났다."""
    from app.solver import diffusion as DF
    tau, n = 0.3, 4001
    vals = [1.0 - DF.concentration_profile(tau, i / (n - 1)) for i in range(n)]
    avg = sum((vals[i] + vals[i + 1]) / 2 for i in range(n - 1)) / (n - 1)   # 사다리꼴
    d = srv.compute_moisture_uptake(_lam([0, 90, 90, 0], mat=_WET), diffusion=_DIFF,
                                    time_s=_time_for_tau(tau), mode="desorption")["data"]
    assert d["state"]["remaining_fraction"] == pytest.approx(avg, rel=1e-4)


def test_desorption_start_is_uniformly_saturated():
    d = srv.compute_moisture_uptake(_lam([0, 90, 90, 0], mat=_WET), diffusion=_DIFF,
                                    time_s=0.0, mode="desorption")["data"]
    inner = [p["c_over_cinf"] for p in d["profile"][1:-1]]
    assert all(v == pytest.approx(1.0, abs=1e-12) for v in inner)
    assert d["profile"][0]["c_over_cinf"] == pytest.approx(0.0, abs=1e-12)
