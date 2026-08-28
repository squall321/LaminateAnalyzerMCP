# V4 확장 — 미시역학·수분확산·경계조건·지배모드 게이트 (계획서 §19.3~§19.7)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import diffusion as DF
from app.solver import micromechanics as MM
from app.solver import plate_navier as NAV

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
T300F = dict(T300, alpha1=0.02e-6, alpha2=22.5e-6, rho=1.6e-9)
STRONG = dict(T300, strength={"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0})

FIBER = {"E1": 231e9, "E2": 14.8e9, "G12": 22e9, "nu12": 0.2,
         "alpha1": -0.5e-6, "alpha2": 10e-6, "rho": 1790.0}
MATRIX = {"E": 3.45e9, "nu": 0.35, "alpha": 45e-6, "rho": 1265.0}


def lam(angles, mat=T300, t=0.125):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": mat, "angle_deg": a, "thickness": t} for a in angles]}


# ── §19.3 미시역학 ──────────────────────────────────────────────────────────

def test_halpin_tsai_xi_limits_are_reuss_and_voigt():
    """Halpin–Tsai 는 ξ→0 에서 Reuss(하한), ξ→∞ 에서 Voigt(상한)로 정확히 수렴한다."""
    ef, em, vf = 14.8e9, 3.45e9, 0.6
    assert MM.halpin_tsai(ef, em, vf, 1e-12) == pytest.approx(MM.reuss(ef, em, vf), rel=1e-9)
    assert MM.halpin_tsai(ef, em, vf, 1e12) == pytest.approx(MM.voigt(ef, em, vf), rel=1e-9)
    assert MM.reuss(ef, em, vf) < MM.halpin_tsai(ef, em, vf, 2.0) < MM.voigt(ef, em, vf)


@pytest.mark.parametrize("model", ["halpin_tsai", "chamis"])
def test_micromechanics_volume_fraction_limits(model):
    """Vf→1 은 섬유 물성, Vf→0 은 수지 물성으로 환원된다."""
    hi = MM.lamina_from_constituents(FIBER, MATRIX, 1.0, model=model)
    assert hi["E1"] == pytest.approx(FIBER["E1"], rel=1e-12)
    assert hi["E2"] == pytest.approx(FIBER["E2"], rel=1e-9)
    # Chamis 는 √Vf 로 수렴하므로 Halpin–Tsai(Vf 선형)보다 느리다 — 그 차이를 반영한다
    lo = MM.lamina_from_constituents(FIBER, MATRIX, 1e-12, model=model)
    assert lo["E1"] == pytest.approx(MATRIX["E"], rel=1e-9)
    assert lo["E2"] == pytest.approx(MATRIX["E"], rel=1e-5)
    tiny = MM.lamina_from_constituents(FIBER, MATRIX, 1e-20, model=model)
    assert tiny["E2"] == pytest.approx(MATRIX["E"], rel=1e-8)


def test_micromechanics_tool_reports_bounds_and_uncertainty():
    """추정값이 Reuss–Voigt 사이에 있고, 기지 지배 물성의 불확실성을 W120으로 알린다."""
    env = srv.derive_lamina_from_constituents(FIBER, MATRIX, 0.6)
    d = env["data"]
    assert env["errors"] == []
    for key in ("E2", "G12"):
        assert d["bounds"][key]["reuss"] <= d["material"][key] <= d["bounds"][key]["voigt"]
    assert d["material"]["source"]["type"] == "estimated"
    assert any(w["code"] == "W120" and "기지 지배" in w["message"] for w in env["warnings"])
    assert any(w["code"] == "W130" for w in
               srv.derive_lamina_from_constituents(FIBER, MATRIX, 0.8)["warnings"])


def test_micromechanics_chains_into_laminate_analysis():
    """만든 material 을 그대로 laminae 에 넣어 해석이 이어진다 — 이게 이 도구의 목적이다."""
    mat = srv.derive_lamina_from_constituents(FIBER, MATRIX, 0.6)["data"]["material"]
    payload = {"unit_system": "SI",
               "laminae": [{"material": mat, "angle_deg": a, "thickness": 0.125e-3}
                           for a in (0.0, 90.0, 90.0, 0.0)]}
    env = srv.analyze_laminate(payload)
    assert env["errors"] == []
    assert any("estimated" in a for a in env["assumptions"])


def test_micromechanics_errors():
    assert srv.derive_lamina_from_constituents(FIBER, MATRIX, 0.0)["errors"][0]["code"] == "E100"
    assert srv.derive_lamina_from_constituents(FIBER, MATRIX, 1.5)["errors"][0]["code"] == "E100"
    assert srv.derive_lamina_from_constituents({"E1": 1.0}, MATRIX, 0.6)["errors"][0]["code"] == "E100"
    assert srv.derive_lamina_from_constituents(
        FIBER, MATRIX, 0.6, model="nope")["errors"][0]["code"] == "E100"


def test_homogenize_reverse_gate_without_false_positive():
    """§19.3 역게이트 — 섬유/수지 오용은 경고하되 정당한 동박층은 침묵해야 한다."""
    cu = srv.homogenize_layer([
        {"material": {"type": "isotropic", "E": 117000.0, "nu": 0.34}, "volume_fraction": 0.7},
        {"material": {"type": "isotropic", "E": 3500.0, "nu": 0.35}, "volume_fraction": 0.3}])
    assert cu["warnings"] == []                       # 주용도에 경고 피로를 만들지 않는다
    assert "derive_lamina_from_constituents" in cu["data"]["scope"]
    fiber = srv.homogenize_layer([
        {"material": {"type": "isotropic", "E": 230000.0, "nu": 0.2}, "volume_fraction": 0.6},
        {"material": {"type": "isotropic", "E": 3500.0, "nu": 0.35}, "volume_fraction": 0.4}])
    assert any(w["code"] == "W130" and "derive_lamina_from_constituents" in w["message"]
               for w in fiber["warnings"])


# ── §19.4 수분 확산 ─────────────────────────────────────────────────────────

def test_diffusion_limits_and_sqrt_asymptote():
    """τ→0 은 0, τ→∞ 는 1. 초기 √t 점근이 급수해와 일치한다 (독립 경로 대조)."""
    assert DF.uptake_fraction(0.0) == 0.0
    assert DF.uptake_fraction(10.0) == pytest.approx(1.0, abs=1e-12)
    for tau in (1e-5, 1e-4, 1e-3, 1e-2):
        assert DF.uptake_fraction(tau) == pytest.approx(DF.early_time_fraction(tau), rel=1e-9)


def test_diffusion_characteristic_time_matches_literature():
    """M/M∞ = 0.5 는 τ ≈ 0.049 (문헌 표준값)."""
    tau50 = DF.tau_for_fraction(0.5)
    assert tau50 == pytest.approx(0.0492, rel=1e-2)
    assert DF.uptake_fraction(tau50) == pytest.approx(0.5, rel=1e-9)
    assert DF.tau_for_fraction(1.5) is None


def test_diffusion_time_scales_with_thickness_squared():
    """τ = D·t/h² 이므로 두께가 2배면 시간이 4배다."""
    thin = srv.compute_moisture_uptake(lam((0.0, 90.0), mat=T300F, t=0.5),
                                       diffusion={"D": 5e-7, "M_inf": 0.6})["data"]
    thick = srv.compute_moisture_uptake(lam((0.0, 90.0), mat=T300F, t=1.0),
                                        diffusion={"D": 5e-7, "M_inf": 0.6})["data"]
    assert (thick["characteristic_times"]["t50"]["seconds"]
            / thin["characteristic_times"]["t50"]["seconds"]) == pytest.approx(4.0, rel=1e-9)


def test_diffusion_chains_into_thermal_tool():
    """확산 결과의 delta_C 가 compute_thermal_response 로 그대로 이어진다."""
    payload = lam((0.0, 90.0, 90.0, 0.0), mat=dict(T300F, beta1=0.002, beta2=0.002), t=0.4)
    s = srv.compute_moisture_uptake(payload, diffusion={"D": 5e-7, "M_inf": 0.6},
                                    time_s=86400 * 7)["data"]["state"]
    assert 0.0 < s["uptake_fraction"] < 1.0
    env = srv.compute_thermal_response(payload, delta_C=s["delta_C_for_thermal_tool"])
    assert env["errors"] == []
    assert env["data"]["response"]["epsilon0"][0] > 0.0     # 흡습 팽창


def test_diffusion_desorption_is_complement():
    """베이크(탈습)는 흡습의 여집합 — 같은 τ 에서 남은 양 = 1 − 흡습률."""
    payload = lam((0.0, 90.0), mat=T300F, t=0.8)
    kw = dict(diffusion={"D": 5e-7, "M_inf": 0.6}, time_s=86400 * 3)
    a = srv.compute_moisture_uptake(payload, **kw)["data"]["state"]
    b = srv.compute_moisture_uptake(payload, mode="desorption", **kw)["data"]["state"]
    assert b["remaining_fraction"] == pytest.approx(1.0 - a["uptake_fraction"], rel=1e-12)
    assert b["delta_C_for_thermal_tool"] < 0.0             # 건조 = 수축


def test_diffusion_profile_is_wettest_at_surface():
    """두께 분포는 노출면이 가장 젖고 중앙이 가장 늦다."""
    d = srv.compute_moisture_uptake(lam((0.0, 90.0), mat=T300F, t=0.8),
                                    diffusion={"D": 5e-7, "M_inf": 0.6},
                                    time_s=86400)["data"]
    prof = [p["c_over_cinf"] for p in d["profile"]]
    assert prof[0] == pytest.approx(1.0, rel=1e-9)
    assert prof[-1] == pytest.approx(1.0, rel=1e-9)
    assert min(prof) == prof[len(prof) // 2]


def test_diffusion_errors_and_arrhenius():
    payload = lam((0.0, 90.0), mat=T300F)
    assert srv.compute_moisture_uptake(payload, diffusion={})["errors"][0]["code"] == "E100"
    assert srv.compute_moisture_uptake(
        payload, diffusion={"D": -1.0, "M_inf": 0.6})["errors"][0]["code"] == "E100"
    assert srv.compute_moisture_uptake(
        payload, diffusion={"D": 5e-7, "M_inf": 0.6}, mode="x")["errors"][0]["code"] == "E100"
    arr = srv.compute_moisture_uptake(payload, diffusion={
        "D0": 1e-3, "Ed": 50000.0, "temperature_K": 358.0, "M_inf": 0.6})
    assert arr["errors"] == [] and arr["data"]["diffusivity_SI_m2_per_s"] > 0.0


# ── §19.5 경계조건 ──────────────────────────────────────────────────────────

def test_clamped_beam_constants_match_literature():
    """고정-고정 보 고유값과 q = λ⁴ 항등식 (수치 적분의 독립 검증)."""
    for n, expect in ((1, 4.730041), (2, 7.853205), (3, 10.995608)):
        p_, q_ = NAV.clamped_pq(n)
        lam_n = NAV._cc_eigenvalue(n)
        assert lam_n == pytest.approx(expect, rel=1e-6)
        assert q_ == pytest.approx(lam_n ** 4, rel=1e-8)
    assert NAV.clamped_pq(1)[0] == pytest.approx(12.3026, rel=1e-4)


@pytest.mark.parametrize("load_ratio", [0.0, 0.5, 1.0])
def test_ritz_reduces_to_navier_for_simply_supported(load_ratio):
    """새 Ritz 경로에 SS 상수를 넣으면 기존 Navier 해와 정확히 일치한다."""
    import numpy as np
    E, nu, h, L = 70e9, 0.3, 2e-3, 0.5
    Dv = E * h ** 3 / (12 * (1 - nu ** 2))
    Dm = np.array([[Dv, nu * Dv, 0.0], [nu * Dv, Dv, 0.0], [0.0, 0.0, (1 - nu) * Dv / 2]])
    a = NAV.scan_ritz_buckling(Dm, L, L, load_ratio, "simply_supported", 10)
    b = NAV.buckling_ncr(Dm, L, L, load_ratio)
    assert a["N_cr"] == pytest.approx(b["N_cr"], rel=1e-12)
    assert (a["mode_m"], a["mode_n"]) == (b["mode_m"], b["mode_n"])


def test_clamped_is_stiffer_and_flagged_as_upper_bound():
    """고정단은 SS보다 뻣뻣하지만 1항 Ritz 는 상계라 비보수 — 반드시 경고한다."""
    payload = lam((0.0, 90.0, 90.0, 0.0), mat=T300F)
    panel = {"Lx": 300.0, "Ly": 300.0}
    ss = srv.compute_buckling(payload, panel=panel)["data"]
    cl = srv.compute_buckling(payload, panel=panel, boundary="clamped")
    assert cl["data"]["N_cr"] > ss["N_cr"]
    assert cl["data"]["boundary"] == "clamped"
    assert any(w["code"] == "W130" and "상계" in w["message"] for w in cl["warnings"])
    f_ss = srv.compute_natural_frequencies(payload, panel=panel)["data"]["modes"][0]["f_hz"]
    f_cl = srv.compute_natural_frequencies(payload, panel=panel,
                                           boundary="clamped")["data"]["modes"][0]["f_hz"]
    assert f_cl > f_ss


def test_isotropic_clamped_coefficients_match_known_values():
    """등방 정사각: SS k=4.000(정확), 고정단 k=10.74(정해 10.07 대비 1항 Ritz 상계)."""
    import numpy as np
    E, nu, h, L = 70e9, 0.3, 2e-3, 0.5
    Dv = E * h ** 3 / (12 * (1 - nu ** 2))
    Dm = np.array([[Dv, nu * Dv, 0.0], [nu * Dv, Dv, 0.0], [0.0, 0.0, (1 - nu) * Dv / 2]])
    k_ss = NAV.scan_ritz_buckling(Dm, L, L, 0.0, "simply_supported", 10)["N_cr"] * L * L / (math.pi ** 2 * Dv)
    k_cl = NAV.scan_ritz_buckling(Dm, L, L, 0.0, "clamped", 8)["N_cr"] * L * L / (math.pi ** 2 * Dv)
    assert k_ss == pytest.approx(4.0, rel=1e-9)
    assert k_cl == pytest.approx(10.74, rel=1e-3)
    assert k_cl / 10.07 == pytest.approx(1.066, rel=1e-2)      # 알려진 +6.6% 상계 오차


def test_panel_rejects_unknown_keys():
    """§19.5 게이트 — panel 의 미지 키를 조용히 무시하지 않는다.

    적대 검증에서 panel={"Lx","Ly","edge_condition":"clamped"} 가 status=ok 로 통과해
    에이전트가 고정단을 요청했다고 믿고 단순지지 값을 받았다.
    """
    payload = lam((0.0, 90.0, 90.0, 0.0), mat=T300F)
    assert srv.compute_buckling(payload, panel={"Lx": 300.0, "Ly": 200.0})["errors"] == []
    for bad in ({"Lx": 300.0, "Ly": 200.0, "edge_condition": "clamped"},
                {"Lx": 300.0, "Ly": 200.0, "typo_Lx": 999.0}):
        assert srv.compute_buckling(payload, panel=bad)["errors"][0]["code"] == "E100"
    assert srv.compute_buckling(payload, panel={"Lx": 300.0, "Ly": 300.0},
                                boundary="nope")["errors"][0]["code"] == "E100"


# ── §19.6 지배 파손모드 게이트 ──────────────────────────────────────────────

def test_compression_without_panel_warns():
    """압축인데 판 크기가 없으면 좌굴을 볼 수 없다고 알린다."""
    env = srv.recover_ply_stresses(lam((0.0, 90.0, 90.0, 0.0), mat=STRONG),
                                   loads={"N": [-60.0, 0.0, 0.0]})
    assert any(w["code"] == "W130" and "압축" in w["message"] and "compute_buckling" in w["message"]
               for w in env["warnings"])


def test_compression_with_panel_ranks_buckling_first():
    """실측 410배 모순 — 강도 R=7.07 인데 좌굴 여유 0.017 이다."""
    env = srv.recover_ply_stresses(lam((0.0, 90.0, 90.0, 0.0), mat=STRONG),
                                   loads={"N": [-60.0, 0.0, 0.0]},
                                   panel={"Lx": 150.0, "Ly": 150.0})
    g = env["data"]["governing_mode"]
    assert g["governing_mode"] == "buckling"
    assert g["margins"]["strength"] > 5.0 and g["margins"]["buckling"] < 0.1
    assert g["margins"]["strength"] / g["margins"]["buckling"] > 100
    assert g["margin"] == min(g["margins"].values())
    assert any(w["code"] == "W130" and "좌굴이 지배한다" in w["message"] for w in env["warnings"])


def test_tension_stays_silent():
    """인장이면 좌굴과 무관하므로 기존 응답이 그대로여야 한다 (경고 피로 방지)."""
    env = srv.recover_ply_stresses(lam((0.0, 90.0, 90.0, 0.0), mat=STRONG),
                                   loads={"N": [60.0, 0.0, 0.0]},
                                   panel={"Lx": 150.0, "Ly": 150.0})
    assert "governing_mode" not in env["data"]
    assert not any("압축" in w["message"] or "좌굴이 지배" in w["message"] for w in env["warnings"])


def test_interlaminar_free_edge_warning_points_to_the_tool():
    """§19.6 — 자유단 W130 이 '3D 해석 필요'가 아니라 자기 서버의 도구를 지목해야 한다."""
    env = srv.compute_interlaminar_stresses(lam((0.0, 45.0, -45.0, 90.0)), shear={"Vx": 10.0})
    msgs = [w["message"] for w in env["warnings"] if "자유단" in w["message"]]
    assert msgs and all("assess_free_edge_delamination" in m for m in msgs)


# ── §19.7 구속 열좌굴 게이트 ────────────────────────────────────────────────

def test_thermal_restrained_buckling_gate():
    """자유 경계 해가 '휨 없음'이라 답해도 면내 구속이면 좌굴한다.

    실측: [±45/0/90]s 1.0mm, ΔT=+100, 200×200 → warpage 7.1e-16 mm·경고 0건 인데
    완전구속 ΔT_cr = 24.9 K 다.
    """
    env = srv.compute_thermal_response(
        lam((45.0, -45.0, 0.0, 90.0, 90.0, 0.0, -45.0, 45.0), mat=T300F),
        delta_T=100.0, panel={"Lx": 200.0, "Ly": 200.0})
    d = env["data"]
    assert abs(d["warpage"]["range"]) < 1e-9          # 자유 경계로는 휨이 없다
    rb = d["restrained_buckling"]
    assert rb["load_factor_to_buckling"] < 1.0
    assert rb["delta_T_critical"] == pytest.approx(100.0 * rb["load_factor_to_buckling"], rel=1e-12)
    assert rb["restrained_Nx"] == pytest.approx(-d["thermal_loads"]["N_thermal"][0], rel=1e-12)
    assert any(w["code"] == "W130" and "이미 좌굴한다" in w["message"] for w in env["warnings"])


def test_thermal_cooling_has_no_buckling_gate():
    """냉각은 구속 시 인장이라 좌굴과 무관 — 게이트가 침묵해야 한다."""
    env = srv.compute_thermal_response(lam((0.0, 90.0, 90.0, 0.0), mat=T300F),
                                       delta_T=-150.0, panel={"Lx": 200.0, "Ly": 200.0})
    assert "restrained_buckling" not in env["data"]
    assert "N_thermal" in env["data"]["thermal_loads"]      # 합력 자체는 항상 노출


def test_thermal_loads_exposed_and_symmetric_has_zero_moment():
    """N^T·M^T 는 항상 응답에 실린다. 대칭 적층이면 M^T = 0 이 불변식이다."""
    d = srv.compute_thermal_response(lam((0.0, 90.0, 90.0, 0.0), mat=T300F),
                                     delta_T=100.0)["data"]["thermal_loads"]
    assert d["N_thermal"][0] > 0.0                          # 가열 → 팽창 합력
    assert all(abs(v) < 1e-9 for v in d["M_thermal"])
    unsym = srv.compute_thermal_response(lam((0.0, 90.0), mat=T300F),
                                         delta_T=100.0)["data"]["thermal_loads"]
    assert abs(unsym["M_thermal"][0]) > 1e-9                # 비대칭이면 0이 아니다
