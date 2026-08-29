# §19.22~19.24 접착 겹치기·노치 강도·응력 이완 (2순위 17·14·10번)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import lap_joint as LJ
from app.solver import notched as NT

AL = {"unit_system": "SI_mm",
      "laminae": [{"material": {"type": "isotropic", "E": 70000.0, "nu": 0.33},
                   "angle_deg": 0.0, "thickness": 2.0}]}
ADH = {"G": 1000.0, "thickness": 0.2, "shear_strength": 35.0}
T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
UTG = {"type": "isotropic", "E": 71000.0, "nu": 0.22}
PSA = {"type": "isotropic", "E": 1.0, "nu": 0.49,
       "viscoelastic": {"E0": 1.0, "Einf": 0.05, "tau_s": 3600.0}}
PI = {"type": "isotropic", "E": 4000.0, "nu": 0.35}


def lam(angles, mat=T300, t=0.125):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": mat, "angle_deg": a, "thickness": t} for a in angles]}


def fold():
    return {"unit_system": "SI_mm", "laminae": [
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.05},
        {"material": PSA, "angle_deg": 0.0, "thickness": 0.025},
        {"material": PI, "angle_deg": 0.0, "thickness": 0.05}]}


# ── §19.22 접착 겹치기 ──────────────────────────────────────────────────────

def test_uniform_shear_when_adhesive_is_very_soft():
    """G_a→0 이면 전단이 균일해진다 (피크/평균 = 1)."""
    assert LJ.peak_over_average(1e-12, 25e-3) == pytest.approx(1.0, rel=1e-9)
    d = srv.assess_bonded_lap_joint(AL, adhesive={"G": 1e-6, "thickness": 0.2},
                                    overlap=25.0, load=1000.0)["data"]
    assert d["peak_over_average"] == pytest.approx(1.0, rel=1e-6)
    assert d["tau_peak"] == pytest.approx(d["tau_avg"], rel=1e-6)


def test_peak_saturates_with_overlap():
    """**겹침을 늘려도 피크가 줄지 않는다** — 이 도구의 존재 이유."""
    peaks = {}
    for L in (25.0, 50.0, 100.0):
        peaks[L] = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=L,
                                               load=1000.0)["data"]["tau_peak"]
    assert peaks[25.0] / peaks[50.0] == pytest.approx(1.0, abs=0.01)   # 2배 늘려도 0.3%
    assert peaks[50.0] / peaks[100.0] == pytest.approx(1.0, abs=1e-4)
    env = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=50.0, load=1000.0)
    assert any(w["code"] == "W130" and "더 늘려도 피크 응력이 거의" in w["message"]
               for w in env["warnings"])


def test_overlap_efficiency_and_saturation_length():
    d = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=100.0, load=1000.0)["data"]
    assert d["overlap_efficiency"] == pytest.approx(1.0 / d["peak_over_average"], rel=1e-12)
    assert d["overlap_efficiency"] < 0.1                      # 100mm 겹침의 7.5% 만 실효
    sat = d["saturation_overlap"]
    assert 10.0 < sat < 40.0
    # 포화 길이에서 ωL/2 = 3 이어야 한다
    at_sat = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=sat,
                                         load=1000.0)["data"]
    assert at_sat["shear_lag"]["omega_L_half"] == pytest.approx(
        LJ.SATURATION_OMEGA_L_HALF, rel=1e-9)


def test_profile_is_symmetric_and_peaks_at_ends():
    d = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=25.0, load=1000.0)["data"]
    p = d["profile"]
    assert p[0]["tau"] == pytest.approx(p[-1]["tau"], rel=1e-9)       # 대칭
    assert p[0]["tau"] == pytest.approx(d["tau_peak"], rel=1e-9)      # 끝단이 피크
    assert min(r["tau"] for r in p) == pytest.approx(p[len(p) // 2]["tau"], rel=1e-9)


def test_single_lap_peel_warning():
    """단일 겹치기는 peel 이 지배하는데 Volkersen 은 모델링하지 않는다 — 비보수."""
    env = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=25.0, load=1000.0,
                                      joint_type="single_lap")
    assert any(w["code"] == "W130" and "peel" in w["message"] and "비보수" in w["message"]
               for w in env["warnings"])
    dbl = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=25.0, load=1000.0)
    assert not any("peel" in w["message"] for w in dbl["warnings"])


def test_dissimilar_adherends_and_errors():
    thin = {"unit_system": "SI_mm",
            "laminae": [{"material": {"type": "isotropic", "E": 70000.0, "nu": 0.33},
                         "angle_deg": 0.0, "thickness": 1.0}]}
    env = srv.assess_bonded_lap_joint(AL, adhesive=ADH, overlap=25.0, load=1000.0,
                                      laminate_2=thin)
    assert env["data"]["peak_over_average"] > srv.assess_bonded_lap_joint(
        AL, adhesive=ADH, overlap=25.0, load=1000.0)["data"]["peak_over_average"]
    assert any(w["code"] == "W120" and "피착재 축강성이 다르다" in w["message"]
               for w in env["warnings"])
    for kw in ({"overlap": -1.0}, {"load": 0.0}, {"joint_type": "x"},
               {"adhesive": {"G": 1000.0}}, {"adhesive": {"G": 1000.0, "thickness": 0.2, "typo": 1}}):
        base = dict(adhesive=ADH, overlap=25.0, load=1000.0)
        base.update(kw)
        assert srv.assess_bonded_lap_joint(AL, **base)["errors"][0]["code"] == "E100"


# ── §19.23 노치 강도 ────────────────────────────────────────────────────────

def test_kt_isotropic_limit_is_exactly_three():
    E, nu = 70e9, 0.3
    assert NT.kt_infinite(E, E, nu, E / (2 * (1 + nu))) == pytest.approx(3.0, rel=1e-12)
    d = srv.assess_notched_strength(AL, hole_diameter=6.35)["data"]
    assert d["K_T"] == pytest.approx(3.0, rel=1e-9)


def test_kt_grows_with_anisotropy():
    qi = lam((45.0, -45.0, 0.0, 90.0, 90.0, 0.0, -45.0, 45.0))
    cp = lam((0.0, 90.0, 90.0, 0.0))
    ud = lam((0.0,) * 4)
    kts = [srv.assess_notched_strength(x, hole_diameter=6.35)["data"]["K_T"]
           for x in (qi, cp, ud)]
    assert kts[0] == pytest.approx(3.0, rel=1e-2)     # 준등방은 등방과 같다
    assert kts[0] < kts[1] < kts[2]
    assert kts[2] > 6.0
    assert any(w["code"] == "W130" and "K_T" in str(w.get("field", ""))
               for w in srv.assess_notched_strength(ud, hole_diameter=6.35)["warnings"])


@pytest.mark.parametrize("kt", [3.0, 4.9, 6.75])
def test_whitney_nuismer_limits(kt):
    """d0·a0 → 0 이면 1/K_T, → ∞ 이면 1 로 환원된다."""
    r = 3.175e-3
    assert NT.point_stress_ratio(kt, r, 1e-12) == pytest.approx(1.0 / kt, rel=1e-4)
    assert NT.point_stress_ratio(kt, r, 1e9) == pytest.approx(1.0, rel=1e-6)
    assert NT.average_stress_ratio(kt, r, 1e-12) == pytest.approx(1.0 / kt, rel=1e-4)
    assert NT.average_stress_ratio(kt, r, 1e9) == pytest.approx(1.0, rel=1e-6)


def test_notched_strength_withheld_without_fitted_constants():
    """d0/a0 가 없으면 σ_OH 를 **아예 내지 않는다** — 임의 기본값은 그럴듯한 오답이다."""
    env = srv.assess_notched_strength(lam((0.0, 90.0, 90.0, 0.0)), hole_diameter=6.35,
                                      unnotched_strength=516.0)
    assert "notched_strength" not in env["data"]
    assert env["data"]["confidence"]["K_T"] == "closed_form"
    assert any(w["code"] == "W120" and "의도적이다" in w["message"] for w in env["warnings"])


def test_notched_strength_sensitivity_is_flagged():
    """d0 가 답을 지배한다 — 0.5~2.0mm 에서 σ_OH 가 크게 갈린다."""
    qi = lam((45.0, -45.0, 0.0, 90.0, 90.0, 0.0, -45.0, 45.0))
    vals = {}
    for d0 in (0.5, 1.0, 2.0):
        d = srv.assess_notched_strength(qi, hole_diameter=6.35, unnotched_strength=516.0,
                                        d0=d0)["data"]["notched_strength"]
        vals[d0] = d["point_stress"]["sigma_OH"]
    assert vals[2.0] / vals[0.5] > 1.5              # 4배 범위에서 1.5배 이상 갈린다
    env = srv.assess_notched_strength(qi, hole_diameter=6.35, unnotched_strength=516.0, d0=1.0)
    assert any(w["code"] == "W130" and "시험 피팅 상수" in w["message"] for w in env["warnings"])


def test_notched_errors():
    for kw in ({"hole_diameter": -1.0}, {"hole_diameter": 6.35, "d0": 0.0},
               {"hole_diameter": 6.35, "unnotched_strength": -5.0}):
        assert srv.assess_notched_strength(AL, **kw)["errors"][0]["code"] == "E100"


# ── §19.24 응력 이완 ────────────────────────────────────────────────────────

def test_relaxation_needs_prescribed_curvature():
    """이완은 변위를 고정한 상태에서만 의미가 있다."""
    assert srv.solve_stress_relaxation(fold(), times_s=[0.0, 3600.0])["errors"][0]["code"] == "E100"
    no_ve = {"unit_system": "SI_mm",
             "laminae": [{"material": UTG, "angle_deg": 0.0, "thickness": 0.1}]}
    assert srv.solve_stress_relaxation(no_ve, times_s=[0.0], bend_radius=5.0
                                       )["errors"][0]["code"] == "E100"


def test_modulus_decays_exponentially():
    """E(t) = E∞ + (E0−E∞)exp(−t/τ) — τ 에서 63% 이완."""
    d = srv.solve_stress_relaxation(fold(), times_s=[0.0, 3600.0, 1e9],
                                    bend_radius=5.0)["data"]
    e0, e_tau, e_inf = [r["E_viscoelastic"][0] for r in d["times"]]
    assert e0 == pytest.approx(1.0, rel=1e-9)
    assert e_inf == pytest.approx(0.05, rel=1e-6)
    assert e_tau == pytest.approx(0.05 + 0.95 * math.exp(-1.0), rel=1e-9)


def test_clt_path_shows_almost_no_relaxation():
    """순응층은 ABD 기여가 작아 CLT 로는 이완이 거의 안 보인다 — 이 사실 자체가 결과다."""
    d = srv.solve_stress_relaxation(fold(), times_s=[0.0, 36000.0],
                                    bend_radius=5.0)["data"]
    assert d["relaxation"]["M_ratio"] > 0.99


def test_partial_composite_path_shows_the_real_relaxation():
    """**전단 결합 경로로 보면 유효 굽힘강성이 0.37배로 무너진다** — 이 도구의 요점."""
    env = srv.solve_stress_relaxation(fold(), times_s=[0.0, 3600.0, 10800.0, 36000.0],
                                      bend_radius=5.0, span=10.0)
    d = env["data"]
    rl = d["relaxation"]
    assert rl["composite_action_initial"] > 0.7
    assert rl["composite_action_final"] < 0.2
    assert rl["EI_effective_ratio"] == pytest.approx(0.373, rel=0.02)
    # 합성도가 시간에 따라 단조 감소
    fs = [r["partial_composite"]["composite_action"] for r in d["times"]]
    assert fs == sorted(fs, reverse=True)
    assert any(w["code"] == "W130" and "부분합성 유효 굽힘강성은" in w["message"]
               for w in env["warnings"])


def test_span_omission_is_warned():
    env = srv.solve_stress_relaxation(fold(), times_s=[0.0, 36000.0], bend_radius=5.0)
    assert any(w["code"] == "W120" and "span 이 없어 부분합성" in w["message"]
               for w in env["warnings"])


def test_relaxation_deterministic_and_sorted():
    a = srv.solve_stress_relaxation(fold(), times_s=[36000.0, 0.0, 3600.0],
                                    bend_radius=5.0, span=10.0)
    b = srv.solve_stress_relaxation(fold(), times_s=[0.0, 3600.0, 36000.0],
                                    bend_radius=5.0, span=10.0)
    assert [r["time_s"] for r in a["data"]["times"]] == [0.0, 3600.0, 36000.0]
    assert a["data"]["times"] == b["data"]["times"]     # 입력 순서 무관
