# V2-2차 스프린트 테스트 — 설계규칙·좌굴·진동·진행성 파손·흡습 (계획서 §17.5)
from __future__ import annotations

import math

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import plate_navier as NAV

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
CFRP_S = {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}


def lam(angles, t=0.125, mat=None, **extra_mat):
    m = dict(mat or T300, **extra_mat)
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": t, "angle_deg": a, "material": dict(m)} for a in angles]}


def rule_map(env):
    return {r["rule"]: r for r in env["data"]["rules"]}


# ── 설계 규칙 ────────────────────────────────────────────────────────────────

def test_rules_good_quad_layup_passes():
    """[45/0/-45/90]s — 인접각 45° 이하까지 전 규칙 통과하는 정석 배열.

    주의: 흔한 [45/-45/0/90]s는 45/-45(90° 점프)·0/90 계면에서 adjacent_angle 위반 —
    검사기가 이를 잡아내는 것이 정상이다(별도 테스트).
    """
    env = srv.check_design_rules(lam([45, 0, -45, 90, 90, -45, 0, 45]))
    r = rule_map(env)
    for name in ("symmetry", "balance", "ten_percent", "contiguity", "adjacent_angle", "outer_protection"):
        assert r[name]["pass"] is True, name
    assert env["data"]["summary"]["hard_fails"] == []


def test_rules_pm45_adjacency_flagged():
    """[45/-45/0/90]s — 45/-45와 0/90 계면의 90° 점프를 adjacent_angle이 지적해야 한다."""
    env = srv.check_design_rules(lam([45, -45, 0, 90, 90, 0, -45, 45]))
    r = rule_map(env)
    assert r["adjacent_angle"]["pass"] is False
    assert r["symmetry"]["pass"] is True and r["balance"]["pass"] is True


def test_rules_violations_detected():
    # 비대칭 + 미짝 +30 + 0/90 인접 + 외층 0° + 90° 없음(10%)
    env = srv.check_design_rules(lam([0, 0, 0, 0, 0, 30]))
    r = rule_map(env)
    assert r["symmetry"]["pass"] is False
    assert r["balance"]["pass"] is False and "30" in r["balance"]["found"]
    assert r["ten_percent"]["pass"] is None      # 30° 포함 → 비쿼드 not_applicable
    assert r["outer_protection"]["pass"] is False

    env2 = srv.check_design_rules(lam([0, 0, 0, 0, 0, 90]))   # 쿼드지만 45 계열 0%
    r2 = rule_map(env2)
    assert r2["ten_percent"]["pass"] is False
    assert r2["contiguity"]["pass"] is False                   # 0° 5연속 > 4
    assert r2["adjacent_angle"]["pass"] is False               # 0/90 직접 접촉
    assert "symmetry" in env2["data"]["summary"]["hard_fails"]


def test_rules_contiguity_limit_param():
    env = srv.check_design_rules(lam([0, 0, 0, 90, 90, 90]), contiguity_limit=2)
    assert rule_map(env)["contiguity"]["pass"] is False
    env2 = srv.check_design_rules(lam([0, 0, 0, 90, 90, 90]), contiguity_limit=3)
    assert rule_map(env2)["contiguity"]["pass"] is True
    assert srv.check_design_rules(lam([0]), contiguity_limit=0)["errors"][0]["code"] == "E100"


# ── 좌굴 (폐형해) ────────────────────────────────────────────────────────────

def iso_plate(E=70000.0, nu=0.3, t=2.0, rho=None):
    m = {"type": "isotropic", "E": E, "nu": nu}
    if rho is not None:
        m["rho"] = rho
    return {"unit_system": "SI_mm", "laminae": [{"thickness": t, "angle_deg": 0, "material": m}]}


def test_buckling_isotropic_square_closed_form():
    """등방 정사각 SS 판: N_cr = 4π²D/b², D = Eh³/(12(1−ν²)) — m=n=1."""
    E, nu, t, b = 70000.0, 0.3, 2.0, 200.0
    env = srv.compute_buckling(iso_plate(E, nu, t), panel={"Lx": b, "Ly": b})
    D = E * t**3 / (12 * (1 - nu * nu))
    assert env["data"]["N_cr"] == pytest.approx(4 * math.pi**2 * D / b**2, rel=1e-9)
    assert (env["data"]["mode"]["m"], env["data"]["mode"]["n"]) == (1, 1)
    assert env["data"]["mode"]["at_scan_boundary"] is False
    assert env["status"] == "ok"          # 등방 대칭 → W130 없음


def test_buckling_long_plate_mode_switch_k4():
    """a/b=2 장판: m=2로 전환하되 k=4 유지 (N_cr 동일) — 고전 결과."""
    E, nu, t, b = 70000.0, 0.3, 2.0, 200.0
    sq = srv.compute_buckling(iso_plate(E, nu, t), panel={"Lx": b, "Ly": b})["data"]
    lg = srv.compute_buckling(iso_plate(E, nu, t), panel={"Lx": 2 * b, "Ly": b})["data"]
    assert lg["mode"]["m"] == 2 and lg["mode"]["n"] == 1
    assert lg["N_cr"] == pytest.approx(sq["N_cr"], rel=1e-9)


def test_buckling_biaxial_and_margin():
    env = srv.compute_buckling(iso_plate(), panel={"Lx": 200.0, "Ly": 200.0},
                               load_ratio=1.0, applied_Nx=10.0)
    d = env["data"]
    # 등2축 정사각: N_cr = π²D[(1/a)²+(1/b)²]²/[(1/a)²+(1/b)²] = 2π²D/b² (단축의 절반)
    uni = srv.compute_buckling(iso_plate(), panel={"Lx": 200.0, "Ly": 200.0})["data"]["N_cr"]
    assert d["N_cr"] == pytest.approx(uni / 2.0, rel=1e-9)
    assert d["margin"]["factor"] == pytest.approx(d["N_cr"] / 10.0, rel=1e-12)


def test_buckling_unsymmetric_uses_reduced_stiffness_with_warning():
    env = srv.compute_buckling(lam([0, 90], t=0.5), panel={"Lx": 100.0, "Ly": 100.0})
    assert env["status"] == "warning"
    assert any(w["code"] == "W130" and "D*" in w["message"] + w["detail"] if "detail" in w else "D*" in w["message"]
               for w in env["warnings"]) or any("D*" in w["message"] for w in env["warnings"])
    assert env["data"]["applicability"]["reduced_stiffness_used"] is True
    # D* 축소는 D보다 유연 → 대칭 가정(D 그대로)보다 N_cr 낮아야 함
    sym = srv.compute_buckling(lam([0, 90, 90, 0], t=0.25), panel={"Lx": 100.0, "Ly": 100.0})["data"]["N_cr"]
    assert env["data"]["N_cr"] < sym


def test_buckling_validation():
    assert srv.compute_buckling(iso_plate(), panel={"Lx": -1, "Ly": 2})["errors"][0]["code"] == "E100"
    assert srv.compute_buckling(iso_plate(), panel={"Lx": 100, "Ly": 100},
                                applied_Nx=-5)["errors"][0]["code"] == "E100"


# ── 고유진동수 (폐형해) ──────────────────────────────────────────────────────

def test_frequencies_isotropic_closed_form():
    """등방 SS 판: ω_mn = π²[(m/a)²+(n/b)²]√(D/ρ_a)."""
    E, nu, t, rho = 70000.0, 0.3, 2.0, 2.7e-9    # SI_mm: MPa, mm, t/mm³
    a = b = 200.0
    env = srv.compute_natural_frequencies(iso_plate(E, nu, t, rho), panel={"Lx": a, "Ly": b}, n_modes=4)
    modes = env["data"]["modes"]
    # SI 값으로 폐형해 계산
    D_si = (E * 1e6) * (t * 1e-3)**3 / (12 * (1 - nu * nu))
    rho_a = (rho * 1e12) * (t * 1e-3)
    a_si = a * 1e-3
    for md in modes:
        m, n = md["m"], md["n"]
        omega = math.pi**2 * ((m / a_si)**2 + (n / a_si)**2) * math.sqrt(D_si / rho_a)
        assert md["f_hz"] == pytest.approx(omega / (2 * math.pi), rel=1e-9)
    assert modes[0]["m"] == 1 and modes[0]["n"] == 1
    assert modes[0]["f_hz"] < modes[-1]["f_hz"]


def test_frequencies_requires_rho():
    env = srv.compute_natural_frequencies(iso_plate(), panel={"Lx": 100, "Ly": 100})
    assert env["errors"][0]["code"] == "E100" and "rho" in env["errors"][0]["message"]


# ── 진행성 파손 ──────────────────────────────────────────────────────────────

def test_progressive_cross_ply_classic_sequence():
    """[0/90]s Nx: 90° 기지 파손(FPF) → 0° 섬유 파손(한계). ultimate > FPF."""
    env = srv.run_progressive_failure(lam([0, 90, 90, 0], mat=T300, strength=dict(CFRP_S)),
                                      loads={"N": [100.0, 0, 0]})
    d = env["data"]
    ev = d["events"]
    assert ev[0]["ply"] in (1, 2) and ev[0]["mode"] == "transverse_tension"
    assert any(e["mode"] == "fiber_tension" and e["ply"] in (0, 3) for e in ev)
    assert d["ultimate_R"] > d["first_ply_failure_R"]
    assert d["termination"] in ("load_carrying_collapse", "all_plies_failed",
                                "no_failable_plies", "stiffness_singular")
    # 유효 Ex는 사건마다 단조 감소
    exs = d["ex_eff_after_events"]
    assert all(exs[i + 1] <= exs[i] + 1e-6 for i in range(len(exs) - 1))
    # FPF R은 recover_ply_stresses의 R과 일치 (경로 정합)
    fpf = srv.recover_ply_stresses(lam([0, 90, 90, 0], mat=T300, strength=dict(CFRP_S)),
                                   loads={"N": [100.0, 0, 0]})["data"]["first_ply_failure"]
    assert d["first_ply_failure_R"] == pytest.approx(fpf["tsai_wu_R"], rel=1e-9)


def test_progressive_determinism_and_validation():
    payload = lam([0, 90], mat=T300, strength=dict(CFRP_S))
    from app.services.envelope import canonical_json
    a = srv.run_progressive_failure(payload, loads={"N": [50.0, 0, 0]})
    b = srv.run_progressive_failure(payload, loads={"N": [50.0, 0, 0]})
    assert canonical_json(a) == canonical_json(b)
    assert srv.run_progressive_failure(payload, loads={"N": [1, 0, 0]},
                                       discount=0.9)["errors"][0]["code"] == "E100"
    assert srv.run_progressive_failure(lam([0, 90]), loads={"N": [1, 0, 0]})["errors"][0]["code"] == "E100"


# ── 흡습 (열 등가) ───────────────────────────────────────────────────────────

def test_hygro_equivalence_with_thermal():
    """αΔT = βΔc 로 맞추면 열/흡습 곡률이 정확히 동일 (§17.5.4 등가 검증)."""
    dT, dC = 100.0, 2.0
    def bi(key_lo, key_hi, v1, v2):
        return {"unit_system": "SI_mm", "laminae": [
            {"thickness": 1.0, "angle_deg": 0, "material":
             {"type": "isotropic", "E": 100000.0, "nu": 0.3, key_lo: v1}},
            {"thickness": 1.0, "angle_deg": 0, "material":
             {"type": "isotropic", "E": 50000.0, "nu": 0.3, key_hi: v2}}]}
    th = srv.compute_thermal_response(bi("alpha", "alpha", 10e-6, 20e-6), delta_T=dT)
    hy = srv.compute_thermal_response(bi("beta", "beta", 10e-6 * dT / dC, 20e-6 * dT / dC), delta_C=dC)
    assert hy["data"]["response"]["kappa"][0] == pytest.approx(
        th["data"]["response"]["kappa"][0], rel=1e-12)
    assert "effective_cme" in hy["data"] and "effective_cte" in th["data"]
    assert hy["data"]["effective_cme"]["beta_x"] * dC == pytest.approx(
        th["data"]["effective_cte"]["alpha_x"] * dT, rel=1e-12)


def test_hygro_missing_beta_e203_and_both_none():
    payload = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 1.0, "angle_deg": 0, "material": {"type": "isotropic", "E": 70000.0, "nu": 0.3}}]}
    assert srv.compute_thermal_response(payload, delta_C=1.0)["errors"][0]["code"] == "E203"
    assert srv.compute_thermal_response(payload)["errors"][0]["code"] == "E100"


# ── 적대 검증(Workflow) 확인 결함 19건 회귀 ─────────────────────────────────

def test_nav1_negative_load_ratio_no_crash():
    """NAV-1: a/b=10, R=-1은 크래시(E501)가 아니라 적응 스캔으로 유효 해를 반환."""
    iso = {"unit_system": "SI", "laminae": [{"thickness": 0.002, "angle_deg": 0.0,
           "material": {"type": "isotropic", "E": 70e9, "nu": 0.3}}]}
    env = srv.compute_buckling(iso, panel={"Lx": 1.0, "Ly": 0.1}, load_ratio=-1.0)
    assert env["status"] in ("ok", "warning") and env["data"]["N_cr"] > 0
    assert env["data"]["mode"]["m"] > 10          # 스캔 확장으로 발견된 고차 모드
    # 압축 지배 모드가 아예 없으면 E501이 아니라 E100
    bad = srv.compute_buckling(iso, panel={"Lx": 1.0, "Ly": 1.0}, load_ratio=-1e6)
    assert bad["errors"][0]["code"] == "E100" and bad["errors"][0]["field"] == "load_ratio"


def test_nav2_long_plate_recovers_k4():
    """NAV-2: a/b=20 장판에서 스캔 경계에 갇히지 않고 참값 k=4를 회복."""
    E, nu, t, b = 70e9, 0.3, 0.002, 0.5
    iso = {"unit_system": "SI", "laminae": [{"thickness": t, "angle_deg": 0.0,
           "material": {"type": "isotropic", "E": E, "nu": nu}}]}
    env = srv.compute_buckling(iso, panel={"Lx": 10.0, "Ly": b})
    D = E * t**3 / (12 * (1 - nu * nu))
    k = env["data"]["N_cr"] * b**2 / (math.pi**2 * D)
    assert k == pytest.approx(4.0, rel=1e-6)
    assert env["data"]["mode"]["m"] == 20 and env["data"]["mode"]["at_scan_boundary"] is False


def test_nav3_high_mode_list_correct():
    """NAV-3: n_modes=25 요청 시 스캔이 함께 늘어 11번째 모드가 (11,1)로 정확."""
    iso = {"unit_system": "SI", "laminae": [{"thickness": 0.002, "angle_deg": 0.0,
           "material": {"type": "isotropic", "E": 70e9, "nu": 0.3, "rho": 2700.0}}]}
    env = srv.compute_natural_frequencies(iso, panel={"Lx": 10.0, "Ly": 0.5}, n_modes=25)
    m11 = env["data"]["modes"][10]
    assert (m11["m"], m11["n"]) == (11, 1)
    assert env["data"]["scan_limit"] >= 25
    fs = [m["f_hz"] for m in env["data"]["modes"]]
    assert fs == sorted(fs)


def test_edge01_nonfinite_inputs_are_e100():
    """EDGE-01/F2: inf·NaN 입력이 E501/E400이 아니라 E100으로 거부된다."""
    iso = {"unit_system": "SI", "laminae": [{"thickness": 0.002, "angle_deg": 0.0,
           "material": {"type": "isotropic", "E": 70e9, "nu": 0.3, "alpha": 23e-6}}]}
    assert srv.compute_buckling(iso, panel={"Lx": float("inf"), "Ly": 1.0})["errors"][0]["code"] == "E100"
    assert srv.compute_buckling(iso, panel={"Lx": 1.0, "Ly": 1.0},
                                load_ratio=float("inf"))["errors"][0]["code"] == "E100"
    assert srv.compute_thermal_response(iso, delta_T=float("nan"))["errors"][0]["code"] == "E100"
    assert srv.compute_thermal_response(iso, delta_T=float("inf"))["errors"][0]["code"] == "E100"


def test_prog1_ultimate_r_is_physical_lpf():
    """PROG-1: ultimate_R이 도달 불가 사건(σ1≈0의 거대 R)에 오염되지 않는다."""
    mat = dict(T300, strength=dict(CFRP_S))
    env = srv.run_progressive_failure(lam([0, 90, 90, 0], mat=mat), loads={"N": [100.0, 0, 0]})
    d = env["data"]
    assert d["ultimate_R"] < 10.0                       # 수정 전 2012.7
    assert d["ultimate_R"] == pytest.approx(max(e["R"] for e in d["events"]), rel=1e-12)
    assert d["termination"] == "load_carrying_collapse"
    assert any(e["mode"] == "fiber_tension" for e in d["events"])
    # 단일 90° ply: σ1 노이즈로 R~1e18이 나오면 안 됨
    solo = srv.run_progressive_failure(lam([90], t=1.0, mat=mat), loads={"N": [10.0, 0, 0]})["data"]
    assert solo["ultimate_R"] == pytest.approx(CFRP_S["Yt"] / 10.0, rel=1e-9)


def test_prog2_pure_bending_detects_failure():
    """PROG-2: 순수 굽힘에서 ply 중앙 σ≈0 때문에 파손이 누락되지 않는다(3점 평가)."""
    mat = dict(T300, strength=dict(CFRP_S))
    d = srv.run_progressive_failure(lam([0], t=1.0, mat=mat), loads={"M": [10.0, 0, 0]})["data"]
    assert d["events"] and d["events"][0]["mode"] in ("fiber_compression", "fiber_tension")
    assert d["ultimate_R"] > 0


def test_prog3_termination_distinguishes_elastic_plies():
    """PROG-3: strength 없는 ply가 남아 있으면 all_plies_failed로 보고하지 않는다."""
    mixed = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 0.5, "angle_deg": 0, "material": dict(T300, strength=dict(CFRP_S))},
        {"thickness": 0.5, "angle_deg": 90, "material": dict(T300)}]}
    d = srv.run_progressive_failure(mixed, loads={"N": [100.0, 0, 0]})["data"]
    assert d["termination"] in ("all_failable_plies_failed", "load_carrying_collapse",
                                "no_failable_plies")
    assert d["termination"] != "all_plies_failed"


def test_f1_balance_checks_thickness_and_material():
    """F1: 각도 짝만 맞고 두께가 다르면 balance는 fail (사양 §17.5.1)."""
    unbal = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 1.0, "angle_deg": 45, "material": dict(T300)},
        {"thickness": 0.1, "angle_deg": -45, "material": dict(T300)}]}
    assert rule_map(srv.check_design_rules(unbal))["balance"]["pass"] is False
    bal = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 0.5, "angle_deg": 45, "material": dict(T300)},
        {"thickness": 0.5, "angle_deg": -45, "material": dict(T300)}]}
    assert rule_map(srv.check_design_rules(bal))["balance"]["pass"] is True


def test_f6_balance_angle_tolerance():
    """F6: 0.3° 어긋난 ±45 짝은 tol 내이므로 거짓 경보를 내지 않는다."""
    near = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 0.5, "angle_deg": 44.7, "material": dict(T300)},
        {"thickness": 0.5, "angle_deg": -45.0, "material": dict(T300)}]}
    assert rule_map(srv.check_design_rules(near))["balance"]["pass"] is True


def test_f3_ten_percent_is_thickness_based():
    """F3: 매수로는 25%지만 두께로는 1.5%인 방향을 잡아낸다."""
    thin90 = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 0.03, "angle_deg": 90, "material": dict(T300)},
        {"thickness": 1.0, "angle_deg": 0, "material": dict(T300)},
        {"thickness": 0.5, "angle_deg": 45, "material": dict(T300)},
        {"thickness": 0.5, "angle_deg": -45, "material": dict(T300)}]}
    r = rule_map(srv.check_design_rules(thin90))["ten_percent"]
    assert r["pass"] is False and "두께 기준" in r["found"]


def test_f4_f5_f8_rule_contract():
    """F4(규칙 이름)·F5(단일 ply 외층)·F8(실행 가능한 인접각 힌트)."""
    names = [r["rule"] for r in srv.check_design_rules(lam([0]))["data"]["rules"]]
    assert "single_ply_angle_group" in names and "angle_distribution" not in names
    assert rule_map(srv.check_design_rules(lam([45], t=1.0)))["outer_protection"]["pass"] is True
    hint = rule_map(srv.check_design_rules(lam([45, -45, 45, -45])))["adjacent_angle"]["fix_hint"]
    assert "삽입" in hint and "±45" not in hint          # ±45 계면에 ±45 삽입 지시 금지


def test_f5_assumptions_no_self_contradiction():
    """F5: 열해석 응답에 '열 잔류변형 미포함' 가정이 남아 있지 않다."""
    iso = {"unit_system": "SI_mm", "laminae": [{"thickness": 1.0, "angle_deg": 0,
           "material": {"type": "isotropic", "E": 70000.0, "nu": 0.3, "alpha": 23e-6}}]}
    env = srv.compute_thermal_response(iso, delta_T=100.0)
    assert not any("미포함" in a for a in env["assumptions"])


def test_f10_version_bumped():
    assert srv.get_server_info()["server_version"] == "0.20.0"


def test_reference_cases_for_sprint_tools():
    """F2(UX): 신규 도구 few-shot이 실제 엔진과 일치."""
    bk = srv.get_reference_cases("isotropic_square_buckling")
    env = srv.compute_buckling(bk["input"]["laminate"], panel=bk["input"]["panel"])
    assert env["data"]["N_cr"] == pytest.approx(bk["expected"]["N_cr_N_per_mm"], rel=1e-9)
    assert [env["data"]["mode"]["m"], env["data"]["mode"]["n"]] == bk["expected"]["mode"]

    dr = srv.get_reference_cases("design_rules_contrast")
    good = srv.check_design_rules(dr["input"]["laminate"])
    judged = [r for r in good["data"]["rules"] if r["pass"] is not None]
    assert all(r["pass"] for r in judged)
    bad_lam = {"unit_system": "SI_mm",
               "laminae": [{"thickness": 0.125, "angle_deg": a, "material": dict(T300)}
                           for a in dr["input"]["contrast_bad_layup"]]}
    assert "symmetry" in srv.check_design_rules(bad_lam)["data"]["summary"]["hard_fails"]
