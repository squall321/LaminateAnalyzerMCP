# V3 기하 비선형 테스트 — Hyer 쌍안정·von Karman 대처짐·좌굴후 (계획서 §18)
from __future__ import annotations

import math

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import nonlinear as NL
from app.solver import thermal as TH

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
T300C = {**T300, "alpha1": 0.02e-6, "alpha2": 22.5e-6}


def lam(angles, mat=T300C, t=0.125):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": mat, "angle_deg": a, "thickness": t} for a in angles]}


def hyer_setup(angles, dT=-150.0, t=0.125e-3):
    """SI 단위 솔버 직접 호출용 (A,B,D,N_f,M_f)."""
    qb = [MAT.qbar_matrix(MAT.q_matrix(181e9, 10.3e9, 7.17e9, 0.28), a) for a in angles]
    al = [TH.alpha_vector(0.02e-6, 22.5e-6, a) for a in angles]
    z = ABD.z_coordinates([t] * len(angles))
    A, B, D = ABD.abd_matrices(qb, z)
    N_f, M_f = TH.thermal_loads(qb, al, z, dT)
    return A, B, D, N_f, M_f


# ── 1) Hyer 쌍안정: 모델 정합성 ─────────────────────────────────────────────

def test_hyer_strain_family_satisfies_compatibility():
    """가정한 면내 변형족이 von Karman 적합조건 ε_x,yy + ε_y,xx = −κxκy 를 만족한다.

    이 조건이 깨지면 원통(전개 가능면)에 없는 막 벌점이 붙어 분기가 사라진다 —
    실제로 첫 구현이 그렇게 틀렸다.
    """
    a, b, c1, c2, d1 = 3.0, -2.0, 1e-4, -2e-4, 5e-3
    h = 1e-5

    def ex(x, y):
        return c1 - d1 * y * y

    def ey(x, y):
        return c2 + (d1 - a * b / 2.0) * x * x

    ex_yy = (ex(0.0, h) - 2 * ex(0.0, 0.0) + ex(0.0, -h)) / h ** 2
    ey_xx = (ey(h, 0.0) - 2 * ey(0.0, 0.0) + ey(-h, 0.0)) / h ** 2
    assert ex_yy + ey_xx == pytest.approx(-a * b, rel=1e-6)


def test_hyer_gradient_matches_energy_derivative():
    """포락선 정리로 구한 해석적 gradient가 에너지의 수치 미분과 일치한다."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    en = NL.HyerEnergy(A, B, D, N_f, M_f, 0.15, 0.15)
    h = 1e-6
    for a, b in ((3.0, -2.0), (0.5, 0.5), (-7.0, 7.0)):
        fd = np.array([(en.energy(a + h, b) - en.energy(a - h, b)) / (2 * h),
                       (en.energy(a, b + h) - en.energy(a, b - h)) / (2 * h)])
        assert en.gradient(a, b) == pytest.approx(fd, rel=1e-6)


def test_hyer_small_panel_reduces_to_linear_clt():
    """작은 판 극한에서 비선형 해가 선형 CLT 곡률로 수렴한다 (극한 축약 검증)."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    kx, ky = NL.linear_curvatures(A, B, D, N_f, M_f)
    en = NL.HyerEnergy(A, B, D, N_f, M_f, 2e-3, 2e-3)
    sols = NL.find_equilibria(en, NL.search_span(en, kx, ky))
    assert len(sols) == 1 and sols[0]["stable"]
    assert sols[0]["a"] == pytest.approx(kx, rel=2e-3)
    assert sols[0]["b"] == pytest.approx(ky, rel=2e-3)


def test_hyer_large_panel_bifurcates_to_two_cylinders():
    """임계 크기를 넘으면 안정 원통 2개 + 불안정 안장 1개로 분기한다."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    kx, ky = NL.linear_curvatures(A, B, D, N_f, M_f)
    en = NL.HyerEnergy(A, B, D, N_f, M_f, 0.1, 0.1)
    sols = NL.find_equilibria(en, NL.search_span(en, kx, ky))
    stable = [s for s in sols if s["stable"]]
    unstable = [s for s in sols if not s["stable"]]
    assert len(stable) == 2 and len(unstable) == 1
    assert all(NL.classify_shape(s["a"], s["b"]) == "cylindrical" for s in stable)
    assert NL.classify_shape(unstable[0]["a"], unstable[0]["b"]) == "saddle"
    # 안정 해의 에너지가 더 낮다 (에너지 감소 판정)
    assert stable[0]["energy"] < unstable[0]["energy"]


def test_hyer_two_stable_shapes_are_mirror_pair():
    """[0/90]의 두 원통해는 (a,b)→(−b,−a) 대칭이고 에너지가 같다.

    x↔y 교환은 적층 순서를 뒤집는 것(=z 반전, 곡률 부호 반전)과 짝을 이룬다.
    """
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    kx, ky = NL.linear_curvatures(A, B, D, N_f, M_f)
    en = NL.HyerEnergy(A, B, D, N_f, M_f, 0.1, 0.1)
    s = [x for x in NL.find_equilibria(en, NL.search_span(en, kx, ky)) if x["stable"]]
    assert s[0]["a"] == pytest.approx(-s[1]["b"], rel=1e-9)
    assert s[0]["b"] == pytest.approx(-s[1]["a"], rel=1e-9)
    assert s[0]["energy"] == pytest.approx(s[1]["energy"], rel=1e-12)


def test_hyer_symmetric_laminate_has_unique_flat_solution():
    """대칭 적층은 B=0·M_th=0 이라 평평한 해 하나뿐 — 분기 자체가 없다."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0, 90.0, 0.0))
    en = NL.HyerEnergy(A, B, D, N_f, M_f, 0.1, 0.1)
    sols = NL.find_equilibria(en, NL.search_span(en, 0.0, 0.0))
    assert len(sols) == 1
    assert sols[0]["a"] == pytest.approx(0.0, abs=1e-9)
    assert sols[0]["b"] == pytest.approx(0.0, abs=1e-9)
    assert NL.critical_scale(A, B, D, N_f, M_f, 0.1, 0.1)["scale"] is None


def test_hyer_critical_size_scales_with_thickness():
    """임계 판 크기는 두께에 정비례한다 (L/h 무차원 불변)."""
    ratios = []
    for t in (0.125e-3, 0.25e-3, 0.5e-3):
        A, B, D, N_f, M_f = hyer_setup((0.0, 90.0), t=t)
        c = NL.critical_scale(A, B, D, N_f, M_f, 0.1, 0.1)
        ratios.append(c["lx"] / (2 * t))
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-6)
    assert ratios[1] == pytest.approx(ratios[2], rel=1e-6)


def test_hyer_critical_size_brackets_equilibrium_count():
    """임계 크기 아래는 해 1개, 위는 3개 — 독립 경로(전수 스캔)와 교차 확인."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    kx, ky = NL.linear_curvatures(A, B, D, N_f, M_f)
    lc = NL.critical_scale(A, B, D, N_f, M_f, 0.1, 0.1)["lx"]
    for factor, expected in ((0.9, 1), (1.1, 3)):
        en = NL.HyerEnergy(A, B, D, N_f, M_f, lc * factor, lc * factor)
        assert len(NL.find_equilibria(en, NL.search_span(en, kx, ky))) == expected


# ── 1) Hyer 쌍안정: Tool 계층 ───────────────────────────────────────────────

def test_bistable_tool_reports_bifurcation_and_critical_panel():
    env = srv.compute_bistable_shapes(lam((0.0, 90.0)), panel={"Lx": 100.0, "Ly": 100.0},
                                      delta_T=-150.0)
    assert env["errors"] == []
    d = env["data"]
    assert d["bistable"] is True and d["stable_count"] == 2
    assert d["linear_reference"]["shape"] == "saddle"
    assert d["critical_panel"]["Lx"] < 100.0
    assert d["energy_barrier"]["min_barrier"] > 0
    assert len(d["energy_barrier"]["per_stable_shape"]) == 2
    assert any(w["code"] == "W130" for w in env["warnings"])


def test_bistable_tool_small_panel_is_not_bistable():
    env = srv.compute_bistable_shapes(lam((0.0, 90.0)), panel={"Lx": 10.0, "Ly": 10.0},
                                      delta_T=-150.0)
    d = env["data"]
    assert d["bistable"] is False and d["stable_count"] == 1
    assert d["equilibria"][0]["shape"] == "saddle"
    assert "energy_barrier" not in d


def test_bistable_tool_unit_systems_agree():
    """SI와 SI_mm이 같은 물리를 준다 (곡률 1/m ↔ 1/mm, 길이 m ↔ mm)."""
    si_mat = {"type": "orthotropic_2d", "E1": 181e9, "E2": 10.3e9, "G12": 7.17e9,
              "nu12": 0.28, "alpha1": 0.02e-6, "alpha2": 22.5e-6}
    si = {"unit_system": "SI",
          "laminae": [{"material": si_mat, "angle_deg": a, "thickness": 0.125e-3}
                      for a in (0.0, 90.0)]}
    a = srv.compute_bistable_shapes(si, panel={"Lx": 0.1, "Ly": 0.1}, delta_T=-150.0)["data"]
    b = srv.compute_bistable_shapes(lam((0.0, 90.0)), panel={"Lx": 100.0, "Ly": 100.0},
                                    delta_T=-150.0)["data"]
    assert a["equilibria"][0]["kappa_x"] == pytest.approx(
        b["equilibria"][0]["kappa_x"] * 1e3, rel=1e-9)
    assert a["critical_panel"]["Lx"] == pytest.approx(b["critical_panel"]["Lx"] * 1e-3, rel=1e-9)


def test_bistable_tool_warns_on_twist_limitation():
    """[+45/−45] 반대칭의 실제 경화 형상은 비틀림 — 모델이 표현 못 함을 알려야 한다."""
    env = srv.compute_bistable_shapes(lam((45.0, -45.0)), panel={"Lx": 100.0, "Ly": 100.0},
                                      delta_T=-150.0)
    assert any(w["code"] == "W130" and "비틀림" in w["message"] for w in env["warnings"])


def test_bistable_tool_errors():
    assert srv.compute_bistable_shapes(lam((0.0, 90.0)), panel=None,
                                       delta_T=-150.0)["errors"][0]["code"] == "E100"
    assert srv.compute_bistable_shapes(lam((0.0, 90.0)), panel={"Lx": 100.0, "Ly": 100.0},
                                       delta_T=None)["errors"][0]["code"] == "E100"
    assert srv.compute_bistable_shapes(lam((0.0, 90.0), mat=T300),
                                       panel={"Lx": 100.0, "Ly": 100.0},
                                       delta_T=-150.0)["errors"][0]["code"] == "E203"


def test_bistable_tool_is_deterministic():
    kw = dict(panel={"Lx": 100.0, "Ly": 100.0}, delta_T=-150.0)
    a = srv.compute_bistable_shapes(lam((0.0, 90.0)), **kw)
    b = srv.compute_bistable_shapes(lam((0.0, 90.0)), **kw)
    assert a["data"] == b["data"]
    assert a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]


# ── §18.5 선형 유효범위 게이트 ──────────────────────────────────────────────

def test_thermal_tool_gates_on_large_deflection():
    """선형 열해석이 w/h ≫ 1 을 낼 때 비선형 도구로 안내해야 한다."""
    env = srv.compute_thermal_response(lam((0.0, 90.0)), delta_T=-150.0,
                                       panel={"Lx": 100.0, "Ly": 100.0})
    assert env["data"]["warpage"]["w_over_thickness"] > 1.0
    assert any(w["code"] == "W130" and "compute_bistable_shapes" in w["message"]
               for w in env["warnings"])


def test_thermal_tool_no_gate_when_small():
    env = srv.compute_thermal_response(lam((0.0, 90.0)), delta_T=-1.0,
                                       panel={"Lx": 5.0, "Ly": 5.0})
    assert env["data"]["warpage"]["w_over_thickness"] < 0.3
    assert not any("compute_bistable_shapes" in w["message"] for w in env["warnings"])


# ── 2) von Karman 대처짐 ────────────────────────────────────────────────────

def iso_plate(E=70e9, nu=0.3, h=2e-3):
    G = E / (2 * (1 + nu))
    Q = E / (1 - nu ** 2)
    A = np.array([[Q * h, nu * Q * h, 0.0], [nu * Q * h, Q * h, 0.0], [0.0, 0.0, G * h]])
    Dv = E * h ** 3 / (12 * (1 - nu ** 2))
    D = np.array([[Dv, nu * Dv, 0.0], [nu * Dv, Dv, 0.0], [0.0, 0.0, (1 - nu) * Dv / 2]])
    return A, D, Dv, h


def test_large_deflection_linear_limit_matches_navier():
    """q→0 극한이 1항 Navier 해 w = (4/π⁶)·qL⁴/D 와 정확히 일치한다.

    Levy 정해는 0.004062 — 1항 근사의 +2.4% 오차는 알려진 값이고 응답 assumptions에 적는다.
    """
    A, D, Dv, h = iso_plate()
    L = 0.5
    r = NL.large_deflection(D, A, L, L, 0.0)
    coeff = (16.0 / math.pi ** 2) / r["alpha"] * Dv / L ** 4
    assert coeff == pytest.approx(4.0 / math.pi ** 6, rel=1e-12)
    assert coeff / 0.0040624 == pytest.approx(1.024, rel=1e-2)


def test_large_deflection_cardano_root_satisfies_equation():
    """Cardano 폐형해가 실제로 αW + βW³ = 16q/π² 를 만족한다."""
    A, D, _, _ = iso_plate()
    L, q = 0.5, 5000.0
    for immovable in (False, True):
        r = NL.large_deflection(D, A, L, L, q, immovable)
        w = r["w_center"]
        res = r["alpha"] * w + r["beta"] * w ** 3 - 16.0 * q / math.pi ** 2
        assert abs(res) < 1e-6 * abs(16.0 * q / math.pi ** 2)


def test_large_deflection_stiffens_and_converges_to_linear():
    """막 효과로 비선형 처짐 < 선형 처짐이고, 하중이 작아지면 비가 1로 간다."""
    A, D, _, _ = iso_plate()
    L = 0.5
    prev = None
    for q in (5000.0, 500.0, 50.0, 5.0, 0.5):
        r = NL.large_deflection(D, A, L, L, q)
        assert r["w_center"] < r["w_linear"]
        assert r["stiffening_ratio"] > 1.0
        if prev is not None:
            assert r["stiffening_ratio"] < prev
        prev = r["stiffening_ratio"]
    assert prev == pytest.approx(1.0, abs=1e-3)


def test_large_deflection_immovable_edges_are_stiffer():
    """면내 구속이면 β가 (3−ν)/(1−ν) 배 커진다 (등방 정사각 폐형해)."""
    nu = 0.3
    A, D, _, _ = iso_plate(nu=nu)
    L = 0.5
    mv = NL.large_deflection(D, A, L, L, 5000.0, False)
    im = NL.large_deflection(D, A, L, L, 5000.0, True)
    assert im["beta"] / mv["beta"] == pytest.approx((3 - nu) / (1 - nu), rel=1e-12)
    assert im["alpha"] == pytest.approx(mv["alpha"], rel=1e-12)
    assert im["w_center"] < mv["w_center"]


def test_large_deflection_sign_antisymmetry():
    """압력 부호를 뒤집으면 처짐도 정확히 뒤집힌다 (홀함수)."""
    A, D, _, _ = iso_plate()
    L = 0.5
    p = NL.large_deflection(D, A, L, L, 5000.0)["w_center"]
    m = NL.large_deflection(D, A, L, L, -5000.0)["w_center"]
    assert m == pytest.approx(-p, rel=1e-12)


def test_large_deflection_tool_warns_and_errors():
    sym = lam((0.0, 90.0, 90.0, 0.0), mat=T300)
    panel = {"Lx": 300.0, "Ly": 300.0}
    small = srv.compute_large_deflection(sym, panel=panel, pressure=1e-7)
    assert any("< 0.3" in w["message"] for w in small["warnings"])
    big = srv.compute_large_deflection(sym, panel=panel, pressure=5e-3)
    assert big["data"]["membrane_dominant"] is True
    assert any("유효 범위를 크게 벗어났다" in w["message"] for w in big["warnings"])
    assert srv.compute_large_deflection(sym, panel=panel, pressure=1e-3,
                                        edge_condition="x")["errors"][0]["code"] == "E100"
    assert srv.compute_large_deflection(sym, panel={"Lx": -1.0, "Ly": 1.0},
                                        pressure=1e-3)["errors"][0]["code"] == "E100"


# ── 3) 좌굴 후 ──────────────────────────────────────────────────────────────

def test_postbuckling_isotropic_square_stiffness_ratio_is_half():
    """등방 정사각 SS 판의 좌굴 후 면내 접선강성비 = 0.5 (고전값, 유도로 재현)."""
    A, D, Dv, h = iso_plate()
    L = 0.5
    n_cr = math.pi ** 2 * Dv * 4 / L ** 2
    pb = NL.postbuckling(D, A, L, L, 1, 1, n_cr, 2.0 * n_cr, h)
    assert pb["stiffness_ratio"] == pytest.approx(0.5, rel=1e-12)


def test_postbuckling_amplitude_and_effective_width():
    """W ∝ √(N/N_cr − 1), b_eff/b = √(N_cr/N), 좌굴 전에는 buckled=False."""
    A, D, Dv, h = iso_plate()
    L = 0.5
    n_cr = math.pi ** 2 * Dv * 4 / L ** 2
    assert NL.postbuckling(D, A, L, L, 1, 1, n_cr, 0.5 * n_cr, h)["buckled"] is False
    w2 = NL.postbuckling(D, A, L, L, 1, 1, n_cr, 2.0 * n_cr, h)
    w5 = NL.postbuckling(D, A, L, L, 1, 1, n_cr, 5.0 * n_cr, h)
    assert w5["amplitude"] / w2["amplitude"] == pytest.approx(math.sqrt(4.0 / 1.0), rel=1e-12)
    assert w5["effective_width_ratio"] == pytest.approx(math.sqrt(1 / 5.0), rel=1e-12)


def test_postbuckling_tool_ncr_matches_buckling_tool():
    """N_cr은 compute_buckling과 동일한 값·모드여야 한다 (같은 스캔 사용)."""
    sym = lam((0.0, 90.0, 90.0, 0.0), mat=T300)
    panel = {"Lx": 300.0, "Ly": 300.0}
    b = srv.compute_buckling(sym, panel=panel)["data"]
    p = srv.compute_postbuckling(sym, panel=panel, applied_Nx=3.0)["data"]
    assert p["N_cr"] == pytest.approx(b["N_cr"], rel=1e-12)
    assert p["mode"] == {"m": b["mode"]["m"], "n": b["mode"]["n"]}
    assert p["load_over_critical"] == pytest.approx(3.0 / b["N_cr"], rel=1e-12)


def test_postbuckling_tool_unbalanced_ratio_differs_from_half():
    """면내 비대칭(a11≠a22) 적층은 0.5가 아니다 — 상수 하드코딩이 아님을 확인."""
    unb = lam((0.0, 0.0, 0.0, 90.0), mat=T300)
    d = srv.compute_postbuckling(unb, panel={"Lx": 300.0, "Ly": 300.0},
                                 applied_Nx=3.0)["data"]
    assert d["stiffness_ratio"] < 0.49


def test_postbuckling_tool_errors():
    sym = lam((0.0, 90.0, 90.0, 0.0), mat=T300)
    panel = {"Lx": 300.0, "Ly": 300.0}
    assert srv.compute_postbuckling(sym, panel=panel,
                                    applied_Nx=-1.0)["errors"][0]["code"] == "E100"
    assert srv.compute_postbuckling(sym, panel=panel, applied_Nx=1.0,
                                    load_ratio=float("nan"))["errors"][0]["code"] == "E100"
    assert srv.compute_postbuckling(sym, panel={"Lx": 0.0, "Ly": 1.0},
                                    applied_Nx=1.0)["errors"][0]["code"] == "E100"


# ── 독립 오라클: 정사각 판 폐형해 (계획서 §8 다중경로 검증) ─────────────────
#
# 격자 스캔 + 뉴턴과 **완전히 다른 경로**로 유도된 폐형해다.
#
# ⚠ 이 폐형해는 γxy⁰ = 0 **제약 모델**의 해다. 출하 엔진은 γxy⁰ = k·x·y 자유도를 하나 더
# 갖고 있어(적대 검증에서 그게 없으면 없는 쌍안정을 만든다는 게 확인됐다) 임계 크기가 더
# 크게 나온다. 그래서 오라클 대조는 A66 → ∞ 로 k = 0 을 강제해 **제약 모델을 재현**한 뒤
# 한다. κ_lin(L→0)과 κ_∞(L→∞)는 두 모델에서 동일하다 — 원통·평면 극한에는 면내 전단이
# 개입하지 않기 때문이다. 면내 자유도를 c₁,c₂만 남기고 d₁ = κxκy/4 로 고정하면 에너지가
#     Π/S = ½κᵀD*κ − M*·κ + Q·κx²κy²,  D* = D − BA⁻¹B, M* = M_f − BA⁻¹N_f,
#     Q = (A11·Ly⁴ + A22·Lx⁴)/5760        (A12는 소거 후 사라진다)
# 로 축소되고, 대칭 크로스플라이 정사각에서는 분기가 pitchfork로 인수분해된다.
#
# **정사각 전용이다.** d₁ = κxκy/4 는 정사각에서만 3자유도 최소화의 최적값과 일치한다
# (아래 test_closed_form_oracle_is_square_only 가 그 경계를 못박는다). 직사각에
# 잘못 적용하면 임계 크기가 50~65% 틀린다.


def _restricted(A):
    """γxy⁰ = 0 제약 모델 재현 — A66 → ∞ 이면 k 자유도의 벌점이 무한대라 k = 0 이 된다."""
    Ar = np.array(A, dtype=float, copy=True)
    Ar[2, 2] *= 1e12
    return Ar


def _dstar_mstar(A, B, D, N_f, M_f):
    A_inv = np.linalg.inv(A)
    return D - B @ A_inv @ B, M_f - B @ A_inv @ N_f


def _closed_form_landmarks(A, B, D, N_f, M_f):
    """정사각 대칭 크로스플라이의 폐형해 랜드마크."""
    Ds, Ms = _dstar_mstar(A, B, D, N_f, M_f)
    return {
        "kappa_linear": Ms[0] / (Ds[0, 0] - Ds[0, 1]),          # L→0 안장
        "kappa_crit": Ms[0] / (2 * Ds[0, 0]),                   # 분기점
        "kappa_inf": Ms[0] / Ds[0, 0],                          # L→∞ 원통
        "L_crit": (5760 * Ds[0, 0] ** 2 * (Ds[0, 0] + Ds[0, 1])
                   / (A[0, 0] * Ms[0] ** 2)) ** 0.25,
    }


@pytest.mark.parametrize("dT,t", [(-150.0, 0.125e-3), (-150.0, 0.25e-3),
                                  (-100.0, 0.125e-3), (-250.0, 0.25e-3)])
def test_closed_form_oracle_critical_size(dT, t):
    """수치 연속추적으로 구한 임계 판 크기가 폐형해와 일치한다 (독립 경로)."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0), dT=dT, t=t)
    closed = _closed_form_landmarks(A, B, D, N_f, M_f)["L_crit"]
    restricted = NL.critical_scale(_restricted(A), B, D, N_f, M_f, 0.1, 0.1)["lx"]
    assert restricted == pytest.approx(closed, rel=1e-8)
    # 출하 모델은 제약을 하나 풀었으므로 안장이 더 오래 버틴다 → 임계 크기가 더 크다
    shipped = NL.critical_scale(A, B, D, N_f, M_f, 0.1, 0.1)["lx"]
    assert shipped > restricted


def test_closed_form_oracle_curvature_landmarks():
    """κ_lin(L→0)·κ_c(분기)·κ_∞(L→∞) 세 랜드마크가 수치해와 일치한다.

    κ_c = κ_∞/2 는 정확한 항등식이다 — 설계 판단에 쓸 수 있는 기억할 만한 사실.
    """
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    lm = _closed_form_landmarks(A, B, D, N_f, M_f)
    assert lm["kappa_crit"] == pytest.approx(lm["kappa_inf"] / 2, rel=1e-12)
    # L→0: 선형 CLT 6×6 해와 일치
    assert NL.linear_curvatures(A, B, D, N_f, M_f)[0] == pytest.approx(
        lm["kappa_linear"], rel=1e-12)
    # L→∞: 원통은 막 벌점이 없어 γxy 자유도와 무관 — 출하 모델에서도 그대로 성립한다
    en = NL.HyerEnergy(A, B, D, N_f, M_f, 1.0, 1.0)
    kx, ky = NL.linear_curvatures(A, B, D, N_f, M_f)
    stable = [s for s in NL.find_equilibria(en, NL.search_span(en, kx, ky)) if s["stable"]]
    assert stable[0]["a"] == pytest.approx(lm["kappa_inf"], rel=1e-6)
    # 임계 크기에서는 세 해가 한 점으로 모인다 (pitchfork) — 제약 모델 기준
    Ar = _restricted(A)
    en_c = NL.HyerEnergy(Ar, B, D, N_f, M_f, lm["L_crit"], lm["L_crit"])
    for s in NL.find_equilibria(en_c, NL.search_span(en_c, kx, ky)):
        assert s["a"] == pytest.approx(lm["kappa_crit"], rel=1e-3)


def test_closed_form_oracle_is_square_only():
    """폐형해는 정사각 전용 — 직사각에 쓰면 크게 틀린다는 경계를 못박는다.

    d₁ = κxκy/4 고정은 정사각에서만 3자유도 최소화의 최적값과 같다. 직사각에서는
    최적 d₁이 그 2배 가까이 되고(면내 에너지를 더 낮춘다), 폐형해 임계 크기는 50%↑
    빗나간다. 미래 세션이 이 폐형해를 일반식으로 오해해 채택하는 것을 막는 테스트다.
    """
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    Ar = _restricted(A)
    # 제약 모델 · 정사각: 최적 d₁ == ab/4
    sq = NL.HyerEnergy(Ar, B, D, N_f, M_f, 0.1, 0.1)
    for a, b in ((3.0, -3.0), (5.0, -2.0)):
        assert sq.membrane_dofs(a, b)[2] == pytest.approx(a * b / 4.0, rel=1e-9)
    # 제약 모델 · 직사각: 최적 d₁ != ab/4
    rect = NL.HyerEnergy(Ar, B, D, N_f, M_f, 0.2, 0.05)
    assert rect.membrane_dofs(3.0, -3.0)[2] != pytest.approx(3.0 * -3.0 / 4.0, rel=1e-2)
    # 그리고 임계 크기 폐형해도 직사각에서는 크게 빗나간다
    Ds, Ms = _dstar_mstar(A, B, D, N_f, M_f)
    q = (A[0, 0] * 0.05 ** 4 + A[1, 1] * 0.2 ** 4) / 5760.0
    q_crit = 2 * Ds[0, 0] ** 2 * (Ds[0, 0] + Ds[0, 1]) / Ms[0] ** 2
    numeric = NL.critical_scale(A, B, D, N_f, M_f, 0.2, 0.05)["scale"]
    assert abs((q_crit / q) ** 0.25 - numeric) / numeric > 0.3


def test_critical_size_governed_by_short_side():
    """직사각 판의 분기는 짧은 변이 지배한다 (폐형해가 못 잡는 실제 거동)."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    short_sides = []
    for lx in (0.2, 0.15):
        c = NL.critical_scale(A, B, D, N_f, M_f, lx, 0.05)
        short_sides.append(c["ly"])
    assert short_sides[0] == pytest.approx(short_sides[1], rel=5e-2)


# ── 기준 케이스(few-shot)가 실제 엔진과 일치하는가 ──────────────────────────

def test_reference_case_bistable_cross_ply():
    """R11: 폐형해 랜드마크가 compute_bistable_shapes 결과와 맞는다."""
    c = srv.get_reference_cases("bistable_cross_ply")
    env = srv.compute_bistable_shapes(c["input"]["laminate"], panel=c["input"]["panel"],
                                      delta_T=c["input"]["delta_T"])
    d, e, tol = env["data"], c["expected"], c["tolerance"]
    assert env["errors"] == []
    assert d["bistable"] is e["bistable"] and d["stable_count"] == e["stable_count"]
    # 엔진은 γxy⁰ 자유도를 하나 더 가지므로 폐형해 하한보다 큰 임계 크기를 준다
    assert d["critical_panel"]["Lx"] > e["critical_panel_Lx_mm_lower_bound"]
    assert d["critical_panel"]["Lx"] < 100.0
    stable = [x for x in d["equilibria"] if x["stable"]]
    assert len(stable) == 2
    assert all(x["shape"] == "cylindrical" for x in stable)
    # 거울쌍 (κx,κy) → (−κy,−κx)
    assert stable[0]["kappa_x"] == pytest.approx(-stable[1]["kappa_y"], rel=1e-9)
    # L→∞ 극한값과 유한 판 값의 근접
    assert stable[0]["kappa_x"] == pytest.approx(e["kappa_inf_per_mm"],
                                                rel=tol["kappa_inf_per_mm"])
    assert e["kappa_crit_per_mm"] == pytest.approx(e["kappa_inf_per_mm"] / 2, rel=1e-12)


def test_reference_case_large_deflection():
    """R13: 1항 Navier 폐형해와 immovable/movable β비가 맞는다."""
    c = srv.get_reference_cases("large_deflection_isotropic")
    i, e, tol = c["input"], c["expected"], c["tolerance"]
    env = srv.compute_large_deflection(i["laminate"], panel=i["panel"], pressure=i["pressure"],
                                       edge_condition=i["edge_condition"])
    d = env["data"]
    assert env["errors"] == []
    assert d["w_center_linear"] == pytest.approx(e["w_center_linear_mm"],
                                                rel=tol["w_center_linear_mm"])
    assert d["w_center"] < d["w_center_linear"]
    imm = srv.compute_large_deflection(i["laminate"], panel=i["panel"], pressure=i["pressure"],
                                       edge_condition="immovable")["data"]
    assert imm["w_center"] < d["w_center"]


# ── 적대 검증에서 확정된 결함의 회귀 방지 ───────────────────────────────────

def test_regression_twist_gate_fires_on_kappa_xy():
    """HYER-1: κxy가 지배적인 적층이 아무 경고 없이 통과하던 문제.

    A/B/D의 16·26 성분비는 대리지표라 [0/90/45/90]을 놓쳤다. 실제 κxy로 판정해야 한다.
    """
    env = srv.compute_bistable_shapes(lam((0.0, 90.0, 45.0, 90.0)),
                                      panel={"Lx": 100.0, "Ly": 100.0}, delta_T=-150.0)
    d = env["data"]
    kx, ky, kxy = (d["linear_reference"]["kappa_x"], d["linear_reference"]["kappa_y"],
                   d["linear_reference"]["kappa_xy"])
    assert abs(kxy) / max(abs(kx), abs(ky)) > 0.3
    assert any(w["code"] == "W130" and "형상 판정을 신뢰하지 말 것" in w["message"]
               for w in env["warnings"])
    # 순수 크로스플라이는 κxy=0 이라 이 경고가 뜨지 않는다
    clean = srv.compute_bistable_shapes(lam((0.0, 90.0)), panel={"Lx": 100.0, "Ly": 100.0},
                                        delta_T=-150.0)
    assert clean["data"]["linear_reference"]["kappa_xy"] == pytest.approx(0.0, abs=1e-12)
    assert not any("비틀림이 지배적" in w["message"] for w in clean["warnings"])


def test_regression_energy_barrier_is_per_stable_shape():
    """HYER-2: 깊은 우물 기준 단일 장벽은 얕은 우물의 실제 장벽을 과대평가했다."""
    env = srv.compute_bistable_shapes(lam((0.0, 0.0, 90.0), t=0.2),
                                      panel={"Lx": 200.0, "Ly": 200.0}, delta_T=-150.0)
    eb = env["data"].get("energy_barrier")
    if eb is None:
        pytest.skip("이 조합은 쌍안정이 아니다")
    bars = [b["barrier"] for b in eb["per_stable_shape"]]
    assert eb["min_barrier"] == pytest.approx(min(bars), rel=1e-12)
    assert all(b > 0 for b in bars)


def test_regression_gamma_xy_dof_removes_spurious_bistability():
    """γxy⁰=0 으로 묶으면 [0/0/90]에 없는 쌍안정이 생긴다 — 4자유도가 이를 없앤다."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 0.0, 90.0))
    kx, ky = NL.linear_curvatures(A, B, D, N_f, M_f)

    def n_stable(a_mat):
        en = NL.HyerEnergy(a_mat, B, D, N_f, M_f, 0.1, 0.1)
        return sum(1 for s in NL.find_equilibria(en, NL.search_span(en, kx, ky)) if s["stable"])

    restricted = np.array(A, dtype=float, copy=True)
    restricted[2, 2] *= 1e12                      # A66 → ∞ ⇒ γxy⁰ = 0
    assert n_stable(restricted) == 2              # 제약 모델의 허위 쌍안정
    assert n_stable(A) == 1                       # 출하 모델은 단안정


def test_regression_postbuckling_uses_load_ratio():
    """PB-1: 2축 압축에서 강성비가 R을 무시해 남은 강성을 과대보고했다."""
    A, D, Dv, h = iso_plate()
    L = 0.5
    n_cr = math.pi ** 2 * Dv * 4 / L ** 2
    ratios = [NL.postbuckling(D, A, L, L, 1, 1, n_cr, 2.0 * n_cr, h, R)["stiffness_ratio"]
              for R in (0.0, 0.5, 1.0, 2.0)]
    assert ratios[0] == pytest.approx(0.5, rel=1e-12)     # R=0 고전값은 유지
    assert ratios[0] > ratios[1] > ratios[2] > ratios[3]  # 2축 압축일수록 강성이 더 준다
    assert ratios[2] < 0.3                                # R=1 에서 0.5가 아니어야 한다


def test_regression_vk_compliance_uses_full_inverse():
    """VK-1: 컴플라이언스를 2×2 블록으로 뒤집으면 불균형 적층에서 β를 과대평가했다."""
    ang = (30.0, 30.0, 30.0, 30.0)
    qb = [MAT.qbar_matrix(MAT.q_matrix(181e9, 10.3e9, 7.17e9, 0.28), a) for a in ang]
    z = ABD.z_coordinates([0.125e-3] * 4)
    A, _B, D = ABD.abd_matrices(qb, z)
    p = q = math.pi / 0.3
    _, beta = NL.vk_coefficients(D, A, p, q, immovable=False)
    a_full = np.linalg.inv(A)[:2, :2]
    a_blk = np.linalg.inv(A[:2, :2])
    beta_full = p ** 4 / (16 * a_full[0, 0]) + q ** 4 / (16 * a_full[1, 1])
    beta_blk = p ** 4 / (16 * a_blk[0, 0]) + q ** 4 / (16 * a_blk[1, 1])
    assert beta == pytest.approx(beta_full, rel=1e-12)
    assert beta_blk / beta_full > 1.5          # 옛 방식은 크게 어긋난다 (여기서 1.9배)


def test_regression_classify_shape_underflow():
    """SHAPE-1: a*b 곱이 언더플로해 안장이 spherical 로 뒤집혔다."""
    assert (-1e-200) * (1e-200) == 0.0          # 곱은 실제로 0으로 언더플로한다
    assert NL.classify_shape(-1e-200, 1e-200) == "saddle"
    assert NL.classify_shape(-3.0, 2.0) == "saddle"
    assert NL.classify_shape(3.0, 2.0) == "spherical"
    assert NL.classify_shape(1e-20, -1e-20, scale=10.0) == "flat"


def test_regression_large_deflection_aspect_ratio_warning():
    """LD-1: 1항 Galerkin은 세장 판에서 처짐을 과소평가하는데 경고가 없었다."""
    sym = lam((0.0, 90.0, 90.0, 0.0), mat=T300)
    slim = srv.compute_large_deflection(sym, panel={"Lx": 1000.0, "Ly": 100.0}, pressure=1e-3)
    assert any(w["code"] == "W130" and "종횡비" in w["message"] for w in slim["warnings"])
    square = srv.compute_large_deflection(sym, panel={"Lx": 300.0, "Ly": 300.0}, pressure=1e-3)
    assert not any("종횡비" in w["message"] for w in square["warnings"])


def test_regression_extreme_input_maps_to_e100_not_e501():
    """HYER-4/HYER-3: 유한·양수 입력의 수치 초과가 E501(내부 오류)로 샜다."""
    sym = lam((0.0, 90.0, 90.0, 0.0), mat=T300)
    for panel, pressure in (({"Lx": 1e30, "Ly": 1e30}, 1e-3), ({"Lx": 300.0, "Ly": 300.0}, 1e30)):
        env = srv.compute_large_deflection(sym, panel=panel, pressure=pressure)
        if env["errors"]:
            assert env["errors"][0]["code"] != "E501", (panel, pressure, env["errors"])


def test_regression_marginal_stability_is_flagged():
    """분기점 바로 근처에서 stable/unstable 단정 대신 marginal 을 표시한다."""
    A, B, D, N_f, M_f = hyer_setup((0.0, 90.0))
    lc = NL.critical_scale(A, B, D, N_f, M_f, 0.1, 0.1)["lx"]
    kx, ky = NL.linear_curvatures(A, B, D, N_f, M_f)
    en = NL.HyerEnergy(A, B, D, N_f, M_f, lc, lc)
    states = {s["stability"] for s in NL.find_equilibria(en, NL.search_span(en, kx, ky))}
    assert states <= {"stable", "unstable", "marginal"}
    # 임계 크기 정확히 위에서는 한계 판정이 나와야 한다
    assert "marginal" in states
