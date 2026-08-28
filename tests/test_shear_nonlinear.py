# V3 항목 4 — 재료 면내 전단 비선형 (Hahn–Tsai) 테스트 (계획서 §18.7)
from __future__ import annotations

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import abd as ABD
from app.solver import shear_nonlinear as SNL

BASE = {"E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
S_MM = 4.0e-8          # 1/MPa^3 (CFRP 자릿수)


def mat(s6666=S_MM):
    m = {"type": "orthotropic_2d", **BASE}
    if s6666 is not None:
        m["shear_nonlinear"] = {"S6666": s6666}
    return m


def lam(angles=(45.0, -45.0, -45.0, 45.0), s6666=S_MM, t=0.125):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": mat(s6666), "angle_deg": a, "thickness": t} for a in angles]}


def si_plies(angles, s6666_si, t=0.125e-3):
    return ([{**{k: v * (1e6 if k in ("E1", "E2", "G12") else 1.0) for k, v in BASE.items()},
              "angle_deg": a, "s6666": s6666_si} for a in angles],
            ABD.z_coordinates([t] * len(angles)))


# ── 구성식 자체 ─────────────────────────────────────────────────────────────

def test_secant_modulus_definition():
    """G_sec = τ/γ 가 γ = τ/G + S·τ³ 와 정확히 정합한다."""
    g12, s, tau = 7.17e9, 4.0e-26, 50.0e6
    g_sec = SNL.secant_g12(g12, s, tau)
    assert tau / g_sec == pytest.approx(tau / g12 + s * tau ** 3, rel=1e-14)


def test_zero_s6666_reduces_to_linear():
    """S6666 = 0 이면 선형 CLT와 완전히 동일하다 (극한 환원)."""
    plies, z = si_plies((45.0, -45.0, -45.0, 45.0), 0.0)
    N, M = np.array([1.0e5, 0.0, 0.0]), np.zeros(3)
    nl = SNL.solve_secant(plies, z, N, M)
    lin = SNL.linear_reference(plies, z, N, M)
    assert nl["eps0"] == pytest.approx(lin["eps0"], rel=1e-12)
    assert nl["kappa"] == pytest.approx(lin["kappa"], abs=1e-12)


@pytest.mark.parametrize("nx", [1.0e4, 5.0e4, 1.0e5, 2.0e5, 4.0e5])
def test_constitutive_law_satisfied_at_convergence(nx):
    """수렴한 해가 γ12 = τ12/G12 + S·τ12³ 을 실제로 만족한다 (수렴의 진짜 척도)."""
    plies, z = si_plies((45.0, -45.0, -45.0, 45.0), 4.0e-26)
    res = SNL.solve_secant(plies, z, np.array([nx, 0.0, 0.0]), np.zeros(3))
    assert res["converged"], res["residual"]
    assert res["residual"] < 1e-9


def test_bending_dominated_softens_outer_plies_more():
    """굽힘 지배 하중에서는 τ12가 큰 외층이 더 많이 연화한다."""
    plies, z = si_plies((45.0, -45.0, -45.0, 45.0), 4.0e-26)
    res = SNL.solve_secant(plies, z, np.zeros(3), np.array([2.0, 0.0, 0.0]))
    r = [p["secant_ratio"] for p in res["per_ply"]]
    assert r[0] < r[1] and r[3] < r[2]
    assert res["residual"] < 1e-9


# ── 폐형해 대조 ─────────────────────────────────────────────────────────────

def test_pm45_tension_shear_stress_is_closed_form():
    """[±45]s 단축인장에서 ply 전단은 τ12 = σx/2 = Nx/(2h) — 정확한 폐형해."""
    nx, t, n = 50.0, 0.125, 4
    env = srv.solve_nonlinear_shear_response(lam(t=t), loads={"N": [nx, 0.0, 0.0]})
    assert env["errors"] == []
    for p in env["data"]["per_ply"]:
        assert p["tau12"] == pytest.approx(nx / (2.0 * n * t), rel=1e-12)


def test_pm45_laminate_shear_stiffness_is_invariant():
    """[±45]에서 Q̄66은 Q66와 무관하다(45°에서 정확히 상쇄) → 적층 Gxy가 안 변한다.

    반대로 Q̄11은 Q66에 강하게 의존하므로 Ex는 크게 떨어진다. 이 비대칭이
    할선 갱신이 올바른 성분에 들어갔는지 보는 강한 불변식이다.
    """
    d = srv.solve_nonlinear_shear_response(lam(), loads={"N": [50.0, 0.0, 0.0]})["data"]
    assert d["softening"]["Gxy_secant_over_linear"] == pytest.approx(1.0, rel=1e-12)
    assert d["softening"]["Ex_secant_over_linear"] < 0.7


def test_softening_is_monotone_in_load():
    """하중이 커질수록 할선비가 단조 감소한다."""
    prev = 1.0
    for nx in (10.0, 25.0, 50.0, 100.0):
        d = srv.solve_nonlinear_shear_response(lam(), loads={"N": [nx, 0.0, 0.0]})["data"]
        r = d["softening"]["worst_ply"]["secant_ratio"]
        assert r < prev
        prev = r
    assert prev < 0.3


def test_small_load_converges_to_linear():
    """하중을 낮추면 비선형 해가 선형 해로 수렴한다."""
    d = srv.solve_nonlinear_shear_response(lam(), loads={"N": [0.05, 0.0, 0.0]})["data"]
    assert d["response"]["epsilon0"][0] == pytest.approx(
        d["linear_response"]["epsilon0"][0], rel=1e-6)
    assert d["softening"]["Ex_secant_over_linear"] == pytest.approx(1.0, rel=1e-6)


def test_cross_ply_is_barely_affected():
    """[0/90]s 는 전단이 거의 없어 전단 비선형의 영향이 미미하다 (물리 감별)."""
    d = srv.solve_nonlinear_shear_response(lam(angles=(0.0, 90.0, 90.0, 0.0)),
                                           loads={"N": [50.0, 0.0, 0.0]})["data"]
    assert d["softening"]["max_gamma12"] < 1e-12
    assert d["softening"]["Ex_secant_over_linear"] == pytest.approx(1.0, rel=1e-12)


# ── Tool 계층 ───────────────────────────────────────────────────────────────

def test_unit_systems_agree():
    """SI와 SI_mm이 같은 물리를 준다 (S6666 은 1/응력³ 이라 1e-18 배)."""
    si = {"unit_system": "SI",
          "laminae": [{"material": {"type": "orthotropic_2d", "E1": 181e9, "E2": 10.3e9,
                                    "G12": 7.17e9, "nu12": 0.28,
                                    "shear_nonlinear": {"S6666": S_MM * 1e-18}},
                       "angle_deg": a, "thickness": 0.125e-3}
                      for a in (45.0, -45.0, -45.0, 45.0)]}
    a = srv.solve_nonlinear_shear_response(si, loads={"N": [50000.0, 0.0, 0.0]})["data"]
    b = srv.solve_nonlinear_shear_response(lam(), loads={"N": [50.0, 0.0, 0.0]})["data"]
    assert a["response"]["epsilon0"] == pytest.approx(b["response"]["epsilon0"], rel=1e-12)
    assert a["per_ply"][0]["tau12"] == pytest.approx(b["per_ply"][0]["tau12"] * 1e6, rel=1e-12)
    assert a["softening"]["Ex_secant_over_linear"] == pytest.approx(
        b["softening"]["Ex_secant_over_linear"], rel=1e-12)


def test_validity_gate_on_large_shear_strain():
    """γ12 > 0.05 면 강도 한계 없는 3차식의 외삽임을 W130으로 알린다."""
    env = srv.solve_nonlinear_shear_response(lam(), loads={"N": [200.0, 0.0, 0.0]})
    assert env["data"]["softening"]["max_gamma12"] > 0.05
    assert any(w["code"] == "W130" and "강도 한계가 없어" in w["message"] for w in env["warnings"])


def test_mixed_plies_warn_and_stay_linear():
    """일부 ply만 S6666이 있으면 나머지는 선형으로 두고 W120으로 알린다."""
    mixed = {"unit_system": "SI_mm", "laminae": [
        {"material": mat(S_MM), "angle_deg": 45.0, "thickness": 0.125},
        {"material": mat(None), "angle_deg": -45.0, "thickness": 0.125},
        {"material": mat(None), "angle_deg": -45.0, "thickness": 0.125},
        {"material": mat(S_MM), "angle_deg": 45.0, "thickness": 0.125}]}
    env = srv.solve_nonlinear_shear_response(mixed, loads={"N": [50.0, 0.0, 0.0]})
    assert env["errors"] == []
    flags = [p["nonlinear"] for p in env["data"]["per_ply"]]
    assert flags == [True, False, False, True]
    assert [p["secant_ratio"] for p in env["data"]["per_ply"]][1] == pytest.approx(1.0, rel=1e-12)
    assert any(w["code"] == "W120" and "선형 G12로 다룹니다" in w["message"] for w in env["warnings"])


def test_linear_tool_warns_when_data_present():
    """§18.7 게이트 — solve_load_response가 S6666을 무시하고 있음을 알려야 한다."""
    env = srv.solve_load_response(lam(), loads={"N": [50.0, 0.0, 0.0]})
    assert any(w["code"] == "W130" and "solve_nonlinear_shear_response" in w["message"]
               for w in env["warnings"])
    # 물성이 없으면 경고하지 않는다
    clean = srv.solve_load_response(lam(s6666=None), loads={"N": [50.0, 0.0, 0.0]})
    assert not any("solve_nonlinear_shear_response" in w["message"] for w in clean["warnings"])


def test_tool_errors():
    assert srv.solve_nonlinear_shear_response(
        lam(s6666=None), loads={"N": [50.0, 0.0, 0.0]})["errors"][0]["code"] == "E100"
    assert srv.solve_nonlinear_shear_response(
        lam(), loads={"N": [0.0, 0.0, 0.0]})["errors"][0]["code"] == "E100"
    assert srv.solve_nonlinear_shear_response(
        lam(), loads={"N": [float("inf"), 0.0, 0.0]})["errors"][0]["code"] == "E100"
    assert srv.solve_nonlinear_shear_response(
        lam(), loads="nope")["errors"][0]["code"] == "E100"
    bad = lam()
    bad["laminae"][0]["material"]["shear_nonlinear"] = {"S6666": -1.0}
    assert srv.solve_nonlinear_shear_response(
        bad, loads={"N": [50.0, 0.0, 0.0]})["errors"][0]["code"] == "E100"


def test_tool_is_deterministic():
    a = srv.solve_nonlinear_shear_response(lam(), loads={"N": [50.0, 0.0, 0.0]})
    b = srv.solve_nonlinear_shear_response(lam(), loads={"N": [50.0, 0.0, 0.0]})
    assert a["data"] == b["data"]
    assert a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]
