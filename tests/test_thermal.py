# 열탄성 테스트 — Timoshenko 바이메탈 폐형해, 대칭·단일재 극한, 평형 불변식, 휨 기하, ROM (계획서 §17.1)
from __future__ import annotations

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import thermal as TH


def iso(E, nu, alpha):
    return {"type": "isotropic", "E": E, "nu": nu, "alpha": alpha}


def lam(layers, unit_system="SI_mm"):
    return {"unit_system": unit_system,
            "laminae": [{"thickness": t, "angle_deg": 0, "material": m} for t, m in layers]}


def test_timoshenko_bimetal_closed_form():
    """등두께 바이메탈: |κ| = 24·Δα·ΔT / (h(14+n+1/n)), n=E1/E2 (동일 ν ⇒ 판=보, math_spec)."""
    E1, E2 = 100e9, 50e9
    a1, a2 = 10e-6, 20e-6
    t, dT = 1.0e-3, 100.0
    payload = {"unit_system": "SI",
               "laminae": [{"thickness": t, "angle_deg": 0, "material": iso(E1, 0.3, a1)},
                           {"thickness": t, "angle_deg": 0, "material": iso(E2, 0.3, a2)}]}
    env = srv.compute_thermal_response(payload, delta_T=dT)
    assert env["status"] == "ok"
    kx, ky, kxy = env["data"]["response"]["kappa"]

    n = E1 / E2
    h = 2 * t
    expected = 24.0 * (a2 - a1) * dT / (h * (14.0 + n + 1.0 / n))
    assert abs(kx) == pytest.approx(expected, rel=1e-6)
    assert ky == pytest.approx(kx, rel=1e-9)      # 등방층 → 등2축(구면) 곡률
    assert kxy == pytest.approx(0.0, abs=1e-15)
    # 부호: 위층(z>0)의 α가 크고 가열 → 위가 더 늘어나야 하므로 κx > 0 (ε=ε0+zκ)
    assert kx > 0


def test_symmetric_stack_no_curvature_and_mixture_cte():
    a, b = iso(70000.0, 0.33, 23e-6), iso(200000.0, 0.30, 12e-6)
    env = srv.compute_thermal_response(lam([(0.5, a), (0.5, b), (0.5, b), (0.5, a)]), delta_T=80.0)
    d = env["data"]
    assert all(abs(k) < 1e-12 for k in d["response"]["kappa"])
    # 유효 CTE는 강성 가중 혼합값 — 두 성분 사이
    assert 12e-6 < d["effective_cte"]["alpha_x"] < 23e-6


def test_single_material_recovers_alpha_and_zero_stress():
    env = srv.compute_thermal_response(lam([(1.0, iso(200000.0, 0.3, 17e-6))]), delta_T=-150.0)
    d = env["data"]
    assert d["effective_cte"]["alpha_x"] == pytest.approx(17e-6, rel=1e-12)
    assert all(abs(k) < 1e-15 for k in d["response"]["kappa"])
    assert d["residual_stress"]["max_abs"]["value"] == pytest.approx(0.0, abs=1e-9)


def test_residual_stress_force_equilibrium():
    """ply 중앙값 힘 평형 Σσ·t = 0 (ply 내 σ가 z에 선형 → 중앙값 적분은 정확)."""
    plies = [(0.3e-3, (110e9, 0.34, 17e-6)), (0.7e-3, (22e9, 0.3, 15e-6)),
             (0.2e-3, (70e9, 0.33, 23e-6))]
    qbars, alphas, ts = [], [], []
    for t, (E, nu, al) in plies:
        E1, E2, G12, nu12 = MAT.isotropic_to_orthotropic(E, nu)
        qbars.append(MAT.qbar_matrix(MAT.q_matrix(E1, E2, G12, nu12), 0.0))
        alphas.append(TH.alpha_vector(al, al, 0.0))
        ts.append(t)
    z = ABD.z_coordinates(ts)
    A, B, D = ABD.abd_matrices(qbars, z)
    N_th, M_th = TH.thermal_loads(qbars, alphas, z, 100.0)
    eps0, kappa = TH.thermal_response(A, B, D, N_th, M_th)
    sig = TH.residual_ply_stresses(qbars, alphas, z, eps0, kappa, 100.0)
    force = sum(s * t for s, t in zip(sig, ts))
    scale = max(float(np.max(np.abs(s))) for s in sig) * sum(ts)
    np.testing.assert_allclose(force, 0.0, atol=1e-9 * max(scale, 1e-300))


def test_warpage_cylinder_geometry():
    wp = TH.warpage_over_panel(np.array([2.0, 0.0, 0.0]), lx=0.1, ly=0.05)
    assert wp["warpage_range"] == pytest.approx(2.0 * 0.1**2 / 8.0, rel=1e-12)


def test_warpage_via_tool_units_bridge():
    """SI vs SI_mm 동일 물리 → warpage(mm) = warpage(m)×1e3."""
    def payload(us, t, E_scale, L):
        return ({"unit_system": us,
                 "laminae": [{"thickness": t, "angle_deg": 0, "material": iso(110e3 * E_scale, 0.34, 17e-6)},
                             {"thickness": t, "angle_deg": 0, "material": iso(22e3 * E_scale, 0.3, 60e-6)}]},
                {"Lx": L, "Ly": L})
    p_mm, panel_mm = payload("SI_mm", 0.5, 1.0, 50.0)
    p_si, panel_si = payload("SI", 0.5e-3, 1e6, 0.05)
    w_mm = srv.compute_thermal_response(p_mm, delta_T=210.0, panel=panel_mm)["data"]["warpage"]["range"]
    w_si = srv.compute_thermal_response(p_si, delta_T=210.0, panel=panel_si)["data"]["warpage"]["range"]
    assert w_mm == pytest.approx(w_si * 1e3, rel=1e-9)
    assert w_mm > 0


def test_homogenize_rom():
    cu = {"material": {"type": "isotropic", "E": 110000.0, "nu": 0.34, "alpha": 17e-6, "rho": 8.9e-9}}
    resin = {"material": {"type": "isotropic", "E": 3500.0, "nu": 0.35, "alpha": 60e-6, "rho": 1.2e-9}}
    env = srv.homogenize_layer([{**cu, "volume_fraction": 0.7}, {**resin, "volume_fraction": 0.3}])
    m = env["data"]["material"]
    assert m["E"] == pytest.approx(0.7 * 110000 + 0.3 * 3500, rel=1e-12)
    # α는 강성 가중 → 산술평균보다 Cu 쪽으로 치우침
    assert 17e-6 < m["alpha"] < 0.7 * 17e-6 + 0.3 * 60e-6
    # 극한: f=1 → 성분 그대로
    one = srv.homogenize_layer([{**cu, "volume_fraction": 0.5}, {**cu, "volume_fraction": 0.5}])
    assert one["data"]["material"]["E"] == pytest.approx(110000.0, rel=1e-12)


def test_homogenize_validation():
    cu = {"material": {"type": "isotropic", "E": 110000.0, "nu": 0.34}, "volume_fraction": 0.7}
    assert srv.homogenize_layer([cu])["errors"][0]["code"] == "E100"                    # 1개
    bad = [cu, {"material": {"type": "isotropic", "E": 3500.0, "nu": 0.35}, "volume_fraction": 0.2}]
    assert srv.homogenize_layer(bad)["errors"][0]["code"] == "E100"                     # Σf≠1


def test_e203_missing_cte_and_bad_delta_t():
    no_cte = lam([(0.5, {"type": "isotropic", "E": 70000.0, "nu": 0.33})])
    assert srv.compute_thermal_response(no_cte, delta_T=100.0)["errors"][0]["code"] == "E203"
    ok = lam([(0.5, iso(70000.0, 0.33, 23e-6))])
    assert srv.compute_thermal_response(ok, delta_T=0.0)["errors"][0]["code"] == "E100"
    assert srv.compute_thermal_response(ok, delta_T=100.0, panel={"Lx": -1, "Ly": 2})["errors"][0]["code"] == "E100"


def test_ppm_mistake_warning():
    env = srv.compute_thermal_response(lam([(0.5, iso(70000.0, 0.33, 23.0))]), delta_T=10.0)
    assert "W110" in [w["code"] for w in env["warnings"]]
