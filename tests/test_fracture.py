# 크랙 차폐 해석해 테스트 — Griffith↔터널 항등, Dundurs, He-Hutchinson, shear-lag, 점탄성 (계획서 §17.2)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import fracture as FR


def test_tunnel_gss_equals_half_griffith_tip():
    """터널 G_ss = 평면변형 중앙크랙(2a=h) 선단 G의 절반 — K=σ√(πa) 경유 독립 산식과 대조."""
    sigma, h, E, nu = 80e6, 0.4e-3, 70e9, 0.22
    a = h / 2.0
    K = sigma * math.sqrt(math.pi * a)
    g_tip = K * K / FR.plane_strain_modulus(E, nu)
    assert FR.tunnel_crack_gss(sigma, h, E, nu) == pytest.approx(g_tip / 2.0, rel=1e-12)


def test_critical_stress_roundtrip_and_thin_layer_benefit():
    h, E, nu, gamma = 0.2e-3, 70e9, 0.22, 8.0
    sc = FR.critical_channeling_stress(gamma, h, E, nu)
    assert FR.tunnel_crack_gss(sc, h, E, nu) == pytest.approx(gamma, rel=1e-12)
    # 박층 유리: h 절반 → σ_c ×√2
    assert FR.critical_channeling_stress(gamma, h / 2, E, nu) == pytest.approx(sc * math.sqrt(2), rel=1e-12)


def test_dundurs_signs():
    a0, b0 = FR.dundurs_parameters(70e9, 0.3, 70e9, 0.3)
    assert a0 == pytest.approx(0.0, abs=1e-15) and b0 == pytest.approx(0.0, abs=1e-15)
    a_stiff_nb, _ = FR.dundurs_parameters(70e9, 0.3, 200e9, 0.3)   # 이웃이 강성 → α<0 (차폐)
    assert a_stiff_nb < 0
    a_soft_nb, _ = FR.dundurs_parameters(70e9, 0.3, 3e9, 0.35)     # 유연 이웃 → α>0 (증폭)
    assert a_soft_nb > 0


def test_he_hutchinson_quarter_rule():
    assert FR.interface_deflection_verdict(2.0, 10.0)["deflects_into_interface"] is True   # 0.2 < 0.25
    assert FR.interface_deflection_verdict(3.0, 10.0)["deflects_into_interface"] is False  # 0.3
    assert "deflects_into_interface" not in FR.interface_deflection_verdict(None, 10.0)


def test_shear_lag_limits():
    l0 = FR.shear_lag_transfer_length(70e9, 0.2e-3, 0.05e-3, 1e9)
    assert FR.shear_lag_transfer_length(70e9, 0.2e-3, 0.05e-3, 4e9) == pytest.approx(l0 / 2, rel=1e-12)
    assert FR.shear_lag_transfer_length(70e9, 0.2e-3, 0.05e-3, 1e15) < 1e-6  # G→∞ ⇒ ℓ→0


def test_viscoelastic_relaxation_factor():
    ve = FR.viscoelastic_relaxation_factor(2.0e9, 0.5e9)
    assert ve["transfer_length_growth"] == pytest.approx(2.0, rel=1e-12)  # √(4)=2


# ── Tool E2E: 보호층(점탄성 PSA) / 취성 피보호층(유리) / 보호층 ────────────


def _stack():
    psa = {"type": "isotropic", "E": 50.0, "nu": 0.45, "name": "PSA",
           "viscoelastic": {"E0": 50.0, "Einf": 5.0, "tau_s": 10.0}}
    glass = {"type": "isotropic", "E": 70000.0, "nu": 0.22, "name": "UTG"}
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": 0.05, "angle_deg": 0, "material": dict(psa)},
                        {"thickness": 0.03, "angle_deg": 0, "material": glass},
                        {"thickness": 0.05, "angle_deg": 0, "material": dict(psa)}]}


def test_assess_crack_shielding_e2e():
    env = srv.assess_crack_shielding(
        _stack(), target_ply=1,
        fracture={"applied_strain": 0.005, "gamma_target": 0.004,   # Γ [N/mm] = 4 J/m²
                  "gamma_interface": 0.001, "gamma_next_layer": 0.02})
    assert env["status"] == "ok"
    d = env["data"]
    assert d["target"]["ply"] == 1 and len(d["neighbors"]) == 2
    for nb in d["neighbors"]:
        assert nb["dundurs_alpha"] > 0.9            # 유리 vs PSA: 극단 유연 이웃
        assert nb["transfer_length"] > 0
        ve = nb["viscoelastic"]
        assert ve["transfer_length_growth"] == pytest.approx(math.sqrt(10.0), rel=1e-9)
    assert d["crack_driving"]["sigma_target"] == pytest.approx(70000.0 * 0.005, rel=1e-9)  # MPa
    assert d["crack_driving"]["G_ss_tunnel"] > 0
    assert d["initiation_threshold"]["sigma_critical"] > 0
    assert d["interface_deflection"]["deflects_into_interface"] is True  # 0.001/0.02=0.05 < 0.25


def test_assess_gamma_units_bridge():
    """Γ 변환: SI(J/m²)와 SI_mm(N/mm) 동일 물리 → σ_c 동일 물리값."""
    si_mm = srv.assess_crack_shielding(_stack(), 1, {"gamma_target": 0.004})       # 4 J/m²
    stack_si = _stack()
    stack_si["unit_system"] = "SI"
    for p in stack_si["laminae"]:
        p["thickness"] *= 1e-3
        p["material"]["E"] *= 1e6
        if "viscoelastic" in p["material"]:
            p["material"]["viscoelastic"]["E0"] *= 1e6
            p["material"]["viscoelastic"]["Einf"] *= 1e6
    si = srv.assess_crack_shielding(stack_si, 1, {"gamma_target": 4.0})            # 4 J/m²
    assert si_mm["data"]["initiation_threshold"]["sigma_critical"] == pytest.approx(
        si["data"]["initiation_threshold"]["sigma_critical"] * 1e-6, rel=1e-9)


def test_assess_validation():
    assert srv.assess_crack_shielding(_stack(), 5)["errors"][0]["code"] == "E100"
    assert srv.assess_crack_shielding(_stack(), 1, {"gamma_target": -1})["errors"][0]["code"] == "E100"
    bad_ve = _stack()
    bad_ve["laminae"][0]["material"]["viscoelastic"] = {"E0": 5.0, "Einf": 50.0}
    assert srv.assess_crack_shielding(bad_ve, 1)["errors"][0]["code"] == "E100"  # Einf > E0
