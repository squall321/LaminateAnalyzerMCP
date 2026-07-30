# 크랙 발생·차폐 해석해 — 터널크랙 G_ss, Dundurs, He-Hutchinson, shear-lag, 점탄성 이완 (계획서 §17.2)
"""전부 문헌 확립 폐형해(Hutchinson & Suo 계열)의 균질/1차 근사 구현.

정직성 원칙: 채널링 g(α,β) 수치 보정표·모드믹스 의존 계면인성은 미탑재 —
Dundurs 부호와 폐형해로 '경향'을 판단하고, 수치 보정이 필요하면 출처와 함께 별도 탑재한다.
"""
from __future__ import annotations

import math


def plane_strain_modulus(E: float, nu: float) -> float:
    """Ē = E/(1−ν²)."""
    return E / (1.0 - nu * nu)


def dundurs_parameters(E1: float, nu1: float, E2: float, nu2: float) -> tuple[float, float]:
    """Dundurs α, β (1=target, 2=이웃). α<0 ⇒ 이웃이 더 강성 → 차폐, α>0 ⇒ 유연 이웃 → 증폭."""
    Eb1, Eb2 = plane_strain_modulus(E1, nu1), plane_strain_modulus(E2, nu2)
    alpha = (Eb1 - Eb2) / (Eb1 + Eb2)
    mu1, mu2 = E1 / (2 * (1 + nu1)), E2 / (2 * (1 + nu2))
    beta = (mu1 * (1 - 2 * nu2) - mu2 * (1 - 2 * nu1)) / \
           (2 * (mu1 * (1 - nu2) + mu2 * (1 - nu1)))
    return alpha, beta


def tunnel_crack_gss(sigma: float, h: float, E: float, nu: float) -> float:
    """층을 관통하는 터널(채널) 크랙의 정상상태 에너지해방률 (균질 근사).

    G_ss = πσ²h/(4Ē). (평면변형 중앙크랙 2a=h의 선단값 πσ²(h/2)/Ē의 절반 — 정상상태 평균.)
    이웃 불일치 보정 g(α,β)는 미탑재 — Dundurs α 부호로 경향 판단.
    """
    return math.pi * sigma * sigma * h / (4.0 * plane_strain_modulus(E, nu))


def critical_channeling_stress(gamma: float, h: float, E: float, nu: float) -> float:
    """G_ss = Γ 로 놓은 임계 채널링 응력 σ_c = √(4ĒΓ/(πh)). 박층일수록 σ_c↑ (§17.2)."""
    return math.sqrt(4.0 * plane_strain_modulus(E, nu) * gamma / (math.pi * h))


def crack_opening_max(sigma: float, h: float, E: float, nu: float) -> float:
    """층 관통 크랙(2a=h)의 최대 개구 δ_max = 2σh/Ē (중앙크랙 타원 프로파일)."""
    return 2.0 * sigma * h / plane_strain_modulus(E, nu)


HE_HUTCHINSON_DEFLECT_RATIO = 0.25  # α≈0에서 Γ_i/Γ_ℓ < 1/4 ⇒ 크랙이 계면으로 편향(저지)


def interface_deflection_verdict(gamma_interface: float | None, gamma_layer: float | None) -> dict:
    """He–Hutchinson 계면 편향 판정 (α≈0 기준 1/4 법칙)."""
    out = {"threshold_alpha0": HE_HUTCHINSON_DEFLECT_RATIO,
           "rule": "Γ_interface/Γ_next_layer < 0.25 (α≈0) ⇒ 크랙이 계면으로 편향되어 저지"}
    if gamma_interface is not None and gamma_layer is not None and gamma_layer > 0:
        ratio = gamma_interface / gamma_layer
        out["ratio"] = ratio
        out["deflects_into_interface"] = bool(ratio < HE_HUTCHINSON_DEFLECT_RATIO)
    return out


def shear_lag_transfer_length(E_t: float, h_t: float, h_n: float, G_n: float) -> float:
    """크랙 난 취성층의 응력 회복(전달) 길이 ℓ = √(E_t h_t h_n / G_n).

    ℓ가 길수록 이웃(보호층)의 개구 구속이 약함. 멀티플 크래킹 간격의 척도 (§17.2).
    """
    return math.sqrt(E_t * h_t * h_n / G_n)


def viscoelastic_relaxation_factor(E0: float, Einf: float) -> dict:
    """준탄성 근사 — 보호층 이완(E0→E∞)에 따른 차폐 저하.

    전달길이 ℓ ∝ 1/√G_n 이므로 ℓ(∞)/ℓ(0) = √(E0/E∞) ≥ 1.
    """
    return {"modulus_ratio_E0_over_Einf": E0 / Einf,
            "transfer_length_growth": math.sqrt(E0 / Einf)}
