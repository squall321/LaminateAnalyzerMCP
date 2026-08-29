# Fickian 수분 확산 동역학 — 흡습에 시간 축을 준다 (계획서 §19.4)
"""compute_thermal_response 는 흡습 **변형**(delta_C)만 다루고 **시간**이 없다.
"85/85 에서 며칠이면 포화하나", "리플로우 전 몇 시간 베이크해야 하나"에 답할 수 없었다.

양면 노출 1D Fick 확산의 해석해(Shen–Springer):

    M(t)/M∞ = 1 − (8/π²)·Σ_{n=0,1,2,…} exp(−(2n+1)²·π²·D·t/h²) / (2n+1)²
    c(z,t)/c∞ = 1 − (4/π)·Σ_{n} sin((2n+1)πζ)/(2n+1) · exp(−(2n+1)²π²Dt/h²),  ζ=(z+h/2)/h

무차원 시간 τ = D·t/h² 하나가 전부를 지배한다 — 두께가 2배면 시간이 4배다.

결정론: 급수 항수를 고정하고(SERIES_TERMS), 역산(목표 흡습률까지의 시간)은 고정 반복
이분법만 쓴다. 수렴 판정으로 항수·반복수를 바꾸지 않는다.
"""
from __future__ import annotations

import math

SERIES_TERMS = 500      # 고정 항수 (사양의 일부)
# **τ 가 작으면 급수를 쓰지 않는다.** 500항으로 수렴하려면 τ ≳ 1e-6 이 필요한데
# (τ=1e-8 에서 실측 97% 오차) 그 영역은 반무한체 폐형해가 오히려 **정확**하다
# (τ ≤ 0.01 에서 수렴 급수와 배정도 한계까지 일치, τ=0.05 에서도 0.11%).
# 적응 판정이 아니라 고정 문턱이므로 결정론 계약은 그대로다 (적대 검증 DIF-01).
TAU_SERIES_MIN = 0.01
BISECT_STEPS = 200      # 고정 이분 반복수
TAU_MAX = 10.0          # 역산 상한 (τ=10 이면 사실상 완전 포화)


def uptake_fraction(tau: float) -> float:
    """M(t)/M∞ — 무차원 시간 τ = D·t/h² 의 함수."""
    if tau <= 0.0:
        return 0.0
    if tau < TAU_SERIES_MIN:
        return min(1.0, early_time_fraction(tau))
    s = 0.0
    for n in range(SERIES_TERMS):
        m = 2 * n + 1
        e = m * m * math.pi * math.pi * tau
        if e > 700.0:               # exp 언더플로 — 남은 항은 무시 가능
            break
        s += math.exp(-e) / (m * m)
    return min(1.0, max(0.0, 1.0 - (8.0 / (math.pi * math.pi)) * s))


def concentration_profile(tau: float, zeta: float) -> float:
    """c(z,t)/c∞ — ζ = (z + h/2)/h ∈ [0,1] (0·1 이 노출면)."""
    if tau <= 0.0:
        return 1.0 if zeta <= 0.0 or zeta >= 1.0 else 0.0
    if tau < TAU_SERIES_MIN:
        # 반무한체 중첩 — 양 노출면에서 들어온 두 확산 전선의 합
        r = 2.0 * math.sqrt(tau)
        return min(1.0, max(0.0, math.erfc(zeta / r) + math.erfc((1.0 - zeta) / r)))
    s = 0.0
    for n in range(SERIES_TERMS):
        m = 2 * n + 1
        e = m * m * math.pi * math.pi * tau
        if e > 700.0:
            break
        s += math.sin(m * math.pi * zeta) / m * math.exp(-e)
    return min(1.0, max(0.0, 1.0 - (4.0 / math.pi) * s))


def tau_for_fraction(target: float) -> float | None:
    """주어진 흡습률에 도달하는 무차원 시간 (고정 반복 이분법)."""
    if not (0.0 < target < 1.0):
        return None
    if uptake_fraction(TAU_MAX) < target:
        return None
    lo, hi = 0.0, TAU_MAX
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if uptake_fraction(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def early_time_fraction(tau: float) -> float:
    """초기 √t 해 M/M∞ = 4·√(τ/π).

    '근사'라는 이름이지만 τ ≲ 0.01 에서는 보정항이 exp(−1/(4τ)) 규모라 사실상 **정확**하다.
    uptake_fraction 이 그 구간에서 이 식을 쓴다(급수가 되레 못 미친다).
    """
    return 4.0 * math.sqrt(max(tau, 0.0) / math.pi)


def arrhenius_diffusivity(d0: float, ed: float, temperature_k: float) -> float:
    """D = D0·exp(−Ed/(R·T)). Ed [J/mol], T [K]."""
    return d0 * math.exp(-ed / (8.314462618 * temperature_k))
