# 접착 겹치기 이음 — Volkersen 전단지연 (계획서 §19.22)
"""서버가 가진 것은 shear-lag 전달길이 스칼라 하나뿐이라 "겹침을 늘리면 강해진다"는
직관을 반박할 수단이 없었다.

Volkersen 단순 겹치기(피착재 축변형만, 굽힘 없음):

    ω² = (G_a/t_a)·(1/(E₁t₁) + 1/(E₂t₂))
    τ(x) = (P·ω/2)·[cosh(ωx)/sinh X + ψ·sinh(ωx)/cosh X],  X = ωL/2
    τ_peak/τ_avg = X·[coth X + |ψ|·tanh X],   ψ = (EA₁ − EA₂)/(EA₁ + EA₂)

**피착재가 다르면 분포가 비대칭이다.** ψ 항을 빼고 대칭 해만 쓰면 피크를
1 + |ψ|tanh²X 배만큼 과소평가한다(EA비 2 → 1.333배, 5 → 1.667배; 적대 검증 LJ-01).
경계조건 τ′(−L/2) = (G/t)P/EA₁, τ′(+L/2) = −(G/t)P/EA₂ 를 풀면 위 형태가 나오고
∫τ dx = P 로 평형이 닫힌다 — 독립 유한요소와 6자리 일치.

**겹침을 늘려도 피크가 줄지 않는다.** 실측(Al 2mm, G_a=1 GPa, t_a=0.2mm):
L=25→50 mm 에서 평균은 절반이 되지만 피크는 **1.003배**만 준다. ωL/2 ≫ 1 이면
τ_peak → P·ω/2 로 **겹침 길이에 무관**해지기 때문이다.

⚠ **단일 겹치기(single lap)에는 이 식만으로 판정하면 안 된다.** 하중선이 어긋나 굽힘이
생기고 그 peel 응력(σ_z)이 보통 전단보다 먼저 파손시킨다 — Volkersen 은 peel 을 아예
모델링하지 않으므로 **비보수**다. 이중 겹치기(double lap)는 편심이 없어 유효하다.
"""
from __future__ import annotations

import math

# ωL/2 가 이 값을 넘으면 τ_peak 가 겹침 길이에 사실상 무관해진다 (coth → 1)
SATURATION_OMEGA_L_HALF = 3.0


def shear_lag_omega(g_adhesive: float, t_adhesive: float,
                    ea1: float, ea2: float) -> float:
    """ω = √((G_a/t_a)(1/EA₁ + 1/EA₂)) — 단위 폭당 축강성 EA = E·t."""
    return math.sqrt((g_adhesive / t_adhesive) * (1.0 / ea1 + 1.0 / ea2))


def stiffness_imbalance(ea1: float, ea2: float) -> float:
    """ψ = (EA₁ − EA₂)/(EA₁ + EA₂) — 0 이면 대칭, ±1 이면 한쪽이 강체."""
    s = ea1 + ea2
    return (ea1 - ea2) / s if s > 0 else 0.0


def peak_over_average(omega: float, overlap: float, psi: float = 0.0) -> float:
    """τ_peak/τ_avg = X·[coth X + |ψ|·tanh X], X = ωL/2. X→0 이면 1(균일)."""
    x = omega * overlap / 2.0
    if x < 1e-9:
        return 1.0
    coth = 1.0 if x > 350.0 else 1.0 / math.tanh(x)
    tanh = 1.0 if x > 350.0 else math.tanh(x)
    return x * (coth + abs(psi) * tanh)


def shear_profile(omega: float, overlap: float, p_load: float,
                  psi: float = 0.0, n_points: int = 9) -> list[dict]:
    """τ(x) 분포 (고정 점수 — 결정론). x 는 겹침 중앙 원점.

    **비율로 정리해 평가한다.** cosh(ωx)/sinh(X) 를 항마다 따로 계산하면 X ≳ 695 에서
    넘쳐 응답 전체가 E403/E501 로 사라졌다(적대 검증 LJ-02). cosh(ωx)/sinh X 는
    X 가 크면 exp(−ω(L/2 − |x|)) 로 수렴하므로 어디서나 유한하다.
    """
    x_half = overlap / 2.0
    x_big = omega * x_half
    out = []
    for i in range(n_points):
        xi = -x_half + 2.0 * x_half * i / (n_points - 1)
        if omega <= 0.0 or x_big < 1e-12:
            tau = p_load / overlap
        elif x_big > 350.0:
            # 포화 구간 — 양 끝단에서 지수적으로 감쇠한다. sinh≈cosh≈e^X/2.
            d_lo, d_hi = omega * (x_half + xi), omega * (x_half - xi)
            c_over_s = math.exp(-d_hi) + math.exp(-d_lo)          # cosh(ωx)/sinh X
            s_over_c = math.exp(-d_hi) - math.exp(-d_lo)          # sinh(ωx)/cosh X
            tau = (p_load * omega / 2.0) * (c_over_s + psi * s_over_c)
        else:
            c_over_s = math.cosh(omega * xi) / math.sinh(x_big)
            s_over_c = math.sinh(omega * xi) / math.cosh(x_big)
            tau = (p_load * omega / 2.0) * (c_over_s + psi * s_over_c)
        out.append({"x_over_L": xi / overlap, "tau": tau})
    return out


def saturation_overlap(omega: float) -> float:
    """이 겹침을 넘으면 피크가 더 줄지 않는다 (ωL/2 = 3 기준)."""
    return 2.0 * SATURATION_OMEGA_L_HALF / omega if omega > 0 else float("inf")
