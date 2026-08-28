# 자유 가장자리 박리 — O'Brien ERR 폐형해 + 계면별 구동력 지표 (계획서 §19.1)
"""면내 강도로는 안전한 적층이 **자유 가장자리에서 먼저 뜯기는** 경우를 잡는다.

두 가지를 함께 준다.

1. **O'Brien(1985) 가장자리 박리 에너지방출률** — 엄밀한 폐형해다.

       G = (ε_x²·h/2)·(E_LAM − E*),   E* = Σ E_i·t_i / h

   E_LAM 은 적층 전체의 축방향 막 유효탄성계수, E* 는 그 계면에서 갈라진 부분적층들의
   두께 가중 평균이다. **박리 길이에 무관**하다(정상상태) — 그래서 임계 변형률이 닫힌
   형태로 나온다: ε_c = √(2G_c / (h·(E_LAM − E*))).

2. **계면별 구동력 지표** — G 는 "그 계면이 뜯기면 방출될 에너지"만 말할 뿐 **어느 계면이
   실제로 먼저 뜯기는지**는 말하지 않는다. 자유 가장자리에서 각 ply 의 σ_y·τ_xy 가 0이
   되어야 하므로, 그 불균형이 두께 규모 경계층에서 층간 응력으로 넘어간다. 계면 위쪽
   부분적층이 지고 있는 불균형 힘·모멘트를 적분해 어느 성분이 지배하는지 알려준다.

   - σ_y 의 모멘트 → σ_z (peel, 개구) 구동
   - σ_y 의 합력   → τ_yz 구동
   - τ_xy 의 합력  → τ_xz 구동  (각도적층 [±θ] 는 이쪽이 지배한다 — σ_y 는 0 이다)

   이건 **경계층 평형 논증에 근거한 크기 규모 지표**이지 응력장이 아니다. 실제 자유
   가장자리 응력은 특이점을 갖는 3D 탄성 문제다.

한계: 혼합모드 분리(G_I/G_II)를 하지 않는다 — O'Brien 의 G 는 총 에너지방출률이다.
"""
from __future__ import annotations

import math

import numpy as np

from app.solver import abd as ABD
from app.solver import response as RESP


def sublaminate_axial_modulus(qbars: list[np.ndarray], thicknesses: list[float],
                              i0: int, i1: int) -> tuple[float, float]:
    """plies[i0:i1] 부분적층의 축방향 막 유효탄성계수와 두께.

    부분적층이 자체로 비대칭이면 자유단에서 굽는다 — 그 연화를 포함한 막 Ex 를 쓴다
    (자체 ABD 의 α 블록 기준). 이것이 박리 후 실제로 남는 강성이다.
    """
    sub_t = list(thicknesses[i0:i1])
    z = ABD.z_coordinates(sub_t)
    A, B, D = ABD.abd_matrices(list(qbars[i0:i1]), z)
    alpha, _, delta = RESP.compliance_blocks(A, B, D)
    h = float(sum(sub_t))
    return float(RESP.effective_constants(alpha, delta, h)["membrane"]["Ex"]), h


MODEL_TOL = 1.0e-9      # ΔE 유효성 판정 (E_LAM 대비 상대)


def obrien_err(eps_x: float, h: float, e_lam: float, e_star: float) -> float | None:
    """G = (ε²·h/2)·(E_LAM − E*).

    ΔE < 0 이면 **0으로 자르지 않고 None** 을 준다. 음수는 이 모델이 그 적층에 적용되지
    않는다는 뜻이지 "구동력이 없다"가 아니다 — 0으로 자르면 응답이 '안전'으로 읽혀 위험하다.
    비대칭 적층에서 실제로 발생한다(B 커플링으로 온전한 적층의 막강성이 이미 연화돼 있어,
    갈라진 부분적층이 오히려 더 뻣뻣하다. 실측: [0/45/−45/90] 계면1 에서 ΔE = −31.6 GPa).
    단일방향처럼 ΔE 가 정확히 0 인 경우는 구동력이 없는 것이 맞으므로 0.0 을 준다.
    """
    d_e = e_lam - e_star
    if d_e < -MODEL_TOL * abs(e_lam):
        return None
    return 0.5 * eps_x * eps_x * h * max(0.0, d_e)


def onset_strain(g_c: float, h: float, delta_e: float) -> float | None:
    """ε_c = √(2·G_c/(h·ΔE)). 구동력이 없으면(ΔE≈0) 박리 개시가 정의되지 않는다."""
    if delta_e <= 0.0 or h <= 0.0 or g_c <= 0.0:
        return None
    return math.sqrt(2.0 * g_c / (h * delta_e))


def interface_driving(sigma_plies: list[np.ndarray], thicknesses: list[float],
                      z: list[float], k: int) -> dict:
    """계면 k 위쪽 부분적층이 지고 있는 불균형 힘·모멘트 (경계층 평형 논증)."""
    z_if = z[k]
    f_y = f_s = m_y = 0.0
    for i in range(k, len(thicknesses)):
        z_mid = 0.5 * (z[i] + z[i + 1])
        f_y += float(sigma_plies[i][1]) * thicknesses[i]
        f_s += float(sigma_plies[i][2]) * thicknesses[i]
        m_y += float(sigma_plies[i][1]) * thicknesses[i] * (z_mid - z_if)
    return {"transverse_force": f_y, "shear_force": f_s, "peel_moment": m_y}


def dominant_driver(drive: dict, h: float, stress_scale: float) -> str:
    """peel(σz) / transverse_shear(τyz) / in_plane_shear(τxz) 중 무엇이 지배하는가.

    셋을 응력 규모로 환산해 비교한다 — 경계층 폭을 h 로 잡으면
    σz ~ M_y/h², τ_yz ~ F_y/h, τ_xz ~ F_s/h.
    """
    if h <= 0:
        return "none"
    mags = {
        "peel": abs(drive["peel_moment"]) / (h * h),
        "transverse_shear": abs(drive["transverse_force"]) / h,
        "in_plane_shear": abs(drive["shear_force"]) / h,
    }
    top = max(mags, key=lambda k: mags[k])
    # 잔여 수치오차를 구동력으로 오인하지 않도록 ply 응력 규모 대비 상대 문턱을 쓴다
    return "none" if mags[top] <= 1.0e-6 * max(abs(stress_scale), 1e-300) else top
