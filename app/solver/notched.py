# 노치(원공) 강도 — Lekhnitskii K_T + Whitney–Nuismer (계획서 §19.23)
"""K_T 는 **순수 폐형해**로 `solve_load_response` 가 이미 주는 유효상수만으로 나온다:

    K_T∞ = 1 + √(2(√(E_x/E_y) − ν_xy) + E_x/G_xy)

등방 극한에서 정확히 3.0 이다(테스트로 10자리 고정). 준등방 3.00, [0/90]s 4.92,
UD 6.75 — 적층에 따라 2배 이상 갈린다.

**그런데 노치 강도 σ_OH 는 다르다.** Whitney–Nuismer 는 특성거리 d0(점응력)·a0(평균응력)를
쓰는데 이건 **재료·적층별 시험 피팅 상수**이고 답이 여기에 강하게 민감하다 — 실측
[±45/0/90]s φ6.35 에서 d0 = 0.5~2.0 mm 만 바꿔도 σ_OH 가 234~368 MPa 로 갈린다.

그래서 **확정값(K_T)과 조건부값(σ_OH)을 응답에서 분리 보고**하고, d0/a0 가 없으면
σ_OH 를 아예 내지 않는다. 있어도 "이 값은 그 상수에 종속"임을 W130 으로 명시한다.
"""
from __future__ import annotations

import math


def kt_infinite(ex: float, ey: float, nu_xy: float, g_xy: float) -> float:
    """무한 직교이방 판 원공의 응력집중계수. 등방이면 정확히 3."""
    return 1.0 + math.sqrt(2.0 * (math.sqrt(ex / ey) - nu_xy) + ex / g_xy)


# 절단급수가 물리 상한을 깨기 시작하는 K_T. 그 위에서는 (K_T−3)(5ξ⁶−7ξ⁸) 항이 지배해
# σ_y(R+d0) < σ∞ 라는 불가능한 값이 나온다 (ξ=0.6494 에서 최소, 적대 검증 NT-01).
KT_SERIES_VALID = 9.2191


def point_stress_ratio(kt: float, radius: float, d0: float) -> float:
    """점응력 기준 σ_OH/σ_un. d0→0 이면 1/K_T, d0→∞ 이면 1.

    **물리 상한 [1/K_T, 1] 안으로 자른다.** Lekhnitskii σ_y 의 8차 절단급수는 K_T 가
    크면 σ_y(R+d0) < σ∞ 를 내놓아 비 > 1 (구멍 뚫은 판이 무노치보다 강함)이 됐다.
    비가 잘렸는지는 series_exceeds_bound 로 알 수 있다.
    """
    xi = radius / (radius + d0)
    den = 2.0 + xi ** 2 + 3.0 * xi ** 4 - (kt - 3.0) * (5.0 * xi ** 6 - 7.0 * xi ** 8)
    raw = 2.0 / den if den > 0 else float("inf")
    return min(max(raw, 1.0 / kt), 1.0) if kt > 0 else raw


def average_stress_ratio(kt: float, radius: float, a0: float) -> float:
    """평균응력 기준 σ_OH/σ_un. a0→0 이면 1/K_T, a0→∞ 이면 1 (같은 상한을 건다)."""
    xi = radius / (radius + a0)
    den = 2.0 - xi ** 2 - xi ** 4 + (kt - 3.0) * (xi ** 6 - xi ** 8)
    raw = 2.0 * (1.0 - xi) / den if den > 0 else float("inf")
    return min(max(raw, 1.0 / kt), 1.0) if kt > 0 else raw


def series_exceeds_bound(kt: float, radius: float, dist: float, average: bool = False) -> bool:
    """절단급수가 물리 상한 [1/K_T, 1] 을 벗어났는가 (자르기 전 값 기준)."""
    xi = radius / (radius + dist)
    if average:
        den = 2.0 - xi ** 2 - xi ** 4 + (kt - 3.0) * (xi ** 6 - xi ** 8)
        raw = 2.0 * (1.0 - xi) / den if den > 0 else float("inf")
    else:
        den = 2.0 + xi ** 2 + 3.0 * xi ** 4 - (kt - 3.0) * (5.0 * xi ** 6 - 7.0 * xi ** 8)
        raw = 2.0 / den if den > 0 else float("inf")
    return not (1.0 / kt - 1e-12 <= raw <= 1.0 + 1e-12) if kt > 0 else False
