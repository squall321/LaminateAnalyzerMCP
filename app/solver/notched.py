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


def point_stress_ratio(kt: float, radius: float, d0: float) -> float:
    """점응력 기준 σ_OH/σ_un. d0→0 이면 1/K_T, d0→∞ 이면 1."""
    xi = radius / (radius + d0)
    den = 2.0 + xi ** 2 + 3.0 * xi ** 4 - (kt - 3.0) * (5.0 * xi ** 6 - 7.0 * xi ** 8)
    return 2.0 / den if den > 0 else float("inf")


def average_stress_ratio(kt: float, radius: float, a0: float) -> float:
    """평균응력 기준 σ_OH/σ_un. a0→0 이면 1/K_T, a0→∞ 이면 1."""
    xi = radius / (radius + a0)
    den = 2.0 - xi ** 2 - xi ** 4 + (kt - 3.0) * (xi ** 6 - xi ** 8)
    return 2.0 * (1.0 - xi) / den if den > 0 else float("inf")
