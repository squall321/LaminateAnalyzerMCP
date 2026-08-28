# 샌드위치 국부 파손 — 면재 주름·셀 딤플링·코어 전단 (계획서 §19.19)
"""§17.6.3 이 "샌드위치는 CLT 밖"을 R_s 로 정량 경고해 놓고 **지목할 도구가 없었다** —
게이트는 있는데 출구가 없는 가장 나쁜 형태다.

세 가지 국부 파손은 전부 대수식이라 이 서버 계약에 맞는다.

1. **면재 주름(face wrinkling)** — Hoff–Mautner: σ_wr = k·(E_f·E_z·G_c)^(1/3).
   **계수 k 는 문헌마다 0.5~0.825 로 갈린다**(1.65배). 단일값을 내면 그 자체가 오답원이라
   범위로 준다.
2. **셀 내 딤플링(intracell dimpling)** — 허니콤 전용: σ_d = 2E_f/(1−ν²)·(t_f/s)².
   얇은 면재 + 큰 셀에서 지배한다.
3. **코어 전단** — τ_c = V/d (d = 면재 중심 거리, 단위 폭당 횡전단력 V).

셋 다 **면재 재료강도와 무관하게** 먼저 올 수 있다 — 실측으로 CFRP 1mm 면재 + 노멕스
코어에서 주름 376 MPa 가 재료 압축강도 600 MPa 보다 먼저 온다.
"""
from __future__ import annotations

import math

WRINKLING_K_RANGE = (0.5, 0.825)     # 문헌 계수 폭 — 단일값을 쓰지 않는다


def face_wrinkling_stress(e_face: float, e_core_z: float, g_core: float,
                          k: float) -> float:
    """Hoff–Mautner σ_wr = k·(E_f·E_z·G_c)^(1/3)."""
    return k * (e_face * e_core_z * g_core) ** (1.0 / 3.0)


def intracell_dimpling_stress(e_face: float, nu_face: float, t_face: float,
                              cell_size: float) -> float:
    """허니콤 셀 내 딤플링 σ_d = 2E_f/(1−ν²)·(t_f/s)²."""
    return 2.0 * e_face / (1.0 - nu_face * nu_face) * (t_face / cell_size) ** 2


def core_shear_stress(v_shear: float, face_distance: float) -> float:
    """코어 전단 τ_c = V/d — 단위 폭당 횡전단력이 코어에 균일 분포한다는 표준 근사."""
    return abs(v_shear) / face_distance if face_distance > 0 else float("inf")
