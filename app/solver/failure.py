# 층별 응력 복원과 파손 판정 — 재료축 변환, Max Stress, Tsai-Wu(강도비 R), FPF (계획서 §17.4)
"""ply별 응력 복원: ε(z) = ε0 + zκ → σ_xyz = Q̄(ε(z) − αΔT) → 재료축 σ1/σ2/τ12.

파손 판정은 강도 5종(Xt, Xc, Yt, Yc, S — 압축은 양수 크기 관례)이 있는 ply에만 적용.
Tsai-Wu 강도비 R: 하중을 R배 하면 파손면에 도달 (R>1 안전, 여유율). FI = 1/R.
"""
from __future__ import annotations

import math

import numpy as np


def stress_to_material_axes(sigma_xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """적층판축 [σx, σy, τxy] → 재료축 [σ1, σ2, τ12] (표준 응력 변환)."""
    th = np.deg2rad(angle_deg)
    m, n = np.cos(th), np.sin(th)
    sx, sy, txy = float(sigma_xyz[0]), float(sigma_xyz[1]), float(sigma_xyz[2])
    return np.array([
        sx * m * m + sy * n * n + 2.0 * txy * m * n,
        sx * n * n + sy * m * m - 2.0 * txy * m * n,
        -sx * m * n + sy * m * n + txy * (m * m - n * n),
    ], dtype=np.float64)


def ply_stresses_at(qbar: np.ndarray, eps0: np.ndarray, kappa: np.ndarray, z_loc: float,
                    alpha_vec: np.ndarray | None = None, delta_t: float = 0.0) -> np.ndarray:
    """z 위치의 적층판축 응력 σ = Q̄(ε0 + zκ − αΔT)."""
    eps = eps0 + z_loc * kappa
    if alpha_vec is not None and delta_t != 0.0:
        eps = eps - alpha_vec * delta_t
    return qbar @ eps


# ── 파손 기준 ────────────────────────────────────────────────────────────────

MODES = ("fiber_tension", "fiber_compression",
         "transverse_tension", "transverse_compression", "in_plane_shear")


def max_stress(sigma_12: np.ndarray, Xt: float, Xc: float, Yt: float, Yc: float, S: float) -> dict:
    """Max Stress — 성분별 지수와 지배 모드. FI ≥ 1 이면 파손."""
    s1, s2, t12 = float(sigma_12[0]), float(sigma_12[1]), float(sigma_12[2])
    terms = {
        "fiber_tension": s1 / Xt if s1 > 0 else 0.0,
        "fiber_compression": -s1 / Xc if s1 < 0 else 0.0,
        "transverse_tension": s2 / Yt if s2 > 0 else 0.0,
        "transverse_compression": -s2 / Yc if s2 < 0 else 0.0,
        "in_plane_shear": abs(t12) / S,
    }
    mode = max(terms, key=terms.get)
    return {"criterion": "max_stress", "failure_index": terms[mode], "mode": mode, "terms": terms}


def tsai_wu(sigma_12: np.ndarray, Xt: float, Xc: float, Yt: float, Yc: float, S: float) -> dict:
    """Tsai-Wu (평면응력) — 강도비 R (하중 R배에서 파손면 도달), FI = 1/R.

    F12 = −½√(F11 F22) (표준 기본 상호작용 계수 — 응답 assumption에 명시).
    """
    s1, s2, t12 = float(sigma_12[0]), float(sigma_12[1]), float(sigma_12[2])
    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S * S)
    F12 = -0.5 * math.sqrt(F11 * F22)

    a = (F11 * s1 * s1 + F22 * s2 * s2 + F66 * t12 * t12 + 2.0 * F12 * s1 * s2)  # R² 계수
    b = F1 * s1 + F2 * s2                                                        # R 계수
    if a <= 0.0 and b <= 0.0:
        return {"criterion": "tsai_wu", "strength_ratio": None, "failure_index": 0.0,
                "note": "무응력 또는 파손면 도달 불가 방향"}
    if a <= 1e-300:
        R = 1.0 / b
    else:
        R = (-b + math.sqrt(b * b + 4.0 * a)) / (2.0 * a)
    return {"criterion": "tsai_wu", "strength_ratio": R, "failure_index": 1.0 / R if R > 0 else None}


def assess_ply(sigma_12: np.ndarray, strength: tuple[float, float, float, float, float]) -> dict:
    """두 기준 동시 평가 — Tsai-Wu R을 주지표로, Max Stress 지배 모드를 설명자로."""
    Xt, Xc, Yt, Yc, S = strength
    ms = max_stress(sigma_12, Xt, Xc, Yt, Yc, S)
    tw = tsai_wu(sigma_12, Xt, Xc, Yt, Yc, S)
    return {"tsai_wu": tw, "max_stress": ms,
            "governing_mode": ms["mode"],
            "fails": bool(tw.get("strength_ratio") is not None and tw["strength_ratio"] <= 1.0)}
