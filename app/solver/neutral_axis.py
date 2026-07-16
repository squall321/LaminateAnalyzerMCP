# 중립면 계산 — beam_equivalent(E_x 가중 도심)와 clt_weighted(B11/A11, B22/A22) (계획서 §4.5).
from __future__ import annotations

import numpy as np


def beam_equivalent(ex_per_ply: list[float], thicknesses: list[float], z: np.ndarray) -> float:
    """환산단면 중립축: z_ns = Σ E_x·t·z̄ / Σ E_x·t (midplane 기준, §4.5a)."""
    ex = np.asarray(ex_per_ply, dtype=np.float64)
    t = np.asarray(thicknesses, dtype=np.float64)
    zbar = (z[:-1] + z[1:]) / 2.0
    return float(np.sum(ex * t * zbar) / np.sum(ex * t))


def clt_weighted(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """Q̄11/Q̄22 가중 도심과 동치인 (B11/A11, B22/A22) (§4.5b)."""
    return float(B[0, 0] / A[0, 0]), float(B[1, 1] / A[1, 1])


def representations(z_ns_mid: float, h: float) -> dict:
    """midplane 기준 값을 세 표기(midplane/bottom/무차원 ζ)로 병기 (D5 규약)."""
    return {
        "z_from_midplane": z_ns_mid,
        "z_from_bottom": z_ns_mid + h / 2.0,
        "zeta": (z_ns_mid + h / 2.0) / h,
    }
