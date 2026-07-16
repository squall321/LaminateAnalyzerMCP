# 독립 오라클 — sympy 변환행렬(T·Reuter) 방식으로 CLT를 재구현. 운영 코드(app/*)를 import하지 않는다 (계획서 §8.3).
#
# 운영 엔진은 Q̄ 전개식(계획서 §4.2)을 쓰지만, 오라클은 의도적으로 다른 경로인
# Q̄ = T⁻¹ · Q · R · T · R⁻¹ (T: 응력 변환, R = diag(1,1,2) Reuter) 를 쓴다.
# 알고리즘·표현이 달라야 공통 오류(수식 오타)가 상쇄되지 않는다.
from __future__ import annotations

import numpy as np
import sympy as sp

_PREC = 30


def _q_sym(E1, E2, G12, nu12) -> sp.Matrix:
    E1, E2, G12, nu12 = (sp.Float(x, _PREC) for x in (E1, E2, G12, nu12))
    nu21 = nu12 * E2 / E1
    d = 1 - nu12 * nu21
    return sp.Matrix([[E1 / d, nu12 * E2 / d, 0],
                      [nu12 * E2 / d, E2 / d, 0],
                      [0, 0, G12]])


def _qbar_sym(Q: sp.Matrix, angle_deg: float) -> sp.Matrix:
    th = sp.rad(sp.Float(angle_deg, _PREC))
    m, n = sp.cos(th), sp.sin(th)
    T = sp.Matrix([[m**2, n**2, 2 * m * n],
                   [n**2, m**2, -2 * m * n],
                   [-m * n, m * n, m**2 - n**2]])
    R = sp.diag(1, 1, 2)
    return T.inv() * Q * R * T * R.inv()


def _to_np(M: sp.Matrix) -> np.ndarray:
    return np.array([[float(M[i, j].evalf(_PREC)) for j in range(3)] for i in range(3)], dtype=np.float64)


def oracle_abd(plies: list[tuple[float, float, tuple[float, float, float, float]]]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """plies: [(t, angle_deg, (E1,E2,G12,nu12)), ...] (index 0 = 최하단) → (A, B, D)."""
    ts = [sp.Float(t, _PREC) for t, _, _ in plies]
    h = sum(ts)
    A, B, D = sp.zeros(3), sp.zeros(3), sp.zeros(3)
    z0 = -h / 2
    for t_f, (_, ang, mat) in zip(ts, plies):
        qb = _qbar_sym(_q_sym(*mat), ang)
        z1 = z0 + t_f
        A += qb * (z1 - z0)
        B += qb * (z1**2 - z0**2) / 2
        D += qb * (z1**3 - z0**3) / 3
        z0 = z1
    return _to_np(A), _to_np(B), _to_np(D)
