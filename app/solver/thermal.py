# 열탄성 확장 — CTE 변환, 열하중, 자유 열응답, 잔류응력, 휨(coplanarity), ROM 균질화 (계획서 §17.1)
from __future__ import annotations

import numpy as np

from app.solver import response as RESP


def alpha_vector(alpha1: float, alpha2: float, angle_deg: float) -> np.ndarray:
    """재료축 CTE(α1, α2) → 적층판축 [αx, αy, αxy] (공학 전단 규약, §17.1)."""
    th = np.deg2rad(angle_deg)
    m, n = np.cos(th), np.sin(th)
    return np.array([
        alpha1 * m * m + alpha2 * n * n,
        alpha1 * n * n + alpha2 * m * m,
        2.0 * m * n * (alpha1 - alpha2),
    ], dtype=np.float64)


def thermal_loads(qbars: list[np.ndarray], alphas: list[np.ndarray],
                  z: np.ndarray, delta_t: float) -> tuple[np.ndarray, np.ndarray]:
    """N_th = ΔT Σ Q̄α t_k, M_th = ΔT Σ Q̄α z̄_k t_k."""
    N = np.zeros(3)
    M = np.zeros(3)
    for k, (qb, a) in enumerate(zip(qbars, alphas)):
        t_k = float(z[k + 1] - z[k])
        zbar = float(z[k + 1] + z[k]) / 2.0
        qa = qb @ a
        N += qa * t_k
        M += qa * zbar * t_k
    return N * delta_t, M * delta_t


def thermal_response(A: np.ndarray, B: np.ndarray, D: np.ndarray,
                     N_th: np.ndarray, M_th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """자유 열변형: K[ε0;κ] = [N_th;M_th]."""
    return RESP.solve_response(A, B, D, N_th, M_th)


def residual_ply_stresses(qbars: list[np.ndarray], alphas: list[np.ndarray],
                          z: np.ndarray, eps0: np.ndarray, kappa: np.ndarray,
                          delta_t: float) -> list[np.ndarray]:
    """ply 중앙면 잔류응력 σ = Q̄(ε0 + z̄κ − αΔT). 평형(ΣF=ΣM=0)이 검증 불변식."""
    out = []
    for k, (qb, a) in enumerate(zip(qbars, alphas)):
        zbar = float(z[k + 1] + z[k]) / 2.0
        out.append(qb @ (eps0 + zbar * kappa - a * delta_t))
    return out


def warpage_over_panel(kappa: np.ndarray, lx: float, ly: float) -> dict:
    """곡률장 w(x,y) = −½(κx x² + κy y² + κxy xy)의 판(Lx×Ly, 원점 중심) 9점 범위.

    2차 곡면의 사각영역 극값은 모서리/변 극점에 위치 — 모서리4+변중점4+중심 평가로 근사
    (전형 케이스에서 정확). 순수 κx 실린더에서 range = |κx|Lx²/8 (§17.1 검증식).
    """
    kx, ky, kxy = float(kappa[0]), float(kappa[1]), float(kappa[2])
    xs = (-lx / 2.0, 0.0, lx / 2.0)
    ys = (-ly / 2.0, 0.0, ly / 2.0)
    w = [-(kx * x * x + ky * y * y + kxy * x * y) / 2.0 for x in xs for y in ys]
    return {"warpage_range": float(max(w) - min(w)),
            "w_center": float(-0.0),
            "w_corner": float(w[0])}


def homogenize_voigt(components: list[tuple[float, float, float, float | None, float | None]]
                     ) -> dict:
    """면내 병렬(Voigt) 균질화 — components: [(f, E, nu, alpha|None, rho|None)].

    E = Σf·E, ν = Σf·ν, α = Σf·E·α/Σf·E (힘 평형 가중), ρ = Σf·ρ.
    동박층 = {Cu f=동박률, 수지 f=1−동박률}. 상한(Voigt) 모델 (§17.1).
    """
    fsum = sum(f for f, *_ in components)
    E = sum(f * e for f, e, *_ in components)
    nu = sum(f * n for f, _, n, *_ in components)
    has_alpha = all(a is not None for _, _, _, a, _ in components)
    alpha = (sum(f * e * a for f, e, _, a, _ in components) / E) if has_alpha and E > 0 else None
    has_rho = all(r is not None for *_, r in components)
    rho = sum(f * r for f, *_, r in components) if has_rho else None
    return {"fraction_sum": fsum, "E": E, "nu": nu, "alpha": alpha, "rho": rho}
