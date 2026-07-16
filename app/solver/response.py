# 하중 응답 — 완전 ABD 역산, 유효 공학 상수, 주강성 방향 (계획서 §4.7, §5.2). 순수 numpy 계층.
from __future__ import annotations

import numpy as np


class SingularSystemError(Exception):
    """강성 행렬 역산 불가 (E400으로 매핑)."""


def assemble_k(A: np.ndarray, B: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.block([[A, B], [B, D]])


def solve_response(A: np.ndarray, B: np.ndarray, D: np.ndarray,
                   N: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N; M] → (ε0, κ). 역행렬 대신 solve 사용 (계획서 §4.7)."""
    K = assemble_k(A, B, D)
    try:
        sol = np.linalg.solve(K, np.concatenate([N, M]))
    except np.linalg.LinAlgError as e:
        raise SingularSystemError(str(e)) from e
    if not np.all(np.isfinite(sol)):
        raise SingularSystemError("비유한 해")
    return sol[:3], sol[3:]


def compliance_blocks(A: np.ndarray, B: np.ndarray, D: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """완전 역산 컴플라이언스 [α β; βᵀ δ] (계획서 §4.7)."""
    K = assemble_k(A, B, D)
    try:
        C = np.linalg.inv(K)
    except np.linalg.LinAlgError as e:
        raise SingularSystemError(str(e)) from e
    return C[:3, :3], C[:3, 3:], C[3:, 3:]


def effective_constants(alpha: np.ndarray, delta: np.ndarray, h: float) -> dict:
    """막/굽힘 유효 공학 상수 (SI Pa). 비대칭 적층은 α·δ에 B 커플링 효과 포함 (§4.7)."""
    return {
        "membrane": {
            "Ex": 1.0 / (h * alpha[0, 0]),
            "Ey": 1.0 / (h * alpha[1, 1]),
            "Gxy": 1.0 / (h * alpha[2, 2]),
            "nu_xy": -alpha[0, 1] / alpha[0, 0],
        },
        "bending": {
            "Ex_b": 12.0 / (h**3 * delta[0, 0]),
            "Ey_b": 12.0 / (h**3 * delta[1, 1]),
        },
    }


def membrane_bending_leakage(A: np.ndarray, B: np.ndarray, D: np.ndarray, h: float) -> float:
    """단위 막하중 N=(1,0,0), M=0 에서 (h/2)·‖κ‖₂/‖ε0‖₂. 대칭이면 0 (§5.2)."""
    eps0, kappa = solve_response(A, B, D, np.array([1.0, 0.0, 0.0]), np.zeros(3))
    n_eps = float(np.linalg.norm(eps0))
    if n_eps == 0.0:
        return 0.0
    return float(0.5 * h * np.linalg.norm(kappa) / n_eps)


def twist_under_bending(A: np.ndarray, B: np.ndarray, D: np.ndarray) -> float | None:
    """M=(1,0,0), N=0 에서 |κ_xy|/|κ_x|. κ_x 소멸 시 None (§5.2)."""
    _, kappa = solve_response(A, B, D, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    if abs(kappa[0]) < 1e-300:
        return None
    return float(abs(kappa[2]) / abs(kappa[0]))


def principal_membrane_direction(build_abd_at, scan_step_deg: float = 1.0) -> dict:
    """E_x^eff(φ)를 φ∈[0°,180°) 그리드 스캔한 최대/최소 막강성 방향 (결정론적, §5.2).

    build_abd_at(phi) → (A, B, D, h): 전 ply를 -φ 회전시킨 적층의 ABD를 반환하는 콜백.
    """
    best = (None, -np.inf)
    worst = (None, np.inf)
    phi = 0.0
    while phi < 180.0:
        A, B, D, h = build_abd_at(phi)
        alpha, _, _ = compliance_blocks(A, B, D)
        ex = 1.0 / (h * alpha[0, 0])
        if ex > best[1]:
            best = (phi, ex)
        if ex < worst[1]:
            worst = (phi, ex)
        phi += scan_step_deg
    return {"angle_deg_of_max_Ex": best[0], "Ex_max": best[1],
            "angle_deg_of_min_Ex": worst[0], "Ex_min": worst[1]}
