# ABD 적분과 합동변환 정규화 K̂ — z좌표 생성 포함 (계획서 §4.3, §4.4, §4.6). 순수 numpy 계층.
from __future__ import annotations

import numpy as np

SQRT12 = float(np.sqrt(12.0))


def z_coordinates(thicknesses: list[float] | np.ndarray) -> np.ndarray:
    """midplane 기준 계면 좌표 z_0..z_n. laminae[0]=최하단 (D1 규약, §4.3)."""
    t = np.asarray(thicknesses, dtype=np.float64)
    h = float(t.sum())
    return np.concatenate(([-h / 2.0], -h / 2.0 + np.cumsum(t)))


def abd_matrices(qbars: list[np.ndarray], z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A/B/D 두께 적분 (계획서 §4.4)."""
    A = np.zeros((3, 3), dtype=np.float64)
    B = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((3, 3), dtype=np.float64)
    for k, qb in enumerate(qbars):
        z0, z1 = z[k], z[k + 1]
        A += qb * (z1 - z0)
        B += qb * (z1 * z1 - z0 * z0) / 2.0
        D += qb * (z1**3 - z0**3) / 3.0
    return A, B, D


def normalized_stiffness(A: np.ndarray, B: np.ndarray, D: np.ndarray, h: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """프로젝트 표준 정규화 Â=A/h, B̂=√12·B/h², D̂=12D/h³ 과 결합 K̂(6x6).

    B의 계수가 √12인 이유: K̂ = S·[A B; B D]·S (S=diag(h^-1/2·I, (12/h³)^1/2·I))의
    합동변환이 되어 양정치성이 원행렬과 정확히 보존된다 (계획서 §4.6).
    """
    A_hat = A / h
    B_hat = SQRT12 * B / (h * h)
    D_hat = 12.0 * D / (h**3)
    K_hat = np.block([[A_hat, B_hat], [B_hat, D_hat]])
    return A_hat, B_hat, D_hat, K_hat


def condition_number(K_hat: np.ndarray) -> float:
    """정규화 결합 강성의 2-노름 조건수 (계획서 §5.1 abd_condition_number)."""
    return float(np.linalg.cond(K_hat, 2))


def is_positive_definite(K_hat: np.ndarray) -> bool:
    """Cholesky 성공 여부로 양정치 판정 (대칭화 후, 계획서 §4.6)."""
    sym = 0.5 * (K_hat + K_hat.T)
    try:
        np.linalg.cholesky(sym)
        return True
    except np.linalg.LinAlgError:
        return False
