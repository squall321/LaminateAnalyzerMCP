# 재료 강성 계산 — Q, Q̄, S̄11, E_x(θ)와 물성 유효성 (계획서 §4.1, §4.2). MCP/pydantic 비의존 순수 계층.
from __future__ import annotations

import numpy as np


def isotropic_to_orthotropic(E: float, nu: float) -> tuple[float, float, float, float]:
    """등방성 (E, ν) → 등가 직교이방 상수 (E1, E2, G12, ν12). 각도 무관."""
    G = E / (2.0 * (1.0 + nu))
    return E, E, G, nu


def check_orthotropic_validity(E1: float, E2: float, G12: float, nu12: float) -> str | None:
    """열역학적 안정 조건 검사. 위반 사유 문자열을 반환하고 유효하면 None (계획서 §4.1)."""
    if not (E1 > 0 and E2 > 0 and G12 > 0):
        return f"E1={E1}, E2={E2}, G12={G12} 중 양수가 아닌 값이 있습니다"
    nu21 = nu12 * E2 / E1
    delta = 1.0 - nu12 * nu21
    if delta <= 0:
        return f"1 - nu12*nu21 = {delta:.3e} <= 0 (|nu12| < sqrt(E1/E2) = {np.sqrt(E1 / E2):.4f} 필요)"
    return None


def q_matrix(E1: float, E2: float, G12: float, nu12: float) -> np.ndarray:
    """평면응력 감축 강성 Q (3x3, 재료 주축 기준). 유효성은 호출 전 검사 전제."""
    nu21 = nu12 * E2 / E1
    delta = 1.0 - nu12 * nu21
    Q = np.zeros((3, 3), dtype=np.float64)
    Q[0, 0] = E1 / delta
    Q[1, 1] = E2 / delta
    Q[0, 1] = Q[1, 0] = nu12 * E2 / delta
    Q[2, 2] = G12
    return Q


def qbar_matrix(Q: np.ndarray, angle_deg: float) -> np.ndarray:
    """적층각 θ(도, +x→1축 CCW 양수, D2 규약)로 변환한 Q̄ (공학 전단 규약, 계획서 §4.2)."""
    th = np.deg2rad(angle_deg)
    m, n = np.cos(th), np.sin(th)
    m2, n2 = m * m, n * n
    m3, n3 = m2 * m, n2 * n
    m4, n4 = m2 * m2, n2 * n2
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

    qb = np.zeros((3, 3), dtype=np.float64)
    qb[0, 0] = Q11 * m4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * n4
    qb[1, 1] = Q11 * n4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * m4
    qb[0, 1] = qb[1, 0] = (Q11 + Q22 - 4.0 * Q66) * m2 * n2 + Q12 * (m4 + n4)
    qb[2, 2] = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * m2 * n2 + Q66 * (m4 + n4)
    qb[0, 2] = qb[2, 0] = (Q11 - Q12 - 2.0 * Q66) * m3 * n + (Q12 - Q22 + 2.0 * Q66) * m * n3
    qb[1, 2] = qb[2, 1] = (Q11 - Q12 - 2.0 * Q66) * m * n3 + (Q12 - Q22 + 2.0 * Q66) * m3 * n
    return qb


def ex_engineering(E1: float, E2: float, G12: float, nu12: float, angle_deg: float) -> float:
    """각도 θ에서의 공학 축방향 탄성계수 E_x(θ) = 1/S̄11 (beam 모드용, 계획서 §4.2)."""
    th = np.deg2rad(angle_deg)
    m, n = np.cos(th), np.sin(th)
    S11 = 1.0 / E1
    S22 = 1.0 / E2
    S12 = -nu12 / E1
    S66 = 1.0 / G12
    sbar11 = S11 * m**4 + (2.0 * S12 + S66) * (m * n) ** 2 + S22 * n**4
    return 1.0 / sbar11


def normalize_angle_deg(angle_deg: float) -> float:
    """(-360, 360) 입력을 등가 적층각 (-90, 90]로 정규화 (D2 규약)."""
    a = ((angle_deg + 90.0) % 180.0) - 90.0
    return 90.0 if a == -90.0 else a
