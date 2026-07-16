# 폐형해 기준 케이스 R4/R5 — 대칭 크로스플라이와 반대칭 앵글플라이 (계획서 §8.1)
import numpy as np
import pytest

from app.solver import material as MAT
from tests.conftest import T300, solve_stack


def test_r4_symmetric_cross_ply(q_t300):
    t = 0.125e-3
    Q11, Q22 = q_t300[0, 0], q_t300[1, 1]
    # [0/90]s = [0/90/90/0], 최하단이 0°
    A, B, D, _ = solve_stack([(t, 0.0, T300), (t, 90.0, T300), (t, 90.0, T300), (t, 0.0, T300)])

    assert np.allclose(B, 0.0, atol=1e-9 * np.linalg.norm(A) * t)
    assert A[0, 0] == pytest.approx(2 * t * (Q11 + Q22), rel=1e-12)
    assert A[1, 1] == pytest.approx(A[0, 0], rel=1e-12)
    assert D[0, 0] == pytest.approx((14 * Q11 + 2 * Q22) * t**3 / 3.0, rel=1e-10)
    assert D[1, 1] == pytest.approx((2 * Q11 + 14 * Q22) * t**3 / 3.0, rel=1e-10)
    # 외측 0° ply가 굽힘 지배 → D11 > D22
    assert D[0, 0] > D[1, 1]


@pytest.mark.parametrize("theta", [15.0, 30.0, 45.0, 60.0])
def test_r5_antisymmetric_angle_ply(q_t300, theta):
    t = 0.2e-3
    qb = MAT.qbar_matrix(q_t300, theta)
    # [+θ/−θ], +θ가 최하단
    A, B, D, _ = solve_stack([(t, theta, T300), (t, -theta, T300)])

    scaleA = np.linalg.norm(A)
    scaleB = scaleA * t
    scaleD = scaleA * t * t
    assert abs(A[0, 2]) < 1e-9 * scaleA and abs(A[1, 2]) < 1e-9 * scaleA
    assert abs(D[0, 2]) < 1e-9 * scaleD and abs(D[1, 2]) < 1e-9 * scaleD
    assert abs(B[0, 0]) < 1e-9 * scaleB and abs(B[0, 1]) < 1e-9 * scaleB
    assert abs(B[1, 1]) < 1e-9 * scaleB and abs(B[2, 2]) < 1e-9 * scaleB
    assert B[0, 2] == pytest.approx(-t**2 * qb[0, 2], rel=1e-10)
    assert B[1, 2] == pytest.approx(-t**2 * qb[1, 2], rel=1e-10)
