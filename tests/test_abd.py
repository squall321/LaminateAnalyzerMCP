# ABD 적분·정규화 K̂ 테스트 — 폐형해 R1/R2와 스케일링 법칙 (계획서 §8.1, §4.6)
import numpy as np
import pytest

from app.solver import abd as ABD
from app.solver import material as MAT
from tests.conftest import T300, ALU_E, ALU_NU, solve_stack, assert_close


def test_r1_single_isotropic_ply():
    h = 2.0e-3
    Q = MAT.q_matrix(*MAT.isotropic_to_orthotropic(ALU_E, ALU_NU))
    A, B, D, z = solve_stack([(h, 0.0, MAT.isotropic_to_orthotropic(ALU_E, ALU_NU))])
    assert_close(A, h * Q)
    assert np.allclose(B, 0.0, atol=1e-12 * np.linalg.norm(A) * h)
    assert_close(D, h**3 / 12.0 * Q)


def test_r2_single_orthotropic_ply_0deg(q_t300):
    h = 1.0e-3
    A, B, D, _ = solve_stack([(h, 0.0, T300)])
    assert_close(A, h * q_t300)
    assert np.allclose(B, 0.0, atol=1e-12 * np.linalg.norm(A) * h)
    assert_close(D, h**3 / 12.0 * q_t300)


def test_r3_cross_ply_unsymmetric_abd(q_t300):
    t = 0.125e-3
    Q11, Q12, Q22, Q66 = q_t300[0, 0], q_t300[0, 1], q_t300[1, 1], q_t300[2, 2]
    A, B, D, _ = solve_stack([(t, 0.0, T300), (t, 90.0, T300)])  # [0/90], 0°가 최하단

    assert A[0, 0] == pytest.approx((Q11 + Q22) * t, rel=1e-12)
    assert A[1, 1] == pytest.approx((Q11 + Q22) * t, rel=1e-12)
    assert A[0, 1] == pytest.approx(2 * Q12 * t, rel=1e-12)
    assert A[2, 2] == pytest.approx(2 * Q66 * t, rel=1e-12)
    assert abs(A[0, 2]) < 1e-9 * A[0, 0]

    assert B[0, 0] == pytest.approx((Q22 - Q11) * t**2 / 2.0, rel=1e-10)
    assert B[1, 1] == pytest.approx(-B[0, 0], rel=1e-10)
    assert abs(B[0, 1]) < 1e-9 * abs(B[0, 0])
    assert abs(B[2, 2]) < 1e-9 * abs(B[0, 0])

    assert D[0, 0] == pytest.approx((Q11 + Q22) * t**3 / 3.0, rel=1e-10)
    assert D[1, 1] == pytest.approx(D[0, 0], rel=1e-10)


def test_z_coordinates_bottom_first():
    z = ABD.z_coordinates([1.0e-3, 2.0e-3, 1.0e-3])
    np.testing.assert_allclose(z, [-2.0e-3, -1.0e-3, 1.0e-3, 2.0e-3], rtol=1e-15)


def test_scaling_law(q_t300):
    plies = [(0.125e-3, 0.0, T300), (0.125e-3, 90.0, T300)]
    plies2 = [(2 * t, a, m) for t, a, m in plies]
    A1, B1, D1, _ = solve_stack(plies)
    A2, B2, D2, _ = solve_stack(plies2)
    assert_close(A2, 2.0 * A1, rtol=1e-12)
    assert_close(B2, 4.0 * B1, rtol=1e-12)
    assert_close(D2, 8.0 * D1, rtol=1e-12)


def test_normalized_khat_congruence_and_scale_invariance():
    plies = [(0.125e-3, 30.0, T300), (0.25e-3, -45.0, T300), (0.125e-3, 0.0, T300)]
    A, B, D, z = solve_stack(plies)
    h = float(z[-1] - z[0])
    A_hat, B_hat, D_hat, K_hat = ABD.normalized_stiffness(A, B, D, h)

    # 합동변환 정확성: K̂ == S·[A B; B D]·S
    S = np.zeros((6, 6))
    S[:3, :3] = np.eye(3) / np.sqrt(h)
    S[3:, 3:] = np.eye(3) * np.sqrt(12.0 / h**3)
    K_full = np.block([[A, B], [B, D]])
    assert_close(K_hat, S @ K_full @ S, rtol=1e-12)

    # 양정치 보존
    assert ABD.is_positive_definite(K_hat)

    # 두께 균등 스케일 불변성 (계획서 P3)
    plies3 = [(3 * t, a, m) for t, a, m in plies]
    A3, B3, D3, z3 = solve_stack(plies3)
    h3 = float(z3[-1] - z3[0])
    _, _, _, K_hat3 = ABD.normalized_stiffness(A3, B3, D3, h3)
    assert_close(K_hat3, K_hat, rtol=1e-9)
    assert ABD.condition_number(K_hat3) == pytest.approx(ABD.condition_number(K_hat), rel=1e-6)


def test_single_homogeneous_ply_hat_equals_q(q_t300):
    # 균질 단일재 판에서 Â = D̂ = Q̄, B̂ = 0 (정규화의 물리적 앵커, §4.6)
    h = 0.7e-3
    A, B, D, _ = solve_stack([(h, 0.0, T300)])
    A_hat, B_hat, D_hat, _ = ABD.normalized_stiffness(A, B, D, h)
    assert_close(A_hat, q_t300, rtol=1e-12)
    assert_close(D_hat, q_t300, rtol=1e-12)
    assert np.allclose(B_hat, 0.0, atol=1e-9 * np.linalg.norm(A_hat))
