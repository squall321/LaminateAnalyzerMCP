# 하중 응답 solver 테스트 — 유효 상수 폐형해, 누출/비틀림 지표, 특이계 (계획서 §4.7, §5.2)
import numpy as np
import pytest

from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import response as RESP
from tests.conftest import T300, ALU_E, ALU_NU, solve_stack


def test_effective_constants_isotropic_recover_inputs():
    h = 2.0e-3
    A, B, D, _ = solve_stack([(h, 0.0, MAT.isotropic_to_orthotropic(ALU_E, ALU_NU))])
    alpha, _, delta = RESP.compliance_blocks(A, B, D)
    eff = RESP.effective_constants(alpha, delta, h)
    assert eff["membrane"]["Ex"] == pytest.approx(ALU_E, rel=1e-10)
    assert eff["membrane"]["Ey"] == pytest.approx(ALU_E, rel=1e-10)
    assert eff["membrane"]["nu_xy"] == pytest.approx(ALU_NU, rel=1e-10)
    assert eff["membrane"]["Gxy"] == pytest.approx(ALU_E / (2 * (1 + ALU_NU)), rel=1e-10)
    assert eff["bending"]["Ex_b"] == pytest.approx(ALU_E, rel=1e-10)


def test_effective_constants_orthotropic_0deg():
    E1, E2, G12, nu12 = T300
    h = 1.0e-3
    A, B, D, _ = solve_stack([(h, 0.0, T300)])
    alpha, _, delta = RESP.compliance_blocks(A, B, D)
    eff = RESP.effective_constants(alpha, delta, h)
    assert eff["membrane"]["Ex"] == pytest.approx(E1, rel=1e-10)
    assert eff["membrane"]["Ey"] == pytest.approx(E2, rel=1e-10)
    assert eff["membrane"]["Gxy"] == pytest.approx(G12, rel=1e-10)
    assert eff["membrane"]["nu_xy"] == pytest.approx(nu12, rel=1e-10)


def test_leakage_zero_for_symmetric_positive_for_unsymmetric():
    t = 0.125e-3
    A, B, D, z = solve_stack([(t, 0.0, T300), (t, 90.0, T300), (t, 90.0, T300), (t, 0.0, T300)])
    h = float(z[-1] - z[0])
    assert RESP.membrane_bending_leakage(A, B, D, h) == pytest.approx(0.0, abs=1e-12)

    A2, B2, D2, z2 = solve_stack([(t, 0.0, T300), (t, 90.0, T300)])
    h2 = float(z2[-1] - z2[0])
    assert RESP.membrane_bending_leakage(A2, B2, D2, h2) > 0.01


def test_twist_under_bending_symmetric_angle_ply():
    # [30/-30/-30/30]: 대칭이지만 D16≠0 → 굽힘-비틀림 커플링 존재
    t = 0.2e-3
    A, B, D, _ = solve_stack([(t, 30.0, T300), (t, -30.0, T300), (t, -30.0, T300), (t, 30.0, T300)])
    tw = RESP.twist_under_bending(A, B, D)
    assert tw is not None and tw > 0.05
    # 크로스플라이는 D16=0 → 비틀림 없음
    A2, B2, D2, _ = solve_stack([(t, 0.0, T300), (t, 90.0, T300), (t, 90.0, T300), (t, 0.0, T300)])
    assert RESP.twist_under_bending(A2, B2, D2) == pytest.approx(0.0, abs=1e-9)


def test_solve_response_equilibrium_roundtrip():
    t = 0.125e-3
    A, B, D, _ = solve_stack([(t, 30.0, T300), (t, -45.0, T300), (t, 90.0, T300)])
    N = np.array([1000.0, -200.0, 50.0])
    M = np.array([3.0, 1.0, -0.5])
    eps0, kappa = RESP.solve_response(A, B, D, N, M)
    K = RESP.assemble_k(A, B, D)
    back = K @ np.concatenate([eps0, kappa])
    np.testing.assert_allclose(back, np.concatenate([N, M]), rtol=1e-9, atol=1e-9 * np.linalg.norm(N))


def test_singular_system_raises():
    Z = np.zeros((3, 3))
    with pytest.raises(RESP.SingularSystemError):
        RESP.solve_response(Z, Z, Z, np.array([1.0, 0, 0]), np.zeros(3))
    with pytest.raises(RESP.SingularSystemError):
        RESP.compliance_blocks(Z, Z, Z)


def test_principal_direction_single_off_axis_ply():
    E1, E2, G12, nu12 = T300
    theta = 30.0
    t = 1.0e-3
    plies = [(t, theta, T300)]
    _, _, _, z = solve_stack(plies)
    h = float(z[-1] - z[0])

    def build_at(phi):
        qb = [MAT.qbar_matrix(MAT.q_matrix(E1, E2, G12, nu12), theta - phi)]
        A, B, D = ABD.abd_matrices(qb, z)
        return A, B, D, h

    res = RESP.principal_membrane_direction(build_at)
    assert res["angle_deg_of_max_Ex"] == pytest.approx(theta, abs=0.51)  # 1° 그리드
    assert res["Ex_max"] == pytest.approx(E1, rel=1e-6)
    assert res["Ex_min"] < res["Ex_max"]
