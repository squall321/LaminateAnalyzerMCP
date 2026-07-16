# Q/Q̄/E_x(θ) 단위 테스트 — 항등식과 변환 대칭성 (계획서 §4.1, §4.2)
import numpy as np
import pytest

from app.solver import material as MAT
from tests.conftest import T300, ALU_E, ALU_NU


def test_q_matrix_identities(q_t300):
    E1, E2, G12, nu12 = T300
    nu21 = nu12 * E2 / E1
    delta = 1.0 - nu12 * nu21
    assert q_t300[0, 0] == pytest.approx(E1 / delta, rel=1e-14)
    assert q_t300[1, 1] == pytest.approx(E2 / delta, rel=1e-14)
    assert q_t300[0, 1] == pytest.approx(nu12 * E2 / delta, rel=1e-14)
    assert q_t300[2, 2] == pytest.approx(G12, rel=1e-14)
    assert q_t300[0, 2] == 0.0 and q_t300[1, 2] == 0.0
    np.testing.assert_allclose(q_t300, q_t300.T)


def test_isotropic_reduction():
    E1, E2, G12, nu12 = MAT.isotropic_to_orthotropic(ALU_E, ALU_NU)
    Q = MAT.q_matrix(E1, E2, G12, nu12)
    assert Q[0, 0] == pytest.approx(ALU_E / (1 - ALU_NU**2), rel=1e-14)
    assert Q[2, 2] == pytest.approx(ALU_E / (2 * (1 + ALU_NU)), rel=1e-14)
    # 등방 재료는 어떤 각도로 돌려도 Q̄ = Q
    for ang in (0.0, 17.3, 45.0, -60.0, 90.0):
        np.testing.assert_allclose(MAT.qbar_matrix(Q, ang), Q, rtol=1e-12, atol=1e-3)


def test_qbar_at_principal_angles(q_t300):
    qb0 = MAT.qbar_matrix(q_t300, 0.0)
    np.testing.assert_allclose(qb0, q_t300, rtol=1e-14, atol=1e-6)
    qb90 = MAT.qbar_matrix(q_t300, 90.0)
    assert qb90[0, 0] == pytest.approx(q_t300[1, 1], rel=1e-12)  # 11↔22 교환
    assert qb90[1, 1] == pytest.approx(q_t300[0, 0], rel=1e-12)
    assert qb90[0, 1] == pytest.approx(q_t300[0, 1], rel=1e-12)
    assert qb90[2, 2] == pytest.approx(q_t300[2, 2], rel=1e-12)
    assert abs(qb90[0, 2]) < 1e-6 * q_t300[0, 0]


def test_qbar16_odd_in_angle(q_t300):
    for ang in (15.0, 30.0, 45.0, 75.0):
        qb_p = MAT.qbar_matrix(q_t300, ang)
        qb_m = MAT.qbar_matrix(q_t300, -ang)
        assert qb_p[0, 2] == pytest.approx(-qb_m[0, 2], rel=1e-10)
        assert qb_p[1, 2] == pytest.approx(-qb_m[1, 2], rel=1e-10)
        # 짝수 성분은 부호 불변
        assert qb_p[0, 0] == pytest.approx(qb_m[0, 0], rel=1e-12)


def test_ex_engineering_limits():
    E1, E2, G12, nu12 = T300
    assert MAT.ex_engineering(E1, E2, G12, nu12, 0.0) == pytest.approx(E1, rel=1e-12)
    assert MAT.ex_engineering(E1, E2, G12, nu12, 90.0) == pytest.approx(E2, rel=1e-12)
    # 중간 각도는 두 극한 사이보다 작을 수도 있으나(전단 지배) 항상 양수
    assert MAT.ex_engineering(E1, E2, G12, nu12, 45.0) > 0


def test_orthotropic_validity():
    assert MAT.check_orthotropic_validity(*T300) is None
    assert MAT.check_orthotropic_validity(-1e9, 10e9, 5e9, 0.3) is not None   # E1<=0
    assert MAT.check_orthotropic_validity(10e9, 10e9, 5e9, 1.5) is not None   # delta<=0


@pytest.mark.parametrize("raw,expected", [
    (45.0, 45.0), (135.0, -45.0), (-135.0, 45.0), (180.0, 0.0),
    (90.0, 90.0), (-90.0, 90.0), (270.0, 90.0), (0.0, 0.0),
])
def test_normalize_angle(raw, expected):
    assert MAT.normalize_angle_deg(raw) == pytest.approx(expected)
