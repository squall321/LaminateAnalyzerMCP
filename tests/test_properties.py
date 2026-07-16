# 성질 기반 테스트 P1~P7 — hypothesis 무작위 적층 (계획서 §8.2)
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from app.solver import abd as ABD
from app.solver import material as MAT
from tests.conftest import solve_stack

# E2 상한 30GPa·E1 하한 10GPa이면 sqrt(E1/E2) >= 0.577 > nu12 상한 0.4 → 항상 열역학적 유효
finite = dict(allow_nan=False, allow_infinity=False)
mat_st = st.tuples(
    st.floats(10e9, 300e9, **finite),
    st.floats(3e9, 30e9, **finite),
    st.floats(2e9, 30e9, **finite),
    st.floats(0.05, 0.4, **finite),
)
ply_st = st.tuples(st.floats(5e-5, 2e-3, **finite), st.floats(-90.0, 90.0, **finite), mat_st)
stack_st = st.lists(ply_st, min_size=1, max_size=8)


@st.composite
def stack_and_perm(draw):
    stack = draw(st.lists(ply_st, min_size=2, max_size=8))
    perm = draw(st.permutations(range(len(stack))))
    return stack, list(perm)


def _tols(A):
    return dict(rtol=1e-9, atol=1e-9 * np.linalg.norm(A))


@settings(max_examples=60, deadline=None)
@given(stack_and_perm())
def test_p1_membrane_stiffness_permutation_invariant(sp):
    stack, perm = sp
    A1, _, _, _ = solve_stack(stack)
    A2, _, _, _ = solve_stack([stack[i] for i in perm])
    np.testing.assert_allclose(A2, A1, **_tols(A1))


@settings(max_examples=60, deadline=None)
@given(stack_st)
def test_p2_reversal_flips_b_sign(stack):
    A1, B1, D1, z = solve_stack(stack)
    A2, B2, D2, _ = solve_stack(list(reversed(stack)))
    h = float(z[-1] - z[0])
    np.testing.assert_allclose(A2, A1, **_tols(A1))
    np.testing.assert_allclose(D2, D1, rtol=1e-9, atol=1e-9 * np.linalg.norm(D1))
    np.testing.assert_allclose(B2, -B1, rtol=1e-9, atol=1e-9 * np.linalg.norm(A1) * h)


@settings(max_examples=60, deadline=None)
@given(stack_st, st.floats(1.5, 5.0, **finite))
def test_p3_uniform_scaling(stack, alpha):
    A1, B1, D1, z1 = solve_stack(stack)
    scaled = [(alpha * t, a, m) for t, a, m in stack]
    A2, B2, D2, z2 = solve_stack(scaled)
    h1, h2 = float(z1[-1] - z1[0]), float(z2[-1] - z2[0])
    np.testing.assert_allclose(A2, alpha * A1, rtol=1e-10, atol=1e-10 * np.linalg.norm(A1))
    np.testing.assert_allclose(B2, alpha**2 * B1, rtol=1e-10, atol=1e-10 * np.linalg.norm(A1) * h1)
    np.testing.assert_allclose(D2, alpha**3 * D1, rtol=1e-10, atol=1e-10 * np.linalg.norm(D1))
    # 정규화 K̂은 두께 균등 스케일에 불변 (계획서 §4.6)
    K1 = ABD.normalized_stiffness(A1, B1, D1, h1)[3]
    K2 = ABD.normalized_stiffness(A2, B2, D2, h2)[3]
    np.testing.assert_allclose(K2, K1, rtol=1e-8, atol=1e-8 * np.linalg.norm(K1))


@settings(max_examples=60, deadline=None)
@given(stack_st, st.floats(-90.0, 90.0, **finite))
def test_p4_rotation_invariants(stack, phi):
    """전 ply 공통 회전 φ에 대해 I1=M11+M22+2M12, I2=M66−M12 불변 (A/B/D 각각)."""
    res1 = solve_stack(stack)
    rotated = [(t, a + phi, m) for t, a, m in stack]
    res2 = solve_stack(rotated)
    scale = np.linalg.norm(res1[0])
    z = res1[3]
    h = float(z[-1] - z[0])
    for idx, s in ((0, scale), (1, scale * h), (2, scale * h * h)):
        M1, M2 = res1[idx], res2[idx]
        i1_1 = M1[0, 0] + M1[1, 1] + 2 * M1[0, 1]
        i1_2 = M2[0, 0] + M2[1, 1] + 2 * M2[0, 1]
        i2_1 = M1[2, 2] - M1[0, 1]
        i2_2 = M2[2, 2] - M2[0, 1]
        assert abs(i1_2 - i1_1) < 1e-9 * s + 1e-9 * abs(i1_1)
        assert abs(i2_2 - i2_1) < 1e-9 * s + 1e-9 * abs(i2_1)


@settings(max_examples=60, deadline=None)
@given(st.lists(ply_st, min_size=1, max_size=4))
def test_p5_symmetric_stack_decouples(half):
    stack = half + list(reversed(half))
    A, B, D, z = solve_stack(stack)
    h = float(z[-1] - z[0])
    A_hat, B_hat, _, _ = ABD.normalized_stiffness(A, B, D, h)
    assert np.linalg.norm(B_hat) <= 1e-9 * np.linalg.norm(A_hat)
    from app.solver import neutral_axis as NA
    z_x, z_y = NA.clt_weighted(A, B)
    assert abs(z_x) < 1e-9 * h and abs(z_y) < 1e-9 * h


@settings(max_examples=60, deadline=None)
@given(stack_st)
def test_p6_positive_definite(stack):
    A, B, D, z = solve_stack(stack)
    h = float(z[-1] - z[0])
    K_hat = ABD.normalized_stiffness(A, B, D, h)[3]
    assert ABD.is_positive_definite(K_hat)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.floats(5e-5, 2e-3, **finite),
                          st.floats(1.0, 89.0, **finite),
                          mat_st),
                min_size=1, max_size=4),
       st.randoms(use_true_random=False))
def test_p7_balanced_stack_kills_a16(pairs, rng):
    """±θ 짝(동일 두께·재료)으로 구성된 balanced 적층 → A16=A26≈0 (순서 무관)."""
    stack = []
    for t, a, m in pairs:
        stack.append((t, a, m))
        stack.append((t, -a, m))
    rng.shuffle(stack)
    A, _, _, _ = solve_stack(stack)
    assert abs(A[0, 2]) < 1e-9 * np.linalg.norm(A)
    assert abs(A[1, 2]) < 1e-9 * np.linalg.norm(A)
