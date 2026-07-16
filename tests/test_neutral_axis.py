# 중립면 폐형해 테스트 — R3 [0/90]과 대칭 적층 (계획서 §8.1, §4.5)
import numpy as np
import pytest

from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import neutral_axis as NA
from tests.conftest import T300, solve_stack


def test_r3_clt_weighted(q_t300):
    t = 0.125e-3
    Q11, Q22 = q_t300[0, 0], q_t300[1, 1]
    A, B, D, _ = solve_stack([(t, 0.0, T300), (t, 90.0, T300)])
    z_x, z_y = NA.clt_weighted(A, B)
    expected_x = t * (Q22 - Q11) / (2.0 * (Q11 + Q22))
    assert z_x == pytest.approx(expected_x, rel=1e-10)
    assert z_y == pytest.approx(-expected_x, rel=1e-10)  # 대칭성: x/y 교환
    # 강성이 큰 0° ply(하단) 쪽으로 이동 → 음수
    assert z_x < 0


def test_r3_beam_equivalent():
    E1, E2, G12, nu12 = T300
    t = 0.125e-3
    plies = [(t, 0.0, T300), (t, 90.0, T300)]
    z = ABD.z_coordinates([t, t])
    ex = [MAT.ex_engineering(E1, E2, G12, nu12, a) for _, a, _ in plies]
    z_ns = NA.beam_equivalent(ex, [t, t], z)
    expected = t * (E2 - E1) / (2.0 * (E1 + E2))
    assert z_ns == pytest.approx(expected, rel=1e-10)


def test_symmetric_stack_all_definitions_zero():
    t = 0.125e-3
    plies = [(t, 0.0, T300), (t, 90.0, T300), (t, 90.0, T300), (t, 0.0, T300)]
    A, B, D, z = solve_stack(plies)
    h = float(z[-1] - z[0])
    z_x, z_y = NA.clt_weighted(A, B)
    assert abs(z_x) < 1e-12 * h and abs(z_y) < 1e-12 * h
    E1, E2, G12, nu12 = T300
    ex = [MAT.ex_engineering(E1, E2, G12, nu12, a) for _, a, _ in plies]
    assert abs(NA.beam_equivalent(ex, [p[0] for p in plies], z)) < 1e-12 * h


def test_representations():
    h = 1.0e-3
    rep = NA.representations(-0.25e-3, h)
    assert rep["z_from_midplane"] == pytest.approx(-0.25e-3)
    assert rep["z_from_bottom"] == pytest.approx(0.25e-3)
    assert rep["zeta"] == pytest.approx(0.25)
