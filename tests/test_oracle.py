# 이중 구현 대조 — numpy 엔진 vs sympy 독립 오라클, 결정론적 무작위 50케이스 (계획서 §8.3)
from __future__ import annotations

import random

import numpy as np

from tests.conftest import solve_stack
from tests.oracle.clt_sympy import oracle_abd

RNG_SEED = 20260716
N_CASES = 50


def _random_case(rng: random.Random):
    n = rng.randint(1, 6)
    plies = []
    for _ in range(n):
        t = rng.uniform(5e-5, 2e-3)
        ang = rng.uniform(-90.0, 90.0)
        E1 = rng.uniform(10e9, 300e9)
        E2 = rng.uniform(3e9, 30e9)
        G12 = rng.uniform(2e9, 30e9)
        nu12 = rng.uniform(0.05, 0.4)
        plies.append((t, ang, (E1, E2, G12, nu12)))
    return plies


def test_engine_matches_independent_oracle():
    rng = random.Random(RNG_SEED)
    for case_no in range(N_CASES):
        plies = _random_case(rng)
        A_e, B_e, D_e, z = solve_stack(plies)
        A_o, B_o, D_o = oracle_abd(plies)
        h = float(z[-1] - z[0])
        sA = np.linalg.norm(A_o)
        for label, eng, ora, atol in (
            ("A", A_e, A_o, 1e-9 * sA),
            ("B", B_e, B_o, 1e-9 * sA * h),
            ("D", D_e, D_o, 1e-9 * sA * h * h),
        ):
            np.testing.assert_allclose(
                eng, ora, rtol=1e-9, atol=atol,
                err_msg=f"case {case_no} {label} 불일치 (plies={plies})")
