# 공용 픽스처 — 표준 물성(SI)과 solver 직접 호출 헬퍼
from __future__ import annotations

import numpy as np
import pytest

from app.solver import abd as ABD
from app.solver import material as MAT

# T300/5208급 흑연/에폭시 (Jones 계열 교과서 표준 물성, SI Pa)
T300 = (181.0e9, 10.3e9, 7.17e9, 0.28)  # (E1, E2, G12, nu12)
ALU_E, ALU_NU = 70.0e9, 0.33


def solve_stack(plies):
    """plies: [(t_m, angle_deg, (E1,E2,G12,nu12)), ...] (index 0 = 최하단) → (A, B, D, z)."""
    qbars = [MAT.qbar_matrix(MAT.q_matrix(*mat), ang) for _, ang, mat in plies]
    z = ABD.z_coordinates([t for t, _, _ in plies])
    A, B, D = ABD.abd_matrices(qbars, z)
    return A, B, D, z


@pytest.fixture
def t300():
    return T300


@pytest.fixture
def q_t300():
    return MAT.q_matrix(*T300)


def assert_close(actual, expected, rtol=1e-12, atol_scale=None):
    """행렬 비교 — atol은 expected 노름 기준 상대 스케일."""
    expected = np.asarray(expected, dtype=np.float64)
    scale = np.linalg.norm(expected)
    atol = (atol_scale if atol_scale is not None else rtol) * max(scale, 1e-300)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
