# 문헌 벤치마크 — 교과서 공표값·불변량 이론과의 독립 대조 (계획서 §8.4, P4 완결)
"""검증 정답의 4번째 출처. 앞선 세 출처(폐형해 수식·성질 기반·sympy 오라클)와 달리
**교과서에 인쇄된 값**과 **엔진과 무관한 이론 경로**로 대조한다.

- T300/5208 물성: E1=181 GPa, E2=10.3 GPa, G12=7.17 GPa, ν12=0.28
  (Jones, *Mechanics of Composite Materials* 2nd ed. 계열의 표준 예제 물성)
- Q 상수 공표값: Q11=181.8, Q12=2.897, Q22=10.35, Q66=7.17 GPa
- 준등방 적층의 유효 상수는 **Tsai–Pagano 불변량**으로 독립 계산한다 —
  U1=(3Q11+3Q22+2Q12+4Q66)/8, U4=(Q11+Q22+6Q12−4Q66)/8 →
  E_x = (U1²−U4²)/U1, ν_xy = U4/U1. 이 경로는 엔진의 Q̄ 변환·ABD 적분을 전혀 쓰지 않으므로
  '적층각을 돌려 적분하는' 엔진 경로 전체에 대한 교차 검증이 된다.
공표값 반올림을 감안해 허용오차는 rtol 5e-3 (계획서 §8.5의 문헌 대조 기준).
"""
from __future__ import annotations

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import material as MAT

LIT_RTOL = 5e-3
T300_SI = (181.0e9, 10.3e9, 7.17e9, 0.28)
# 교과서 공표 Q 상수 [GPa]
Q_PUBLISHED = {"Q11": 181.8, "Q12": 2.897, "Q22": 10.35, "Q66": 7.17}


def test_q_matrix_matches_published_values():
    """감축 강성 Q가 교과서 인쇄값과 일치 (rtol 5e-3)."""
    Q = MAT.q_matrix(*T300_SI) / 1e9
    assert Q[0, 0] == pytest.approx(Q_PUBLISHED["Q11"], rel=LIT_RTOL)
    assert Q[0, 1] == pytest.approx(Q_PUBLISHED["Q12"], rel=LIT_RTOL)
    assert Q[1, 1] == pytest.approx(Q_PUBLISHED["Q22"], rel=LIT_RTOL)
    assert Q[2, 2] == pytest.approx(Q_PUBLISHED["Q66"], rel=LIT_RTOL)


def _invariants(Q: np.ndarray) -> tuple[float, float]:
    """Tsai–Pagano 불변량 U1, U4 (엔진의 변환·적분 경로를 쓰지 않는 독립 계산)."""
    q11, q12, q22, q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    u1 = (3 * q11 + 3 * q22 + 2 * q12 + 4 * q66) / 8.0
    u4 = (q11 + q22 + 6 * q12 - 4 * q66) / 8.0
    return u1, u4


def test_quasi_isotropic_effective_constants_match_invariant_theory():
    """[0/45/-45/90]s 준등방 적층의 유효 E_x·ν_xy가 불변량 이론값과 일치.

    준등방이면 A11/h = U1, A12/h = U4 이므로 E_x = (U1²−U4²)/U1, ν = U4/U1.
    T300/5208에서 각각 약 69.7 GPa, 0.296 — 문헌에 널리 인용되는 값이다.
    """
    Q = MAT.q_matrix(*T300_SI)
    u1, u4 = _invariants(Q)
    ex_theory = (u1 * u1 - u4 * u4) / u1
    nu_theory = u4 / u1
    assert ex_theory / 1e9 == pytest.approx(69.7, rel=LIT_RTOL)      # 문헌 인용값
    assert nu_theory == pytest.approx(0.296, rel=LIT_RTOL)

    lam = {"unit_system": "SI", "name": "quasi_isotropic_literature",
           "laminae": [{"thickness": 0.125e-3, "angle_deg": a,
                        "material": {"type": "orthotropic_2d", "E1": T300_SI[0], "E2": T300_SI[1],
                                     "G12": T300_SI[2], "nu12": T300_SI[3]}}
                       for a in (0, 45, -45, 90, 90, -45, 45, 0)]}
    eff = srv.solve_load_response(lam, loads={"N": [1.0, 0, 0]},
                                  scan_principal_direction=False)["data"]["effective_constants"]
    assert eff["membrane"]["Ex"] == pytest.approx(ex_theory, rel=LIT_RTOL)
    assert eff["membrane"]["Ey"] == pytest.approx(ex_theory, rel=LIT_RTOL)   # 준등방 → 방향 무관
    assert eff["membrane"]["nu_xy"] == pytest.approx(nu_theory, rel=LIT_RTOL)
    # 전단도 등방 관계 G = E/(2(1+ν))를 만족해야 준등방
    assert eff["membrane"]["Gxy"] == pytest.approx(ex_theory / (2 * (1 + nu_theory)), rel=LIT_RTOL)


def test_quasi_isotropy_index_agrees_with_theory():
    """엔진의 quasi_isotropy_score가 이론적 준등방 적층에서 1.0."""
    lam = {"unit_system": "SI",
           "laminae": [{"thickness": 0.125e-3, "angle_deg": a,
                        "material": {"type": "orthotropic_2d", "E1": T300_SI[0], "E2": T300_SI[1],
                                     "G12": T300_SI[2], "nu12": T300_SI[3]}}
                       for a in (0, 45, -45, 90, 90, -45, 45, 0)]}
    idx = srv.evaluate_laminate(lam)["data"]["indices"]
    assert idx["quasi_isotropy_score"]["value"] == pytest.approx(1.0, abs=1e-9)
    assert idx["quasi_isotropy_score"]["grade"] == "quasi_isotropic"


def test_unidirectional_engineering_constants_recovered():
    """단방향 [0]8 적층의 유효 상수는 입력 ply 상수 그대로 (문헌 정의의 자명한 요구)."""
    lam = {"unit_system": "SI",
           "laminae": [{"thickness": 0.125e-3, "angle_deg": 0.0,
                        "material": {"type": "orthotropic_2d", "E1": T300_SI[0], "E2": T300_SI[1],
                                     "G12": T300_SI[2], "nu12": T300_SI[3]}} for _ in range(8)]}
    eff = srv.solve_load_response(lam, loads={"N": [1.0, 0, 0]},
                                  scan_principal_direction=False)["data"]["effective_constants"]
    assert eff["membrane"]["Ex"] == pytest.approx(T300_SI[0], rel=1e-9)
    assert eff["membrane"]["Ey"] == pytest.approx(T300_SI[1], rel=1e-9)
    assert eff["membrane"]["Gxy"] == pytest.approx(T300_SI[2], rel=1e-9)
    assert eff["membrane"]["nu_xy"] == pytest.approx(T300_SI[3], rel=1e-9)


def test_pm45_shear_dominant_stiffness():
    """[±45]s의 유효 E_x는 U1−U4 관계로 결정 — 앵글플라이 문헌 결과 대조.

    [±45]s에서 A11/h = U1 − U5(=(U1−U4)/2)·0 ... 실제로는 A11/h = U1 − U4 관계가 아니라
    Q̄ 평균이므로, 여기서는 '전단 지배로 준등방보다 낮고 90°판보다 높다'는 순서 관계를 검증한다.
    """
    def build(angles):
        return {"unit_system": "SI",
                "laminae": [{"thickness": 0.125e-3, "angle_deg": a,
                             "material": {"type": "orthotropic_2d", "E1": T300_SI[0],
                                          "E2": T300_SI[1], "G12": T300_SI[2], "nu12": T300_SI[3]}}
                            for a in angles]}
    ex = lambda angles: srv.solve_load_response(
        build(angles), loads={"N": [1.0, 0, 0]},
        scan_principal_direction=False)["data"]["effective_constants"]["membrane"]["Ex"]
    e_90 = ex((90, 90, 90, 90))
    e_pm45 = ex((45, -45, -45, 45))
    e_qi = ex((0, 45, -45, 90, 90, -45, 45, 0))
    e_0 = ex((0, 0, 0, 0))
    assert e_90 < e_pm45 < e_qi < e_0
    # ±45의 E_x는 전단계수 지배 — 4G12 근방(고전 근사)
    assert e_pm45 == pytest.approx(4 * T300_SI[2], rel=0.35)
