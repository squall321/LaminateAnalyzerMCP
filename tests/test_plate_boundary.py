# §19.13 임의 경계조건 Rayleigh-Ritz — S·C 조합 9종 (2순위 4번)
from __future__ import annotations

import math

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import plate_navier as NAV

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0,
        "nu12": 0.28, "rho": 1.6e-9}
PANEL = {"Lx": 300.0, "Ly": 300.0}


def lam(angles=(0.0, 90.0, 90.0, 0.0)):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": T300, "angle_deg": a, "thickness": 0.125} for a in angles]}


def iso_D(E=70e9, nu=0.3, h=2e-3):
    Dv = E * h ** 3 / (12 * (1 - nu ** 2))
    return np.array([[Dv, nu * Dv, 0.0], [nu * Dv, Dv, 0.0], [0.0, 0.0, (1 - nu) * Dv / 2]]), Dv


# ── 보 함수 검증 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair,lam1", [("SS", math.pi), ("CC", 4.730041), ("CS", 3.926602)])
def test_beam_eigenvalues_match_literature(pair, lam1):
    assert NAV._eigenvalue(pair, 1) == pytest.approx(lam1, rel=1e-6)


@pytest.mark.parametrize("pair", ["SS", "CC", "CS"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_q_equals_lambda_fourth(pair, n):
    """q = λ⁴ 는 S·C 어느 조합에서도 성립한다 — 수치적분의 독립 검증이다.

    부분적분의 경계항이 S(φ=φ''=0)·C(φ=φ'=0) 양쪽에서 0이기 때문이다.
    """
    _p, q = NAV.beam_pq(pair, n)
    assert q == pytest.approx(NAV._eigenvalue(pair, n) ** 4, rel=1e-8)


def test_boundary_normalization():
    assert NAV.normalize_boundary("simply_supported") == ("SS", "SS")
    assert NAV.normalize_boundary("clamped") == ("CC", "CC")
    assert NAV.normalize_boundary("CCSS") == ("CC", "SS")
    assert NAV.normalize_boundary("scss") == ("CS", "SS")      # 소문자·뒤집힘 정규화
    for bad in ("XX", "CC", "SSSSS", "SSSF", "CFCF", 5, None):
        assert NAV.normalize_boundary(bad) is None


def test_free_edge_is_rejected_not_approximated():
    """자유변은 1항 Ritz 가 강체 모드를 놓쳐 6.4배 비보수가 된다 — 거부해야 한다."""
    env = srv.compute_buckling(lam(), panel=PANEL, boundary="SSSF")
    assert env["errors"][0]["code"] == "E100"
    assert "자유변" in env["errors"][0]["message"]


# ── 문헌 대조 ───────────────────────────────────────────────────────────────

def test_ssss_is_exact_and_others_are_upper_bounds():
    """SSSS 는 Navier 정해와 정확히 일치하고, 나머지는 알려진 상계 오차를 갖는다."""
    Dm, Dv = iso_D()
    L = 0.5

    def k_of(bc):
        return NAV.scan_ritz_buckling(Dm, L, L, 0.0, bc, 8)["N_cr"] * L * L / (math.pi ** 2 * Dv)

    assert k_of("SSSS") == pytest.approx(4.0, rel=1e-9)          # 정해
    assert k_of("CCCC") == pytest.approx(10.07 * 1.066, rel=1e-2)  # 상계 +6.6%
    assert k_of("SSCC") == pytest.approx(6.97 * 1.116, rel=1e-2)   # 상계 +11.6%
    # 상계이므로 항상 정해보다 크다
    assert k_of("CCCC") > 10.07 and k_of("SSCC") > 6.97


def test_frequency_ratios_match_literature():
    """진동수는 정확도가 좋다 — CCCC 비 1.830 (문헌 1.83)."""
    Dm, _ = iso_D()
    L, rho = 0.5, 1600.0 * 2e-3
    f0 = math.sqrt(NAV.ritz_omega2(Dm, rho, L, L, "SSSS", 1, 1))
    assert math.sqrt(NAV.ritz_omega2(Dm, rho, L, L, "CCCC", 1, 1)) / f0 == pytest.approx(1.830, rel=1e-3)


@pytest.mark.parametrize("load_ratio", [0.0, 0.5, 1.0])
def test_ssss_reduces_to_navier(load_ratio):
    """SSSS 를 넣으면 기존 Navier 경로와 정확히 일치한다 (공짜 회귀 테스트)."""
    Dm, _ = iso_D()
    L = 0.5
    a = NAV.scan_ritz_buckling(Dm, L, L, load_ratio, "SSSS", 10)
    b = NAV.buckling_ncr(Dm, L, L, load_ratio)
    assert a["N_cr"] == pytest.approx(b["N_cr"], rel=1e-12)
    assert (a["mode_m"], a["mode_n"]) == (b["mode_m"], b["mode_n"])


# ── 물리 순서 ───────────────────────────────────────────────────────────────

def test_stiffness_ordering_is_physical():
    """구속이 많을수록 N_cr·f 가 크다: SSSS < 혼합 < CCCC."""
    vals = {}
    for bc in ("SSSS", "SSCS", "SSCC", "CSCS", "CCSS", "CCCC"):
        b = srv.compute_buckling(lam(), panel=PANEL, boundary=bc)["data"]
        f = srv.compute_natural_frequencies(lam(), panel=PANEL, n_modes=1,
                                            boundary=bc)["data"]["modes"][0]["f_hz"]
        vals[bc] = (b["N_cr"], f)
    assert vals["SSSS"][0] < vals["SSCC"][0] < vals["CCCC"][0]
    assert vals["SSSS"][1] < vals["CSCS"][1] < vals["CCCC"][1]
    assert vals["SSSS"][1] == min(v[1] for v in vals.values())
    assert vals["CCCC"][1] == max(v[1] for v in vals.values())


def test_axis_assignment_is_not_symmetric():
    """CCSS 와 SSCC 는 다른 결과여야 한다 — x변·y변 배정이 실제로 반영되는지."""
    a = srv.compute_buckling(lam(), panel=PANEL, boundary="CCSS")["data"]
    b = srv.compute_buckling(lam(), panel=PANEL, boundary="SSCC")["data"]
    assert a["N_cr"] != pytest.approx(b["N_cr"], rel=1e-6)
    assert a["boundary_pairs"] == {"x_edges": "CC", "y_edges": "SS"}
    assert b["boundary_pairs"] == {"x_edges": "SS", "y_edges": "CC"}


def test_upper_bound_warning_and_backward_compat():
    """비-SSSS 경계는 상계 경고를 붙이고, 기존 인자값은 그대로 동작한다."""
    env = srv.compute_buckling(lam(), panel=PANEL, boundary="CCSS")
    assert any(w["code"] == "W130" and "상계" in w["message"] for w in env["warnings"])
    plain = srv.compute_buckling(lam(), panel=PANEL)
    assert not any("상계" in w["message"] for w in plain["warnings"])
    # 별칭이 4글자 코드와 같은 답
    assert srv.compute_buckling(lam(), panel=PANEL,
                                boundary="clamped")["data"]["N_cr"] == pytest.approx(
        srv.compute_buckling(lam(), panel=PANEL, boundary="CCCC")["data"]["N_cr"], rel=1e-12)
    assert srv.compute_buckling(lam(), panel=PANEL)["data"]["N_cr"] == pytest.approx(
        srv.compute_buckling(lam(), panel=PANEL, boundary="SSSS")["data"]["N_cr"], rel=1e-12)


def test_deterministic():
    a = srv.compute_buckling(lam(), panel=PANEL, boundary="CSCS")
    b = srv.compute_buckling(lam(), panel=PANEL, boundary="CSCS")
    assert a["data"] == b["data"]
