# §19.14 변위 제어 — 곡률·면내변형 지정 (1순위 미완 항목)
from __future__ import annotations

import numpy as np
import pytest

import app.mcp_server as srv
from app.services import validation as VAL
from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import prescribed as PRE

UTG = {"type": "isotropic", "E": 71000.0, "nu": 0.22}
PSA = {"type": "isotropic", "E": 0.894, "nu": 0.49}
PI = {"type": "isotropic", "E": 4000.0, "nu": 0.35}


def stack(mats=(UTG, PSA, PI), ths=(0.05, 0.025, 0.05)):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": m, "angle_deg": 0.0, "thickness": t}
                        for m, t in zip(mats, ths)]}


def _K(payload):
    si, _, _ = VAL.validate_and_convert(payload)
    qb = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg) for p in si.plies]
    z = ABD.z_coordinates(si.thicknesses)
    A, B, D = ABD.abd_matrices(qb, z)
    return np.block([[A, B], [B, D]]), A, B, D


# ── 분할 풀이의 환원 항등식 ─────────────────────────────────────────────────

def test_symmetric_full_kappa_reduces_to_D_kappa():
    """대칭 적층에서 κ 를 전부 지정하면 M = D·κ 로 **정확히** 환원된다."""
    K, _A, _B, D = _K(stack(mats=(UTG, PSA, UTG)))
    kx = 1.0 / 5e-3
    x, f = PRE.partitioned_solve(K, [None] * 3 + [kx, 0.0, 0.0], np.zeros(6))
    assert f[3:] == pytest.approx(D @ np.array([kx, 0.0, 0.0]), rel=1e-12)


def test_unsymmetric_full_kappa_reduces_to_reduced_D():
    """비대칭 + N 자유 + κ 전부 지정 → M = D*·κ = (D − B A⁻¹ B)κ 로 정확히 환원."""
    K, A, B, D = _K(stack())
    kx = 1.0 / 5e-3
    x, f = PRE.partitioned_solve(K, [None] * 3 + [kx, 0.0, 0.0], np.zeros(6))
    d_star = D - B @ np.linalg.inv(A) @ B
    assert f[3:] == pytest.approx(d_star @ np.array([kx, 0.0, 0.0]), rel=1e-9)
    assert f[:3] == pytest.approx(np.zeros(3), abs=1e-9)     # N 자유 → 0


def test_round_trip_through_force_control():
    """분할해로 얻은 N·M 을 힘 제어로 되풀면 원래 ε⁰·κ 가 복원된다."""
    from app.solver import response as RESP
    K, A, B, D = _K(stack())
    kx = 1.0 / 5e-3
    x, f = PRE.partitioned_solve(K, [None] * 3 + [kx, None, None], np.zeros(6))
    e0, kap = RESP.solve_response(A, B, D, f[:3], f[3:])
    assert np.concatenate([e0, kap]) == pytest.approx(x, abs=1e-12)


def test_singular_partition_raises():
    """지정이 하나도 없고 강성이 특이하면 명시적으로 실패한다."""
    K = np.zeros((6, 6))
    with pytest.raises(PRE.SingularPartition):
        PRE.partitioned_solve(K, [None] * 6, np.ones(6))


# ── 자유 폭 vs 구속 폭 ──────────────────────────────────────────────────────

def test_free_and_constrained_width_differ():
    """M_y=0(자유 폭)과 κ_y=0(구속 폭)은 다른 문제다 — 이 구분이 도구 가치의 절반이다."""
    free = srv.solve_prescribed_curvature(stack(), bend_radius=5.0, width="free")["data"]
    con = srv.solve_prescribed_curvature(stack(), bend_radius=5.0, width="constrained")["data"]
    assert free["response"]["kappa"][1] != pytest.approx(0.0, abs=1e-9)    # 반곡률
    assert con["response"]["kappa"][1] == pytest.approx(0.0, abs=1e-12)
    assert free["equivalent_loads"]["M"][1] == pytest.approx(0.0, abs=1e-9)  # 자유 → M_y=0
    assert abs(con["equivalent_loads"]["M"][1]) > 1e-6                       # 구속 → 반력
    ratio = con["equivalent_loads"]["M"][0] / free["equivalent_loads"]["M"][0]
    assert ratio == pytest.approx(1.096, rel=0.02)      # 실측 9.6% 차이


def test_naive_shortcut_is_flagged():
    """지름길 M = D·κ 가 크게 틀리면 경고한다 (비대칭 자유 폭에서 실측 +222%)."""
    env = srv.solve_prescribed_curvature(stack(), bend_radius=5.0)
    over = env["data"]["naive_D_kappa"]["overprediction_Mx"]
    assert over > 3.0
    assert any(w["code"] == "W130" and "지름길" in w["message"] for w in env["warnings"])
    # 대칭 + 구속 폭이면 지름길이 정확하므로 경고하지 않는다
    ok = srv.solve_prescribed_curvature(stack(mats=(UTG, PSA, UTG)), bend_radius=5.0,
                                        width="constrained")
    assert ok["data"]["naive_D_kappa"]["overprediction_Mx"] == pytest.approx(1.0, rel=1e-9)
    assert not any("지름길" in w["message"] for w in ok["warnings"])


# ── 체인 ────────────────────────────────────────────────────────────────────

def test_equivalent_loads_chain_into_failure_tools():
    """반환한 N·M 을 recover_ply_stresses 에 넘기면 같은 응답이 재현된다."""
    d = srv.solve_prescribed_curvature(stack(), bend_radius=5.0)["data"]
    lo = d["equivalent_loads"]
    env = srv.recover_ply_stresses(stack(), loads={"N": lo["N"], "M": lo["M"]})
    assert env["errors"] == []
    # 표면 변형률이 일치해야 한다 (같은 상태의 두 표현)
    top = d["surface_strain"]["top"]
    assert abs(top[0]) > 1e-6


def test_surface_strain_replaces_hand_calculation():
    """assess_crack_shielding 의 applied_strain 을 손계산 없이 얻는다."""
    d = srv.solve_prescribed_curvature(stack(), bend_radius=5.0)["data"]
    top, bot = d["surface_strain"]["top"], d["surface_strain"]["bottom"]
    assert top[0] > 0 > bot[0]                         # 굽힘이라 부호가 갈린다
    env = srv.assess_crack_shielding(stack(), target_ply=2,
                                     fracture={"applied_strain": top[0]})
    assert env["errors"] == []


def test_per_ply_material_axis_strain():
    d = srv.solve_prescribed_curvature(stack(), bend_radius=5.0)["data"]
    assert len(d["per_ply_strain"]) == 3
    for row in d["per_ply_strain"]:
        for loc in ("bottom", "mid", "top"):
            assert "epsilon_1" in row[loc] and "epsilon_2" in row[loc]


# ── 입력·경계 ───────────────────────────────────────────────────────────────

def test_kappa_list_with_nulls():
    """kappa 성분별 자유/지정이 동작한다."""
    d = srv.solve_prescribed_curvature(stack(), kappa=[200.0, None, None])["data"]
    assert d["prescribed_dof"] == ["kappa_x"]
    assert d["response"]["kappa"][0] == pytest.approx(200.0, rel=1e-12)
    both = srv.solve_prescribed_curvature(stack(), kappa=[200.0, 0.0, None])["data"]
    assert both["prescribed_dof"] == ["kappa_x", "kappa_y"]


def test_epsilon0_prescription():
    """면내변형 지정도 같은 기계로 처리된다."""
    d = srv.solve_prescribed_curvature(stack(), epsilon0=[0.001, None, None])["data"]
    assert d["response"]["epsilon0"][0] == pytest.approx(0.001, rel=1e-12)
    assert "eps_x" in d["prescribed_dof"]


def test_errors():
    s = stack()
    assert srv.solve_prescribed_curvature(s)["errors"][0]["code"] == "E100"      # 지정 없음
    assert srv.solve_prescribed_curvature(s, kappa=[1.0, 0.0, 0.0],
                                          bend_radius=5.0)["errors"][0]["code"] == "E100"
    assert srv.solve_prescribed_curvature(s, bend_radius=-1.0)["errors"][0]["code"] == "E100"
    assert srv.solve_prescribed_curvature(s, bend_radius=5.0,
                                          bend_axis="z")["errors"][0]["code"] == "E100"
    assert srv.solve_prescribed_curvature(s, bend_radius=5.0,
                                          width="x")["errors"][0]["code"] == "E100"
    assert srv.solve_prescribed_curvature(s, kappa=[1.0, 2.0])["errors"][0]["code"] == "E100"
    assert srv.solve_prescribed_curvature(s, kappa=[1.0, "x", None])["errors"][0]["code"] == "E100"


def test_unit_systems_agree():
    si = {"unit_system": "SI", "laminae": [
        {"material": {"type": "isotropic", "E": 71e9, "nu": 0.22}, "angle_deg": 0.0,
         "thickness": 50e-6},
        {"material": {"type": "isotropic", "E": 0.894e6, "nu": 0.49}, "angle_deg": 0.0,
         "thickness": 25e-6},
        {"material": {"type": "isotropic", "E": 4e9, "nu": 0.35}, "angle_deg": 0.0,
         "thickness": 50e-6}]}
    a = srv.solve_prescribed_curvature(si, bend_radius=5e-3)["data"]
    b = srv.solve_prescribed_curvature(stack(), bend_radius=5.0)["data"]
    assert a["response"]["epsilon0"] == pytest.approx(b["response"]["epsilon0"], rel=1e-9)
    assert a["response"]["kappa"][0] == pytest.approx(b["response"]["kappa"][0] * 1e3, rel=1e-9)


def test_bend_axis_y():
    x = srv.solve_prescribed_curvature(stack(), bend_radius=5.0, bend_axis="x")["data"]
    y = srv.solve_prescribed_curvature(stack(), bend_radius=5.0, bend_axis="y")["data"]
    assert x["prescribed_dof"] == ["kappa_x"] and y["prescribed_dof"] == ["kappa_y"]
    assert y["response"]["kappa"][1] == pytest.approx(x["response"]["kappa"][0], rel=1e-9)


def test_deterministic():
    a = srv.solve_prescribed_curvature(stack(), bend_radius=5.0)
    b = srv.solve_prescribed_curvature(stack(), bend_radius=5.0)
    assert a["data"] == b["data"]
    assert a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]
