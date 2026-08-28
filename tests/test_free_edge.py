# 자유 가장자리 박리 테스트 — O'Brien ERR + 계면별 구동력 (계획서 §19.1)
from __future__ import annotations

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import abd as ABD
from app.solver import free_edge as FE
from app.solver import material as MAT

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}


def lam(angles, t=0.125):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": T300, "angle_deg": a, "thickness": t} for a in angles]}


def _si(angles, t=0.125e-3):
    qb = [MAT.qbar_matrix(MAT.q_matrix(181e9, 10.3e9, 7.17e9, 0.28), a) for a in angles]
    return qb, [t] * len(angles)


# ── 폐형해 성질 ─────────────────────────────────────────────────────────────

def test_unidirectional_has_no_driving_force():
    """단일방향 적층은 부분적층 강성이 전체와 같아 ΔE = 0 → G = 0 (물리 감별)."""
    qb, th = _si((0.0,) * 4)
    e_lam, h = FE.sublaminate_axial_modulus(qb, th, 0, 4)
    for k in range(1, 4):
        e1, t1 = FE.sublaminate_axial_modulus(qb, th, 0, k)
        e2, t2 = FE.sublaminate_axial_modulus(qb, th, k, 4)
        e_star = (e1 * t1 + e2 * t2) / h
        assert e_star == pytest.approx(e_lam, rel=1e-12)
        assert FE.obrien_err(0.004, h, e_lam, e_star) == 0.0


@pytest.mark.parametrize("angles", [(0.0, 90.0, 90.0, 0.0), (45.0, -45.0, -45.0, 45.0),
                                    (30.0, -30.0, 90.0, 90.0, -30.0, 30.0),
                                    (0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0)])
def test_delaminated_stiffness_never_exceeds_intact(angles):
    """**대칭** 적층에서는 박리가 구속을 푸는 것이므로 E* ≤ E_LAM 이다 (G ≥ 0 의 근거).

    비대칭 적층에서는 성립하지 않는다 — 아래 test_unsymmetric_is_flagged_not_silently_zero 참조.
    """
    qb, th = _si(angles)
    n = len(angles)
    e_lam, h = FE.sublaminate_axial_modulus(qb, th, 0, n)
    for k in range(1, n):
        e1, t1 = FE.sublaminate_axial_modulus(qb, th, 0, k)
        e2, t2 = FE.sublaminate_axial_modulus(qb, th, k, n)
        assert (e1 * t1 + e2 * t2) / h <= e_lam * (1 + 1e-12)


def test_unsymmetric_is_flagged_not_silently_zero():
    """비대칭 적층은 ΔE 가 음수가 될 수 있다 — 0으로 잘라 '안전'으로 보이면 안 된다.

    B 커플링으로 온전한 적층의 막강성이 이미 연화돼 있어 갈라진 부분적층이 오히려
    더 뻣뻣하다(실측 [0/45/−45/90] 계면1 에서 ΔE = −31.6 GPa). 모델이 적용 불가라는
    뜻이지 구동력이 없다는 뜻이 아니므로 G = null 과 W130 으로 알려야 한다.
    """
    env = srv.assess_free_edge_delamination(lam((0.0, 45.0, -45.0, 90.0)),
                                            loads={"N": [150.0, 0.0, 0.0]},
                                            fracture={"G_c": 0.25})
    d = env["data"]
    assert any(r["delta_E"] < 0 for r in d["interfaces"])
    for r in d["interfaces"]:
        if r["delta_E"] < 0:
            assert r["model_valid"] is False
            assert r["G"] is None and r["onset_strain"] is None
    assert any(w["code"] == "W130" and "구동력이 없다는 뜻이 아니다" in w["message"]
               for w in env["warnings"])
    # 단일방향은 ΔE = 0 이고 이건 '구동력 없음'이 맞으므로 G = 0 (null 아님)
    uni = srv.assess_free_edge_delamination(lam((0.0,) * 4),
                                            loads={"N": [150.0, 0.0, 0.0]})["data"]
    assert all(r["model_valid"] and r["G"] == 0.0 for r in uni["interfaces"])


def test_err_scales_with_strain_squared():
    """G ∝ ε² (O'Brien 식의 정의)."""
    g1 = FE.obrien_err(0.004, 5e-4, 70e9, 40e9)
    g2 = FE.obrien_err(0.008, 5e-4, 70e9, 40e9)
    assert g2 / g1 == pytest.approx(4.0, rel=1e-12)


def test_onset_strain_scales_with_sqrt_toughness():
    """ε_c ∝ √G_c."""
    a = FE.onset_strain(250.0, 5e-4, 40e9)
    b = FE.onset_strain(1000.0, 5e-4, 40e9)
    assert b / a == pytest.approx(2.0, rel=1e-12)
    assert FE.onset_strain(250.0, 5e-4, 0.0) is None      # 구동력 없으면 정의 안 됨


# ── 계면별 구동력 판별 ──────────────────────────────────────────────────────

def test_angle_ply_is_shear_driven_cross_ply_is_peel_driven():
    """[±45]는 σy가 0이라 전단 지배, [0/90]는 peel 지배 — 물리가 갈리는 지점."""
    a = srv.assess_free_edge_delamination(lam((45.0, -45.0, -45.0, 45.0)),
                                          loads={"N": [50.0, 0.0, 0.0]})["data"]
    assert a["interfaces"][0]["dominant_driver"] == "in_plane_shear"
    assert abs(a["interfaces"][0]["driving"]["peel_moment"]) < 1e-9

    b = srv.assess_free_edge_delamination(lam((0.0, 90.0, 90.0, 0.0)),
                                          loads={"N": [150.0, 0.0, 0.0]})["data"]
    mid = [r for r in b["interfaces"] if r["interface"] == 2][0]
    assert mid["dominant_driver"] == "peel"
    assert abs(mid["driving"]["shear_force"]) < 1e-9


def test_identical_ply_interface_has_no_driver():
    """[±45]s 중앙면은 −45/−45 라 재료 불연속이 없다 → 구동력 none (G는 0이 아니어도)."""
    d = srv.assess_free_edge_delamination(lam((45.0, -45.0, -45.0, 45.0)),
                                          loads={"N": [50.0, 0.0, 0.0]})["data"]
    mid = [r for r in d["interfaces"] if r["interface"] == 2][0]
    assert mid["dominant_driver"] == "none"
    assert mid["G"] > 0.0          # 에너지는 있지만 구동력이 없다 — 이 구분이 이 도구의 요점


def test_symmetric_laminate_gives_mirror_interfaces():
    """대칭 적층은 계면 k 와 n−k 의 G 가 같아야 한다."""
    d = srv.assess_free_edge_delamination(lam((30.0, -30.0, 90.0, 90.0, -30.0, 30.0)),
                                          loads={"N": [150.0, 0.0, 0.0]})["data"]
    g = {r["interface"]: r["G"] for r in d["interfaces"]}
    assert g[1] == pytest.approx(g[5], rel=1e-12)
    assert g[2] == pytest.approx(g[4], rel=1e-12)


# ── Tool 계층 ───────────────────────────────────────────────────────────────

def test_onset_and_margin_with_toughness():
    """G_c 를 주면 계면별 개시 변형률과 여유율이 나오고 최소가 지배 계면이 된다."""
    env = srv.assess_free_edge_delamination(lam((30.0, -30.0, 90.0, 90.0, -30.0, 30.0)),
                                            loads={"N": [150.0, 0.0, 0.0]},
                                            fracture={"G_c": 0.25})
    d = env["data"]
    assert env["errors"] == []
    gov = d["governing_interface"]
    assert gov["onset_strain"] == min(r["onset_strain"] for r in d["interfaces"])
    for r in d["interfaces"]:
        assert r["margin"] == pytest.approx(r["onset_strain"] / d["applied_strain_x"], rel=1e-9)


def test_margin_is_load_dependent():
    """하중을 올리면 여유율이 반비례로 떨어진다 (ε_c 는 하중 무관)."""
    a = srv.assess_free_edge_delamination(lam((0.0, 90.0, 90.0, 0.0)),
                                          loads={"N": [100.0, 0.0, 0.0]},
                                          fracture={"G_c": 0.25})["data"]
    b = srv.assess_free_edge_delamination(lam((0.0, 90.0, 90.0, 0.0)),
                                          loads={"N": [200.0, 0.0, 0.0]},
                                          fracture={"G_c": 0.25})["data"]
    assert b["governing_interface"]["onset_strain"] == pytest.approx(
        a["governing_interface"]["onset_strain"], rel=1e-12)
    assert b["governing_interface"]["margin"] == pytest.approx(
        a["governing_interface"]["margin"] / 2.0, rel=1e-9)


def test_unit_systems_agree():
    """SI와 SI_mm이 같은 물리를 준다 (G는 J/m² ↔ N/mm = ×1e-3)."""
    si = {"unit_system": "SI",
          "laminae": [{"material": {"type": "orthotropic_2d", "E1": 181e9, "E2": 10.3e9,
                                    "G12": 7.17e9, "nu12": 0.28},
                       "angle_deg": a, "thickness": 0.125e-3}
                      for a in (0.0, 90.0, 90.0, 0.0)]}
    a = srv.assess_free_edge_delamination(si, loads={"N": [150000.0, 0.0, 0.0]},
                                          fracture={"G_c": 250.0})["data"]
    b = srv.assess_free_edge_delamination(lam((0.0, 90.0, 90.0, 0.0)),
                                          loads={"N": [150.0, 0.0, 0.0]},
                                          fracture={"G_c": 0.25})["data"]
    assert a["applied_strain_x"] == pytest.approx(b["applied_strain_x"], rel=1e-12)
    assert a["interfaces"][1]["G"] == pytest.approx(b["interfaces"][1]["G"] * 1e3, rel=1e-12)
    assert a["governing_interface"]["onset_strain"] == pytest.approx(
        b["governing_interface"]["onset_strain"], rel=1e-12)


def test_warnings_and_errors():
    good = lam((0.0, 90.0, 90.0, 0.0))
    no_gc = srv.assess_free_edge_delamination(good, loads={"N": [150.0, 0.0, 0.0]})
    assert any(w["code"] == "W120" and "G_c" in w["message"] for w in no_gc["warnings"])
    assert any(w["code"] == "W120" and "혼합모드" in w["message"] for w in no_gc["warnings"])
    # 굽힘이 걸리면 축인장 전제를 벗어난다
    bent = srv.assess_free_edge_delamination(good, loads={"N": [150.0, 0.0, 0.0],
                                                          "M": [1.0, 0.0, 0.0]})
    assert any("굽힘 모멘트" in w["message"] for w in bent["warnings"])
    # 비대칭 적층 경고
    unsym = srv.assess_free_edge_delamination(lam((0.0, 90.0)), loads={"N": [150.0, 0.0, 0.0]})
    assert any("비대칭 적층" in w["message"] for w in unsym["warnings"])
    # 압축이면 경고
    comp = srv.assess_free_edge_delamination(good, loads={"N": [-150.0, 0.0, 0.0]})
    assert any("인장이 아니다" in w["message"] for w in comp["warnings"])
    # 오류
    assert srv.assess_free_edge_delamination(
        lam((0.0,)), loads={"N": [150.0, 0.0, 0.0]})["errors"][0]["code"] == "E100"
    assert srv.assess_free_edge_delamination(
        good, loads={"N": [0.0, 0.0, 0.0]})["errors"][0]["code"] == "E100"
    assert srv.assess_free_edge_delamination(
        good, loads={"N": [150.0, 0.0, 0.0]}, fracture={"G_c": -1.0})["errors"][0]["code"] == "E100"
    assert srv.assess_free_edge_delamination(
        good, loads={"N": [float("inf"), 0.0, 0.0]})["errors"][0]["code"] == "E100"


def test_deterministic():
    kw = dict(loads={"N": [150.0, 0.0, 0.0]}, fracture={"G_c": 0.25})
    a = srv.assess_free_edge_delamination(lam((0.0, 90.0, 90.0, 0.0)), **kw)
    b = srv.assess_free_edge_delamination(lam((0.0, 90.0, 90.0, 0.0)), **kw)
    assert a["data"] == b["data"]
    assert a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]


def test_complements_in_plane_check():
    """면내 강도는 통과하는데 가장자리 여유가 더 빡빡한 케이스가 실제로 존재한다.

    이 도구가 없으면 에이전트가 면내 R 만 보고 '안전'이라고 단정하게 된다.
    """
    strong = dict(T300, strength={"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0})
    stack = (30.0, -30.0, 90.0, 90.0, -30.0, 30.0)
    payload = {"unit_system": "SI_mm",
               "laminae": [{"material": strong, "angle_deg": a, "thickness": 0.125} for a in stack]}
    loads = {"N": [150.0, 0.0, 0.0]}
    inplane = srv.recover_ply_stresses(payload, loads=loads)["data"]
    r_min = min(p["stresses"]["mid"]["failure"]["tsai_wu_R"] for p in inplane["plies"])
    # 취성 에폭시의 현실적 G_Ic ≈ 100 J/m² = 0.10 N/mm
    edge = srv.assess_free_edge_delamination(payload, loads=loads,
                                             fracture={"G_c": 0.10})["data"]
    margin = edge["governing_interface"]["margin"]
    assert r_min > 1.0                # 면내는 "1.33배 여유"라고 답하는데
    assert margin < 1.0               # 가장자리는 이미 개시 하중을 넘었다
    assert margin < r_min


def test_reference_case_free_edge():
    """R14: 손유도 폐형해(4×4 대칭 분해)가 엔진과 일치한다 — 독립 경로 교차검증."""
    c = srv.get_reference_cases("free_edge_cross_ply")
    i, e, tol = c["input"], c["expected"], c["tolerance"]
    env = srv.assess_free_edge_delamination(i["laminate"], loads=i["loads"],
                                            fracture=i["fracture"])
    d = env["data"]
    assert env["errors"] == []
    assert d["laminate_modulus_Ex"] == pytest.approx(e["laminate_modulus_Ex_MPa"],
                                                     rel=tol["laminate_modulus_Ex_MPa"])
    mid = [r for r in d["interfaces"] if r["interface"] == 2][0]
    assert mid["delta_E"] == pytest.approx(e["midplane_delta_E_MPa"],
                                           rel=tol["midplane_delta_E_MPa"])
    assert mid["onset_strain"] == pytest.approx(e["midplane_onset_strain"],
                                                rel=tol["midplane_onset_strain"])
    assert mid["dominant_driver"] == e["midplane_dominant_driver"]
    outer = [r for r in d["interfaces"] if r["interface"] == 1][0]
    assert outer["dominant_driver"] == e["outer_interface_dominant_driver"]
    assert outer["G"] == pytest.approx(
        [r for r in d["interfaces"] if r["interface"] == 3][0]["G"], rel=1e-12)


# ── §19.9 혼합모드 분리 ─────────────────────────────────────────────────────

def test_mirror_split_only_at_symmetric_midplane():
    """거울 분할은 대칭 적층의 중앙면에서만 성립한다."""
    for angles, expect in (((0.0, 90.0, 90.0, 0.0), [2]),
                           ((30.0, -30.0, 90.0, 90.0, -30.0, 30.0), [3]),
                           ((0.0, 0.0, 0.0, 90.0), [])):
        th = [0.125] * len(angles)
        got = [k for k in range(1, len(angles)) if FE.is_mirror_split(list(angles), th, k)]
        assert got == expect


def test_mirror_split_means_equal_sublaminate_stiffness():
    """거울 분할이면 두 부분적층 축강성이 정확히 같다 — 상대 미끄러짐 0 의 근거."""
    qb, th = _si((30.0, -30.0, 90.0, 90.0, -30.0, 30.0))
    e1, _ = FE.sublaminate_axial_modulus(qb, th, 0, 3)
    e2, _ = FE.sublaminate_axial_modulus(qb, th, 3, 6)
    assert e1 == pytest.approx(e2, rel=1e-12)


def test_benzeggagh_kenane_limits():
    """B-K 는 Mode II 비 0 에서 G_Ic, 1 에서 G_IIc 로 정확히 환원된다."""
    assert FE.benzeggagh_kenane(0.1, 0.5, 0.0) == pytest.approx(0.1, rel=1e-15)
    assert FE.benzeggagh_kenane(0.1, 0.5, 1.0) == pytest.approx(0.5, rel=1e-15)
    mid = FE.benzeggagh_kenane(0.1, 0.5, 0.5, eta=2.0)
    assert mid == pytest.approx(0.1 + 0.4 * 0.25, rel=1e-15)
    assert 0.1 < mid < 0.5


def test_mode_mix_reported_per_interface():
    """중앙면은 순수 Mode I(정확), 그 외는 범위로 답한다."""
    env = srv.assess_free_edge_delamination(lam((30.0, -30.0, 90.0, 90.0, -30.0, 30.0)),
                                            loads={"N": [150.0, 0.0, 0.0]},
                                            fracture={"G_Ic": 0.10, "G_IIc": 0.50})
    d = env["data"]
    mid = [r for r in d["interfaces"] if r["interface"] == 3][0]
    assert mid["mode_mix"]["mode_II_fraction"] == 0.0
    assert mid["mode_mix"]["basis"] == "mirror_symmetry"
    assert "onset_strain_range" not in mid          # 분할이 확정이라 범위가 없다
    off = [r for r in d["interfaces"] if r["interface"] == 2][0]
    assert off["mode_mix"]["basis"] == "unknown"
    rng = off["onset_strain_range"]
    # 대표값은 보수적인 Mode I 쪽이어야 한다
    assert off["onset_strain"] == pytest.approx(rng["conservative_mode_I"], rel=1e-12)
    assert rng["optimistic_mode_II"] > rng["conservative_mode_I"]
    assert any(w["code"] == "W130" and "분할을 정할 수 없다" in w["message"] for w in env["warnings"])
    assert any(w["code"] == "W120" and "순수 Mode I" in w["message"] for w in env["warnings"])


def test_mode_i_toughness_matches_single_gc_path():
    """거울 계면에서 {G_Ic, G_IIc} 는 {G_c: G_Ic} 와 같은 답을 준다 (B-K 환원)."""
    payload = lam((0.0, 90.0, 90.0, 0.0))
    loads = {"N": [150.0, 0.0, 0.0]}
    a = srv.assess_free_edge_delamination(payload, loads=loads,
                                          fracture={"G_Ic": 0.10, "G_IIc": 0.50})["data"]
    b = srv.assess_free_edge_delamination(payload, loads=loads,
                                          fracture={"G_c": 0.10})["data"]
    mid_a = [r for r in a["interfaces"] if r["interface"] == 2][0]
    mid_b = [r for r in b["interfaces"] if r["interface"] == 2][0]
    assert mid_a["onset_strain"] == pytest.approx(mid_b["onset_strain"], rel=1e-12)


def test_fracture_input_validation():
    payload = lam((0.0, 90.0, 90.0, 0.0))
    loads = {"N": [150.0, 0.0, 0.0]}
    bad = [{"G_Ic": 0.5, "G_IIc": 0.1},      # 뒤바뀐 값
           {"typo": 1.0},                     # 미지 키
           {"G_Ic": 0.1},                     # G_IIc 누락
           {"G_Ic": 0.1, "G_IIc": 0.5, "eta": -1.0}]
    for fr in bad:
        assert srv.assess_free_edge_delamination(
            payload, loads=loads, fracture=fr)["errors"][0]["code"] == "E100"
    assert srv.assess_free_edge_delamination(
        payload, loads=loads, fracture={"G_c": 0.1})["errors"] == []
