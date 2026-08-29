# §19.12 순응층 부분합성 굽힘 — CLT 가 원리적으로 못 보는 것 (2순위 3번)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import partial_composite as PC

UTG = {"type": "isotropic", "E": 71000.0, "nu": 0.22, "rho": 2.5e-9}
OCA = {"type": "isotropic", "E": 0.894, "nu": 0.49, "rho": 1.0e-9}   # G ≈ 0.30 MPa
STIFF_MID = {"type": "isotropic", "E": 30000.0, "nu": 0.3, "rho": 1.5e-9}


def stack(mid=OCA, t_mid=0.05):
    return {"unit_system": "SI_mm", "laminae": [
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.03},
        {"material": mid, "angle_deg": 0.0, "thickness": t_mid},
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.03}]}


def _si_parts():
    e, nu = 71e9, 0.22
    eo, nuo = 0.894e6, 0.49
    th = [30e-6, 50e-6, 30e-6]
    qb = [MAT.qbar_matrix(MAT.q_matrix(e, e, e / (2 * (1 + nu)), nu), 0.0),
          MAT.qbar_matrix(MAT.q_matrix(eo, eo, eo / (2 * (1 + nuo)), nuo), 0.0),
          MAT.qbar_matrix(MAT.q_matrix(e, e, e / (2 * (1 + nu)), nu), 0.0)]
    z = ABD.z_coordinates(th)
    _A, _B, D = ABD.abd_matrices(qb, z)
    ea1, ei1 = PC.sublaminate_ea_ei(qb, th, 0, 1)
    ea2, ei2 = PC.sublaminate_ea_ei(qb, th, 2, 3)
    _e, ei_c = PC.sublaminate_ea_ei(qb, th, 1, 2)
    return ea1, ei1, ea2, ei2, ei_c, float(D[0, 0])


# ── 폐형해 극한 ─────────────────────────────────────────────────────────────

def test_shear_limits_close_exactly():
    """G_c→0 이면 f=0(각자 굼), G_c→∞ 이면 f=1(CLT 와 일치) — 두 극한이 정확히 닫힌다."""
    ea1, ei1, ea2, ei2, ei_c, ei_full = _si_parts()
    lo = PC.composite_action(ea1, ei1, ea2, ei2, ei_c, ei_full, 1e-12, 50e-6, 10e-3)
    assert lo["composite_action"] == pytest.approx(0.0, abs=1e-9)
    assert lo["EI_effective"] == pytest.approx(lo["EI_layered"], rel=1e-9)
    hi = PC.composite_action(ea1, ei1, ea2, ei2, ei_c, ei_full, 1e18, 50e-6, 10e-3)
    assert hi["composite_action"] == pytest.approx(1.0, rel=1e-6)
    assert hi["EI_effective"] == pytest.approx(ei_full, rel=1e-6)


def test_zero_shear_modulus_gives_layered_stiffness():
    ea1, ei1, ea2, ei2, ei_c, ei_full = _si_parts()
    r = PC.composite_action(ea1, ei1, ea2, ei2, ei_c, ei_full, 0.0, 50e-6, 10e-3)
    assert r["composite_action"] == 0.0
    assert r["EI_effective"] == pytest.approx(r["EI_layered"], rel=1e-12)


def test_combination_is_harmonic_not_linear():
    """**결합이 조화형이지 선형이 아니다** — Newmark ODE 재유도로 확인된 것(적대 검증 PC-01).

    두 극한(f=0, f=1)은 두 형태 모두 통과하므로 중간 구간을 고정해야 잡힌다.
    """
    ea1, ei1, ea2, ei2, ei_c, ei_full = _si_parts()
    r = PC.composite_action(ea1, ei1, ea2, ei2, ei_c, ei_full, 0.3e6, 50e-6, 10e-3)
    ei_lay, f = r["EI_layered"], r["composite_action"]
    r_ratio = (ei_full - ei_lay) / ei_full
    harmonic = ei_lay / (1.0 - r_ratio * f)
    linear = ei_lay + f * (ei_full - ei_lay)
    assert r["EI_effective"] == pytest.approx(harmonic, rel=1e-12)
    assert linear / harmonic > 4.0            # 선형결합은 5배 가까이 과대(비보수)였다
    # f 는 중앙처짐 metric: f = 1 − 2(1 − sech X)/X²
    x = r["alpha_L"] / 2.0
    assert f == pytest.approx(1.0 - 2.0 * (1.0 - 1.0 / math.cosh(x)) / (x * x), rel=1e-12)


def test_parallel_axis_identity_gives_face_distance():
    """d² 를 EI_full − EI_layered 항등식에서 역산하므로 두 값과 항상 정합한다.

    UTG/OCA/UTG 의 면재 중심 거리 80µm 에 가까워야 한다(OCA 축강성 기여로 약간 큼).
    """
    ea1, ei1, ea2, ei2, ei_c, ei_full = _si_parts()
    r = PC.composite_action(ea1, ei1, ea2, ei2, ei_c, ei_full, 0.3e6, 50e-6, 10e-3)
    assert r["d"] == pytest.approx(82e-6, rel=0.05)
    # 항등식 재검산
    d2 = (ei_full - (ei1 + ei2 + ei_c)) * (ea1 + ea2) / (ea1 * ea2)
    assert r["d"] ** 2 == pytest.approx(d2, rel=1e-12)


def test_composite_action_increases_with_span():
    """스팬이 길수록 전단 전달 길이가 충분해져 CLT 에 수렴한다."""
    prev = 0.0
    for L in (1.0, 5.0, 10.0, 50.0, 200.0):
        f = srv.assess_partial_composite_bending(stack(), span=L)["data"]["composite_action"]
        assert f > prev
        prev = f
    assert prev > 0.95


# ── Tool 계층 ───────────────────────────────────────────────────────────────

def test_measured_foldable_case():
    """실측 랜드마크: UTG/OCA/UTG, OCA G=0.3 MPa, L=10mm → f≈57.4%, CLT 10.09배 과대.

    (적대 검증 PC-01 이전에는 선형결합이라 f=46.8%·2.03배로 잘못 나왔다.)
    """
    d = srv.assess_partial_composite_bending(stack(), span=10.0)["data"]
    assert d["composite_action"] == pytest.approx(0.574, rel=0.02)
    assert d["clt_overprediction"] == pytest.approx(10.09, rel=0.02)
    assert d["EI_layered"] < d["EI_effective"] < d["EI_full_CLT"]
    assert d["core_ply"]["index"] == 1 and d["core_ply"]["detected"] == "auto"


def test_short_span_is_nearly_uncoupled():
    """L=1mm 면 사실상 각자 굼 — αL<1 경고까지 나와야 한다."""
    env = srv.assess_partial_composite_bending(stack(), span=1.0)
    d = env["data"]
    assert d["composite_action"] < 0.05
    assert d["clt_overprediction"] > 15.0
    assert any(w["code"] == "W130" and "전단 전달이 거의" in w["message"] for w in env["warnings"])


def test_stiff_interlayer_is_not_detected():
    """이웃 대비 10배 이상 무르지 않으면 순응층이 아니다 — CLT 가 이미 맞다."""
    env = srv.assess_partial_composite_bending(stack(mid=STIFF_MID), span=10.0)
    assert env["errors"][0]["code"] == "E100"
    assert "무른 중간층을 찾지 못했" in env["errors"][0]["message"]


def test_explicit_core_ply_and_errors():
    assert srv.assess_partial_composite_bending(stack(), span=10.0,
                                                core_ply=1)["errors"] == []
    for bad in (0, 2, -1, 1.5):
        assert srv.assess_partial_composite_bending(
            stack(), span=10.0, core_ply=bad)["errors"][0]["code"] == "E100"
    for bad_span in (0.0, -1.0, float("inf")):
        assert srv.assess_partial_composite_bending(
            stack(), span=bad_span)["errors"][0]["code"] == "E100"
    two_ply = {"unit_system": "SI_mm", "laminae": [
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.03},
        {"material": OCA, "angle_deg": 0.0, "thickness": 0.05}]}
    assert srv.assess_partial_composite_bending(two_ply, span=10.0)["errors"][0]["code"] == "E100"


def test_unit_systems_agree():
    si = {"unit_system": "SI", "laminae": [
        {"material": {"type": "isotropic", "E": 71e9, "nu": 0.22}, "angle_deg": 0.0,
         "thickness": 30e-6},
        {"material": {"type": "isotropic", "E": 0.894e6, "nu": 0.49}, "angle_deg": 0.0,
         "thickness": 50e-6},
        {"material": {"type": "isotropic", "E": 71e9, "nu": 0.22}, "angle_deg": 0.0,
         "thickness": 30e-6}]}
    a = srv.assess_partial_composite_bending(si, span=10e-3)["data"]
    b = srv.assess_partial_composite_bending(stack(), span=10.0)["data"]
    assert a["composite_action"] == pytest.approx(b["composite_action"], rel=1e-9)
    assert a["clt_overprediction"] == pytest.approx(b["clt_overprediction"], rel=1e-9)


def test_deterministic():
    a = srv.assess_partial_composite_bending(stack(), span=10.0)
    b = srv.assess_partial_composite_bending(stack(), span=10.0)
    assert a["data"] == b["data"]
    assert a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]


# ── 교차 게이트 ─────────────────────────────────────────────────────────────

def test_gate_fires_on_buckling_and_frequencies():
    """판 크기가 있는 도구는 짧은 변을 스팬으로 써서 CLT 과대평가를 알린다."""
    small = {"Lx": 10.0, "Ly": 10.0}
    for env in (srv.compute_buckling(stack(), panel=small),
                srv.compute_natural_frequencies(stack(), panel=small)):
        assert any(w["code"] == "W130" and "순응층" in w["message"] and "과대평가" in w["message"]
                   for w in env["warnings"])


def test_gate_silent_when_span_is_long_or_no_compliant_layer():
    """큰 판은 합성도가 높아 침묵하고, 순응층이 없는 적층도 침묵한다 (경고 피로 방지)."""
    big = srv.compute_buckling(stack(), panel={"Lx": 200.0, "Ly": 200.0})
    assert not any("순응층" in w["message"] for w in big["warnings"])
    T = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0,
         "nu12": 0.28, "rho": 1.6e-9}
    normal = {"unit_system": "SI_mm",
              "laminae": [{"material": T, "angle_deg": a, "thickness": 0.125}
                          for a in (0.0, 90.0, 90.0, 0.0)]}
    env = srv.compute_buckling(normal, panel={"Lx": 10.0, "Ly": 10.0})
    assert not any("순응층" in w["message"] for w in env["warnings"])


# --- NC-07: 쪼갠 코어를 하나로 묶는다 ------------------------------------------

def _split_core(n_split):
    """코어를 n 등분해도 물리적으로 같은 적층이다."""
    return {"unit_system": "SI_mm", "laminae":
            [{"material": UTG, "angle_deg": 0.0, "thickness": 0.03}]
            + [{"material": OCA, "angle_deg": 0.0, "thickness": 0.05 / n_split}] * n_split
            + [{"material": UTG, "angle_deg": 0.0, "thickness": 0.03}]}


def test_split_core_is_detected_as_one_run():
    """인접 동등 순응층을 하나로 묶는다 — 전에는 '순응층 없음' E100 이었다."""
    for n_split in (2, 3, 5):
        d = srv.assess_partial_composite_bending(_split_core(n_split), span=10.0)["data"]
        assert d["core_ply"]["indices"] == list(range(1, n_split + 1))
        assert d["core_ply"]["merged_plies"] == n_split
        assert d["core_ply"]["thickness"] == pytest.approx(0.05, rel=1e-12)


def test_split_core_gives_identical_answer():
    """쪼개도 답이 **완전히 같아야** 한다(직렬 조화평균 G, 두께 합)."""
    one = srv.assess_partial_composite_bending(_split_core(1), span=10.0)["data"]
    for n_split in (2, 3, 5):
        d = srv.assess_partial_composite_bending(_split_core(n_split), span=10.0)["data"]
        assert d["composite_action"] == pytest.approx(one["composite_action"], rel=1e-12)
        assert d["EI_effective"] == pytest.approx(one["EI_effective"], rel=1e-12)
        assert d["clt_overprediction"] == pytest.approx(one["clt_overprediction"], rel=1e-12)


def test_core_shear_series_harmonic():
    """서로 다른 두 코어층은 직렬 조화평균으로 묶인다 (t/G 가 더해진다)."""
    soft = {"type": "isotropic", "E": 0.894, "nu": 0.49}
    softer = {"type": "isotropic", "E": 0.3, "nu": 0.49}
    lam = {"unit_system": "SI_mm", "laminae": [
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.03},
        {"material": soft, "angle_deg": 0.0, "thickness": 0.025},
        {"material": softer, "angle_deg": 0.0, "thickness": 0.025},
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.03}]}
    d = srv.assess_partial_composite_bending(lam, span=10.0)["data"]
    g1 = 0.894 / (2 * 1.49)
    g2 = 0.3 / (2 * 1.49)
    expected = 0.05 / (0.025 / g1 + 0.025 / g2)
    assert d["core_ply"]["G_transverse"] == pytest.approx(expected, rel=1e-9)


def test_stiff_inner_run_is_still_rejected():
    """뻣뻣한 중간층 무리는 여전히 순응층이 아니다 (게이트 오작동 방지)."""
    stiff = {"type": "isotropic", "E": 60000.0, "nu": 0.3}
    lam = {"unit_system": "SI_mm", "laminae": [
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.03},
        {"material": stiff, "angle_deg": 0.0, "thickness": 0.025},
        {"material": stiff, "angle_deg": 0.0, "thickness": 0.025},
        {"material": UTG, "angle_deg": 0.0, "thickness": 0.03}]}
    r = srv.assess_partial_composite_bending(lam, span=10.0)
    assert r["errors"][0]["code"] == "E100"
