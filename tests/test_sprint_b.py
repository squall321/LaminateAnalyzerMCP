# V2-3차 스프린트 테스트 — 층간 전단·모달 감쇠·횡전단 유연성 (계획서 §17.6)
from __future__ import annotations

import numpy as np
import pytest

import app.mcp_server as srv
from app.solver import abd as ABD
from app.solver import interlaminar as IL
from app.solver import material as MAT

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}


def iso_lam(E=70000.0, nu=0.3, t=2.0, **extra):
    m = {"type": "isotropic", "E": E, "nu": nu, **extra}
    return {"unit_system": "SI_mm", "laminae": [{"thickness": t, "angle_deg": 0, "material": m}]}


# ── 층간 전단 (폐형해) ──────────────────────────────────────────────────────

def test_interlaminar_isotropic_parabola():
    """등방 단일층: τmax = 1.5V/h (z=0), 자유표면 0 — 고전 폐형해."""
    V, t = 100.0, 2.0
    env = srv.compute_interlaminar_stresses(iso_lam(t=t), shear={"Vx": V})
    d = env["data"]["tau_xz"]
    assert d["peak"]["tau"] == pytest.approx(1.5 * V / t, rel=1e-9)
    assert d["peak"]["z_from_midplane"] == pytest.approx(0.0, abs=1e-12)
    assert abs(d["profile"][0]["tau"]) < 1e-9 and abs(d["profile"][-1]["tau"]) < 1e-9


def test_interlaminar_scaling_and_direction():
    """τ ∝ V 선형, Vy는 τ_yz 블록으로 분리."""
    a = srv.compute_interlaminar_stresses(iso_lam(), shear={"Vx": 100.0})["data"]["tau_xz"]
    b = srv.compute_interlaminar_stresses(iso_lam(), shear={"Vx": 250.0})["data"]["tau_xz"]
    assert b["peak"]["tau"] == pytest.approx(2.5 * a["peak"]["tau"], rel=1e-12)
    both = srv.compute_interlaminar_stresses(iso_lam(), shear={"Vx": 100.0, "Vy": 50.0})["data"]
    assert "tau_xz" in both and "tau_yz" in both


def test_interlaminar_ilss_margin():
    """margin = ILSS/|τ| — 강도 절반이면 여유도 절반."""
    lam = iso_lam(t=2.0, ilss=60.0)
    env = srv.compute_interlaminar_stresses(
        {"unit_system": "SI_mm", "laminae": [
            {"thickness": 1.0, "angle_deg": 0, "material": dict(lam["laminae"][0]["material"])},
            {"thickness": 1.0, "angle_deg": 0, "material": dict(lam["laminae"][0]["material"])}]},
        shear={"Vx": 100.0})
    d = env["data"]["tau_xz"]
    assert "critical_location" in d
    m = d["critical_location"]
    tau_at = next(p["tau"] for p in d["profile"] if abs(p["z"] - m["z"]) < 1e-12)
    assert m["margin"] == pytest.approx(60.0 / abs(tau_at), rel=1e-9)
    # 최대 τ 지점(여기서는 계면 z=0)이 임계로 잡혀야 한다
    assert m["margin"] == pytest.approx(60.0 / abs(d["peak"]["tau"]), rel=1e-9)


def test_interlaminar_continuity_and_equilibrium():
    """적층 τ는 연속이고 상하 자유표면 0 (평형법 불변식)."""
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.25, "angle_deg": a, "material": dict(T300)}
                       for a in (0, 90, 90, 0)]}
    prof = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0})["data"]["tau_xz"]["profile"]
    assert abs(prof[0]["tau"]) < 1e-9 and abs(prof[-1]["tau"]) < 1e-6
    taus = [p["tau"] for p in prof]
    assert max(abs(t) for t in taus) > 0                      # 비자명
    # 대칭 적층이면 τ 분포도 z에 대칭
    mid = len(prof) // 2
    assert taus[1] == pytest.approx(taus[-2], rel=1e-9)


def test_interlaminar_validation():
    assert srv.compute_interlaminar_stresses(iso_lam(), shear={})["errors"][0]["code"] == "E100"
    assert srv.compute_interlaminar_stresses(iso_lam(),
                                             shear={"Vx": float("inf")})["errors"][0]["code"] == "E100"
    assert srv.compute_interlaminar_stresses(iso_lam(), shear=[1, 2])["errors"][0]["code"] == "E100"


def test_free_edge_ranking():
    """각도 점프·강성 불일치가 큰 계면이 상위 — 0/90이 0/45보다 위험."""
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.25, "angle_deg": a, "material": dict(T300)}
                       for a in (0, 45, 90, 0)]}
    env = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0})
    rank = env["data"]["free_edge_risk_ranking"]
    assert rank[0]["interface"] in ("1/2", "2/3")            # 45/90 또는 90/0
    assert "W130" in [w["code"] for w in env["warnings"]]


# ── 모달 감쇠 ────────────────────────────────────────────────────────────────

def test_modal_damping_homogeneous_reduces_to_eta():
    """균질 단일재: η_modal = η (MSE 환원)."""
    lam = iso_lam(t=2.0, rho=2.7e-9, loss_factor=0.02)
    env = srv.compute_natural_frequencies(lam, panel={"Lx": 200.0, "Ly": 200.0})
    d = env["data"]["damping"]
    assert d["modal_loss_factor"] == pytest.approx(0.02, rel=1e-12)
    assert d["Q_factor"] == pytest.approx(50.0, rel=1e-12)


def test_modal_damping_weighted_by_bending_contribution():
    """감쇠재가 외측(D 기여 큼)일 때가 내측일 때보다 모달 η가 크다."""
    stiff = {"type": "isotropic", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9, "loss_factor": 0.001}
    damp = {"type": "isotropic", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9, "loss_factor": 0.5}
    def build(outer, inner):
        return {"unit_system": "SI_mm", "laminae": [
            {"thickness": 0.5, "angle_deg": 0, "material": dict(outer)},
            {"thickness": 0.5, "angle_deg": 0, "material": dict(inner)},
            {"thickness": 0.5, "angle_deg": 0, "material": dict(inner)},
            {"thickness": 0.5, "angle_deg": 0, "material": dict(outer)}]}
    eta_out = srv.compute_natural_frequencies(build(damp, stiff),
                                              panel={"Lx": 200.0, "Ly": 200.0})["data"]["damping"]["modal_loss_factor"]
    eta_in = srv.compute_natural_frequencies(build(stiff, damp),
                                             panel={"Lx": 200.0, "Ly": 200.0})["data"]["damping"]["modal_loss_factor"]
    assert eta_out > eta_in
    assert 0.001 < eta_in < eta_out < 0.5


def test_damping_absent_without_loss_factor():
    env = srv.compute_natural_frequencies(iso_lam(rho=2.7e-9), panel={"Lx": 200.0, "Ly": 200.0})
    assert "damping" not in env["data"]


# ── 횡전단 유연성 ────────────────────────────────────────────────────────────

def test_shear_flexibility_thin_vs_thick():
    """얇은 판은 R_s≈0(CLT 유효), 두꺼운 판은 R_s 커지고 W130."""
    thin = srv.compute_natural_frequencies(iso_lam(t=1.0, rho=2.7e-9),
                                           panel={"Lx": 500.0, "Ly": 500.0})
    thick = srv.compute_natural_frequencies(iso_lam(t=40.0, rho=2.7e-9),
                                            panel={"Lx": 100.0, "Ly": 100.0})
    rs_thin = thin["data"]["transverse_shear"]["R_s"]
    rs_thick = thick["data"]["transverse_shear"]["R_s"]
    assert rs_thin < 1e-4 and rs_thick > 0.02
    assert "W130" in [w["code"] for w in thick["warnings"]]
    # 보정은 항상 CLT 이하 (전단 유연성은 강성을 낮춤)
    d = thick["data"]
    assert d["transverse_shear"]["corrected_f1_hz"] < d["modes"][0]["f_hz"]
    assert thin["data"]["transverse_shear"]["frequency_factor"] == pytest.approx(1.0, abs=1e-4)


def test_shear_flexibility_limit_and_buckling_correction():
    """G→∞ ⇒ R_s→0 ⇒ 보정 = CLT (극한 검증)."""
    stiff_core = {"type": "orthotropic_2d", "E1": 70000.0, "E2": 70000.0, "G12": 27000.0,
                  "nu12": 0.3, "G13": 1e9, "G23": 1e9, "rho": 2.7e-9}
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 5.0, "angle_deg": 0, "material": stiff_core}]}
    env = srv.compute_natural_frequencies(lam, panel={"Lx": 100.0, "Ly": 100.0})
    ts = env["data"]["transverse_shear"]
    assert ts["R_s"] < 1e-6 and ts["frequency_factor"] == pytest.approx(1.0, abs=1e-6)
    bk = srv.compute_buckling(lam, panel={"Lx": 100.0, "Ly": 100.0})
    assert bk["data"]["transverse_shear"]["corrected_N_cr"] == pytest.approx(
        bk["data"]["N_cr"], rel=1e-6)


def test_g_transverse_assumed_warning():
    """직교이방에서 G13/G23 미지정 시 G12 근사 + W120 표기."""
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 2.0, "angle_deg": 0, "material": dict(T300, rho=1.6e-9)}]}
    env = srv.compute_natural_frequencies(lam, panel={"Lx": 200.0, "Ly": 200.0})
    assert env["data"]["transverse_shear"]["G_transverse_assumed"] is True
    assert "W120" in [w["code"] for w in env["warnings"]]


def test_interlaminar_units_bridge():
    si_mm = srv.compute_interlaminar_stresses(iso_lam(t=2.0), shear={"Vx": 100.0})
    si = srv.compute_interlaminar_stresses(
        {"unit_system": "SI", "laminae": [{"thickness": 2.0e-3, "angle_deg": 0,
         "material": {"type": "isotropic", "E": 70e9, "nu": 0.3}}]}, shear={"Vx": 100.0e3})
    assert si_mm["data"]["tau_xz"]["peak"]["tau"] == pytest.approx(
        si["data"]["tau_xz"]["peak"]["tau"] * 1e-6, rel=1e-12)


# ── 적대 검증(Workflow) 확인 결함 회귀 ──────────────────────────────────────

def _unsym(angles=(90, 0), t=0.5, **extra):
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": t, "angle_deg": a, "material": dict(T300, **extra)}
                        for a in angles]}


def test_il1_free_surface_zero_in_unsymmetric():
    """IL-1/IL-01: 비대칭 적층에서도 상하 자유표면 τ=0 (막-굽힘 연성 포함)."""
    for angles in ((0, 90), (90, 0)):
        prof = srv.compute_interlaminar_stresses(_unsym(angles), shear={"Vx": 100.0})["data"]["tau_xz"]["profile"]
        scale = max(abs(p["tau"]) for p in prof)
        assert abs(prof[0]["tau"]) < 1e-9 * max(scale, 1.0)
        assert abs(prof[-1]["tau"]) < 1e-9 * max(scale, 1.0)


def test_il1_unsymmetric_interface_value_correct():
    """IL-1: [0/90]과 [90/0]의 계면 τ는 같아야 하고(대칭 뒤집기), 검증팀 참값 75.878과 일치."""
    a = srv.compute_interlaminar_stresses(_unsym((0, 90)), shear={"Vx": 100.0})["data"]["tau_xz"]
    b = srv.compute_interlaminar_stresses(_unsym((90, 0)), shear={"Vx": 100.0})["data"]["tau_xz"]
    ta = next(p["tau"] for p in a["profile"] if p["interface"] == "0/1")
    tb = next(p["tau"] for p in b["profile"] if p["interface"] == "0/1")
    assert ta == pytest.approx(tb, rel=1e-9)
    assert abs(ta) == pytest.approx(75.878, rel=1e-3)


def test_il1_symmetric_unchanged():
    """대칭 적층 결과는 회귀하지 않는다 (검증팀 참값 64.977)."""
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.5, "angle_deg": a, "material": dict(T300)}
                       for a in (0, 90, 90, 0)]}
    d = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0})["data"]["tau_xz"]
    interf = max(abs(p["tau"]) for p in d["profile"] if p["interface"])
    assert interf == pytest.approx(64.977, rel=1e-3)


def test_il2_shear_flexibility_is_mode_aware():
    """IL-2/TS-01/FLEX-01: 좁은 판·고차 모드에서 R_s가 커지고 W130이 뜬다."""
    iso = lambda t: {"unit_system": "SI_mm", "laminae": [{"thickness": t, "angle_deg": 0,
                     "material": {"type": "isotropic", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9}}]}
    # 좁은 판(Ly 짧음) — 이전엔 R_s≈0.001로 침묵했음
    env = srv.compute_natural_frequencies(iso(20.0), panel={"Lx": 1000.0, "Ly": 50.0})
    assert env["data"]["transverse_shear"]["R_s"] > 0.02
    assert "W130" in [w["code"] for w in env["warnings"]]
    # 좌굴: 임계 모드가 m=10이면 그 모드 기준으로 평가
    bk = srv.compute_buckling(iso(20.0), panel={"Lx": 1000.0, "Ly": 100.0})
    assert bk["data"]["mode"]["m"] > 1
    assert bk["data"]["transverse_shear"]["critical_mode"]["m"] == bk["data"]["mode"]["m"]
    assert bk["data"]["transverse_shear"]["corrected_N_cr"] < bk["data"]["N_cr"] * 0.9
    # 판을 90° 돌려도 물리적으로 같은 R_s (이전엔 (Lx/Ly)²배 달라졌음)
    a = srv.compute_natural_frequencies(iso(20.0), panel={"Lx": 400.0, "Ly": 100.0})["data"]["transverse_shear"]["R_s"]
    b = srv.compute_natural_frequencies(iso(20.0), panel={"Lx": 100.0, "Ly": 400.0})["data"]["transverse_shear"]["R_s"]
    assert a == pytest.approx(b, rel=1e-9)


def test_ts03_energy_equivalent_a55_sandwich():
    """TS-03: 샌드위치에서 A55가 코어 전단 지배 — (5/6)ΣGt 균질식보다 훨씬 작다."""
    skin = {"type": "orthotropic_2d", "E1": 70000.0, "E2": 70000.0, "G12": 26923.0,
            "nu12": 0.3, "G13": 26923.0, "G23": 26923.0, "rho": 2.7e-9}
    core = {"type": "orthotropic_2d", "E1": 100.0, "E2": 100.0, "G12": 40.0,
            "nu12": 0.3, "G13": 40.0, "G23": 40.0, "rho": 5e-11}
    lam = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 1.0, "angle_deg": 0, "material": dict(skin)},
        {"thickness": 10.0, "angle_deg": 0, "material": dict(core)},
        {"thickness": 1.0, "angle_deg": 0, "material": dict(skin)}]}
    env = srv.compute_natural_frequencies(lam, panel={"Lx": 300.0, "Ly": 300.0})
    a55 = env["data"]["transverse_shear"]["A55"]
    naive = (5.0 / 6.0) * (26923.0 * 1.0 * 2 + 40.0 * 10.0)     # 균질 근사
    assert a55 < naive / 10.0                                    # 코어 지배로 훨씬 작음
    assert env["data"]["transverse_shear"]["R_s"] > 0.02         # 샌드위치는 CLT 부적합
    assert "W130" in [w["code"] for w in env["warnings"]]


def test_ts05_a55_angle_transform():
    """TS-05: G13≠G23인 재료를 90° 돌리면 A55/A44가 교환된다."""
    from app.solver import interlaminar as IL

    class P:
        def __init__(self, ang):
            self.angle_deg, self.thickness = ang, 1.0
    a0 = IL.transverse_shear_stiffness([P(0.0)], [(1000.0, 100.0)])
    a90 = IL.transverse_shear_stiffness([P(90.0)], [(1000.0, 100.0)])
    assert a0[0] == pytest.approx(a90[1], rel=1e-12)
    assert a0[1] == pytest.approx(a90[0], rel=1e-12)


def test_ts04_a55_unit_bridge():
    """IL-3/TS-04: A55가 표시 단위로 변환된다 (SI_mm는 SI의 1e-3배)."""
    mm = srv.compute_natural_frequencies(
        {"unit_system": "SI_mm", "laminae": [{"thickness": 2.0, "angle_deg": 0,
         "material": {"type": "isotropic", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9}}]},
        panel={"Lx": 200.0, "Ly": 200.0})["data"]["transverse_shear"]["A55"]
    si = srv.compute_natural_frequencies(
        {"unit_system": "SI", "laminae": [{"thickness": 0.002, "angle_deg": 0,
         "material": {"type": "isotropic", "E": 70e9, "nu": 0.3, "rho": 2700.0}}]},
        panel={"Lx": 0.2, "Ly": 0.2})["data"]["transverse_shear"]["A55"]
    assert mm == pytest.approx(si * 1e-3, rel=1e-9)


def test_mse_mode_and_neutral_axis():
    """MSE-01/02: 감쇠가 1차 모드·중립면 기준으로 계산된다."""
    lam = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 1.0, "angle_deg": 0, "material": {"type": "isotropic", "E": 70000.0,
         "nu": 0.3, "rho": 2.7e-9, "loss_factor": 0.01}},
        {"thickness": 1.0, "angle_deg": 90, "material": {"type": "orthotropic_2d",
         "E1": 10000.0, "E2": 10000.0, "G12": 3800.0, "nu12": 0.3, "rho": 1.2e-9,
         "loss_factor": 0.3}}]}
    env = srv.compute_natural_frequencies(lam, panel={"Lx": 200.0, "Ly": 100.0})
    d = env["data"]["damping"]
    assert 0.01 < d["modal_loss_factor"] < 0.3
    assert d["mode"] == {"m": env["data"]["modes"][0]["m"], "n": env["data"]["modes"][0]["n"]}


def test_damp_partial_input_warns():
    """IL-7/DAMP-01: loss_factor 부분 입력·전 0이 조용히 사라지지 않는다."""
    base = {"type": "isotropic", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9}
    partial = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 1.0, "angle_deg": 0, "material": dict(base, loss_factor=0.02)},
        {"thickness": 1.0, "angle_deg": 0, "material": dict(base)}]}
    env = srv.compute_natural_frequencies(partial, panel={"Lx": 200.0, "Ly": 200.0})
    assert "damping" not in env["data"]
    assert any(w["code"] == "W120" and "loss_factor" in (w.get("field") or "") for w in env["warnings"])
    zero = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 2.0, "angle_deg": 0, "material": dict(base, loss_factor=0.0)}]}
    env2 = srv.compute_natural_frequencies(zero, panel={"Lx": 200.0, "Ly": 200.0})
    assert any(w["code"] == "W120" for w in env2["warnings"])


def test_ilss_partial_input_reported():
    """IL-03: ilss 부분 입력 시 미평가 위치가 보고된다."""
    m_with = {"type": "isotropic", "E": 70000.0, "nu": 0.3, "ilss": 60.0}
    m_without = {"type": "isotropic", "E": 70000.0, "nu": 0.3}
    lam = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 0.7, "angle_deg": 0, "material": dict(m_without)},
        {"thickness": 0.7, "angle_deg": 0, "material": dict(m_without)},
        {"thickness": 0.7, "angle_deg": 0, "material": dict(m_with)}]}
    d = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0})["data"]["tau_xz"]
    assert "ilss_unevaluated" in d and "ilss_note" in d


def test_units01_w110_covers_new_fields():
    """UNITS-01: G13·ilss에도 단위 자릿수 휴리스틱이 걸린다."""
    lam = {"unit_system": "SI_mm", "laminae": [{"thickness": 1.0, "angle_deg": 0,
           "material": {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0,
                        "nu12": 0.28, "G13": 7.17e9, "ilss": 8e7, "rho": 1.6e-9}}]}
    env = srv.compute_natural_frequencies(lam, panel={"Lx": 200.0, "Ly": 200.0})
    fields = [w.get("field", "") for w in env["warnings"] if w["code"] == "W110"]
    assert any("G13" in f for f in fields) and any("ilss" in f for f in fields)


def test_freq01_no_infinity_in_response():
    """FREQ-01: 비정상 수치가 응답에 실리지 않는다 (E403 방어선)."""
    from app.services.envelope import canonical_json
    lam = {"unit_system": "SI_mm", "laminae": [{"thickness": 2.0, "angle_deg": 0,
           "material": {"type": "isotropic", "E": 70000.0, "nu": 0.3, "rho": 2.7e-9}}]}
    for env in (srv.compute_natural_frequencies(lam, panel={"Lx": 200.0, "Ly": 200.0}),
                srv.compute_buckling(lam, panel={"Lx": 200.0, "Ly": 200.0})):
        canonical_json(env)          # allow_nan=False — Inf/NaN이면 예외


def test_size01_interlaminar_summary():
    """SIZE-01: 두꺼운 적층은 프로파일이 요약되고 detail=full로 전체를 받는다."""
    lam = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.125, "angle_deg": (0, 45, -45, 90)[i % 4],
                        "material": dict(T300)} for i in range(64)]}
    auto = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0})["data"]["tau_xz"]
    full = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0},
                                             detail="full")["data"]["tau_xz"]
    assert "profile_truncation" in auto and len(auto["profile"]) < len(full["profile"])
    assert srv.compute_interlaminar_stresses(lam, shear={"Vx": 1.0},
                                             detail="x")["errors"][0]["code"] == "E100"


def test_interlaminar_determinism():
    from app.services.envelope import canonical_json
    lam = _unsym((0, 90), ilss=60.0)
    a = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0, "Vy": 30.0})
    b = srv.compute_interlaminar_stresses(lam, shear={"Vx": 100.0, "Vy": 30.0})
    assert canonical_json(a) == canonical_json(b)
