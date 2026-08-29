# §19.19 샌드위치 국부 파손 — 면재 주름·셀 딤플링·코어 전단 (2순위 6번)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import sandwich as SW

FACE = {"type": "orthotropic_2d", "E1": 70000.0, "E2": 70000.0, "G12": 26900.0, "nu12": 0.3,
        "strength": {"Xt": 600.0, "Xc": 600.0, "Yt": 600.0, "Yc": 600.0, "S": 300.0}}
CORE_PLY = {"type": "isotropic", "E": 1.0, "nu": 0.3}
CORE = {"Ez": 138.0, "Gc": 44.0, "cell_size": 5.0, "shear_strength": 1.5}


def sw(t_face=1.0, t_core=20.0):
    return {"unit_system": "SI_mm", "laminae": [
        {"material": FACE, "angle_deg": 0.0, "thickness": t_face},
        {"material": CORE_PLY, "angle_deg": 0.0, "thickness": t_core},
        {"material": FACE, "angle_deg": 0.0, "thickness": t_face}]}


# ── 폐형해 ──────────────────────────────────────────────────────────────────

def test_wrinkling_is_cube_root_product():
    """σ_wr = k(E_f·E_z·G_c)^(1/3) — 세 인자 각각을 8배 하면 응력이 2배."""
    base = SW.face_wrinkling_stress(70e9, 138e6, 44e6, 0.5)
    for scaled in (SW.face_wrinkling_stress(8 * 70e9, 138e6, 44e6, 0.5),
                   SW.face_wrinkling_stress(70e9, 8 * 138e6, 44e6, 0.5),
                   SW.face_wrinkling_stress(70e9, 138e6, 8 * 44e6, 0.5)):
        assert scaled / base == pytest.approx(2.0, rel=1e-12)
    assert SW.face_wrinkling_stress(70e9, 138e6, 44e6, 1.0) / base == pytest.approx(2.0, rel=1e-12)


def test_dimpling_scales_with_thickness_over_cell_squared():
    """σ_d ∝ (t_f/s)² — 셀이 2배면 응력이 1/4."""
    a = SW.intracell_dimpling_stress(70e9, 0.3, 1e-3, 5e-3)
    b = SW.intracell_dimpling_stress(70e9, 0.3, 1e-3, 10e-3)
    assert a / b == pytest.approx(4.0, rel=1e-12)
    c = SW.intracell_dimpling_stress(70e9, 0.3, 2e-3, 5e-3)
    assert c / a == pytest.approx(4.0, rel=1e-12)


def test_core_shear_is_force_over_face_distance():
    assert SW.core_shear_stress(20.0, 21.0) == pytest.approx(20.0 / 21.0, rel=1e-15)
    assert SW.core_shear_stress(-20.0, 21.0) == pytest.approx(20.0 / 21.0, rel=1e-15)


# ── Tool 계층 ───────────────────────────────────────────────────────────────

def test_wrinkling_reported_as_range_not_single_value():
    """k 가 문헌마다 0.5~0.825 로 갈린다 — 단일값을 내면 그 자체가 오답원이다."""
    d = srv.assess_sandwich_local_failure(sw(), core=CORE)["data"]
    w = d["modes"]["face_wrinkling"]
    assert w["k_range"] == [0.5, 0.825]
    assert w["sigma_optimistic"] / w["sigma_conservative"] == pytest.approx(0.825 / 0.5, rel=1e-9)
    assert w["sigma_conservative"] == pytest.approx(375.94, rel=1e-3)


def test_local_failure_can_govern_over_material_strength():
    """면재 재료강도는 여유 4.0 인데 국부 모드가 먼저 온다 — 이 도구의 존재 이유."""
    d = srv.assess_sandwich_local_failure(sw(), core=CORE, applied={"N": [-300.0, 0.0, 0.0]},
                                          shear={"Vx": 20.0})["data"]
    g = d["governing"]
    assert g["margins"]["face_material"] > 3.5          # 재료만 보면 안전
    assert g["mode"] != "face_material"                  # 그런데 지배는 다른 모드
    assert g["margin"] < g["margins"]["face_material"]
    assert g["margin"] == min(g["margins"].values())


def test_thin_face_makes_dimpling_govern():
    """면재가 얇아지면 딤플링이 지배로 넘어온다 (σ_d ∝ t_f²)."""
    thick = srv.assess_sandwich_local_failure(sw(t_face=1.0), core=CORE,
                                              applied={"N": [-300.0, 0.0, 0.0]})["data"]
    thin = srv.assess_sandwich_local_failure(sw(t_face=0.1), core=CORE,
                                             applied={"N": [-30.0, 0.0, 0.0]})["data"]
    assert (thin["modes"]["intracell_dimpling"]["sigma"]
            < thick["modes"]["intracell_dimpling"]["sigma"])
    assert thin["governing"]["margins"]["intracell_dimpling"] < \
        thick["governing"]["margins"]["intracell_dimpling"]


def test_core_shear_margin_and_missing_inputs_warn():
    with_shear = srv.assess_sandwich_local_failure(sw(), core=CORE, shear={"Vx": 20.0})
    assert with_shear["data"]["modes"]["core_shear"]["margin"] == pytest.approx(
        1.5 / (20.0 / 21.0), rel=1e-9)
    without = srv.assess_sandwich_local_failure(sw(), core=CORE)
    assert "core_shear" not in without["data"]["modes"]
    assert any(w["code"] == "W120" and "횡전단력이 없어" in w["message"]
               for w in without["warnings"])
    no_cell = srv.assess_sandwich_local_failure(sw(), core={"Ez": 138.0, "Gc": 44.0})
    assert any(w["code"] == "W120" and "cell_size" in w["message"] for w in no_cell["warnings"])


def test_governing_below_one_is_flagged():
    env = srv.assess_sandwich_local_failure(sw(), core=CORE,
                                            applied={"N": [-900.0, 0.0, 0.0]})
    assert env["data"]["governing"]["margin"] < 1.0
    assert any(w["code"] == "W130" and "1 미만" in w["message"] for w in env["warnings"])


def test_core_detection_and_errors():
    d = srv.assess_sandwich_local_failure(sw(), core=CORE)["data"]
    assert d["core_ply"]["index"] == 1 and d["core_ply"]["detected"] == "auto"
    assert srv.assess_sandwich_local_failure(sw(), core=CORE, core_ply=1)["errors"] == []
    bad_cases = [
        ({"Ez": 138.0}, "Gc 누락"),
        ({"Ez": -1.0, "Gc": 44.0}, "음수"),
        ({"Ez": 138.0, "Gc": 44.0, "typo": 1.0}, "미지 키"),
    ]
    for c, _why in bad_cases:
        assert srv.assess_sandwich_local_failure(sw(), core=c)["errors"][0]["code"] == "E100"
    assert srv.assess_sandwich_local_failure(sw(), core=CORE,
                                             core_ply=0)["errors"][0]["code"] == "E100"
    stiff = {"unit_system": "SI_mm", "laminae": [
        {"material": FACE, "angle_deg": 0.0, "thickness": 1.0} for _ in range(3)]}
    assert srv.assess_sandwich_local_failure(stiff, core=CORE)["errors"][0]["code"] == "E100"


def test_interlaminar_warning_points_to_this_tool():
    """§17.6.3 의 R_s 경고가 이제 출구를 지목해야 한다."""
    env = srv.compute_natural_frequencies(
        {"unit_system": "SI_mm", "laminae": [
            {"material": dict(FACE, rho=1.6e-9), "angle_deg": 0.0, "thickness": 1.0},
            {"material": dict(CORE_PLY, rho=5e-11), "angle_deg": 0.0, "thickness": 20.0},
            {"material": dict(FACE, rho=1.6e-9), "angle_deg": 0.0, "thickness": 1.0}]},
        panel={"Lx": 300.0, "Ly": 300.0})
    rs = [w["message"] for w in env["warnings"]
          if w["code"] == "W130" and "횡전단 유연성 R_s" in w["message"]]
    assert rs and all("assess_sandwich_local_failure" in m for m in rs)


def test_unit_systems_agree():
    si = {"unit_system": "SI", "laminae": [
        {"material": {"type": "orthotropic_2d", "E1": 70e9, "E2": 70e9, "G12": 26.9e9,
                      "nu12": 0.3}, "angle_deg": 0.0, "thickness": 1e-3},
        {"material": {"type": "isotropic", "E": 1e6, "nu": 0.3}, "angle_deg": 0.0,
         "thickness": 20e-3},
        {"material": {"type": "orthotropic_2d", "E1": 70e9, "E2": 70e9, "G12": 26.9e9,
                      "nu12": 0.3}, "angle_deg": 0.0, "thickness": 1e-3}]}
    a = srv.assess_sandwich_local_failure(
        si, core={"Ez": 138e6, "Gc": 44e6, "cell_size": 5e-3})["data"]
    b = srv.assess_sandwich_local_failure(
        sw(), core={"Ez": 138.0, "Gc": 44.0, "cell_size": 5.0})["data"]
    assert a["modes"]["face_wrinkling"]["sigma_conservative"] == pytest.approx(
        b["modes"]["face_wrinkling"]["sigma_conservative"] * 1e6, rel=1e-9)


def test_deterministic():
    a = srv.assess_sandwich_local_failure(sw(), core=CORE, applied={"N": [-300.0, 0.0, 0.0]})
    b = srv.assess_sandwich_local_failure(sw(), core=CORE, applied={"N": [-300.0, 0.0, 0.0]})
    assert a["data"] == b["data"]


# --- NC-03 / GATE-04: 지배 모드 판정 ------------------------------------------

_FACE_S = {"type": "orthotropic_2d", "E1": 70000.0, "E2": 70000.0, "G12": 26000.0, "nu12": 0.34,
           "strength": {"Xt": 600.0, "Xc": 570.0, "Yt": 600.0, "Yc": 570.0, "S": 90.0}}
_CORE_S = {"type": "orthotropic_2d", "E1": 200.0, "E2": 200.0, "G12": 77.0, "nu12": 0.3,
           "G13": 60.0, "G23": 40.0,
           "strength": {"Xt": 2.0, "Xc": 1.5, "Yt": 2.0, "Yc": 1.5, "S": 1.2}}
_SW_S = {"unit_system": "SI_mm", "laminae": [
    {"material": _FACE_S, "angle_deg": 0.0, "thickness": 0.5},
    {"material": _CORE_S, "angle_deg": 0.0, "thickness": 10.0},
    {"material": _FACE_S, "angle_deg": 0.0, "thickness": 0.5}]}
_CORE_IN = {"Ez": 300.0, "Gc": 77.0, "shear_strength": 1.2}


def test_core_shear_governs_without_face_compression():
    """면재 압축이 없어도 코어 전단은 독립적으로 지배한다 (NC-03)."""
    d = srv.assess_sandwich_local_failure(_SW_S, core=_CORE_IN, shear={"Vx": 8.0})["data"]
    assert d["governing"]["mode"] == "core_shear"
    assert d["governing"]["margin"] == pytest.approx(
        d["modes"]["core_shear"]["margin"], rel=1e-12)


def test_face_material_margin_excludes_core_strength():
    """면재 재료 여유는 면재 ply 의 재료축 Tsai-Wu 다 (GATE-04 + SW-01)."""
    lam = _SW_S
    d = srv.assess_sandwich_local_failure(
        lam, core=_CORE_IN, applied={"N": [-200.0, 0.0, 0.0]})["data"]
    # 코어 Xc=1.5 가 섞이면 0.008 로 "이미 파손"이 된다 — 면재 기준이어야 한다
    assert d["governing"]["margins"]["face_material"] > 1.0
    # 같은 서버의 다른 경로(recover_ply_stresses)와 일치해야 한다
    rp = srv.recover_ply_stresses(lam, loads={"N": [-200.0, 0.0, 0.0]})["data"]
    face_r = min(loc["failure"]["tsai_wu_R"] for p in rp["plies"] if p["ply"] != 1
                 for loc in p["stresses"].values() if loc["failure"]["tsai_wu_R"] is not None)
    assert d["governing"]["margins"]["face_material"] == pytest.approx(face_r, rel=1e-9)


def test_core_shear_competes_with_face_modes():
    """면재 압축과 코어 전단이 같이 있으면 더 작은 쪽이 지배한다."""
    d = srv.assess_sandwich_local_failure(
        _SW_S, core=_CORE_IN, applied={"N": [-200.0, 0.0, 0.0]}, shear={"Vx": 8.0})["data"]
    m = d["governing"]["margins"]
    assert set(m) == {"face_wrinkling", "face_material", "core_shear"}
    assert d["governing"]["margin"] == pytest.approx(min(m.values()), rel=1e-12)
    assert d["governing"]["mode"] == "core_shear"
    assert d["governing"]["basis"] == sorted(m)


# --- SW-01 / SW-02 / SW-03: 방향·재료축·합력 ----------------------------------

_OFF = {"type": "orthotropic_2d", "E1": 70000.0, "E2": 70000.0, "G12": 26900.0, "nu12": 0.3,
        "strength": {"Xt": 1200.0, "Xc": 1200.0, "Yt": 200.0, "Yc": 200.0, "S": 100.0}}
_CP = {"type": "isotropic", "E": 1.0, "nu": 0.3}
_C_IN = {"Ez": 138.0, "Gc": 44.0, "shear_strength": 1.5}


def _off_sw(ang):
    return {"unit_system": "SI_mm", "laminae": [
        {"material": _OFF, "angle_deg": ang, "thickness": 1.0},
        {"material": _CP, "angle_deg": 0.0, "thickness": 20.0},
        {"material": _OFF, "angle_deg": ang, "thickness": 1.0}]}


def test_face_material_uses_material_axes():
    """off-axis 면재는 Yc·S 로 판정해야 한다 — 전역 σx 를 Xc 와 비교하면 6배 낙관."""
    lam = _off_sw(90.0)
    d = srv.assess_sandwich_local_failure(
        lam, core=_C_IN, applied={"N": [-303.0, 0.0, 0.0]})["data"]
    rp = srv.recover_ply_stresses(lam, loads={"N": [-303.0, 0.0, 0.0]})["data"]
    ref = min(loc["failure"]["tsai_wu_R"] for p in rp["plies"] if p["ply"] != 1
              for loc in p["stresses"].values() if loc["failure"]["tsai_wu_R"] is not None)
    assert d["governing"]["margins"]["face_material"] == pytest.approx(ref, rel=1e-9)
    # Xc/σx 였다면 4.8 이 나온다 — 그 값이 아니어야 한다
    assert d["governing"]["margins"]["face_material"] < 2.0


def test_governing_is_rotation_invariant():
    """같은 물리 문제를 축만 90° 돌리면 같은 답이 나와야 한다 (SW-02)."""
    a = srv.assess_sandwich_local_failure(
        _off_sw(0.0), core=_C_IN, applied={"N": [0.0, -1000.0, 0.0]},
        shear={"Vx": 2.0})["data"]["governing"]
    b = srv.assess_sandwich_local_failure(
        _off_sw(90.0), core=_C_IN, applied={"N": [-1000.0, 0.0, 0.0]},
        shear={"Vx": 2.0})["data"]["governing"]
    assert a["mode"] == b["mode"]
    assert a["margin"] == pytest.approx(b["margin"], rel=1e-9)
    assert a["margin"] < 1.0


def test_ny_compression_does_not_erase_face_modes():
    """Ny 압축이어도 면재 모드가 후보로 남는다 — 전에는 core_shear 만 남았다."""
    d = srv.assess_sandwich_local_failure(
        _off_sw(0.0), core=_C_IN, applied={"N": [0.0, -1000.0, 0.0]},
        shear={"Vx": 2.0})["data"]
    assert d["face_stress"]["by_direction"]["y"]["max_compressive"] > 0
    assert "face_wrinkling" in d["governing"]["margins"]
    assert set(d["governing"]["margins"]) >= {"face_wrinkling", "face_material", "core_shear"}


def test_core_shear_uses_resultant():
    """τ_c 는 √(Vx²+Vy²)/d — max(|Vx|,|Vy|) 는 최대 √2 배 비보수다 (SW-03)."""
    one = srv.assess_sandwich_local_failure(
        _off_sw(0.0), core=_C_IN, shear={"Vx": 5.0})["data"]["modes"]["core_shear"]
    two = srv.assess_sandwich_local_failure(
        _off_sw(0.0), core=_C_IN, shear={"Vx": 5.0, "Vy": 5.0})["data"]["modes"]["core_shear"]
    assert two["tau_core"] / one["tau_core"] == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert two["margin"] < one["margin"]
