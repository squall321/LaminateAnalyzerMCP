# §19.21 방향성 패턴층 직교이방 균질화 (2순위 12번)
from __future__ import annotations

import pytest

import app.mcp_server as srv
from app.solver import micromechanics as MM
from app.solver import thermal as TH

CU = {"type": "isotropic", "E": 117000.0, "nu": 0.34, "alpha": 17e-6, "rho": 8.96e-9}
EP = {"type": "isotropic", "E": 3500.0, "nu": 0.35, "alpha": 60e-6, "rho": 1.2e-9}
FR4 = {"type": "isotropic", "E": 22000.0, "nu": 0.15, "alpha": 16e-6}


def comps(f_cu=0.6):
    return [{"material": CU, "volume_fraction": f_cu},
            {"material": EP, "volume_fraction": 1.0 - f_cu}]


def same_comps(E=70000.0, nu=0.3, alpha=2e-5):
    m = {"type": "isotropic", "E": E, "nu": nu, "alpha": alpha}
    return [{"material": m, "volume_fraction": 0.6}, {"material": m, "volume_fraction": 0.4}]


# ── 폐형해 성질 ─────────────────────────────────────────────────────────────

def test_E1_is_voigt_and_matches_homogenize_layer():
    """E1 은 등변형이라 Voigt 가 정확하다 — 기존 homogenize_layer 의 E 와 같아야 한다."""
    a = srv.homogenize_patterned_layer(comps())["data"]["material"]
    b = srv.homogenize_layer(comps())["data"]["material"]
    assert a["E1"] == pytest.approx(b["E"], rel=1e-12)
    assert a["nu12"] == pytest.approx(b["nu"], rel=1e-12)
    assert a["alpha1"] == pytest.approx(b["alpha"], rel=1e-12)


def test_E2_is_reuss_lower_bound():
    """E2 는 등응력이라 Reuss — 항상 E1 보다 작다."""
    d = srv.homogenize_patterned_layer(comps())["data"]
    m = d["material"]
    reuss = 1.0 / (0.6 / 117000.0 + 0.4 / 3500.0)
    assert m["E2"] == pytest.approx(reuss, rel=1e-12)
    assert m["E2"] < m["E1"]
    assert d["anisotropy"]["E1_over_E2"] == pytest.approx(m["E1"] / m["E2"], rel=1e-12)
    assert d["anisotropy"]["E1_over_E2"] == pytest.approx(8.55, rel=1e-2)


def test_alpha_anisotropy_direction():
    """α2 > α1 이어야 한다 — 가로 방향은 수지가 자유롭게 팽창한다."""
    d = srv.homogenize_patterned_layer(comps())["data"]
    m = d["material"]
    assert m["alpha2"] > m["alpha1"]
    assert d["anisotropy"]["alpha2_over_alpha1"] == pytest.approx(2.238, rel=1e-2)


@pytest.mark.parametrize("E,nu,alpha", [(70000.0, 0.3, 2e-5), (10.0, 0.45, 1e-4)])
def test_isotropic_limit_matches_homogenize_layer(E, nu, alpha):
    """전 상이 같은 물성이면 E1=E2, α1=α2 로 기존 도구와 정확히 일치한다 (회귀 불변식)."""
    p = srv.homogenize_patterned_layer(same_comps(E, nu, alpha))["data"]["material"]
    h = srv.homogenize_layer(same_comps(E, nu, alpha))["data"]["material"]
    assert p["E1"] == pytest.approx(p["E2"], rel=1e-12) == pytest.approx(h["E"], rel=1e-12)
    assert p["alpha1"] == pytest.approx(p["alpha2"], rel=1e-9) == pytest.approx(h["alpha"], rel=1e-9)
    assert p["nu12"] == pytest.approx(h["nu"], rel=1e-12)


def test_volume_fraction_limits():
    """f→1 이면 그 상의 물성으로 환원된다."""
    only_cu = srv.homogenize_patterned_layer(
        [{"material": CU, "volume_fraction": 1.0}, {"material": EP, "volume_fraction": 0.0}])
    assert only_cu["errors"][0]["code"] == "E100"      # f=0 은 허용 안 함
    near = srv.homogenize_patterned_layer(comps(f_cu=1.0 - 1e-9))["data"]["material"]
    assert near["E1"] == pytest.approx(117000.0, rel=1e-6)
    assert near["E2"] == pytest.approx(117000.0, rel=1e-4)


# ── 이 도구의 존재 이유: 패턴 방향이 형상에 반영된다 ────────────────────────

def test_pattern_direction_now_changes_the_answer():
    """등방 균질화로는 스택업 A/B 가 바이트 동일했다 — 이제 갈린다."""
    iso = srv.homogenize_layer(comps())["data"]["material"]
    ortho = srv.homogenize_patterned_layer(comps())["data"]["material"]

    def stack(mat, a_top):
        return {"unit_system": "SI_mm", "laminae": [
            {"material": mat, "angle_deg": 0.0, "thickness": 0.035},
            {"material": FR4, "angle_deg": 0.0, "thickness": 0.2},
            {"material": mat, "angle_deg": a_top, "thickness": 0.035}]}

    def kappa(mat, a_top):
        return srv.compute_thermal_response(stack(mat, a_top), delta_T=-150.0,
                                            panel={"Lx": 100.0, "Ly": 100.0})["data"]["response"]["kappa"]

    # 등방: A 와 B 가 구분 불가
    assert kappa(iso, 90.0) == pytest.approx(kappa(iso, 0.0), abs=1e-12)
    # 직교이방: 갈린다
    ka, kb = kappa(ortho, 90.0), kappa(ortho, 0.0)
    assert abs(ka[0] - kb[0]) > 1e-6


def test_patterned_stackup_produces_saddle():
    """0/90 패턴 비대칭은 안장(κx·κy < 0)을 만든다 — PCB 실측 형상이다."""
    ortho = srv.homogenize_patterned_layer(comps())["data"]["material"]
    lam = {"unit_system": "SI_mm", "laminae": [
        {"material": ortho, "angle_deg": 0.0, "thickness": 0.035},
        {"material": FR4, "angle_deg": 0.0, "thickness": 0.2},
        {"material": ortho, "angle_deg": 90.0, "thickness": 0.035}]}
    d = srv.compute_thermal_response(lam, delta_T=-150.0,
                                     panel={"Lx": 100.0, "Ly": 100.0})["data"]
    kx, ky = d["response"]["kappa"][0], d["response"]["kappa"][1]
    assert kx * ky < 0                       # 안장
    assert d["warpage"]["range"] > 1.0       # 등방일 때는 1e-15 였다


def test_mirror_stackups_give_opposite_curvature():
    """A(0/FR4/90)와 B(90/FR4/0)는 거울상이라 곡률 부호가 뒤집힌다."""
    ortho = srv.homogenize_patterned_layer(comps())["data"]["material"]

    def kappa(a_bot, a_top):
        lam = {"unit_system": "SI_mm", "laminae": [
            {"material": ortho, "angle_deg": a_bot, "thickness": 0.035},
            {"material": FR4, "angle_deg": 0.0, "thickness": 0.2},
            {"material": ortho, "angle_deg": a_top, "thickness": 0.035}]}
        return srv.compute_thermal_response(lam, delta_T=-150.0)["data"]["response"]["kappa"]

    a, b = kappa(0.0, 90.0), kappa(90.0, 0.0)
    assert a[0] == pytest.approx(-b[0], rel=1e-9)
    assert a[1] == pytest.approx(-b[1], rel=1e-9)


# ── 게이트·입력 ─────────────────────────────────────────────────────────────

def test_homogenize_layer_points_to_this_tool():
    """등방 도구의 scope 가 방향성 패턴층에 대해 이 도구를 지목해야 한다."""
    scope = srv.homogenize_layer(comps())["data"]["scope"]
    assert "homogenize_patterned_layer" in scope


def test_near_isotropic_is_flagged():
    env = srv.homogenize_patterned_layer(same_comps())
    assert any(w["code"] == "W130" and "거의 등방" in w["message"] for w in env["warnings"])
    strong = srv.homogenize_patterned_layer(comps())
    assert not any("거의 등방" in w["message"] for w in strong["warnings"])
    assert any(w["code"] == "W130" and "등응력 하한" in w["message"] for w in strong["warnings"])


def test_errors():
    assert srv.homogenize_patterned_layer([])["errors"][0]["code"] == "E100"
    assert srv.homogenize_patterned_layer(comps()[:1])["errors"][0]["code"] == "E100"
    bad_sum = [{"material": CU, "volume_fraction": 0.6},
               {"material": EP, "volume_fraction": 0.6}]
    assert srv.homogenize_patterned_layer(bad_sum)["errors"][0]["code"] == "E100"
    bad_mat = [{"material": {"type": "orthotropic_2d", "E1": 1e5, "E2": 1e4, "G12": 5e3,
                             "nu12": 0.3}, "volume_fraction": 0.5},
               {"material": EP, "volume_fraction": 0.5}]
    assert srv.homogenize_patterned_layer(bad_mat)["errors"][0]["code"] == "E100"


def test_material_is_directly_usable_and_deterministic():
    mat = srv.homogenize_patterned_layer(comps())["data"]["material"]
    lam = {"unit_system": "SI_mm",
           "laminae": [{"material": mat, "angle_deg": 45.0, "thickness": 0.035}]}
    assert srv.analyze_laminate(lam)["errors"] == []
    a = srv.homogenize_patterned_layer(comps())
    b = srv.homogenize_patterned_layer(comps())
    assert a["data"] == b["data"] and a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]
