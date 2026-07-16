# Tool 동작 테스트 — analyze 블록 구성, 기준 판정, 자가 검증 루프, 단위 브리지 (계획서 §6.3)
from __future__ import annotations

import pytest

import app.mcp_server as srv

T300_MM = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}


def sym_cross_ply():
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": 0.125, "angle_deg": a, "material": dict(T300_MM)}
                        for a in (0.0, 90.0, 90.0, 0.0)]}


def test_analyze_full_blocks():
    env = srv.analyze_laminate(sym_cross_ply())
    assert env["status"] == "ok"
    d = env["data"]
    assert d["laminate_summary"]["n_plies"] == 4
    assert d["laminate_summary"]["is_symmetric_stack"] is True
    assert set(d["abd"].keys()) == {"A", "B", "D", "A_hat", "B_hat", "D_hat"}
    assert abs(d["abd"]["B"][0][0]) < 1e-9 * abs(d["abd"]["A"][0][0])
    assert d["neutral_surface"]["x"]["zeta"] == pytest.approx(0.5, abs=1e-9)
    assert d["indices"]["coupling_ratio"]["grade"] == "negligible"
    assert env["metadata"]["units"]["E"] == "MPa"
    assert env["metadata"]["payload_hash"].startswith("sha256:")


def test_include_subset_and_6x6():
    env = srv.compute_abd_matrix(sym_cross_ply(), include_abd_6x6=True)
    d = env["data"]
    assert "abd" in d and "neutral_surface" not in d and "indices" not in d
    assert len(d["abd"]["ABD_6x6_display_units"]) == 6


def test_neutral_axis_modes_agree_for_symmetric():
    for mode in ("clt_weighted", "beam_equivalent"):
        env = srv.compute_neutral_axis(sym_cross_ply(), mode=mode)
        assert env["data"]["neutral_surface"]["x"]["zeta"] == pytest.approx(0.5, abs=1e-9)
        assert env["data"]["neutral_surface"]["mode"] == mode


def test_evaluate_with_criteria():
    env = srv.evaluate_laminate(sym_cross_ply(),
                                criteria={"max_coupling_ratio": 0.05,
                                          "min_quasi_isotropy_score": 0.99,
                                          "미지원기준": 1.0})
    pf = {r["criterion"]: r for r in env["data"]["evaluation"]["pass_fail"]}
    assert pf["max_coupling_ratio"]["pass"] is True
    assert pf["min_quasi_isotropy_score"]["pass"] is False  # 크로스플라이는 A66 조건 미달
    assert pf["미지원기준"]["pass"] is None


def test_reference_case_self_verification_loop():
    """get_reference_cases의 input을 analyze에 넣어 expected와 대조 — 에이전트 자가 검증 경로 그대로."""
    listing = srv.get_reference_cases()
    assert {c["case_id"] for c in listing["cases"]} == {
        "single_isotropic_alu", "cross_ply_symmetric", "cross_ply_unsymmetric"}

    for cid in ("single_isotropic_alu", "cross_ply_symmetric", "cross_ply_unsymmetric"):
        case = srv.get_reference_cases(cid)
        env = srv.analyze_laminate(case["input"]["laminate"])
        assert env["status"] in ("ok", "warning")
        d, exp = env["data"], case["expected"]
        rtol = case["tolerance"]["rtol"]
        if "A11_N_per_mm" in exp:
            assert d["abd"]["A"][0][0] == pytest.approx(exp["A11_N_per_mm"], rel=rtol)
        if "D11_N_mm" in exp:
            assert d["abd"]["D"][0][0] == pytest.approx(exp["D11_N_mm"], rel=rtol)
        if "D22_N_mm" in exp:
            assert d["abd"]["D"][1][1] == pytest.approx(exp["D22_N_mm"], rel=rtol)
        if "B11_N" in exp:
            assert d["abd"]["B"][0][0] == pytest.approx(exp["B11_N"], rel=rtol)
        if exp.get("B_zero"):
            assert abs(d["abd"]["B"][0][0]) < 1e-9 * abs(d["abd"]["A"][0][0])
        if "z_ns_x_from_midplane_mm" in exp:
            assert d["neutral_surface"]["x"]["z_from_midplane"] == pytest.approx(
                exp["z_ns_x_from_midplane_mm"], rel=rtol)
        if "zeta_x" in exp:
            assert d["neutral_surface"]["x"]["zeta"] == pytest.approx(exp["zeta_x"], abs=1e-9)
        if "quasi_isotropy_score" in exp:
            assert d["indices"]["quasi_isotropy_score"]["value"] == pytest.approx(
                exp["quasi_isotropy_score"], abs=1e-9)
        if "is_symmetric_stack" in exp:
            assert d["laminate_summary"]["is_symmetric_stack"] is exp["is_symmetric_stack"]

    assert "error" in srv.get_reference_cases("없는케이스")


def test_units_bridge_si_vs_si_mm():
    """같은 물리 적층을 SI/SI_mm로 넣으면 표시값이 정확히 단위 변환 관계여야 한다 (§3 D4)."""
    si = {"unit_system": "SI",
          "laminae": [{"thickness": 0.125e-3, "angle_deg": a,
                       "material": {"type": "orthotropic_2d", "E1": 181e9, "E2": 10.3e9,
                                    "G12": 7.17e9, "nu12": 0.28}} for a in (0.0, 90.0)]}
    mm = {"unit_system": "SI_mm",
          "laminae": [{"thickness": 0.125, "angle_deg": a, "material": dict(T300_MM)}
                      for a in (0.0, 90.0)]}
    d_si = srv.analyze_laminate(si)["data"]
    d_mm = srv.analyze_laminate(mm)["data"]
    assert d_mm["abd"]["A"][0][0] == pytest.approx(d_si["abd"]["A"][0][0] * 1e-3, rel=1e-12)
    assert d_mm["abd"]["B"][0][0] == pytest.approx(d_si["abd"]["B"][0][0], rel=1e-12)
    assert d_mm["abd"]["D"][0][0] == pytest.approx(d_si["abd"]["D"][0][0] * 1e3, rel=1e-12)
    assert d_mm["neutral_surface"]["x"]["zeta"] == pytest.approx(
        d_si["neutral_surface"]["x"]["zeta"], rel=1e-12)


def test_get_server_info():
    info = srv.get_server_info()
    assert info["server"] == "laminate-analyzer"
    assert set(info["unit_systems"]) == {"SI", "SI_mm"}
    assert info["limits"]["max_plies"] == 512
    assert "E101" in info["error_codes"]


def test_validate_tool_ok_and_error():
    env = srv.validate_laminate_input(sym_cross_ply())
    assert env["status"] == "ok" and env["data"]["valid"] is True
    env2 = srv.validate_laminate_input({"laminae": []})
    assert env2["status"] == "error" and env2["data"] is None
