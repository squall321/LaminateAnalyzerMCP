# 오류·경고 코드 유발 테스트 — 구현된 전 코드에 트리거 존재 (계획서 S4, §6.4)
from __future__ import annotations

import pytest

from app import config
from app.errors import CATALOG
from app.services import pipeline as PIPE

T300_MM = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}


def lam(laminae, unit_system="SI_mm", **kw):
    return {"unit_system": unit_system, "laminae": laminae, **kw}


def ply(t=0.125, a=0.0, mat=None):
    return {"thickness": t, "angle_deg": a, "material": dict(mat or T300_MM)}


def codes_of(env, kind="errors"):
    return [e["code"] for e in env[kind]]


def test_e100_not_a_dict():
    assert "E100" in codes_of(PIPE.run_analysis("nonsense"))


def test_e100_bad_type_in_field():
    env = PIPE.run_analysis(lam([{"thickness": "두껍게", "angle_deg": 0, "material": T300_MM}]))
    assert "E100" in codes_of(env)
    assert any("thickness" in (e.get("field") or "") for e in env["errors"])


def test_e100_bad_include_and_mode():
    good = lam([ply()])
    assert "E100" in codes_of(PIPE.run_analysis(good, include=("abd", "무지개")))
    assert "E100" in codes_of(PIPE.run_analysis(good, neutral_axis_mode="full_coupled"))


def test_e101_unit_system_required():
    env = PIPE.run_analysis({"laminae": [ply()]})
    assert codes_of(env) == ["E101"]
    assert "E101" in codes_of(PIPE.run_analysis(lam([ply()], unit_system="ENG")))


def test_e102_empty_and_e103_too_many():
    assert "E102" in codes_of(PIPE.run_analysis(lam([])))
    many = lam([ply() for _ in range(config.MAX_PLIES + 1)])
    assert "E103" in codes_of(PIPE.run_analysis(many))


def test_e104_payload_too_large():
    big = lam([ply()], name="x" * (config.MAX_PAYLOAD_BYTES + 10))
    assert "E104" in codes_of(PIPE.run_analysis(big))


def test_e200_non_positive_modulus():
    bad = dict(T300_MM, E1=-1.0)
    env = PIPE.run_analysis(lam([ply(mat=bad)]))
    assert "E200" in codes_of(env)


def test_e201_poisson_unstable():
    bad = dict(T300_MM, E1=10000.0, E2=10000.0, nu12=1.5)
    assert "E201" in codes_of(PIPE.run_analysis(lam([ply(mat=bad)])))
    iso_bad = {"type": "isotropic", "E": 70000.0, "nu": 0.7}
    assert "E201" in codes_of(PIPE.run_analysis(lam([ply(mat=iso_bad)])))


def test_e202_invalid_material_type():
    env = PIPE.run_analysis(lam([ply(mat={"type": "viscoelastic", "E": 1.0})]))
    assert "E202" in codes_of(env)


def test_e300_e301_geometry():
    assert "E300" in codes_of(PIPE.run_analysis(lam([ply(t=0.0)])))
    assert "E301" in codes_of(PIPE.run_analysis(lam([ply(a=400.0)])))


def test_e402_not_positive_definite(monkeypatch):
    monkeypatch.setattr(PIPE.ABD, "is_positive_definite", lambda K: False)
    env = PIPE.run_analysis(lam([ply()]))
    assert codes_of(env) == ["E402"] and env["data"] is None


def test_e403_nan_defense(monkeypatch):
    monkeypatch.setattr(PIPE.NA, "clt_weighted", lambda A, B: (float("nan"), 0.0))
    env = PIPE.run_analysis(lam([ply()]))
    assert "E403" in codes_of(env)


def test_e501_internal_error(monkeypatch):
    import app.mcp_server as srv
    monkeypatch.setattr(srv.PIPE, "run_analysis", lambda *a, **k: 1 / 0)
    env = srv.analyze_laminate(lam([ply()]))
    assert "E501" in codes_of(env)
    assert "ZeroDivisionError" in env["errors"][0]["message"]


def test_w110_unit_heuristics():
    si_gpa_mistake = lam([{"thickness": 0.000125, "angle_deg": 0,
                           "material": dict(T300_MM, E1=181.0, E2=10.3, G12=7.17)}], unit_system="SI")
    env = PIPE.run_analysis(si_gpa_mistake)
    assert "W110" in codes_of(env, "warnings")

    si_thick = lam([{"thickness": 2.0, "angle_deg": 0, "material":
                     {"type": "isotropic", "E": 70e9, "nu": 0.33}}], unit_system="SI")
    assert "W110" in codes_of(PIPE.run_analysis(si_thick), "warnings")

    mm_pa_mistake = lam([ply(mat=dict(T300_MM, E1=181e9))])
    assert "W110" in codes_of(PIPE.run_analysis(mm_pa_mistake), "warnings")


def test_w111_thin_ply():
    env = PIPE.run_analysis(lam([ply(t=1.0), ply(t=1.0e-7)]))
    assert "W111" in codes_of(env, "warnings")


def test_w112_extreme_ratio():
    mat = dict(T300_MM, E1=300000.0, E2=1.0, G12=1.0, nu12=0.001)
    env = PIPE.run_analysis(lam([ply(mat=mat)]))
    assert "W112" in codes_of(env, "warnings")


def test_w120_assumed_constant():
    mat = dict(T300_MM, source={"type": "assumed", "ref": "문헌 대표값"})
    env = PIPE.run_analysis(lam([ply(mat=mat)]))
    assert "W120" in codes_of(env, "warnings")
    assert any("assumed" in a for a in env["assumptions"])


def test_w200_high_coupling():
    env = PIPE.run_analysis(lam([ply(a=0.0), ply(a=90.0)]))
    assert "W200" in codes_of(env, "warnings")
    assert env["data"]["indices"]["coupling_ratio"]["grade"] == "high"


def test_w401_ill_conditioned(monkeypatch):
    monkeypatch.setattr(config, "COND_WARN_THRESHOLD", 1.0)
    env = PIPE.run_analysis(lam([ply()]))
    assert "W401" in codes_of(env, "warnings")


def test_w402_index_undefined(monkeypatch):
    monkeypatch.setattr(config, "DENOM_EPS", 1e300)
    env = PIPE.run_analysis(lam([ply()]))
    assert "W402" in codes_of(env, "warnings")
    assert env["data"]["indices"]["coupling_ratio"]["value"] is None


def test_catalog_complete():
    """계획서 §6.4의 코드 전수가 카탈로그에 존재 (E400/E500은 V1 하중응답·타임아웃에서 트리거 예정)."""
    expected = {"E100", "E101", "E102", "E103", "E104", "E200", "E201", "E202", "E203",
                "E300", "E301", "E400", "E402", "E403", "E500", "E501",
                "W110", "W111", "W112", "W120", "W130", "W200", "W401", "W402"}
    assert expected == set(CATALOG)
