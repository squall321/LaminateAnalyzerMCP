# 피로 수명 테스트 — 성분별 부호 보존 S-N + Goodman (계획서 §17.7, 적대 검증 FAT-01~14 반영)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import fatigue as FAT

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
S = {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}
SN_LOGLIN = {"model_type": "log_linear", "k": 0.1}
SN_BASQUIN = {"model_type": "basquin", "b": 0.1}


def lam(angles=(0, 90, 90, 0), t=0.125, sn=None, strength=None):
    m = dict(T300, strength=dict(strength or S), fatigue=dict(sn or SN_LOGLIN))
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": t, "angle_deg": a, "material": dict(m)} for a in angles]}


# ── solver 폐형해 ────────────────────────────────────────────────────────────

def test_sn_models_roundtrip():
    """S-N 역산이 정확 — log_linear·basquin 둘 다."""
    assert FAT.cycles_to_failure(0.5, "log_linear", 0.1) == pytest.approx(10 ** ((1 - 0.5) / 0.1))
    assert FAT.cycles_to_failure(0.5, "basquin", 0.1) == pytest.approx(0.5 ** (-10), rel=1e-9)
    assert FAT.cycles_to_failure(1.0, "log_linear", 0.1) == 1.0     # 정적 파손
    assert FAT.cycles_to_failure(0.0, "log_linear", 0.1) is None    # 진폭 0 → 무한


def test_overflow_capped_not_crash():
    """FAT-04/05: 스키마 허용 범위의 작은 k·b에서 OverflowError가 아니라 cap."""
    for model, param in (("log_linear", 1e-3), ("basquin", 1e-3), ("log_linear", 1e-9)):
        n = FAT.cycles_to_failure(0.5, model, param)
        assert n == FAT.N_CAP


def test_component_sign_preserved():
    """FAT-01의 근본: 부호가 살아 완전반복이 최대 진폭으로 잡힌다."""
    st = (1500.0, 1200.0, 40.0, 246.0, 68.0)
    rev = FAT.assess_component(100.0, -100.0, 0, st, "log_linear", 0.1)   # R=-1
    pul = FAT.assess_component(100.0, 0.0, 0, st, "log_linear", 0.1)      # R=0
    assert rev["sigma_amplitude"] == pytest.approx(100.0)
    assert rev["sigma_mean"] == pytest.approx(0.0)
    assert pul["sigma_amplitude"] == pytest.approx(50.0)
    assert rev["cycles_to_failure"] < pul["cycles_to_failure"]           # 반복이 더 가혹


def test_goodman_tension_only_penalty():
    """표준 Goodman: 인장 평균만 감산, 압축 평균은 무감산."""
    st = (1500.0, 1200.0, 40.0, 246.0, 68.0)
    zero_mean = FAT.assess_component(100.0, -100.0, 0, st, "log_linear", 0.1)
    tens_mean = FAT.assess_component(300.0, 100.0, 0, st, "log_linear", 0.1)   # 진폭 100, 평균 200
    comp_mean = FAT.assess_component(-100.0, -300.0, 0, st, "log_linear", 0.1)  # 진폭 100, 평균 -200
    assert tens_mean["r_alternating"] > zero_mean["r_alternating"]   # 인장 평균 → 가혹
    assert comp_mean["r_alternating"] == pytest.approx(comp_mean["r_amplitude"], rel=1e-12)


def test_static_overload_reported_as_one_cycle():
    """FAT-03: 정적 한계를 넘는 평균은 진폭이 작아도 1사이클 파손."""
    st = (1500.0, 1200.0, 40.0, 246.0, 68.0)
    r = FAT.assess_component(1600.0, 1500.0, 0, st, "log_linear", 0.1)
    assert r["cycles_to_failure"] == 1.0 and "정적 한계" in r["note"]


# ── Tool 통합 (FAT 회귀) ────────────────────────────────────────────────────

def test_fat01_fully_reversed_is_most_severe():
    """FAT-01 회귀: 완전반복(R=−1)이 '무한수명'이 아니라 가장 짧은 수명."""
    rev = srv.estimate_fatigue_life(lam(), loads_max={"N": [100.0, 0, 0]},
                                    loads_min={"N": [-100.0, 0, 0]})["data"]
    pul = srv.estimate_fatigue_life(lam(), loads_max={"N": [100.0, 0, 0]})["data"]
    assert rev["life_cycles"] is not None and rev["life_cycles"] > 1.0
    assert rev["life_cycles"] < pul["life_cycles"]
    assert rev["critical_ply"]["governing_component"] in ("sigma_1", "sigma_2", "tau_12")


def test_fat02_argument_order_irrelevant():
    """FAT-02/04 회귀: loads_max/min 라벨 순서가 결과를 바꾸지 않는다."""
    a = srv.estimate_fatigue_life(lam(), loads_max={"N": [100.0, 0, 0]},
                                  loads_min={"N": [-100.0, 0, 0]})["data"]["life_cycles"]
    b = srv.estimate_fatigue_life(lam(), loads_max={"N": [-100.0, 0, 0]},
                                  loads_min={"N": [100.0, 0, 0]})["data"]["life_cycles"]
    assert a == pytest.approx(b, rel=1e-12)


def test_fat05_compression_only_cycle_supported():
    """FAT-05 회귀: 0 → −200 압축 전용 사이클을 표현할 수 있다."""
    env = srv.estimate_fatigue_life(lam(), loads_max={"N": [0.0, 0, 0]},
                                    loads_min={"N": [-200.0, 0, 0]})
    assert env["status"] in ("ok", "warning") and env["data"]["life_cycles"] > 1.0


def test_fat03_excluded_plies_warn():
    """FAT-03/06 회귀: 평가 제외 ply가 W120으로 보고된다(과대평가 위험 명시)."""
    mixed = {"unit_system": "SI_mm", "laminae": [
        {"thickness": 0.125, "angle_deg": 0,
         "material": dict(T300, strength=dict(S), fatigue=dict(SN_LOGLIN))},
        {"thickness": 0.125, "angle_deg": 90, "material": dict(T300)}]}
    env = srv.estimate_fatigue_life(mixed, loads_max={"N": [100.0, 0, 0]})
    assert env["status"] == "warning"
    w = [x for x in env["warnings"] if x["code"] == "W120"]
    assert w and "과대평가" in w[0]["message"] + w[0]["suggestion"] or "과대평가" in w[0]["message"]
    assert env["data"]["excluded_plies"] == [1]


def test_fat14_nonfinite_loads_are_e100():
    """FAT-14 회귀: NaN/inf 하중이 E400(적층 탓)이 아니라 E100(하중 탓)."""
    for bad in ({"N": [float("nan"), 0, 0]}, {"N": [float("inf"), 0, 0]}):
        env = srv.estimate_fatigue_life(lam(), loads_max=bad)
        assert env["errors"][0]["code"] == "E100"
        assert "loads_max" in env["errors"][0]["field"]
    # FAT-06: 어느 인자가 잘못됐는지 구분된다
    env2 = srv.estimate_fatigue_life(lam(), loads_max={"N": [1.0, 0, 0]},
                                     loads_min={"N": [1, 2]})
    assert "loads_min" in env2["errors"][0]["field"]


def test_identical_states_rejected():
    """사이클 두 끝이 같으면 피로가 정의되지 않는다 — 조용한 무한수명 대신 E100."""
    env = srv.estimate_fatigue_life(lam(), loads_max={"N": [50.0, 0, 0]},
                                    loads_min={"N": [50.0, 0, 0]})
    assert env["errors"][0]["code"] == "E100"


def test_fat08_critical_ply_tie_break():
    """FAT-08 회귀: 수명 동률이면 더 가혹한(r_alternating 큰) ply가 임계로 보고된다."""
    heavy = srv.estimate_fatigue_life(lam(), loads_max={"N": [3000.0, 0, 0]})["data"]
    assert heavy["life_cycles"] == 1.0
    if "tied_plies" in heavy:
        crit = heavy["critical_ply"]
        for r in heavy["plies"]:
            if r.get("cycles_to_failure") == 1.0 and r.get("r_alternating") and crit.get("r_alternating"):
                assert crit["r_alternating"] >= r["r_alternating"]


def test_fat07_response_summary():
    """FAT-07 회귀: 두꺼운 적층은 수명 짧은 순 요약 + detail=full."""
    big = {"unit_system": "SI_mm",
           "laminae": [{"thickness": 0.125, "angle_deg": (0, 45, -45, 90)[i % 4],
                        "material": dict(T300, strength=dict(S), fatigue=dict(SN_LOGLIN))}
                       for i in range(64)]}
    auto = srv.estimate_fatigue_life(big, loads_max={"N": [100.0, 0, 0]})["data"]
    full = srv.estimate_fatigue_life(big, loads_max={"N": [100.0, 0, 0]}, detail="full")["data"]
    assert "truncation" in auto and len(auto["plies"]) < len(full["plies"])
    assert any(r["ply"] == auto["critical_ply"]["ply"] for r in auto["plies"])
    assert srv.estimate_fatigue_life(big, loads_max={"N": [1.0, 0, 0]},
                                     detail="x")["errors"][0]["code"] == "E100"


def test_load_magnitude_monotonic_and_models():
    a = srv.estimate_fatigue_life(lam(), loads_max={"N": [50.0, 0, 0]})["data"]["life_cycles"]
    b = srv.estimate_fatigue_life(lam(), loads_max={"N": [100.0, 0, 0]})["data"]["life_cycles"]
    assert a > b > 1.0
    bq = srv.estimate_fatigue_life(lam(sn=SN_BASQUIN), loads_max={"N": [100.0, 0, 0]})["data"]
    assert bq["critical_ply"]["model"] == "basquin"


def test_validation_and_determinism():
    from app.services.envelope import canonical_json
    bad = lam()
    bad["laminae"][0]["material"]["fatigue"] = {"model_type": "log_linear"}      # k 누락
    assert srv.estimate_fatigue_life(bad, loads_max={"N": [1, 0, 0]})["errors"][0]["code"] == "E100"
    plain = {"unit_system": "SI_mm", "laminae": [{"thickness": 1.0, "angle_deg": 0,
             "material": dict(T300)}]}
    assert srv.estimate_fatigue_life(plain, loads_max={"N": [1, 0, 0]})["errors"][0]["code"] == "E100"
    p = lam()
    assert canonical_json(srv.estimate_fatigue_life(p, loads_max={"N": [80.0, 0, 0]})) == \
           canonical_json(srv.estimate_fatigue_life(p, loads_max={"N": [80.0, 0, 0]}))


def test_cfrp_sn_rule_of_thumb():
    """문헌 경험칙 대조: CFRP k≈0.1이면 10^7 사이클 강도 ≈ 정적의 30%."""
    r_at_1e7 = 1.0 - 0.1 * math.log10(1e7)
    assert r_at_1e7 == pytest.approx(0.30, abs=1e-9)
    assert FAT.cycles_to_failure(0.30, "log_linear", 0.1) == pytest.approx(1e7, rel=1e-9)


def test_fatigue_reference_case():
    """FAT-08 회귀: 피로 few-shot이 존재하고 실제 엔진과 일치."""
    c = srv.get_reference_cases("fatigue_reversed_cycle")
    rev = srv.estimate_fatigue_life(c["input"]["laminate"], loads_max=c["input"]["loads_max"],
                                    loads_min=c["input"]["loads_min"])["data"]
    pul = srv.estimate_fatigue_life(c["input"]["laminate"],
                                    loads_max=c["input"]["loads_max"])["data"]
    assert rev["life_cycles"] < pul["life_cycles"]
    assert rev["critical_ply"]["governing_component"] in c["expected"]["governing_component_in"]
