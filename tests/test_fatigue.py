# 피로 수명 테스트 — S-N 환원·Goodman·단축 고전식 대조 (계획서 §17.7)
from __future__ import annotations

import math

import pytest

import app.mcp_server as srv
from app.solver import fatigue as FAT

T300 = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28}
S = {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}
SN_LOGLIN = {"model_type": "log_linear", "k": 0.1}
SN_BASQUIN = {"model_type": "basquin", "b": 0.1}


def lam(angles=(0, 90, 90, 0), t=0.125, sn=None):
    m = dict(T300, strength=dict(S), fatigue=dict(sn or SN_LOGLIN))
    return {"unit_system": "SI_mm",
            "laminae": [{"thickness": t, "angle_deg": a, "material": dict(m)} for a in angles]}


# ── solver 단위 (폐형해) ────────────────────────────────────────────────────

def test_goodman_and_sn_closed_forms():
    """R=−1(완전 반복): 평균 0이므로 FI_ar = FI_a. log-linear 역산이 정확."""
    eq = FAT.equivalent_alternating(0.5, -0.5)
    assert eq["FI_mean"] == pytest.approx(0.0) and eq["FI_ar"] == pytest.approx(0.5)
    n = FAT.cycles_to_failure(0.5, "log_linear", 0.1)
    assert n == pytest.approx(10 ** ((1 - 0.5) / 0.1), rel=1e-12)      # 1e5
    nb = FAT.cycles_to_failure(0.5, "basquin", 0.1)
    assert nb == pytest.approx(0.5 ** (-10), rel=1e-12)               # 1024

    # R=0(영-인장): 평균 = 진폭 → Goodman이 등가 교번을 키운다
    eq0 = FAT.equivalent_alternating(0.5, 0.0)
    assert eq0["FI_amplitude"] == pytest.approx(0.25) and eq0["FI_mean"] == pytest.approx(0.25)
    assert eq0["FI_ar"] == pytest.approx(0.25 / 0.75, rel=1e-12)
    assert FAT.cycles_to_failure(eq0["FI_ar"], "log_linear", 0.1) > 1e5  # 완전반복보다 길다


def test_static_and_infinite_limits():
    assert FAT.cycles_to_failure(1.0, "log_linear", 0.1) == 1.0        # FI_ar=1 → 정적 파손
    assert FAT.cycles_to_failure(1.5, "basquin", 0.1) == 1.0
    assert FAT.cycles_to_failure(0.0, "log_linear", 0.1) is None       # 무한수명
    assert math.isinf(FAT.equivalent_alternating(1.2, 1.1)["FI_ar"])   # 평균이 정적한계 초과


# ── Tool 통합 ────────────────────────────────────────────────────────────────

def test_fatigue_life_decreases_with_load():
    a = srv.estimate_fatigue_life(lam(), loads_max={"N": [50.0, 0, 0]})["data"]["life_cycles"]
    b = srv.estimate_fatigue_life(lam(), loads_max={"N": [100.0, 0, 0]})["data"]["life_cycles"]
    assert a > b > 1.0


def test_fatigue_critical_ply_is_90_transverse():
    """[0/90]s 인장 피로: 정적과 마찬가지로 90° ply가 임계."""
    env = srv.estimate_fatigue_life(lam(), loads_max={"N": [100.0, 0, 0]})
    d = env["data"]
    assert d["critical_ply"]["ply"] in (1, 2)
    # 정적 FPF와 정합: FI_max = 1/R_static
    fpf = srv.recover_ply_stresses(lam(), loads={"N": [100.0, 0, 0]})["data"]["first_ply_failure"]
    fi_expected = 1.0 / fpf["tsai_wu_R"]
    fi_max = d["critical_ply"]["FI_amplitude"] * 2 + 0.0     # R=0이면 FI_max = 2·진폭
    assert fi_max == pytest.approx(fi_expected, rel=1e-6)


def test_fatigue_matches_uniaxial_hand_calc():
    """단축 케이스에서 손계산과 일치 — FI=1/R, Goodman, log-linear 역산."""
    env = srv.estimate_fatigue_life(lam(), loads_max={"N": [100.0, 0, 0]})
    c = env["data"]["critical_ply"]
    fi_ar = c["FI_ar"]
    assert c["cycles_to_failure"] == pytest.approx(10 ** ((1 - fi_ar) / 0.1), rel=1e-9)
    # R=0 사이클이므로 FI_m = FI_a, Goodman 관계 확인
    assert c["FI_mean"] == pytest.approx(c["FI_amplitude"], rel=1e-12)
    assert fi_ar == pytest.approx(c["FI_amplitude"] / (1 - c["FI_mean"]), rel=1e-12)


def test_fatigue_mean_stress_effect():
    """같은 진폭이라도 평균이 크면 수명이 짧다 (Goodman)."""
    full_rev = srv.estimate_fatigue_life(lam(), loads_max={"N": [60.0, 0, 0]},
                                         loads_min={"N": [-60.0, 0, 0]})["data"]["life_cycles"]
    pulsating = srv.estimate_fatigue_life(lam(), loads_max={"N": [120.0, 0, 0]},
                                          loads_min={"N": [0.0, 0, 0]})["data"]["life_cycles"]
    assert pulsating < full_rev      # 같은 진폭 60, 평균 60 vs 0


def test_fatigue_models_and_validation():
    bq = srv.estimate_fatigue_life(lam(sn=SN_BASQUIN), loads_max={"N": [100.0, 0, 0]})
    assert bq["data"]["critical_ply"]["model"] == "basquin"
    # 파라미터 누락
    bad = lam()
    bad["laminae"][0]["material"]["fatigue"] = {"model_type": "log_linear"}
    assert srv.estimate_fatigue_life(bad, loads_max={"N": [1, 0, 0]})["errors"][0]["code"] == "E100"
    # strength/fatigue 없는 적층
    plain = {"unit_system": "SI_mm", "laminae": [{"thickness": 1.0, "angle_deg": 0,
             "material": dict(T300)}]}
    assert srv.estimate_fatigue_life(plain, loads_max={"N": [1, 0, 0]})["errors"][0]["code"] == "E100"


def test_fatigue_low_load_long_life_and_static_cycle():
    """작은 하중 → 매우 긴 수명. 진폭 0(정적 유지)이면 피로 손상 없음(무한)."""
    tiny = srv.estimate_fatigue_life(lam(), loads_max={"N": [0.5, 0, 0]})["data"]
    assert tiny["life_cycles"] > 1e6          # log-linear은 내구한도 개념이 없어 유한값
    same = srv.estimate_fatigue_life(lam(), loads_max={"N": [50.0, 0, 0]},
                                     loads_min={"N": [50.0, 0, 0]})["data"]
    assert same["life_cycles"] is None        # 진폭 0 → 무한수명
    assert all(r.get("infinite_life") for r in same["plies"])


def test_fatigue_determinism():
    from app.services.envelope import canonical_json
    p = lam()
    assert canonical_json(srv.estimate_fatigue_life(p, loads_max={"N": [80.0, 0, 0]})) == \
           canonical_json(srv.estimate_fatigue_life(p, loads_max={"N": [80.0, 0, 0]}))
