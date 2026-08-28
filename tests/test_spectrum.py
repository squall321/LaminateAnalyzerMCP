# §19.20 하중 스펙트럼 Miner 누적손상 (2순위 7번)
from __future__ import annotations

import pytest

import app.mcp_server as srv

T = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0, "nu12": 0.28,
     "alpha1": 0.02e-6, "alpha2": 22.5e-6,
     "strength": {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0},
     "fatigue": {"model_type": "log_linear", "k": 0.1}}


def lam(angles=(0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0)):
    return {"unit_system": "SI_mm",
            "laminae": [{"material": T, "angle_deg": a, "thickness": 0.125} for a in angles]}


BLOCKS = [{"name": "열사이클", "loads_max": {"N": [30.0, 0.0, 0.0]}, "cycles": 1000.0},
          {"name": "진동", "loads_max": {"N": [0.0, 0.0, 15.0]},
           "loads_min": {"N": [0.0, 0.0, -15.0]}, "cycles": 1e6},
          {"name": "취급충격", "loads_max": {"N": [80.0, 0.0, 0.0]}, "cycles": 50.0}]


def test_miner_sum_and_repeats():
    """D = Σ n_i/N_f,i 이고 repeats = 1/D 다."""
    d = srv.estimate_spectrum_life(lam(), blocks=BLOCKS)["data"]
    assert d["total_damage"] == pytest.approx(sum(b["damage"] for b in d["blocks"]), rel=1e-12)
    assert d["repeats_to_failure"] == pytest.approx(1.0 / d["total_damage"], rel=1e-12)
    for b in d["blocks"]:
        assert b["damage"] == pytest.approx(b["cycles_applied"] / b["cycles_to_failure"],
                                            rel=1e-12)


def test_single_block_matches_estimate_fatigue_life():
    """블록 1개면 estimate_fatigue_life 와 같은 N_f 를 준다 (같은 경로 재사용)."""
    blk = {"loads_max": {"N": [60.0, 0.0, 0.0]}, "cycles": 1.0}
    a = srv.estimate_spectrum_life(lam(), blocks=[blk])["data"]["blocks"][0]
    b = srv.estimate_fatigue_life(lam(), loads_max=blk["loads_max"])["data"]
    assert a["cycles_to_failure"] == pytest.approx(b["life_cycles"], rel=1e-12)
    assert a["critical_ply"] == b["critical_ply"]["ply"]


def test_damage_is_linear_in_cycles():
    """같은 블록의 반복수를 2배 하면 손상이 정확히 2배."""
    one = srv.estimate_spectrum_life(
        lam(), blocks=[{"loads_max": {"N": [60.0, 0.0, 0.0]}, "cycles": 1000.0}])["data"]
    two = srv.estimate_spectrum_life(
        lam(), blocks=[{"loads_max": {"N": [60.0, 0.0, 0.0]}, "cycles": 2000.0}])["data"]
    assert two["total_damage"] == pytest.approx(2.0 * one["total_damage"], rel=1e-12)


def test_order_independence_of_blocks():
    """Miner 는 순서 무관 — 블록 순서를 바꿔도 총 손상이 같다 (그 한계를 W130 으로 알린다)."""
    a = srv.estimate_spectrum_life(lam(), blocks=BLOCKS)
    b = srv.estimate_spectrum_life(lam(), blocks=list(reversed(BLOCKS)))
    assert a["data"]["total_damage"] == pytest.approx(b["data"]["total_damage"], rel=1e-12)
    assert any(w["code"] == "W130" and "순서를 무시" in w["message"] for w in a["warnings"])


def test_mixed_governing_components_are_flagged():
    """블록마다 지배 성분·임계 ply 가 다르면 손합산이 틀린다 — 이 도구의 존재 이유."""
    env = srv.estimate_spectrum_life(lam(), blocks=BLOCKS)
    d = env["data"]
    assert len(d["governing_components"]) > 1
    assert len(d["damage_by_ply"]) > 1
    assert sum(p["fraction"] for p in d["damage_by_ply"]) == pytest.approx(1.0, rel=1e-9)
    assert any(w["code"] == "W130" and "지배 성분이 다르다" in w["message"] for w in env["warnings"])
    assert any(w["code"] == "W120" and "임계 ply 가 다르다" in w["message"] for w in env["warnings"])


def test_thermal_residual_dominates_spectrum():
    """경화 냉각 잔류를 넣으면 손상이 자릿수로 뛴다 (§19.10 과 정합)."""
    dry = srv.estimate_spectrum_life(lam(), blocks=BLOCKS)["data"]["total_damage"]
    wet = srv.estimate_spectrum_life(lam(), blocks=BLOCKS, delta_T=-150.0)["data"]
    assert wet["total_damage"] > 1000 * dry
    assert wet["delta_T"] == -150.0
    assert any(w["code"] == "W130" and "이미 파손" in w["message"]
               for w in srv.estimate_spectrum_life(lam(), blocks=BLOCKS,
                                                   delta_T=-150.0)["warnings"])


def test_errors_propagate_with_block_index():
    assert srv.estimate_spectrum_life(lam(), blocks=[])["errors"][0]["code"] == "E100"
    assert srv.estimate_spectrum_life(lam(), blocks="x")["errors"][0]["code"] == "E100"
    for bad in ({"loads_max": {"N": [10.0, 0.0, 0.0]}, "cycles": -1.0},
                {"loads_max": {"N": [10.0, 0.0, 0.0]}, "cycles": 10.0, "typo": 1},
                {"cycles": 10.0}):
        assert srv.estimate_spectrum_life(lam(), blocks=[bad])["errors"][0]["code"] == "E100"
    # 블록 인덱스가 오류 field 에 실린다
    env = srv.estimate_spectrum_life(lam(), blocks=[
        {"loads_max": {"N": [10.0, 0.0, 0.0]}, "cycles": 10.0},
        {"loads_max": {"N": [0.0, 0.0, 0.0]}, "cycles": 10.0}])
    assert env["errors"] and "blocks[1]" in env["errors"][0]["field"]


def test_deterministic():
    a = srv.estimate_spectrum_life(lam(), blocks=BLOCKS)
    b = srv.estimate_spectrum_life(lam(), blocks=BLOCKS)
    assert a["data"] == b["data"]
    assert a["metadata"]["payload_hash"] == b["metadata"]["payload_hash"]
