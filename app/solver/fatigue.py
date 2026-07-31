# 피로 수명 — 정규화 파손지수 공간의 Basquin/log-linear S-N + Goodman 평균응력 보정 (계획서 §17.7)
"""하중 사이클(최대·최소)에서 ply별 반복 수명 N_f를 추정한다.

**정규화 접근**: 정적 파손 판정(Tsai-Wu 강도비 R)의 역수 FI = 1/R 은 '정적 한계 대비 몇 %'라
FI=1이 정적 파손이다. 이 공간에서 S-N을 세우면 다축 응력·적층각을 별도 처리하지 않아도 되고,
단축 케이스에서 σ/σ_u 기반 고전 S-N과 정확히 일치한다.

  진폭 FI_a = (FI_max − FI_min)/2, 평균 FI_m = (FI_max + FI_min)/2
  Goodman 등가 교번: FI_ar = FI_a / (1 − FI_m)   (FI_m ≥ 1이면 정적 파손)
  log_linear: FI_ar = 1 − k·log10(N)  → N = 10^((1 − FI_ar)/k)   [복합재 관례, k≈0.1]
  basquin:    FI_ar = N^(−b)          → N = FI_ar^(−1/b)          [금속 관례]

한계: 등진폭·비례하중·모드 무관 1차 근사. 층간·박리 피로와 잔류강도 저하는 미포함.
"""
from __future__ import annotations

import math

MODELS = ("log_linear", "basquin")
N_CAP = 1.0e12          # 무한수명 취급 상한 (보고 시 명시)


def equivalent_alternating(fi_max: float, fi_min: float) -> dict:
    """FI 사이클 → 진폭·평균·Goodman 등가 교번 FI_ar."""
    fi_a = (fi_max - fi_min) / 2.0
    fi_m = (fi_max + fi_min) / 2.0
    if fi_a <= 0.0:
        return {"FI_amplitude": fi_a, "FI_mean": fi_m, "FI_ar": 0.0,
                "note": "진폭 0 — 정적 하중(피로 무관)"}
    if fi_m >= 1.0:
        return {"FI_amplitude": fi_a, "FI_mean": fi_m, "FI_ar": float("inf"),
                "note": "평균 응력이 이미 정적 한계 이상 — 첫 사이클 파손"}
    return {"FI_amplitude": fi_a, "FI_mean": fi_m, "FI_ar": fi_a / (1.0 - fi_m)}


def cycles_to_failure(fi_ar: float, model: str, param: float) -> float | None:
    """등가 교번 FI_ar → 수명 N. FI_ar ≥ 1이면 1사이클(정적 파손), 0이면 무한(None)."""
    if fi_ar <= 0.0:
        return None                     # 무한수명
    if fi_ar >= 1.0:
        return 1.0
    if model == "log_linear":           # FI_ar = 1 − k·log10(N)
        return min(10.0 ** ((1.0 - fi_ar) / param), N_CAP)
    if model == "basquin":              # FI_ar = N^(−b)
        return min(fi_ar ** (-1.0 / param), N_CAP)
    raise ValueError(f"unknown S-N model: {model}")


def assess(fi_max: float, fi_min: float, model: str, param: float) -> dict:
    """한 지점의 피로 평가 — 등가 교번과 수명."""
    eq = equivalent_alternating(fi_max, fi_min)
    fi_ar = eq["FI_ar"]
    if math.isinf(fi_ar):
        n = 1.0
    else:
        n = cycles_to_failure(fi_ar, model, param)
    return {**eq, "cycles_to_failure": n,
            "infinite_life": n is None,
            "at_cap": bool(n is not None and n >= N_CAP)}
