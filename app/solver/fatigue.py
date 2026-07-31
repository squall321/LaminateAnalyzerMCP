# 피로 수명 — 재료축 성분별 부호 보존 S-N + Goodman 평균응력 보정 (계획서 §17.7)
"""하중 사이클(최대·최소)에서 ply별 반복 수명 N_f를 추정한다.

**성분별 접근이 필수**: 초기 구현은 Tsai-Wu 강도비의 역수 FI=1/R을 사이클 값으로 썼는데,
FI는 항상 양수라 **부호가 소실**된다. 그 결과 완전반복(R=−1, 피로에서 가장 가혹한 형태)이
FI_max=FI_min → 진폭 0 → '무한수명'으로 뒤집혔다(적대 검증 FAT-01, 최악의 비보수 오류).

따라서 재료축 성분 σ1·σ2·τ12 각각에서 부호를 살려 평가한다.
  σ_a = (σ_max − σ_min)/2,  σ_m = (σ_max + σ_min)/2      (부호 보존)
  정규화: r_a = σ_a/σ_u(진폭 방향 강도), r_m = σ_m/σ_u(평균 부호에 맞는 강도)
  Goodman: r_ar = r_a/(1 − r_m)  단 r_m > 0 일 때만 감산 (음의 평균은 표준 Goodman에서 무감산)
  log_linear: N = 10^((1 − r_ar)/k)   ·   basquin: N = r_ar^(−1/b)
성분 중 최소 수명이 그 지점의 수명이고, 지배 성분을 함께 보고한다.

한계: 등진폭·비례하중, 성분 독립(다축 상호작용 미고려), 데이터 범위 밖 외삽 주의,
층간/박리 피로·잔류강도 저하·하중 순서·환경 효과 미포함.
"""
from __future__ import annotations

import math

MODELS = ("log_linear", "basquin")
N_CAP = 1.0e12          # 무한수명 취급 상한 (거듭제곱 전에 적용 — 오버플로 방어, FAT-04/05)
COMPONENTS = ("sigma_1", "sigma_2", "tau_12")


def _strength_for(comp: int, sign: float, strength) -> float:
    """성분·부호에 맞는 정적 강도. strength = (Xt, Xc, Yt, Yc, S) 모두 양수."""
    Xt, Xc, Yt, Yc, S = strength
    if comp == 0:
        return Xt if sign >= 0 else Xc
    if comp == 1:
        return Yt if sign >= 0 else Yc
    return S                     # 전단은 부호 무관


def cycles_to_failure(r_ar: float, model: str, param: float) -> float | None:
    """정규화 등가 교번 r_ar → 수명 N. r_ar ≥ 1이면 1(정적), ≤ 0이면 None(무한).

    N_CAP은 거듭제곱 '전에' 지수 한계로 적용해 OverflowError를 원천 차단한다.
    """
    if r_ar <= 0.0:
        return None
    if r_ar >= 1.0:
        return 1.0
    if model == "log_linear":                 # r_ar = 1 − k·log10(N)
        exponent = (1.0 - r_ar) / param
        return N_CAP if exponent >= math.log10(N_CAP) else 10.0 ** exponent
    if model == "basquin":                    # r_ar = N^(−b)
        exponent = -math.log10(r_ar) / param
        return N_CAP if exponent >= math.log10(N_CAP) else 10.0 ** exponent
    raise ValueError(f"unknown S-N model: {model}")


def assess_component(s_max: float, s_min: float, comp: int, strength,
                     model: str, param: float) -> dict:
    """한 성분(σ1/σ2/τ12)의 피로 평가 — 부호를 보존한 진폭·평균과 Goodman 보정."""
    s_a = (s_max - s_min) / 2.0                     # 항상 ≥ 0 (max ≥ min 보장 시)
    s_m = (s_max + s_min) / 2.0
    su_a = _strength_for(comp, 1.0 if s_max >= abs(s_min) else -1.0, strength)
    su_m = _strength_for(comp, 1.0 if s_m >= 0 else -1.0, strength)
    r_a = abs(s_a) / su_a
    r_m = s_m / su_m if su_m > 0 else 0.0           # 부호 보존 (압축 평균이면 음수)
    if r_a <= 0.0:
        return {"component": COMPONENTS[comp], "sigma_amplitude": s_a, "sigma_mean": s_m,
                "r_alternating": 0.0, "cycles_to_failure": None,
                "note": "진폭 0 — 이 성분은 피로 손상 없음(정적 하중)"}
    if r_m >= 1.0:
        return {"component": COMPONENTS[comp], "sigma_amplitude": s_a, "sigma_mean": s_m,
                "r_alternating": float("inf"), "cycles_to_failure": 1.0,
                "note": "평균 응력이 이미 정적 한계 이상 — 첫 사이클 파손"}
    # 표준 Goodman: 인장 평균만 감산, 압축 평균은 무감산(보수적 취급)
    r_ar = r_a / (1.0 - r_m) if r_m > 0 else r_a
    return {"component": COMPONENTS[comp], "sigma_amplitude": s_a, "sigma_mean": s_m,
            "r_amplitude": r_a, "r_mean": r_m, "r_alternating": r_ar,
            "cycles_to_failure": cycles_to_failure(r_ar, model, param)}


def assess_point(s12_max, s12_min, strength, model: str, param: float) -> dict:
    """한 지점(ply의 한 z)의 피로 — 세 성분 중 최소 수명과 지배 성분."""
    per = [assess_component(float(s12_max[i]), float(s12_min[i]), i, strength, model, param)
           for i in range(3)]
    finite = [c for c in per if c["cycles_to_failure"] is not None]
    if not finite:
        return {"components": per, "cycles_to_failure": None, "governing": None,
                "infinite_life": True}
    worst = min(finite, key=lambda c: c["cycles_to_failure"])
    return {"components": per, "cycles_to_failure": worst["cycles_to_failure"],
            "governing": worst["component"], "infinite_life": False,
            "r_alternating": worst["r_alternating"],
            "at_cap": bool(worst["cycles_to_failure"] >= N_CAP)}
