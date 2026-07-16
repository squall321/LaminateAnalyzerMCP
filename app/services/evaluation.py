# 평가 지표와 등급·권고 — 원시값 + 등급 밴드 병기, 점수 조작 없음 (계획서 §5.1, §5.4)
from __future__ import annotations

import numpy as np

from app import config
from app.errors import item


def _index(value, grade, definition):
    return {"value": value, "grade": grade, "definition": definition}


def _safe_ratio(num: float, den: float):
    return None if abs(den) < config.DENOM_EPS else float(num / den)


def is_symmetric_stack(fingerprint: list) -> bool:
    """적층 정의의 기하학적 회문 검사 (두께·각도·물성, 계획서 §5.1)."""
    return fingerprint == list(reversed(fingerprint))


def compute_indices(A_hat: np.ndarray, B_hat: np.ndarray, D_hat: np.ndarray,
                    K_hat: np.ndarray, positive_definite: bool,
                    ns_x_mid: float, h: float,
                    warnings: list[dict]) -> dict:
    """정규화 강성 기반 무차원 지표. 분모 소멸 시 값 null + W402 (§5.4)."""
    idx: dict[str, dict] = {}

    nA, nB, nD = (float(np.linalg.norm(m)) for m in (A_hat, B_hat, D_hat))
    cr = _safe_ratio(nB, float(np.sqrt(nA * nD)))
    if cr is None:
        warnings.append(item("W402", field="indices.coupling_ratio", detail="||A_hat||·||D_hat|| ≈ 0"))
        grade = None
    else:
        grade = "negligible" if cr < 0.01 else "low" if cr < 0.05 else "moderate" if cr < config.HIGH_COUPLING_RATIO else "high"
        if cr >= config.HIGH_COUPLING_RATIO:
            warnings.append(item("W200", field="indices.coupling_ratio", detail=f"coupling_ratio = {cr:.3g}"))
    idx["coupling_ratio"] = _index(cr, grade, "||B_hat||_F / sqrt(||A_hat||_F * ||D_hat||_F), B_hat = sqrt(12)*B/h^2")

    ma = _safe_ratio(A_hat[0, 0], A_hat[1, 1])
    idx["membrane_anisotropy"] = _index(
        ma, None if ma is None else ("balanced" if 0.5 <= ma <= 2.0 else "directional"),
        "A_hat11 / A_hat22")
    ba = _safe_ratio(D_hat[0, 0], D_hat[1, 1])
    idx["bending_anisotropy"] = _index(
        ba, None if ba is None else ("balanced" if 0.5 <= ba <= 2.0 else "directional"),
        "D_hat11 / D_hat22")

    sec = _safe_ratio(max(abs(A_hat[0, 2]), abs(A_hat[1, 2])), float(np.sqrt(A_hat[0, 0] * A_hat[1, 1])))
    idx["shear_extension_coupling"] = _index(
        sec, None if sec is None else ("balanced" if sec < 0.01 else "mild" if sec < 0.1 else "strong"),
        "max(|A_hat16|, |A_hat26|) / sqrt(A_hat11 * A_hat22)")
    btc = _safe_ratio(max(abs(D_hat[0, 2]), abs(D_hat[1, 2])), float(np.sqrt(D_hat[0, 0] * D_hat[1, 1])))
    idx["bend_twist_coupling"] = _index(
        btc, None if btc is None else ("balanced" if btc < 0.01 else "mild" if btc < 0.1 else "strong"),
        "max(|D_hat16|, |D_hat26|) / sqrt(D_hat11 * D_hat22)")

    q_ref = (A_hat[0, 0] + A_hat[1, 1]) / 2.0
    if abs(q_ref) < config.DENOM_EPS:
        warnings.append(item("W402", field="indices.quasi_isotropy_score", detail="(A_hat11+A_hat22)/2 ≈ 0"))
        qi = None
    else:
        e1 = abs(A_hat[0, 0] - A_hat[1, 1]) / q_ref
        e2 = (abs(A_hat[0, 2]) + abs(A_hat[1, 2])) / q_ref
        e3 = abs(A_hat[2, 2] - 0.5 * (A_hat[0, 0] - A_hat[0, 1])) / q_ref
        qi = float(1.0 - min(1.0, (e1 + e2 + e3) / 3.0))
    idx["quasi_isotropy_score"] = _index(
        qi, None if qi is None else ("quasi_isotropic" if qi > 0.95 else "near_isotropic" if qi > 0.8 else "anisotropic"),
        "1 - min(1, (e1+e2+e3)/3); e1=|A11-A22|/Qref, e2=(|A16|+|A26|)/Qref, e3=|A66-(A11-A12)/2|/Qref (A=A_hat, Qref=(A11+A22)/2)")

    nso = ns_x_mid / h
    idx["ns_offset_ratio"] = _index(
        float(nso), "centered" if abs(nso) < 0.01 else "offset" if abs(nso) < 0.1 else "strongly_offset",
        "z_ns_x / h (midplane 기준, 0 = 중앙)")

    cond = float(np.linalg.cond(K_hat, 2))
    if cond > config.COND_WARN_THRESHOLD:
        warnings.append(item("W401", field="indices.abd_condition_number", detail=f"cond2(K_hat) = {cond:.3e}"))
    idx["abd_condition_number"] = _index(cond, None, "cond_2(K_hat), K_hat = 합동변환 정규화 6x6 (§4.6)")

    idx["positive_definite"] = _index(bool(positive_definite), None, "Cholesky(K_hat) 성공 여부")

    # 지배 커플링 항 (계획서 §5.2 dominant_coupling_terms) — 커플링이 유의할 때만
    terms = []
    if nB > 1e-9 * nA:
        names = {(0, 0): "B11", (0, 1): "B12", (0, 2): "B16", (1, 1): "B22", (1, 2): "B26", (2, 2): "B66"}
        for (i, j), nm in names.items():
            frac = abs(B_hat[i, j]) / nB
            if frac >= 0.05:
                terms.append({"term": nm, "fraction_of_norm": float(frac),
                              "sign": int(np.sign(B_hat[i, j]))})
        terms.sort(key=lambda x: -x["fraction_of_norm"])
    idx["dominant_coupling_terms"] = {"value": terms[:3],
                                      "definition": "|B_hat_ij|/||B_hat||_F 상위 항 (비대칭 원인 진단용, 대칭이면 빈 목록)"}
    return idx


def recommendations(idx: dict, is_symmetric: bool) -> list[str]:
    """지표 기반 규칙형 권고 (한국어, 최대 4건)."""
    rec: list[str] = []
    cr = idx["coupling_ratio"]["value"]
    if cr is not None and cr >= 0.05:
        rec.append("막-굽힘 커플링이 유의합니다. 경화 후 뒤틀림과 해석 커플링을 줄이려면 대칭 적층([...]s)을 검토하세요.")
    sec = idx["shear_extension_coupling"]["value"]
    if sec is not None and sec >= 0.1:
        rec.append("면내 전단-신장 커플링이 큽니다. ±θ 짝을 맞춘 balanced 구성을 검토하세요.")
    nso = idx["ns_offset_ratio"]["value"]
    if nso is not None and abs(nso) >= 0.1:
        rec.append("중립면이 midplane에서 크게 벗어나 있습니다(강성 비대칭). 굽힘 하중 방향에 따라 응답이 달라짐에 주의하세요.")
    qi = idx["quasi_isotropy_score"]["value"]
    if qi is not None and qi > 0.95:
        rec.append("면내 강성이 준등방성에 근접합니다. 방향별 강성 요구가 없다면 현 구성이 무난합니다.")
    if not rec and is_symmetric:
        rec.append("대칭 적층으로 커플링이 사실상 없습니다(B≈0).")
    return rec[:4]


_CRITERIA_DEFS = {
    "max_coupling_ratio": ("coupling_ratio", lambda v, lim: v <= lim),
    "min_quasi_isotropy_score": ("quasi_isotropy_score", lambda v, lim: v >= lim),
    "max_abs_ns_offset_ratio": ("ns_offset_ratio", lambda v, lim: abs(v) <= lim),
    "max_condition_number": ("abd_condition_number", lambda v, lim: v <= lim),
}


def evaluate_criteria(idx: dict, criteria: dict) -> list[dict]:
    """사용자 기준 대비 pass/fail. 지원 키는 _CRITERIA_DEFS (계획서 §6.3 evaluate_laminate)."""
    out = []
    for key, limit in criteria.items():
        if key not in _CRITERIA_DEFS:
            out.append({"criterion": key, "limit": limit, "actual": None, "pass": None,
                        "note": f"지원하지 않는 기준입니다. 지원 목록: {sorted(_CRITERIA_DEFS)}"})
            continue
        idx_key, fn = _CRITERIA_DEFS[key]
        actual = idx[idx_key]["value"]
        ok = None if actual is None or not isinstance(limit, (int, float)) else bool(fn(actual, float(limit)))
        out.append({"criterion": key, "limit": limit, "actual": actual, "pass": ok})
    return out
