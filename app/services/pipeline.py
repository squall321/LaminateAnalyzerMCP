# 요청 오케스트레이션 — 검증→SI 계산→표시 변환→envelope 조립 (계획서 §6.3, §7.2)
from __future__ import annotations

import time

import numpy as np

from app import units
from app.errors import item
from app.services import envelope as ENV
from app.services import evaluation as EVAL
from app.services import validation as VAL
from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import neutral_axis as NA

VALID_INCLUDE = ("abd", "neutral_axis", "indices")
NA_MODES = ("clt_weighted", "beam_equivalent")

_AXIS_DEF = {
    "clt_weighted": "Q̄11(x)/Q̄22(y) 가중 도심 — B11/A11, B22/A22와 동치. 방향 분리 근사(커플링 미반영)",
    "beam_equivalent": "환산단면 중립축 — ply별 공학계수 E_x(θ) 가중 도심. 1축 응력(보/스트립) 가정",
}


def _mat3(m: np.ndarray, factor: float) -> list[list[float]]:
    return (m * factor).tolist()


def _ns_repr(z_ns_mid: float, h: float, f_z: float) -> dict:
    """중립면 세 표기(D5)를 표시 단위로 변환. zeta는 무차원이라 변환 없음."""
    rep = NA.representations(z_ns_mid, h)
    return {"z_from_midplane": rep["z_from_midplane"] * f_z,
            "z_from_bottom": rep["z_from_bottom"] * f_z,
            "zeta": rep["zeta"]}


def _source_assumptions(si: VAL.SiLaminate) -> list[str]:
    lines = []
    for k, p in enumerate(si.plies):
        if p.source is not None:
            ref = f" ({p.source.ref})" if p.source.ref else ""
            conf = f", confidence={p.source.confidence}" if p.source.confidence else ""
            lines.append(f"laminae[{k}].material({p.name or 'unnamed'}): 물성 출처 {p.source.type}{ref}{conf}")
    return lines


def run_analysis(payload, *, include=None, neutral_axis_mode: str = "clt_weighted",
                 include_abd_6x6: bool = False, include_debug: bool = False,
                 criteria: dict | None = None) -> dict:
    """analyze/compute/evaluate 계열 Tool의 공통 경로. 항상 envelope dict를 반환한다."""
    t0 = time.perf_counter()
    include = tuple(include) if include else VALID_INCLUDE

    bad_inc = [x for x in include if x not in VALID_INCLUDE]
    if bad_inc:
        return ENV.build(data=None, errors=[item("E100", field="include",
                                                 detail=f"include 항목 {bad_inc}는 지원하지 않습니다. 지원: {list(VALID_INCLUDE)}")],
                         warnings=[], payload=payload, include_debug=include_debug, t0=t0)
    if neutral_axis_mode not in NA_MODES:
        return ENV.build(data=None, errors=[item("E100", field="neutral_axis_mode",
                                                 detail=f"neutral_axis_mode = {neutral_axis_mode!r}. 지원: {list(NA_MODES)}")],
                         warnings=[], payload=payload, include_debug=include_debug, t0=t0)

    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    # SI 계산
    h = si.total_thickness
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg) for p in si.plies]
    z = ABD.z_coordinates(si.thicknesses)
    A, B, D = ABD.abd_matrices(qbars, z)
    A_hat, B_hat, D_hat, K_hat = ABD.normalized_stiffness(A, B, D, h)
    pd_ok = ABD.is_positive_definite(K_hat)
    if not pd_ok:
        return ENV.build(data=None, errors=[item("E402", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    ns_x, ns_y = NA.clt_weighted(A, B)
    symmetric = EVAL.is_symmetric_stack(si.fingerprint)

    # 표시 단위 변환
    f = units.FROM_SI[si.unit_system]
    rho_all = [p.rho for p in si.plies]
    areal_mass_si = sum(r * p.thickness for r, p in zip(rho_all, si.plies)) if all(r is not None for r in rho_all) else None

    data: dict = {
        "laminate_summary": {
            "name": si.name,
            "n_plies": len(si.plies),
            "total_thickness": h * f["z"],
            "areal_mass": None if areal_mass_si is None else areal_mass_si * f["areal_mass"],
            "is_symmetric_stack": symmetric,
        }
    }

    if "abd" in include:
        data["abd"] = {
            "A": _mat3(A, f["A"]), "B": _mat3(B, f["B"]), "D": _mat3(D, f["D"]),
            "A_hat": _mat3(A_hat, f["hat"]), "B_hat": _mat3(B_hat, f["hat"]), "D_hat": _mat3(D_hat, f["hat"]),
        }
        if include_abd_6x6:
            top = np.hstack([A * f["A"], B * f["B"]])
            bot = np.hstack([B * f["B"], D * f["D"]])
            data["abd"]["ABD_6x6_display_units"] = np.vstack([top, bot]).tolist()

    if "neutral_axis" in include:
        if neutral_axis_mode == "clt_weighted":
            ns_block = {
                "mode": "clt_weighted",
                "x": _ns_repr(ns_x, h, f["z"]),
                "y": _ns_repr(ns_y, h, f["z"]),
                "axis_definition": _AXIS_DEF["clt_weighted"],
            }
        else:
            ex = [MAT.ex_engineering(p.E1, p.E2, p.G12, p.nu12, p.angle_deg) for p in si.plies]
            nb = NA.beam_equivalent(ex, si.thicknesses, z)
            ns_block = {
                "mode": "beam_equivalent",
                "x": _ns_repr(nb, h, f["z"]),
                "axis_definition": _AXIS_DEF["beam_equivalent"],
            }
        data["neutral_surface"] = ns_block

    if "indices" in include:
        idx = EVAL.compute_indices(A_hat, B_hat, D_hat, K_hat, pd_ok, ns_x, h, warnings)
        data["indices"] = idx
        evaluation: dict = {"recommendations": EVAL.recommendations(idx, symmetric)}
        if criteria:
            evaluation["pass_fail"] = EVAL.evaluate_criteria(idx, criteria)
        data["evaluation"] = evaluation

    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    return ENV.build(data=data, errors=[], warnings=warnings, payload=payload,
                     unit_system=si.unit_system, assumptions_extra=_source_assumptions(si),
                     include_debug=include_debug, t0=t0)


def run_validation(payload, include_debug: bool = False) -> dict:
    """검증만 수행 (validate_laminate_input Tool)."""
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    data = None
    if not errors:
        data = {"valid": True,
                "n_plies": len(si.plies),
                "total_thickness_si_m": si.total_thickness,
                "is_symmetric_stack": EVAL.is_symmetric_stack(si.fingerprint)}
    return ENV.build(data=data, errors=errors, warnings=warnings, payload=payload,
                     unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                     include_debug=include_debug, t0=t0)
