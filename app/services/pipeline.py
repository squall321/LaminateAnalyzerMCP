# 요청 오케스트레이션 — 검증→SI 계산→표시 변환→envelope 조립 (계획서 §6.3, §7.2)
from __future__ import annotations

import math
import time

import numpy as np

from app import config, units
from app.errors import item
from app.services import envelope as ENV
from app.services import evaluation as EVAL
from app.services import validation as VAL
from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import neutral_axis as NA
from app.solver import response as RESP

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


# ── V1: 하중 응답 / 민감도 / 배치 / 리포트 ──────────────────────────────────


def _abd_of_plies(plies) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """SiPly 목록 → (A, B, D, z, h). 검증 이후의 순수 계산 경로."""
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg) for p in plies]
    z = ABD.z_coordinates([p.thickness for p in plies])
    A, B, D = ABD.abd_matrices(qbars, z)
    return A, B, D, z, float(z[-1] - z[0])


def _loads_to_si(loads, us: str) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    """loads {"N":[3], "M":[3]} → SI 벡터. 형식 오류면 (None, None, E100 항목)."""
    if loads is None:
        loads = {}
    if not isinstance(loads, dict):
        return None, None, item("E100", field="loads", detail="loads는 {\"N\": [Nx,Ny,Nxy], \"M\": [Mx,My,Mxy]} 객체여야 합니다")
    f = units.TO_SI[us]
    out = []
    for key, factor in (("N", f["load_n"]), ("M", f["load_m"])):
        v = loads.get(key, [0.0, 0.0, 0.0])
        if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
            return None, None, item("E100", field=f"loads.{key}", detail=f"loads.{key}는 숫자 3개 리스트여야 합니다")
        out.append(np.asarray(v, dtype=np.float64) * factor)
    if float(np.linalg.norm(out[0])) == 0.0 and float(np.linalg.norm(out[1])) == 0.0:
        return None, None, item("E100", field="loads", detail="N과 M이 모두 0입니다. 최소 한 성분은 0이 아니어야 합니다")
    return out[0], out[1], None


def run_load_response(payload, loads=None, scan_principal_direction: bool = True,
                      include_debug: bool = False) -> dict:
    """N/M 하중 응답 + 유효 공학 상수 + V1 지표 (solve_load_response Tool, 계획서 §4.7·§5.2)."""
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    N_si, M_si, load_err = _loads_to_si(loads, si.unit_system)
    if load_err is not None:
        return ENV.build(data=None, errors=[load_err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    # §18.7 게이트 — 전단 비선형 물성이 있는데 선형으로 푸는 중임을 알린다
    nl_plies = [k for k, p_ in enumerate(si.plies) if p_.s6666 is not None]
    if nl_plies:
        warnings.append(item("W130", field="laminae",
                             detail=f"laminae{nl_plies} 에 shear_nonlinear(S6666)가 있으나 이 도구는 "
                                    f"선형 G12로 풉니다. 전단 지배 적층이면 강성을 크게 과대평가합니다 — "
                                    f"solve_nonlinear_shear_response 를 쓰세요"))

    A, B, D, z, h = _abd_of_plies(si.plies)
    f = units.FROM_SI[si.unit_system]
    try:
        eps0, kappa = RESP.solve_response(A, B, D, N_si, M_si)
        alpha, _, delta = RESP.compliance_blocks(A, B, D)
        eff = RESP.effective_constants(alpha, delta, h)
        leakage = RESP.membrane_bending_leakage(A, B, D, h)
        twist = RESP.twist_under_bending(A, B, D)
    except RESP.SingularSystemError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    if twist is None:
        warnings.append(item("W402", field="v1_indices.twist_under_bending", detail="κ_x ≈ 0"))

    fm = f["modulus"]
    data: dict = {
        "response": {
            "epsilon0": eps0.tolist(),
            "kappa": (kappa * f["kappa"]).tolist(),
            "note": "epsilon0 무차원, kappa 단위는 metadata.units 참조. 하중은 단위 폭당 물리량",
        },
        "effective_constants": {
            "membrane": {k: (v * fm if k != "nu_xy" else v) for k, v in eff["membrane"].items()},
            "bending": {k: v * fm for k, v in eff["bending"].items()},
        },
        "v1_indices": {
            "membrane_bending_leakage": {"value": leakage,
                                         "definition": "(h/2)*||kappa||/||eps0|| under unit Nx, 대칭이면 0"},
            "twist_under_bending": {"value": twist,
                                    "definition": "|kappa_xy|/|kappa_x| under unit Mx"},
        },
    }

    rho_all = [p.rho for p in si.plies]
    if all(r is not None for r in rho_all):
        rho_bar = sum(r * p.thickness for r, p in zip(rho_all, si.plies)) / h
        data["v1_indices"]["specific_modulus_Ex_si"] = {
            "value": eff["membrane"]["Ex"] / rho_bar,
            "definition": "Ex_eff/rho_bar [Pa/(kg/m^3)] — 단위계와 무관하게 SI로 보고"}

    if scan_principal_direction:
        def build_at(phi: float):
            qb = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg - phi)
                  for p in si.plies]
            Ar, Br, Dr = ABD.abd_matrices(qb, z)
            return Ar, Br, Dr, h
        try:
            pd_scan = RESP.principal_membrane_direction(build_at)
        except RESP.SingularSystemError:
            return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                             payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
        data["v1_indices"]["in_plane_principal_direction"] = {
            "angle_deg_of_max_Ex": pd_scan["angle_deg_of_max_Ex"],
            "Ex_max": pd_scan["Ex_max"] * fm,
            "angle_deg_of_min_Ex": pd_scan["angle_deg_of_min_Ex"],
            "Ex_min": pd_scan["Ex_min"] * fm,
            "definition": "E_x^eff(phi) 1도 그리드 스캔, phi in [0,180)",
        }

    hash_payload = {"laminate": payload, "loads": loads}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=_source_assumptions(si),
                     include_debug=include_debug, t0=t0)


def _sensitivity_responses(plies, f_hat: float) -> dict:
    """민감도 대상 응답 3종 — D̂11(표시단위), coupling_ratio, zeta_x."""
    A, B, D, z, h = _abd_of_plies(plies)
    A_hat, B_hat, D_hat, _ = ABD.normalized_stiffness(A, B, D, h)
    nA, nB, nD = (float(np.linalg.norm(m)) for m in (A_hat, B_hat, D_hat))
    ns_x, _ = NA.clt_weighted(A, B)
    return {
        "D_hat_11": D_hat[0, 0] * f_hat,
        "coupling_ratio": nB / np.sqrt(nA * nD),
        "zeta_x": (ns_x + h / 2.0) / h,
    }


def run_sensitivity(payload, angle_delta_deg: float = 1.0, thickness_rel: float = 0.01,
                    modulus_rel: float = 0.01, include_debug: bool = False) -> dict:
    """ply별 각도/두께/탄성계수 섭동의 중앙차분 민감도 (run_sensitivity_analysis Tool, §5.2)."""
    import dataclasses
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    for name, v in (("angle_delta_deg", angle_delta_deg), ("thickness_rel", thickness_rel),
                    ("modulus_rel", modulus_rel)):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            return ENV.build(data=None, errors=[item("E100", field=name, detail=f"{name}는 양수여야 합니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)

    f_hat = units.FROM_SI[si.unit_system]["hat"]
    base = _sensitivity_responses(si.plies, f_hat)
    rows: list[dict] = []

    def half_diff(plus_plies, minus_plies) -> dict:
        rp = _sensitivity_responses(plus_plies, f_hat)
        rm = _sensitivity_responses(minus_plies, f_hat)
        return {k: (rp[k] - rm[k]) / 2.0 for k in rp}

    def scale_modulus(ply, factor: float):
        # 등방 재료는 E 스케일 시 E1·E2·G12가 함께 변한다 (ν 고정 → G = E/2(1+ν) 비례)
        if ply.is_isotropic:
            return dataclasses.replace(ply, E1=ply.E1 * factor, E2=ply.E2 * factor, G12=ply.G12 * factor)
        return dataclasses.replace(ply, E1=ply.E1 * factor)

    for k, p in enumerate(si.plies):
        if time.perf_counter() - t0 > config.COMPUTE_TIMEOUT_S:
            return ENV.build(data=None, errors=[item("E500")], warnings=warnings, payload=payload,
                             unit_system=si.unit_system, include_debug=include_debug, t0=t0)
        others = si.plies
        variants = [
            (f"laminae[{k}].angle_deg", f"±{angle_delta_deg} deg",
             dataclasses.replace(p, angle_deg=p.angle_deg + angle_delta_deg),
             dataclasses.replace(p, angle_deg=p.angle_deg - angle_delta_deg)),
            (f"laminae[{k}].thickness", f"±{thickness_rel * 100:g}%",
             dataclasses.replace(p, thickness=p.thickness * (1 + thickness_rel)),
             dataclasses.replace(p, thickness=p.thickness * (1 - thickness_rel))),
            (f"laminae[{k}].material.E" if p.is_isotropic else f"laminae[{k}].material.E1",
             f"±{modulus_rel * 100:g}%",
             scale_modulus(p, 1 + modulus_rel),
             scale_modulus(p, 1 - modulus_rel)),
        ]
        for param, delta_label, pp, pm in variants:
            plus = [pp if i == k else q for i, q in enumerate(others)]
            minus = [pm if i == k else q for i, q in enumerate(others)]
            rows.append({"parameter": param, "delta": delta_label,
                         "response_half_diff": half_diff(plus, minus)})

    data = {"base": base, "rows": rows,
            "note": "response_half_diff = (f(+Δ) - f(-Δ))/2 — +Δ 섭동당 응답 변화량(중앙차분)"}
    hash_payload = {"laminate": payload, "angle_delta_deg": angle_delta_deg,
                    "thickness_rel": thickness_rel, "modulus_rel": modulus_rel}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system, include_debug=include_debug, t0=t0)


def run_batch(laminates, criteria: dict | None = None, include_debug: bool = False) -> dict:
    """복수 적층안 일괄 평가 — 케이스당 핵심 지표 요약 (batch_evaluate_laminates Tool)."""
    t0 = time.perf_counter()
    hash_payload = {"laminates": laminates, "criteria": criteria}
    if not isinstance(laminates, list) or len(laminates) == 0:
        return ENV.build(data=None,
                         errors=[item("E100", field="laminates", detail="laminates는 1개 이상 적층 정의의 리스트여야 합니다")],
                         warnings=[], payload=hash_payload, include_debug=include_debug, t0=t0)
    if len(laminates) > config.MAX_BATCH:
        return ENV.build(data=None,
                         errors=[item("E103", field="laminates", n=len(laminates), max=config.MAX_BATCH)],
                         warnings=[], payload=hash_payload, include_debug=include_debug, t0=t0)

    results = []
    for i, one in enumerate(laminates):
        if time.perf_counter() - t0 > config.COMPUTE_TIMEOUT_S:
            return ENV.build(data=None, errors=[item("E500")], warnings=[],
                             payload=hash_payload, include_debug=include_debug, t0=t0)
        env = run_analysis(one, include=("indices",), criteria=criteria)
        row = {"index": i,
               "name": (one.get("name") if isinstance(one, dict) else None),
               "status": env["status"],
               "error_codes": [e["code"] for e in env["errors"]],
               "warning_codes": [w["code"] for w in env["warnings"]]}
        if env["data"] is not None:
            idx = env["data"]["indices"]
            row.update({
                "coupling_ratio": idx["coupling_ratio"]["value"],
                "quasi_isotropy_score": idx["quasi_isotropy_score"]["value"],
                "ns_offset_ratio": idx["ns_offset_ratio"]["value"],
                "is_symmetric_stack": env["data"]["laminate_summary"]["is_symmetric_stack"],
            })
            if criteria:
                pf = env["data"]["evaluation"].get("pass_fail", [])
                row["pass_all"] = all(r["pass"] is True for r in pf if r["pass"] is not None) if pf else None
        results.append(row)

    n_ok = sum(1 for r in results if not r["error_codes"])
    data = {"results": results, "n_total": len(results), "n_ok": n_ok, "n_error": len(results) - n_ok}
    us0 = laminates[0].get("unit_system") if isinstance(laminates[0], dict) else None
    return ENV.build(data=data, errors=[], warnings=[], payload=hash_payload,
                     unit_system=us0 if us0 in units.SUPPORTED_UNIT_SYSTEMS else None,
                     include_debug=include_debug, t0=t0)


_REPORT_LABELS = {
    "ko": {"title": "적층 평가 리포트", "summary": "요약", "input": "입력 적층",
           "abd": "ABD 강성", "ns": "중립면", "idx": "평가 지표", "rec": "권고",
           "warn": "경고", "assume": "해석 가정", "ply": "ply", "angle": "각도",
           "thick": "두께", "mat": "재료"},
    "en": {"title": "Laminate Evaluation Report", "summary": "Summary", "input": "Layup",
           "abd": "ABD Stiffness", "ns": "Neutral Surface", "idx": "Indices", "rec": "Recommendations",
           "warn": "Warnings", "assume": "Assumptions", "ply": "ply", "angle": "angle",
           "thick": "thickness", "mat": "material"},
}


def run_report(payload, criteria: dict | None = None, language: str = "ko",
               include_debug: bool = False) -> dict:
    """사람용 Markdown 리포트 + LLM용 요약 (generate_design_report Tool, Q4: 한국어 기본)."""
    t0 = time.perf_counter()
    if language not in _REPORT_LABELS:
        return ENV.build(data=None,
                         errors=[item("E100", field="language", detail=f"language는 {sorted(_REPORT_LABELS)} 중 하나여야 합니다")],
                         warnings=[], payload=payload, include_debug=include_debug, t0=t0)
    env = run_analysis(payload, criteria=criteria)
    if env["status"] == "error":
        return env

    L = _REPORT_LABELS[language]
    d = env["data"]
    u = env["metadata"]["units"]
    lam = payload.get("laminae", [])
    s = d["laminate_summary"]
    idx = d["indices"]

    def fmt(v):
        return f"{v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    lines = [f"# {L['title']} — {s.get('name') or 'unnamed'}", ""]
    lines += [f"## {L['summary']}", "",
              f"- {L['ply']}: {s['n_plies']}개, h = {s['total_thickness']:.6g} {u['thickness']}, "
              f"{'대칭' if s['is_symmetric_stack'] else '비대칭'} 적층",
              f"- coupling_ratio: {fmt(idx['coupling_ratio']['value'])} ({idx['coupling_ratio']['grade']})",
              f"- quasi_isotropy: {fmt(idx['quasi_isotropy_score']['value'])} ({idx['quasi_isotropy_score']['grade']})",
              f"- ζ(중립면, x): {fmt(d['neutral_surface']['x']['zeta'])}", ""]
    lines += [f"## {L['input']}", "", f"| # | {L['angle']} [deg] | {L['thick']} [{u['thickness']}] | {L['mat']} |",
              "|---|---|---|---|"]
    for i, p in enumerate(lam):
        m = p.get("material", {})
        lines.append(f"| {i} | {p.get('angle_deg')} | {p.get('thickness')} | {m.get('name') or m.get('type')} |")
    lines += ["", f"## {L['abd']}", ""]
    for key in ("A", "B", "D"):
        unit = u[key]
        rows = d["abd"][key]
        lines += [f"**{key}** [{unit}]", ""]
        lines += ["| " + " | ".join(f"{v:.5g}" for v in r) + " |" for r in rows]
        lines.append("")
    ns = d["neutral_surface"]["x"]
    lines += [f"## {L['ns']}", "",
              f"- z(midplane) = {ns['z_from_midplane']:.6g} {u['z']}, ζ = {ns['zeta']:.4g} "
              f"({d['neutral_surface']['axis_definition']})", ""]
    lines += [f"## {L['idx']}", "", "| index | value | grade |", "|---|---|---|"]
    for k, v in idx.items():
        if isinstance(v, dict) and "value" in v:
            val = v["value"]
            val_s = f"{val:.5g}" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)
            lines.append(f"| {k} | {val_s} | {v.get('grade')} |")
    if d["evaluation"].get("pass_fail"):
        lines += ["", "| criterion | limit | actual | pass |", "|---|---|---|---|"]
        for r in d["evaluation"]["pass_fail"]:
            lines.append(f"| {r['criterion']} | {r['limit']} | {r['actual']} | {r['pass']} |")
    if d["evaluation"]["recommendations"]:
        lines += ["", f"## {L['rec']}", ""] + [f"- {r}" for r in d["evaluation"]["recommendations"]]
    if env["warnings"]:
        lines += ["", f"## {L['warn']}", ""] + [f"- [{w['code']}] {w['message']}" for w in env["warnings"]]
    lines += ["", f"## {L['assume']}", ""] + [f"- {a}" for a in env["assumptions"]]
    lines += ["", "---", f"payload_hash: `{env['metadata']['payload_hash']}` · engine {env['metadata']['engine_version']}"]

    data = {"report_markdown": "\n".join(lines),
            "summary": {"name": s.get("name"), "n_plies": s["n_plies"],
                        "is_symmetric_stack": s["is_symmetric_stack"],
                        "coupling_ratio": idx["coupling_ratio"]["value"],
                        "quasi_isotropy_score": idx["quasi_isotropy_score"]["value"],
                        "zeta_x": d["neutral_surface"]["x"]["zeta"],
                        "recommendations": d["evaluation"]["recommendations"]}}
    hash_payload = {"laminate": payload, "criteria": criteria, "language": language}
    return ENV.build(data=data, errors=[], warnings=env["warnings"], payload=hash_payload,
                     unit_system=env["metadata"]["unit_system_in"],
                     assumptions_extra=[a for a in env["assumptions"] if a not in ENV.BASE_ASSUMPTIONS],
                     include_debug=include_debug, t0=t0)


# ── V2-1차: 열탄성 휨 / 균질화 / 크랙 차폐 (계획서 §17) ─────────────────────


def _cte_vectors_or_error(si) -> tuple[list | None, list[dict]]:
    """전 ply CTE 확보 검사 → [αx,αy,αxy] 목록 또는 E203."""
    from app.solver import thermal as TH
    missing = [k for k, p in enumerate(si.plies) if not p.has_cte]
    if missing:
        return None, [item("E203", field="laminae",
                           detail=f"laminae{missing} 에 CTE가 없습니다")]
    return [TH.alpha_vector(p.alpha1, p.alpha2, p.angle_deg) for p in si.plies], []


def run_thermal(payload, delta_t=None, panel=None, delta_c=None,
                include_debug: bool = False) -> dict:
    """자유 열·흡습 변형: 유효 CTE/CME·곡률·ply 잔류응력·판 휨 (compute_thermal_response, §17.1·§17.5.4)."""
    from app.solver import thermal as TH
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    def _num(v):
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v))
    dT = float(delta_t) if _num(delta_t) else (None if delta_t is None else "bad")
    dC = float(delta_c) if _num(delta_c) else (None if delta_c is None else "bad")
    if dT == "bad" or dC == "bad" or ((dT in (None, 0.0)) and (dC in (None, 0.0))):
        return ENV.build(data=None, errors=[item("E100", field="delta_T/delta_C",
                                                 detail="delta_T [K] 또는 delta_C [%M] 중 최소 하나는 0이 아닌 숫자여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    dT = dT or 0.0
    dC = dC or 0.0
    if panel is not None:
        if _panel_to_si(panel, si.unit_system)[0] is None:
            return ENV.build(data=None, errors=[item("E100", field="panel",
                                                     detail="panel은 {\"Lx\": >0, \"Ly\": >0} 만 허용합니다 (유한 양수, 길이 단위). 경계조건은 panel 이 아니라 boundary 인자로 주세요")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)

    # 자유변형 벡터 = αΔT + βΔc (필요 물성 검사, §17.5.4)
    if dT != 0.0:
        missing = [k for k, p_ in enumerate(si.plies) if not p_.has_cte]
        if missing:
            return ENV.build(data=None, errors=[item("E203", field="laminae",
                                                     detail=f"laminae{missing} 에 CTE(alpha)가 없습니다 (delta_T 해석)")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
    if dC != 0.0:
        missing = [k for k, p_ in enumerate(si.plies) if not p_.has_cme]
        if missing:
            return ENV.build(data=None, errors=[item("E203", field="laminae",
                                                     detail=f"laminae{missing} 에 CME(beta)가 없습니다 (delta_C 해석)")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
    eps_free = []
    for p_ in si.plies:
        e = np.zeros(3)
        if dT != 0.0:
            e = e + TH.alpha_vector(p_.alpha1, p_.alpha2, p_.angle_deg) * dT
        if dC != 0.0:
            e = e + TH.alpha_vector(p_.beta1, p_.beta2, p_.angle_deg) * dC
        eps_free.append(e)

    A, B, D, z, h = _abd_of_plies(si.plies)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]
    N_f, M_f = TH.free_strain_loads(qbars, eps_free, z)
    try:
        eps0, kappa = TH.thermal_response(A, B, D, N_f, M_f)
    except RESP.SingularSystemError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    sig = TH.residual_stresses_free(qbars, eps_free, z, eps0, kappa)
    per_ply = [{"ply": k, "sigma_xyz": (s_ * f["modulus"]).tolist()} for k, s_ in enumerate(sig)]
    worst = max(range(len(sig)), key=lambda k: float(np.max(np.abs(sig[k]))))
    trunc_note = None
    if len(per_ply) > config.SUMMARY_PLY_LIMIT:
        ranked = sorted(per_ply, key=lambda r: -max(abs(v) for v in r["sigma_xyz"]))
        per_ply = sorted(ranked[:config.SUMMARY_TOP_N], key=lambda r: r["ply"])
        trunc_note = (f"ply {len(sig)}개 중 |σ| 상위 {len(per_ply)}개만 반환 (§6.6 토큰 예산). "
                      f"max_abs는 전체 기준")

    data: dict = {
        "response": {
            "delta_T": dT if dT != 0.0 else None,
            "delta_C": dC if dC != 0.0 else None,
            "epsilon0": eps0.tolist(),
            "kappa": (kappa * f["kappa"]).tolist(),
        },
        "residual_stress": {
            "note": "ply 중앙면 값 σ = Q̄(ε0+z̄κ−ε_free). 힘 평형(Σσt=0)은 엔진 불변식",
            "per_ply": per_ply,
            **({"truncation": trunc_note} if trunc_note else {}),
            "max_abs": {"ply": worst, "value": float(np.max(np.abs(sig[worst])) * f["modulus"])},
        },
    }
    if dT != 0.0 and dC == 0.0:
        data["effective_cte"] = {
            "alpha_x": float(eps0[0] / dT), "alpha_y": float(eps0[1] / dT),
            "alpha_xy": float(eps0[2] / dT),
            "definition": "자유 열변형 막변형률/ΔT [1/K] (비대칭이면 곡률 커플링 포함 유효값)",
        }
        data["response"]["kappa_per_K"] = (kappa * f["kappa"] / dT).tolist()
    elif dC != 0.0 and dT == 0.0:
        data["effective_cme"] = {
            "beta_x": float(eps0[0] / dC), "beta_y": float(eps0[1] / dC),
            "beta_xy": float(eps0[2] / dC),
            "definition": "자유 흡습변형 막변형률/Δc [1/%M]",
        }
    else:
        data["note"] = "열+흡습 동시 — 유효 계수는 분리 호출로 산출 (총 응답만 반환)"
    if panel is not None:
        lx_si = panel["Lx"] * units.TO_SI[si.unit_system]["length"]
        ly_si = panel["Ly"] * units.TO_SI[si.unit_system]["length"]
        wp = TH.warpage_over_panel(kappa, lx_si, ly_si)
        data["warpage"] = {
            "panel": {"Lx": panel["Lx"], "Ly": panel["Ly"]},
            "range": wp["warpage_range"] * f["z"],
            "definition": "w(x,y)=−½(κx x²+κy y²+κxy xy)의 판 위 최대−최소 (coplanarity). 9점 평가",
        }
        # §18.5 선형 유효범위 게이트 — w/h가 크면 이 선형 곡률은 실현되지 않는다
        w_over_h = wp["warpage_range"] / h
        data["warpage"]["w_over_thickness"] = w_over_h
        if w_over_h > 0.3:
            warnings.append(item("W130", field="warpage",
                                 detail=f"w/h = {w_over_h:.3g} > 0.3 — 소변형(선형) 가정을 벗어났다. "
                                        f"실제 곡률은 이보다 작고, 비대칭 적층이면 형상이 안장이 아니라 "
                                        f"원통 두 개로 분기할 수 있다. compute_bistable_shapes 로 확인할 것"))

    # §19.7 구속 게이트 — 자유 경계만 풀면 대칭 적층은 κ≈0 이라 "열변형 무시 가능"으로 읽힌다.
    # 그런데 면내가 구속되면 같은 ΔT 가 압축을 만들어 좌굴한다. 반력 N^T 는 이미 손에 있다.
    # 실측: [±45/0/90]s 1.0mm·ΔT=+100·200×200 → warpage 7.1e-16 mm·경고 0건 인데
    # 완전구속 ΔT_cr = 49.7 K (2배 초과). [0/90]s 0.5mm 는 ΔT_cr = 7.7 K.
    data["thermal_loads"] = {
        "N_thermal": (N_f * f["A"]).tolist(),
        "M_thermal": (M_f * f["load_m"]).tolist(),
        "definition": ("자유변형이 만드는 합력·합모멘트. **면내를 완전 구속하면 반력 N = −N_thermal "
                       "이 그대로 작용한다** — 팽창이면 압축이라 좌굴을 유발한다"),
    }
    n_restr = -N_f                       # 완전 면내 구속 시 실제로 작용하는 하중
    if panel is not None and float(n_restr[0]) < 0.0:
        from app.solver import plate_navier as NAV
        d_use, _appl, _hh, _kk = _bending_stiffness_for_navier(si, warnings)
        ratio = float(n_restr[1] / n_restr[0]) if n_restr[0] != 0.0 else 0.0
        bres = NAV.buckling_ncr(d_use, lx_si, ly_si, ratio)
        if bres["N_cr"] is not None:
            factor = bres["N_cr"] / abs(float(n_restr[0]))
            blk = {
                "N_cr": bres["N_cr"] * f["A"],
                "restrained_Nx": float(n_restr[0]) * f["A"],
                "load_factor_to_buckling": factor,
                "mode": {"m": bres["mode_m"], "n": bres["mode_n"]},
                "definition": ("면내 **완전 구속** 극단. load_factor = N_cr/|N_구속| — 1보다 작으면 "
                               "이 ΔT/ΔC 에서 이미 좌굴한다. 실제 구속은 자유와 완전구속 사이다"),
            }
            if dT != 0.0 and dC == 0.0:
                blk["delta_T_critical"] = abs(dT) * factor
            data["restrained_buckling"] = blk
            if factor < 1.0:
                warnings.append(item("W130", field="restrained_buckling",
                                     detail=f"자유 경계 해는 휨이 거의 없다고 답하지만, **면내가 구속되면 "
                                            f"이 조건에서 이미 좌굴한다**(N_cr/|N_구속| = {factor:.3g})."
                                            + (f" ΔT_cr ≈ {blk['delta_T_critical']:.3g} K." if 'delta_T_critical' in blk else "")
                                            + " 실제 구속 정도를 확인하고 compute_buckling 으로 상세 판정할 것"))
            elif factor < 3.0:
                warnings.append(item("W130", field="restrained_buckling",
                                     detail=f"면내 완전 구속 시 좌굴 여유가 {factor:.3g} 배로 얇다 — "
                                            f"구속 조건을 확인할 것"))

    extra = [
        "자유변형 해석 가정: 선형 CTE/CME(온도·수분 무관 — Tg 이상 α 급변 미반영), 자유 경계·소변형",
        "thermal_loads 는 자유변형 합력이다 — 면내 구속 시 반력이 되어 좌굴을 유발할 수 있다",
        "delta_T = T_현재 − T_무응력기준, delta_C = 수분함량 변화 [%M]",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "delta_T": delta_t, "delta_C": delta_c, "panel": panel}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


def run_homogenize(components, include_debug: bool = False) -> dict:
    """면내 병렬(Voigt) 층 균질화 — 동박률 등가 물성 (homogenize_layer Tool, §17.1).

    단위계 무관: 출력 E/α/ρ 단위는 입력과 동일하다.
    """
    from app.solver import thermal as TH
    t0 = time.perf_counter()
    err = None
    parsed = []
    if not isinstance(components, list) or len(components) < 2:
        err = item("E100", field="components",
                   detail="components는 2개 이상 {material(isotropic), volume_fraction} 목록이어야 합니다")
    else:
        for i, c in enumerate(components):
            m = c.get("material") if isinstance(c, dict) else None
            fvol = c.get("volume_fraction") if isinstance(c, dict) else None
            ok = (isinstance(m, dict) and m.get("type") == "isotropic"
                  and isinstance(m.get("E"), (int, float)) and m.get("E") > 0
                  and isinstance(m.get("nu"), (int, float))
                  and isinstance(fvol, (int, float)) and not isinstance(fvol, bool) and 0 < fvol <= 1)
            if not ok:
                err = item("E100", field=f"components[{i}]",
                           detail=f"components[{i}]는 {{material: {{type:'isotropic', E>0, nu}}, volume_fraction∈(0,1]}} 이어야 합니다")
                break
            parsed.append((float(fvol), float(m["E"]), float(m["nu"]),
                           float(m["alpha"]) if isinstance(m.get("alpha"), (int, float)) else None,
                           float(m["rho"]) if isinstance(m.get("rho"), (int, float)) else None))
    if err is None and abs(sum(p[0] for p in parsed) - 1.0) > 1e-6:
        err = item("E100", field="components",
                   detail=f"volume_fraction 합이 1이어야 합니다 (현재 {sum(p[0] for p in parsed):.6f})")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=[], payload={"components": components},
                         include_debug=include_debug, t0=t0)

    hom = TH.homogenize_voigt(parsed)
    material = {"type": "isotropic", "E": hom["E"], "nu": hom["nu"]}
    if hom["alpha"] is not None:
        material["alpha"] = hom["alpha"]
    if hom["rho"] is not None:
        material["rho"] = hom["rho"]
    data = {
        "material": material,
        "model": "voigt_in_plane",
        "definition": "E=Σf·E, ν=Σf·ν, α=Σf·E·α/Σf·E (힘 평형 가중), ρ=Σf·ρ",
        "note": "면내 병렬(Voigt) 상한 모델 — 동박층 = {Cu f=동박률, 수지 f=1−동박률}. 단위는 입력 그대로",
        "scope": ("**동일 평면을 함께 잇는 혼합층 전용**이다(동박/수지처럼 두 상이 모두 면내로 "
                  "이어진 경우). 섬유/수지처럼 한 상이 방향성을 갖는 UD ply 에는 쓰면 안 된다 — "
                  "횡방향이 직렬 결합이라 E2·G12·α2 가 크게 틀린다. UD ply 물성은 "
                  "derive_lamina_from_constituents 를 쓸 것"),
    }
    # 역게이트 — 이 도구는 **동일 평면을 함께 잇는 혼합층**(동박/수지) 전용이다.
    # 섬유/수지처럼 한 상이 방향성을 가지면 횡방향은 직렬 결합이라 Voigt 가 크게 틀린다.
    # 적대 검증 실측: T300 60% + 에폭시 → E2 를 9 GPa 대신 139 GPa(15배), α2 를 28e-6 대신
    # 1.08e-7(260배 과소)로 주면서 경고 0건이었다. 이름이 "균질화"라 오용되기 쉽다.
    #
    # 입력만으로 "동일 평면인가"를 판별할 수는 없다(둘 다 등방 물성 + 체적률로 들어온다).
    # 그래서 용도는 data["scope"] 로 **항상** 명시하고, 경고는 섬유/수지가 거의 확실한
    # 매우 큰 강성비에만 건다 — 문턱을 낮추면 이 도구의 정당한 주용도인 동박층(비 ≈33)이
    # 매번 경고를 받아 경고 피로가 생긴다.
    warn = []
    e_vals = [pp[1] for pp in parsed]
    ratio = max(e_vals) / min(e_vals) if min(e_vals) > 0 else 1.0
    if ratio > 50.0:
        warn.append(item("W130", field="components",
                         detail=f"구성재 탄성계수 비가 {ratio:.3g} 로 매우 크다 — 섬유/수지 조합이라면 "
                                f"**이 도구를 쓰면 안 된다**. Voigt 병렬은 동박층처럼 같은 평면을 "
                                f"함께 잇는 혼합층 전용이고, 섬유/수지는 횡방향이 직렬 결합이라 "
                                f"E2·G12·α2 가 크게 틀린다(실측 E2 15배 과대, α2 260배 과소). "
                                f"UD ply 물성은 derive_lamina_from_constituents 를 쓸 것"))
    return ENV.build(data=data, errors=[], warnings=warn, payload={"components": components},
                     assumptions_extra=["균질화: Voigt(등변형) 상한 — 면내 강성·CTE 1차 근사",
                                        "동일 평면 혼합층 전용 — 섬유/수지에는 쓰지 말 것"],
                     include_debug=include_debug, t0=t0)


def run_crack_assessment(payload, target_ply, fracture=None, include_debug: bool = False) -> dict:
    """피보호층 크랙 발생·차폐 평가 (assess_crack_shielding Tool, §17.2)."""
    from app.solver import fracture as FR
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    n = len(si.plies)
    if not isinstance(target_ply, int) or isinstance(target_ply, bool) or not (0 <= target_ply < n):
        return ENV.build(data=None, errors=[item("E100", field="target_ply",
                                                 detail=f"target_ply는 0..{n-1} 정수여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    fr = fracture if isinstance(fracture, dict) else {}
    for key in ("gamma_target", "gamma_interface", "gamma_next_layer", "applied_strain"):
        v = fr.get(key)
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or
                              (key != "applied_strain" and v <= 0)):
            return ENV.build(data=None, errors=[item("E100", field=f"fracture.{key}",
                                                     detail=f"fracture.{key}는 양수(변형률은 실수)여야 합니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    f_gam = units.TO_SI[si.unit_system]["energy_area"]
    tp = si.plies[target_ply]
    # 하중 방향 x 기준 유효 계수 (취성층의 채널링은 x하중-횡방향 크랙 구도)
    E_t = MAT.ex_engineering(tp.E1, tp.E2, tp.G12, tp.nu12, tp.angle_deg)
    nu_t = tp.nu12 if tp.is_isotropic else 0.0  # 직교이방은 평면변형 보정 생략(1차 근사) — 가정 명시
    h_t = tp.thickness

    target_block = {"ply": target_ply, "name": tp.name,
                    "thickness": h_t * f["z"], "E_x": E_t * f["modulus"]}

    neighbors = []
    for side, idx in (("below", target_ply - 1), ("above", target_ply + 1)):
        if not (0 <= idx < n):
            continue
        np_ = si.plies[idx]
        E_n = MAT.ex_engineering(np_.E1, np_.E2, np_.G12, np_.nu12, np_.angle_deg)
        nu_n = np_.nu12 if np_.is_isotropic else 0.0
        G_n = E_n / (2.0 * (1.0 + nu_n))
        alpha_d, beta_d = FR.dundurs_parameters(E_t, nu_t, E_n, nu_n)
        nb = {"ply": idx, "side": side, "name": np_.name,
              "dundurs_alpha": alpha_d, "dundurs_beta": beta_d,
              "shielding_tendency": ("이웃이 더 강성 → 개구 차폐" if alpha_d < 0
                                     else "유연한 이웃 → 크랙 구동력 증폭 경향"),
              "transfer_length": FR.shear_lag_transfer_length(E_t, h_t, np_.thickness, G_n) * f["z"]}
        if np_.ve_E0 is not None and np_.ve_Einf is not None:
            ve = FR.viscoelastic_relaxation_factor(np_.ve_E0, np_.ve_Einf)
            alpha_rel, _ = FR.dundurs_parameters(
                E_t, nu_t, np_.ve_Einf * (E_n / max(np_.ve_E0, 1e-300)), nu_n)
            nb["viscoelastic"] = {
                "E0": np_.ve_E0 * f["modulus"], "Einf": np_.ve_Einf * f["modulus"],
                "tau_s": np_.ve_tau_s,
                "transfer_length_growth": ve["transfer_length_growth"],
                "dundurs_alpha_relaxed": alpha_rel,
                "meaning": "이완 후 전달길이 ×배 → crack-opening 구속 저하 (준탄성 근사)",
            }
        neighbors.append(nb)

    data: dict = {"target": target_block, "neighbors": neighbors}

    eps = fr.get("applied_strain")
    if eps is not None:
        sigma_t = E_t * float(eps)
        data["crack_driving"] = {
            "applied_strain": float(eps),
            "sigma_target": sigma_t * f["modulus"],
            "G_ss_tunnel": FR.tunnel_crack_gss(sigma_t, h_t, E_t, nu_t) * f["energy_area"],
            "crack_opening_max": FR.crack_opening_max(sigma_t, h_t, E_t, nu_t) * f["z"],
            "definition": "G_ss=πσ²h/(4Ē) (균질 근사), δ_max=2σh/Ē",
        }
    g_t = fr.get("gamma_target")
    if g_t is not None:
        sc = FR.critical_channeling_stress(float(g_t) * f_gam, h_t, E_t, nu_t)
        data["initiation_threshold"] = {
            "sigma_critical": sc * f["modulus"],
            "strain_critical": sc / E_t,
            "definition": "σ_c=√(4ĒΓ/(πh)) — 박층일수록 문턱↑ (h_t 절반 ⇒ σ_c ×√2)",
        }
    if fr.get("gamma_interface") is not None or fr.get("gamma_next_layer") is not None:
        data["interface_deflection"] = FR.interface_deflection_verdict(
            float(fr["gamma_interface"]) * f_gam if fr.get("gamma_interface") is not None else None,
            float(fr["gamma_next_layer"]) * f_gam if fr.get("gamma_next_layer") is not None else None)

    extra = [
        "파괴 해석 가정: 문헌 폐형해의 균질/1차 근사 — 채널링 g(α,β) 수치 보정·모드믹스 의존 계면인성 미탑재 (경향 판단용)",
        "직교이방 ply는 하중방향 유효계수 E_x(θ) 기반 등가 등방 근사",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "target_ply": target_ply, "fracture": fracture}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


def run_ply_stresses(payload, loads=None, delta_t=None, detail: str = "auto",
                     panel=None, include_debug: bool = False) -> dict:
    """층별 기계(+선택 열) 응력 복원과 파손 판정 (recover_ply_stresses Tool, §17.4).

    강도(strength)가 있는 ply는 Tsai-Wu 강도비 R·Max Stress 모드까지, 없는 ply는 응력만.
    delta_t를 주면 열하중을 중첩한다(전 ply CTE 필요 — E203).
    detail: "auto"(기본 — ply 수가 크면 임계 상위만 반환+note) | "full" | "summary".
    """
    from app.solver import failure as FAIL
    from app.solver import thermal as TH
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    N_si, M_si, load_err = _loads_to_si(loads, si.unit_system)
    if load_err is not None and delta_t is None:
        return ENV.build(data=None, errors=[load_err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    if load_err is not None:                      # 열 단독 하중 허용 (loads 생략/0 가능)
        N_si, M_si = np.zeros(3), np.zeros(3)
    if delta_t is not None and (not isinstance(delta_t, (int, float)) or isinstance(delta_t, bool)):
        return ENV.build(data=None, errors=[item("E100", field="delta_T",
                                                 detail="delta_T는 숫자 [K]여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    alphas = None
    dT = float(delta_t) if delta_t is not None else 0.0
    if dT != 0.0:
        alphas, cte_err = _cte_vectors_or_error(si)
        if cte_err:
            return ENV.build(data=None, errors=cte_err, warnings=warnings, payload=payload,
                             unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    A, B, D, z, h = _abd_of_plies(si.plies)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg) for p in si.plies]
    N_tot, M_tot = N_si.copy(), M_si.copy()
    if dT != 0.0:
        N_th, M_th = TH.thermal_loads(qbars, alphas, z, dT)
        N_tot += N_th
        M_tot += M_th
    try:
        eps0, kappa = RESP.solve_response(A, B, D, N_tot, M_tot)
    except RESP.SingularSystemError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    plies_out = []
    fpf = None                      # (R, ply, loc, mode)
    no_strength = []
    for k, p in enumerate(si.plies):
        a_vec = alphas[k] if alphas is not None else None
        locs = {"bottom": float(z[k]), "mid": float(z[k] + z[k + 1]) / 2.0, "top": float(z[k + 1])}
        row = {"ply": k, "angle_deg": p.angle_deg, "name": p.name, "stresses": {}}
        worst_here = None           # (R, loc, assess)
        for loc, zv in locs.items():
            s_xyz = FAIL.ply_stresses_at(qbars[k], eps0, kappa, zv, a_vec, dT)
            s_12 = FAIL.stress_to_material_axes(s_xyz, p.angle_deg)
            entry = {"sigma_xyz": (s_xyz * f["modulus"]).tolist(),
                     "sigma_12": (s_12 * f["modulus"]).tolist()}
            if p.strength is not None:
                assess = FAIL.assess_ply(s_12, p.strength)
                entry["failure"] = {
                    "tsai_wu_R": assess["tsai_wu"].get("strength_ratio"),
                    "max_stress_FI": assess["max_stress"]["failure_index"],
                    "governing_mode": assess["governing_mode"],
                    "fails": assess["fails"],
                }
                R = assess["tsai_wu"].get("strength_ratio")
                if R is not None and (worst_here is None or R < worst_here[0]):
                    worst_here = (R, loc, assess)
            row["stresses"][loc] = entry
        if p.strength is None:
            no_strength.append(k)
        elif worst_here is not None:
            row["min_tsai_wu_R"] = worst_here[0]
            if fpf is None or worst_here[0] < fpf[0]:
                fpf = (worst_here[0], k, worst_here[1], worst_here[2]["governing_mode"])
        plies_out.append(row)

    if detail not in ("auto", "full", "summary"):
        return ENV.build(data=None, errors=[item("E100", field="detail",
                                                 detail="detail은 auto|full|summary 중 하나여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    plies_full = list(plies_out)     # 절단 전 전체 — 항복 게이트는 전 ply 를 봐야 한다
    truncated_note = None
    if detail != "full" and (detail == "summary" or len(plies_out) > config.SUMMARY_PLY_LIMIT):
        # 임계도 순 상위 N만 (무단 절단 금지 — note로 명시, detail="full"로 전체 조회 가능)
        def crit(row):
            if "min_tsai_wu_R" in row:
                return row["min_tsai_wu_R"]                      # R 작을수록 위험
            return -max(abs(v) for e in row["stresses"].values() for v in e["sigma_xyz"])
        keep = sorted(plies_out, key=crit)[:config.SUMMARY_TOP_N]
        keep_ids = {r["ply"] for r in keep}
        truncated_note = (f"ply {len(plies_out)}개 중 임계 상위 {len(keep)}개만 반환 "
                          f"(기준: min Tsai-Wu R 오름차순, 강도 없으면 |σ| 내림차순). "
                          f"전체는 detail=\"full\"로 재호출")
        plies_out = sorted(keep, key=lambda r: r["ply"])

    data: dict = {"load_state": {"N": (N_si / units.TO_SI[si.unit_system]["load_n"]).tolist(),
                                 "M": M_si.tolist(),
                                 "delta_T": dT if dT != 0.0 else None},
                  "plies": plies_out}
    if truncated_note:
        data["truncation"] = truncated_note
    if fpf is not None:
        R, kp, loc, mode = fpf
        data["first_ply_failure"] = {
            "ply": kp, "location": loc, "tsai_wu_R": R, "governing_mode": mode,
            "fails_at_current_load": bool(R <= 1.0),
            "meaning": "현재 하중을 R배 하면 해당 ply가 Tsai-Wu 파손면에 도달 (R>1 = 여유)",
        }
    if no_strength:
        data["note"] = f"laminae{no_strength}는 strength 미입력 — 응력만 복원(파손 판정 제외). " \
                       f"강도는 material.strength {{Xt,Xc,Yt,Yc,S}}로 입력"

    # §19.15 항복 게이트 — 서버 전체가 완전탄성이라 소성이 들어가면 조용히 틀린다.
    # 소성 솔버를 짓기 전에 "탄성 가정이 깨졌다"만 말해도 이 문제 가치의 대부분이다.
    yielded = []
    for row in plies_full:
        p_ = si.plies[row["ply"]]
        if p_.sigma_y is None:
            continue
        peak = max(abs(FAIL.von_mises_plane_stress(np.asarray(v["sigma_xyz"]) / f["modulus"]))
                   for v in (row["stresses"]["bottom"], row["stresses"]["mid"],
                             row["stresses"]["top"]))
        if peak >= p_.sigma_y:
            yielded.append({"ply": row["ply"], "von_mises": peak * f["modulus"],
                            "sigma_y": p_.sigma_y * f["modulus"],
                            "ratio": peak / p_.sigma_y})
    if yielded:
        data["yielding"] = {
            "plies": yielded,
            "definition": "평면응력 von Mises √(σx²−σxσy+σy²+3τxy²) 의 ply 내 3점 최대값 대비 σ_y",
        }
        worst = max(yielded, key=lambda r: r["ratio"])
        warnings.append(item("W130", field="yielding",
                             detail=f"laminae[{worst['ply']}] 의 von Mises 가 항복강도의 "
                                    f"{worst['ratio']:.3g}배다 — **완전탄성 가정이 깨졌다**. "
                                    f"이 서버는 소성을 모델링하지 않으므로 응력·곡률·잔류가 모두 "
                                    f"과대평가된다. 제하 후 잔류 곡률도 실제로는 0이 아니다"))
    elif any(p_.sigma_y is not None for p_ in si.plies):
        data["yielding"] = {"plies": [], "note": "항복강도가 주어진 ply 는 모두 탄성 범위 안이다"}

    # §19.6 지배모드 게이트 — 압축이면 좌굴이 먼저 올 수 있다(실측 최대 410배 모순)
    gov = _stability_gate(si, N_si, panel, warnings,
                          r_strength=(fpf[0] if fpf is not None else None))
    if gov is not None:
        data["governing_mode"] = gov

    extra = [
        "파손 판정: Tsai-Wu(F12=-0.5√(F11F22) 표준 상호작용) 강도비 R 주지표 + Max Stress 지배 모드",
        "응력은 각 ply의 bottom/mid/top 3점 (ply 내 z 선형)",
        *(["열하중 중첩: 선형 CTE, 자유 경계 (§17.1 가정 동일)"] if dT != 0.0 else []),
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "loads": loads, "delta_T": delta_t, "panel": panel}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


# ── V2-2차: 설계규칙 / 좌굴 / 진동 / 진행성 파손 (계획서 §17.5) ─────────────


def _bending_stiffness_for_navier(si, warnings) -> tuple:
    """(D_used, 사용정보) — 비대칭이면 D* = D − B A⁻¹ B + W130, D16/D26 유의 시 W130 (§17.5.2)."""
    from app.solver import plate_navier as NAV
    A, B, D, z, h = _abd_of_plies(si.plies)
    A_hat, B_hat, D_hat, K_hat = ABD.normalized_stiffness(A, B, D, h)
    reduced = False
    D_use = D
    if float(np.linalg.norm(B_hat)) > 1e-6 * float(np.linalg.norm(A_hat)):
        D_use = NAV.reduced_bending_stiffness(A, B, D)
        reduced = True
        warnings.append(item("W130", field="laminate",
                             detail="비대칭 적층 — 축소 굽힘강성 D* = D − B·A⁻¹·B 근사 사용"))
    bt = NAV.bend_twist_significance(D_use)
    if bt > 0.05:
        warnings.append(item("W130", field="laminate",
                             detail=f"굽힘-비틀림 커플링 유의 (max|D16,D26|/√(D11·D22) = {bt:.3g}) — "
                                    f"Navier(specially orthotropic) 해는 비보수적일 수 있음"))
    return D_use, {"reduced_stiffness_used": reduced, "bend_twist_ratio": bt}, h, K_hat


PANEL_KEYS = ("Lx", "Ly")


def _panel_to_si(panel, us):
    """panel {"Lx","Ly"} → SI 길이. 형식·유한성·양수 위반 시 (None, None) (EDGE-01).

    **미지 키는 조용히 무시하지 않는다.** 적대 검증에서 panel={"Lx","Ly","edge_condition"}
    과 panel={"Lx","Ly","typo_Lx":999} 가 둘 다 status=ok 로 통과했다 — 에이전트는 고정단을
    요청했다고 믿고 단순지지 값을 보고하게 된다. 지원하지 않는 키가 있으면 실패시킨다.
    """
    if not (isinstance(panel, dict)
            and all(isinstance(panel.get(k), (int, float)) and not isinstance(panel.get(k), bool)
                    and math.isfinite(panel.get(k)) and panel.get(k) > 0 for k in PANEL_KEYS)):
        return None, None
    if set(panel) - set(PANEL_KEYS):
        return None, None
    f_len = units.TO_SI[us]["length"]
    return panel["Lx"] * f_len, panel["Ly"] * f_len


def run_design_rules(payload, contiguity_limit: int = 4, include_debug: bool = False) -> dict:
    """적층 설계 규칙 검사 (check_design_rules Tool, §17.5.1)."""
    from app.solver import design_rules as DR
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    if not isinstance(contiguity_limit, int) or isinstance(contiguity_limit, bool) or contiguity_limit < 1:
        return ENV.build(data=None, errors=[item("E100", field="contiguity_limit",
                                                 detail="contiguity_limit는 1 이상 정수여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    A, B, D, z, h = _abd_of_plies(si.plies)
    A_hat, B_hat, D_hat, _ = ABD.normalized_stiffness(A, B, D, h)
    nA, nB, nD = (float(np.linalg.norm(m)) for m in (A_hat, B_hat, D_hat))
    cr = nB / np.sqrt(nA * nD) if nA * nD > 0 else None
    a16 = max(abs(A_hat[0, 2]), abs(A_hat[1, 2])) / np.sqrt(A_hat[0, 0] * A_hat[1, 1]) \
        if A_hat[0, 0] * A_hat[1, 1] > 0 else None

    rules = DR.check_rules(si.plies, si.fingerprint, cr, a16, contiguity_limit)
    judged = [r for r in rules if r["pass"] is not None]
    data = {
        "rules": rules,
        "summary": {
            "n_pass": sum(1 for r in judged if r["pass"]),
            "n_fail": sum(1 for r in judged if not r["pass"]),
            "hard_fails": [r["rule"] for r in judged if not r["pass"] and r["severity"] == "hard"],
        },
    }
    return ENV.build(data=data, errors=[], warnings=warnings, payload={"laminate": payload,
                                                                      "contiguity_limit": contiguity_limit},
                     unit_system=si.unit_system,
                     assumptions_extra=["설계 규칙은 업계 관례 휴리스틱 — 위반이 곧 불합격은 아니며 근거와 함께 검토용"],
                     include_debug=include_debug, t0=t0)


def run_buckling(payload, panel, load_ratio: float = 0.0, applied_Nx=None,
                 boundary: str = "simply_supported", include_debug: bool = False) -> dict:
    """직교이방 판 좌굴 임계 (compute_buckling Tool, §17.5.2 · 경계조건 확장 §19.5)."""
    from app.solver import plate_navier as NAV
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    a, b = _panel_to_si(panel, si.unit_system)
    if a is None:
        return ENV.build(data=None, errors=[item("E100", field="panel",
                                                 detail="panel은 {\"Lx\": >0, \"Ly\": >0} 만 허용합니다 (길이 단위). 지원하지 않는 키가 있으면 조용히 무시하지 않고 실패시킵니다 — 경계조건은 boundary 인자로 주세요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    if (not isinstance(load_ratio, (int, float)) or isinstance(load_ratio, bool)
            or not math.isfinite(load_ratio)):
        return ENV.build(data=None, errors=[item("E100", field="load_ratio",
                                                 detail="load_ratio = Ny/Nx 는 유한한 숫자여야 합니다 (압축 양수)")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    if applied_Nx is not None and (not isinstance(applied_Nx, (int, float))
                                   or isinstance(applied_Nx, bool)
                                   or not math.isfinite(applied_Nx) or applied_Nx <= 0):
        return ENV.build(data=None, errors=[item("E100", field="applied_Nx",
                                                 detail="applied_Nx는 압축 크기(양수)여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    bpairs = NAV.normalize_boundary(boundary)
    if bpairs is None:
        return ENV.build(data=None, errors=[item("E100", field="boundary",
                                                 detail=f"boundary는 'simply_supported'·'clamped' 또는 "
                                                        f"S·C 4글자 코드(앞 두 글자=x변, 뒤 두 글자=y변, "
                                                        f"예: 'CCSS')여야 합니다. {NAV.FREE_EDGE_NOTE}")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    D_use, appl, h, _ = _bending_stiffness_for_navier(si, warnings)
    _compliant_core_gate(si, min(a, b), warnings, "짧은 변")   # §19.12
    if bpairs != ("SS", "SS"):
        res = NAV.scan_ritz_buckling(D_use, a, b, float(load_ratio), boundary,
                                     NAV.CLAMPED_MODE_LIMIT)
    else:
        res = NAV.buckling_ncr(D_use, a, b, float(load_ratio))
    if res["N_cr"] is None:
        # 압축 지배 모드가 없음(과도한 음수 load_ratio) — 내부 오류가 아니라 입력 문제 (NAV-1)
        return ENV.build(data=None, errors=[item("E100", field="load_ratio",
                                                 detail=f"load_ratio={load_ratio}에서는 {res['reason']}. "
                                                        f"압축을 양수로 하는 Ny/Nx 값을 확인하세요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    # 전단 유연성은 임계 모드 기준 (TS-02: m=1 고정식은 고차 모드에서 45% 비보수)
    flex_b = _shear_flexibility(si, D_use, a, b, res["mode_m"], res["mode_n"],
                                warnings, si.unit_system)
    if res.get("boundary"):
        warnings.append(item("W130", field="buckling.mode",
                             detail=f"임계 모드가 스캔 상한({res['mode_scan']})에 걸렸습니다 — "
                                    f"N_cr이 비보수적으로 과대평가되었을 수 있습니다"))
    if bpairs != ("SS", "SS"):
        warnings.append(item("W130", field="boundary",
                             detail="단순지지 외 경계는 폐형해가 없어 1항 Rayleigh–Ritz 로 푼다 — "
                                    "**상계**라 N_cr 을 과대평가한다. 등방 정사각 실측 오차: "
                                    "CCCC +6.6%(k=10.74 vs 10.07), SSCC +11.6%(7.78 vs 6.97). "
                                    "혼합 경계일수록 오차가 커진다. boundary='SSSS' 값을 하한으로 "
                                    "함께 보아 참값을 감쌀 것"))
    f = units.FROM_SI[si.unit_system]
    data = {
        "N_cr": res["N_cr"] * f["A"],
        "boundary": boundary,
        "boundary_pairs": {"x_edges": bpairs[0], "y_edges": bpairs[1]},
        "mode": {"m": res["mode_m"], "n": res["mode_n"], "scan_limit": res["mode_scan"],
                 "at_scan_boundary": bool(res.get("boundary"))},
        "load_ratio_Ny_over_Nx": float(load_ratio),
        "applicability": appl,
        **({"transverse_shear": {**flex_b,
                                 "corrected_N_cr": res["N_cr"] * f["A"] * flex_b["buckling_factor"]}}
           if flex_b else {}),
        "definition": "SS 4변, N_cr(m,n)=π²[D11(m/a)⁴+2(D12+2D66)(m/a)²(n/b)²+D22(n/b)⁴]/[(m/a)²+R(n/b)²] 최소. 압축 양수, Nxy 미지원",
    }
    if applied_Nx is not None:
        data["margin"] = {"applied_Nx": applied_Nx, "factor": data["N_cr"] / applied_Nx,
                          "meaning": "factor>1 이면 좌굴 전 (임계/작용)"}
    hash_payload = {"laminate": payload, "panel": panel, "load_ratio": load_ratio, "applied_Nx": applied_Nx}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=["좌굴: 4변 단순지지·specially orthotropic Navier 폐형해, 전단하중(Nxy) 미포함"],
                     include_debug=include_debug, t0=t0)


def run_frequencies(payload, panel, n_modes: int = 5, boundary: str = "simply_supported",
                    include_debug: bool = False) -> dict:
    """직교이방 판 고유진동수 (compute_natural_frequencies Tool, §17.5.2 · 경계조건 확장 §19.5)."""
    from app.solver import plate_navier as NAV
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    a, b = _panel_to_si(panel, si.unit_system)
    if a is None:
        return ENV.build(data=None, errors=[item("E100", field="panel",
                                                 detail="panel은 {\"Lx\": >0, \"Ly\": >0} 만 허용합니다 (길이 단위). 지원하지 않는 키가 있으면 조용히 무시하지 않고 실패시킵니다 — 경계조건은 boundary 인자로 주세요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    if not isinstance(n_modes, int) or isinstance(n_modes, bool) or not (1 <= n_modes <= 25):
        return ENV.build(data=None, errors=[item("E100", field="n_modes",
                                                 detail="n_modes는 1..25 정수여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    missing_rho = [k for k, p_ in enumerate(si.plies) if p_.rho is None]
    if missing_rho:
        return ENV.build(data=None, errors=[item("E100", field="laminae",
                                                 detail=f"고유진동수에는 전 ply 밀도(rho)가 필요합니다 — laminae{missing_rho} 누락")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    bpairs = NAV.normalize_boundary(boundary)
    if bpairs is None:
        return ENV.build(data=None, errors=[item("E100", field="boundary",
                                                 detail=f"boundary는 'simply_supported'·'clamped' 또는 "
                                                        f"S·C 4글자 코드(예: 'CCSS')여야 합니다. "
                                                        f"{NAV.FREE_EDGE_NOTE}")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    D_use, appl, h, _ = _bending_stiffness_for_navier(si, warnings)
    _compliant_core_gate(si, min(a, b), warnings, "짧은 변")   # §19.12
    rho_areal = sum(p_.rho * p_.thickness for p_ in si.plies)   # SI kg/m²
    scan = max(NAV.MODE_SCAN, n_modes)      # f는 m·n 단조 증가 → scan ≥ n_modes면 상위 정확 (NAV-3)
    if bpairs != ("SS", "SS"):
        modes = NAV.ritz_frequencies(D_use, rho_areal, a, b, boundary, n_modes, scan)
        warnings.append(item("W130", field="boundary",
                             detail="단순지지 외 경계는 1항 Rayleigh–Ritz 근사다 — 주파수를 약간 "
                                    "과대평가한다. 등방 정사각 1차 모드 실측: CCCC 비 1.829(문헌 1.83, "
                                    "−0.0%)로 매우 정확하지만 CSCS 는 1.379(문헌 1.265, +9.0%)다. "
                                    "혼합 경계·고차 모드일수록 오차가 커진다"))
    else:
        modes = NAV.natural_frequencies(D_use, rho_areal, a, b, n_modes, mode_scan=scan)

    # 모달 감쇠 (§17.6.2) — 전 ply에 loss_factor가 있을 때만, 1차 모드 정합 (MSE-01/02)
    from app.solver import interlaminar as IL
    damping = None
    m1, n1 = (modes[0]["m"], modes[0]["n"]) if modes else (1, 1)
    have_eta = [k for k, p_ in enumerate(si.plies) if p_.loss_factor is not None]
    if len(have_eta) == len(si.plies):
        Am, Bm, Dm_, zc, _h = _abd_of_plies(si.plies)
        qb = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
              for p_ in si.plies]
        eta = IL.modal_loss_factor(qb, zc, [p_.loss_factor for p_ in si.plies],
                                   a_kk=Am[0, 0], b_kk=Bm[0, 0], d_kk=Dm_[0, 0],
                                   a=a, b=b, m=m1, n=n1)
        if eta is not None and eta > 0:
            damping = {"modal_loss_factor": eta, "Q_factor": 1.0 / eta,
                       "mode": {"m": m1, "n": n1},
                       "definition": "MSE법 η = navier(D_η,m,n)/navier(D,m,n) — 1차 모드 정합, "
                                     "비대칭이면 중립면 기준"}
        elif eta == 0.0:
            warnings.append(item("W120", field="material.loss_factor",
                                 detail="전 ply loss_factor가 0 — 감쇠 없음으로 계산됩니다"))
    elif have_eta:
        warnings.append(item("W120", field="material.loss_factor",
                             detail=f"loss_factor가 laminae"
                                    f"{[k for k in range(len(si.plies)) if k not in have_eta]}에 없어 "
                                    f"모달 감쇠를 산출하지 않았습니다 (감쇠 없는 층은 0으로 명시)"))

    # 횡전단 유연성 (§17.6.3) — 1차 모드 기준
    flex = _shear_flexibility(si, D_use, a, b, m1, n1, warnings, si.unit_system)

    data = {
        "modes": modes,                       # f_hz는 단위계 무관 [Hz]
        "scan_limit": scan,
        **({"damping": damping} if damping else {}),
        **({"transverse_shear": {
            **flex,
            "corrected_f1_hz": modes[0]["f_hz"] * flex["frequency_factor"] if modes else None,
        }} if flex else {}),
        "rho_areal": rho_areal * units.FROM_SI[si.unit_system]["areal_mass"],
        "applicability": appl,
        "definition": "SS 4변, ω²=π⁴·navier(m,n)/ρ_areal, f [Hz]. 감쇠·부가질량 미포함",
    }
    hash_payload = {"laminate": payload, "panel": panel, "n_modes": n_modes}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=["진동: 4변 단순지지 Navier 폐형해 — 경계·감쇠·공기 부가질량 미반영"],
                     include_debug=include_debug, t0=t0)


def run_progressive(payload, loads, discount: float = 0.1, delta_t=None,
                    include_debug: bool = False) -> dict:
    """ply discount 진행성 파손 — FPF→한계하중 (run_progressive_failure Tool, §17.5.3·§19.10)."""
    from app.solver import progressive as PROG
    from app.solver import thermal as TH
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    N_si, M_si, load_err = _loads_to_si(loads, si.unit_system)
    if load_err is not None:
        return ENV.build(data=None, errors=[load_err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    if not isinstance(discount, (int, float)) or isinstance(discount, bool) or not (0.0 < discount <= 0.5):
        return ENV.build(data=None, errors=[item("E100", field="discount",
                                                 detail="discount(강성 잔존율 η)는 (0, 0.5] 실수여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    with_strength = [k for k, p_ in enumerate(si.plies) if p_.strength is not None]
    if not with_strength:
        return ENV.build(data=None, errors=[item("E100", field="laminae",
                                                 detail="strength가 있는 ply가 하나도 없습니다 — material.strength {Xt,Xc,Yt,Yc,S} 필요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    # §19.10 열잔류 — 각 discount 단계마다 재계산한다(강성이 바뀌면 잔류도 바뀐다)
    eps_free = None
    dT = 0.0
    if delta_t is not None:
        if not (isinstance(delta_t, (int, float)) and not isinstance(delta_t, bool)
                and math.isfinite(delta_t)):
            return ENV.build(data=None, errors=[item("E100", field="delta_T",
                                                     detail="delta_T는 유한한 숫자여야 합니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
        dT = float(delta_t)
        if dT != 0.0:
            missing = [k for k, p_ in enumerate(si.plies) if not p_.has_cte]
            if missing:
                return ENV.build(data=None, errors=[item("E203", field="laminae",
                                                         detail=f"laminae{missing} 에 CTE(alpha)가 없습니다 (delta_T 해석)")],
                                 warnings=warnings, payload=payload, unit_system=si.unit_system,
                                 include_debug=include_debug, t0=t0)
            eps_free = [TH.alpha_vector(p_.alpha1, p_.alpha2, p_.angle_deg) * dT
                        for p_ in si.plies]

    res = PROG.run(si.plies, N_si, M_si, float(discount), eps_free=eps_free)
    f = units.FROM_SI[si.unit_system]
    events = res["events"]
    trunc = None
    if len(events) > 50:
        head, tail = events[:25], events[-24:]
        trunc = f"사건 {len(events)}건 중 처음 25·마지막 24건만 표시 (§6.6)"
        events = head + tail
    no_strength = [k for k in range(len(si.plies)) if k not in with_strength]
    data = {
        "events": events,
        **({"truncation": trunc} if trunc else {}),
        "first_ply_failure_R": res["events"][0]["R"] if res["events"] else None,
        "ultimate_R": res["ultimate_R"],
        "last_ply_R": res["last_ply_R"],
        "ex_eff_after_events": [e * f["modulus"] for e in res["ex_eff_after_events"]],
        "termination": res["termination"],
        "meaning": "R = 입력 하중 패턴의 배수. ultimate_R×loads = 하중 제어 용량(최대 지지 하중)",
    }
    data["delta_T"] = dT if dT != 0.0 else None
    if no_strength:
        data["note"] = f"laminae{no_strength}는 strength 미입력 — 탄성 유지(비파손) 가정"
    if dT == 0.0 and any(p_.has_cte for p_ in si.plies):
        warnings.append(item("W130", field="delta_T",
                             detail="ply 에 CTE 가 있는데 delta_T 를 주지 않았다 — 경화 냉각 잔류응력이 "
                                    "빠져 있다. 실측으로 [0/90]s CFRP 는 ΔT=−150 K 에서 FPF 하중이 "
                                    "81% 낮아진다. delta_T 를 주면 각 단계마다 잔류를 재계산한다"))
    if res["events"] and res["events"][0]["R"] <= 0.0:
        warnings.append(item("W130", field="first_ply_failure_R",
                             detail="FPF 하중 배수가 0 이다 — **기계 하중 없이 잔류응력만으로 이미 "
                                    "파손**한다. 냉각 폭이나 강도 입력을 확인할 것"))
    hash_payload = {"laminate": payload, "loads": loads, "discount": discount,
                    "delta_T": delta_t}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=[
                         "진행성 파손: ply discount(하중 제어 quasi-static restart) 표준 근사 — "
                         f"기지/전단 파손 시 E2·G12·ν12×{discount}, 섬유 파손 시 전 성분×{discount}",
                         "열잔류응력 미중첩, ply 중앙 응력 기준, Tsai-Wu(첫 파손)+섬유 항(후속) 판정"],
                     include_debug=include_debug, t0=t0)


# ── V2-3차: 층간 응력 / 감쇠 / 횡전단 유연성 (계획서 §17.6) ─────────────────


def _shear_flexibility(si, D_use, a, b, m, n, warnings, unit_system) -> dict | None:
    """모드 정합 횡전단 유연성 R_s와 1차 FSDT 보정 (§17.6.3, 적대 검증 IL-2/TS-01~05 반영).

    A55/A44는 ply 각도 변환 + 에너지등가(Whitney) — 샌드위치에서 코어 전단을 정확히 반영한다.
    """
    from app.solver import interlaminar as IL
    A, B, D, z, h = _abd_of_plies(si.plies)
    a55, a44 = IL.transverse_shear_stiffness(
        si.plies, [(p_.g13, p_.g23) for p_ in si.plies],
        qbars=[MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
               for p_ in si.plies],
        z=z, a_kk=(A[0, 0], A[1, 1]), b_kk=(B[0, 0], B[1, 1]), d_kk=(D[0, 0], D[1, 1]))
    if a55 <= 0 and a44 <= 0:
        return None
    rs = IL.shear_flexibility_ratio(D_use, a55, a44, a, b, m, n)
    if not math.isfinite(rs):
        return None
    assumed = any(p_.g_transverse_assumed for p_ in si.plies)
    if rs > 0.02:
        warnings.append(item("W130", field="transverse_shear",
                             detail=f"횡전단 유연성 R_s = {rs:.3g} (>0.02, 임계모드 m={m},n={n}) — "
                                    f"CLT 진동수·좌굴이 {100*(1-1/np.sqrt(1+rs)):.0f}% 이상 "
                                    f"비보수적일 수 있습니다 (두꺼운 판·샌드위치 코어)"))
    if assumed:
        warnings.append(item("W120", field="material.G13/G23",
                             detail="G13/G23 미지정 — 면내 G12로 근사했습니다 "
                                    "(실제 G23는 보통 더 작아 R_s를 과소평가 = 비보수)"))
    fA = units.FROM_SI[unit_system]["A"]
    return {"A55": a55 * fA, "A44": a44 * fA, "R_s": rs,
            "critical_mode": {"m": m, "n": n},
            "G_transverse_assumed": assumed,
            "frequency_factor": 1.0 / float(np.sqrt(1.0 + rs)),
            "buckling_factor": 1.0 / (1.0 + rs),
            "definition": "R_s = π²·navier(m,n)/(k²·S_eff), S_eff = (A55α²+A44β²)/k², "
                          "α=m/a·β=n/b. A55/A44는 각도변환+에너지등가(Whitney). 보정은 1차 근사"}


def run_interlaminar(payload, shear=None, detail: str = "auto",
                    include_debug: bool = False) -> dict:
    """평형법 층간 전단응력과 ILSS 여유 (compute_interlaminar_stresses Tool, §17.6.1)."""
    from app.solver import interlaminar as IL
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    sh = shear if isinstance(shear, dict) else {}
    if shear is not None and not isinstance(shear, dict):
        return ENV.build(data=None, errors=[item("E100", field="shear",
                                                 detail='shear는 {"Vx": , "Vy": } (단위 폭당 횡전단력) 객체여야 합니다')],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    vx, vy = sh.get("Vx", 0.0), sh.get("Vy", 0.0)
    for nm, v in (("Vx", vx), ("Vy", vy)):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return ENV.build(data=None, errors=[item("E100", field=f"shear.{nm}",
                                                     detail=f"shear.{nm}는 유한한 숫자여야 합니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
    if vx == 0.0 and vy == 0.0:
        return ENV.build(data=None, errors=[item("E100", field="shear",
                                                 detail="Vx·Vy가 모두 0입니다 — 최소 하나는 0이 아니어야 합니다 "
                                                        "(굽힘 모멘트 구배 = 횡전단력)")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    if detail not in ("auto", "full"):
        return ENV.build(data=None, errors=[item("E100", field="detail",
                                                 detail="detail은 auto|full 중 하나여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    A, B, D, z, h = _abd_of_plies(si.plies)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg) for p in si.plies]
    f = units.FROM_SI[si.unit_system]
    fv = units.TO_SI[si.unit_system]["load_n"]        # V도 N/m 계열 (A와 동일 차원)

    data: dict = {"note": "평형법 사후 복원 — 원통형 굽힘(∂M/∂x=V) 가정, 막-굽힘 연성 포함, 자유단 3D 효과 미포함"}
    for comp, (nm, v, akk, bkk, dkk) in enumerate((("x", vx, A[0, 0], B[0, 0], D[0, 0]),
                                                   ("y", vy, A[1, 1], B[1, 1], D[1, 1]))):
        if v == 0.0:
            continue
        try:
            prof = IL.transverse_shear_profile(qbars, z, akk, bkk, dkk, v * fv, comp=comp)
        except ZeroDivisionError:
            return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                             payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
        pk = IL.peak_shear(prof)
        block = {
            "applied_V": v,
            "peak": {"tau": pk["tau"] * f["modulus"], "z_from_midplane": pk["z"] * f["z"],
                     "interface": pk["interface"]},
            "profile": [{"z": p_["z"] * f["z"], "tau": p_["tau"] * f["modulus"],
                         "interface": p_["interface"]} for p_ in prof],
        }
        if detail == "auto" and len(prof) > config.SUMMARY_PLY_LIMIT:
            keep = sorted(prof, key=lambda p_: -abs(p_["tau"]))[:config.SUMMARY_TOP_N]
            keep_z = {p_["z"] for p_ in keep} | {float(z[0]), float(z[-1])}
            block["profile"] = [pt for pt in block["profile"]
                                if any(abs(pt["z"] / f["z"] - kz) < 1e-15 for kz in keep_z)]
            block["profile_truncation"] = (f"프로파일 {len(prof)}점 중 |τ| 상위 "
                                           f"{len(block['profile'])}점만 반환 — 전체는 detail=\"full\"")
        # ILSS 여유 — 계면뿐 아니라 ply 내부 극값까지 (IL-4/IL-02: 최대 τ가 내부면 누락됐음)
        margins, unevaluated = [], []
        for p_ in prof:
            if abs(p_["tau"]) < 1e-300:
                continue
            if p_["interface"]:
                k = int(p_["interface"].split("/")[0])
                cands = [si.plies[k].ilss, si.plies[k + 1].ilss]
                loc = f"interface {p_['interface']}"
            else:
                kk = next((i for i in range(len(si.plies)) if z[i] <= p_["z"] <= z[i + 1]), None)
                if kk is None:
                    continue
                cands = [si.plies[kk].ilss]
                loc = f"ply {kk} 내부"
            avail = [c for c in cands if c is not None]
            if not avail:
                unevaluated.append({"location": loc, "tau": p_["tau"] * f["modulus"],
                                    "reason": "해당 위치 ply에 ilss 미입력"})
                continue
            margins.append({"location": loc, "z": p_["z"] * f["z"],
                            "ilss": min(avail) * f["modulus"],
                            "margin": min(avail) / abs(p_["tau"]),
                            **({"one_sided": True} if len(avail) < len(cands) else {})})
        if margins:
            worst = min(margins, key=lambda m: m["margin"])
            block["ilss_margins"] = margins
            block["critical_location"] = worst
            block["meaning"] = "margin = ILSS/|τ| — 1 미만이면 층간 전단 파손(박리) 예상"
        if unevaluated:
            block["ilss_unevaluated"] = unevaluated[:5]
            block["ilss_note"] = ("일부 위치는 ilss 미입력으로 평가되지 않았습니다 — "
                                  "critical_location이 실제 최악이 아닐 수 있습니다")
        data[f"tau_{nm}z"] = block

    # 자유단 박리 경향 (정성 순위)
    mism = []
    for k in range(len(si.plies) - 1):
        q1, q2 = float(qbars[k][0, 0]), float(qbars[k + 1][0, 0])
        gap = abs(si.plies[k + 1].angle_deg - si.plies[k].angle_deg)
        mism.append({"interface": f"{k}/{k+1}",
                     "angle_jump_deg": min(gap, 180.0 - gap) if gap else 0.0,
                     "stiffness_mismatch": abs(q1 - q2) / max(q1, q2)})
    if mism:
        mism.sort(key=lambda m: -(m["stiffness_mismatch"] + m["angle_jump_deg"] / 90.0))
        data["free_edge_risk_ranking"] = mism[:5]
        data["free_edge_note"] = ("정성 순위 — 자유단 층간응력은 각도 점프·강성 불일치가 클수록 집중된다. "
                                  "**정량 판정은 assess_free_edge_delamination 을 쓸 것** "
                                  "(O'Brien ERR 폐형해 + 계면별 지배 구동력)")
        warnings.append(item("W130", field="free_edge_risk_ranking",
                             detail="자유단 박리는 이 도구(CLT 평형법)로는 정량화되지 않습니다 — 순위는 "
                                    "정성 지표다. 정량 판정은 assess_free_edge_delamination "
                                    "(O'Brien ERR + 계면별 구동력)을 호출할 것"))

    hash_payload = {"laminate": payload, "shear": shear}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=[
                         "층간 전단: 3D 평형식 사후 복원 — N=0·dM/dx=V의 2×2 연성계(det=AD−B²)를 풀어 "
                         "τ(z) = −∫Q̄(ζ)(ε0'+κ'ζ)dζ (막-굽힘 연성 포함, 비대칭에서도 유효)",
                         "상하 자유표면 τ=0은 자동 만족(엔진 불변식). 자유단 경계층·σz는 미포함"],
                     include_debug=include_debug, t0=t0)


def run_fatigue(payload, loads_max, loads_min=None, detail: str = "auto", delta_t=None,
                include_debug: bool = False) -> dict:
    """하중 사이클의 ply별 피로 수명 (estimate_fatigue_life Tool, §17.7).

    재료축 성분(σ1/σ2/τ12)별로 **부호를 보존한** 진폭·평균에 Goodman + S-N을 적용한다
    (적대 검증 FAT-01: 부호 없는 FI 기반은 완전반복을 '무한수명'으로 뒤집었다).
    """
    from app.solver import failure as FAIL
    from app.solver import fatigue as FAT
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    if detail not in ("auto", "full"):
        return ENV.build(data=None, errors=[item("E100", field="detail",
                                                 detail="detail은 auto|full 중 하나여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    def _parse_state(loads, label, allow_zero):
        """하중 상태 → SI 벡터. 전부 0도 허용(사이클의 한쪽 끝), 오류 field는 인자명으로 (FAT-05/06/14)."""
        if loads is None:
            return np.zeros(3), np.zeros(3), None
        if not isinstance(loads, dict):
            return None, None, item("E100", field=label,
                                    detail=f'{label}는 {{"N": [Nx,Ny,Nxy], "M": [Mx,My,Mxy]}} 객체여야 합니다')
        out = []
        f_ = units.TO_SI[si.unit_system]
        for key, factor in (("N", f_["load_n"]), ("M", f_["load_m"])):
            v = loads.get(key, [0.0, 0.0, 0.0])
            if not (isinstance(v, list) and len(v) == 3
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            and math.isfinite(x) for x in v)):
                return None, None, item("E100", field=f"{label}.{key}",
                                        detail=f"{label}.{key}는 유한한 숫자 3개 리스트여야 합니다")
            out.append(np.asarray(v, dtype=np.float64) * factor)
        if not allow_zero and float(np.linalg.norm(out[0])) == 0.0 and float(np.linalg.norm(out[1])) == 0.0:
            return None, None, item("E100", field=label,
                                    detail=f"{label}의 N·M이 모두 0입니다 — 사이클의 두 끝이 같으면 "
                                           f"피로 손상이 없습니다")
        return out[0], out[1], None

    N_max, M_max, err = _parse_state(loads_max, "loads_max", allow_zero=True)
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    N_min, M_min, err = _parse_state(loads_min, "loads_min", allow_zero=True)
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    if (float(np.linalg.norm(N_max - N_min)) == 0.0
            and float(np.linalg.norm(M_max - M_min)) == 0.0):
        return ENV.build(data=None, errors=[item("E100", field="loads_max/loads_min",
                                                 detail="두 하중 상태가 동일합니다 — 사이클 진폭이 0이면 "
                                                        "피로 손상이 없습니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    usable = [k for k, p_ in enumerate(si.plies) if p_.strength is not None and p_.fatigue is not None]
    if not usable:
        return ENV.build(data=None, errors=[item("E100", field="laminae",
                                                 detail="피로 평가에는 ply에 strength{Xt..S}와 "
                                                        "fatigue{model_type,k|b}가 모두 필요합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    skipped = [k for k in range(len(si.plies)) if k not in usable]
    if skipped:
        # 임계 ply가 빠지면 수명이 과대평가된다 — 침묵 금지 (FAT-03/06)
        warnings.append(item("W120", field="laminae",
                             detail=f"laminae{skipped}는 strength/fatigue 미입력으로 평가에서 제외됩니다 — "
                                    f"제외된 ply가 실제 임계면 life_cycles가 과대평가(비보수)입니다"))

    A, B, D, z, h = _abd_of_plies(si.plies)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]
    try:
        e_max, k_max = RESP.solve_response(A, B, D, N_max, M_max)
        e_min, k_min = RESP.solve_response(A, B, D, N_min, M_min)
    except RESP.SingularSystemError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    # §19.10 열잔류 — 잔류는 진폭을 바꾸지 않고 **평균응력만 이동**시킨다. Goodman 이
    # 인장 평균에 벌점을 주므로 냉각 잔류가 인장인 ply 는 수명이 크게 줄어든다.
    dT = 0.0
    res_state = None
    if delta_t is not None:
        from app.solver import thermal as TH
        if not (isinstance(delta_t, (int, float)) and not isinstance(delta_t, bool)
                and math.isfinite(delta_t)):
            return ENV.build(data=None, errors=[item("E100", field="delta_T",
                                                     detail="delta_T는 유한한 숫자여야 합니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
        dT = float(delta_t)
        if dT != 0.0:
            missing = [k for k, p_ in enumerate(si.plies) if not p_.has_cte]
            if missing:
                return ENV.build(data=None, errors=[item("E203", field="laminae",
                                                         detail=f"laminae{missing} 에 CTE(alpha)가 없습니다 (delta_T 해석)")],
                                 warnings=warnings, payload=payload, unit_system=si.unit_system,
                                 include_debug=include_debug, t0=t0)
            eps_free = [TH.alpha_vector(p_.alpha1, p_.alpha2, p_.angle_deg) * dT
                        for p_ in si.plies]
            n_th, m_th = TH.free_strain_loads(qbars, eps_free, z)
            try:
                e_r, k_r = RESP.solve_response(A, B, D, n_th, m_th)
            except RESP.SingularSystemError:
                return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                                 payload=payload, unit_system=si.unit_system,
                                 include_debug=include_debug, t0=t0)
            res_state = (e_r, k_r, eps_free)

    fmod = units.FROM_SI[si.unit_system]["modulus"]
    fz = units.FROM_SI[si.unit_system]["z"]
    rows = []
    for k in usable:
        p_ = si.plies[k]
        model, param = p_.fatigue
        worst = None
        for zv in (float(z[k]), float(z[k] + z[k + 1]) / 2.0, float(z[k + 1])):
            s_hi = FAIL.stress_to_material_axes(
                FAIL.ply_stresses_at(qbars[k], e_max, k_max, zv), p_.angle_deg)
            s_lo = FAIL.stress_to_material_axes(
                FAIL.ply_stresses_at(qbars[k], e_min, k_min, zv), p_.angle_deg)
            if res_state is not None:
                e_r, k_r, eps_free = res_state
                sr = FAIL.stress_to_material_axes(
                    qbars[k] @ (e_r + zv * k_r - eps_free[k]), p_.angle_deg)
                s_hi, s_lo = s_hi + sr, s_lo + sr    # 두 끝에 같은 값 → 평균만 이동
            # 성분별로 큰 쪽을 max로 정렬 — loads_max/min 라벨 순서에 결과가 의존하지 않게 (FAT-02/04)
            hi = np.maximum(s_hi, s_lo)
            lo = np.minimum(s_hi, s_lo)
            res = FAT.assess_point(hi, lo, p_.strength, model, param)
            n = res["cycles_to_failure"]
            if n is None:
                continue
            if worst is None or n < worst[0]["cycles_to_failure"]:
                worst = (res, zv)
        if worst is None:
            rows.append({"ply": k, "angle_deg": p_.angle_deg, "infinite_life": True,
                         "model": model, "param": param,
                         "note": "전 성분 진폭 0 — 이 ply는 피로 손상 없음"})
        else:
            res, zv = worst
            gov = next(c for c in res["components"] if c["component"] == res["governing"])
            rows.append({
                "ply": k, "angle_deg": p_.angle_deg, "z": zv * fz,
                "governing_component": res["governing"],
                "sigma_amplitude": gov["sigma_amplitude"] * fmod,
                "sigma_mean": gov["sigma_mean"] * fmod,
                "r_alternating": gov["r_alternating"] if math.isfinite(gov["r_alternating"]) else None,
                "cycles_to_failure": res["cycles_to_failure"],
                "at_cap": res["at_cap"], "model": model, "param": param,
                **({"note": gov["note"]} if "note" in gov else {}),
            })

    finite = [r for r in rows if r.get("cycles_to_failure") is not None]
    # 동률이면 가장 가혹한(r_alternating 큰) ply를 임계로 (FAT-08: first-wins가 실제 최악을 가림)
    critical = min(finite, key=lambda r: (r["cycles_to_failure"],
                                          -(r.get("r_alternating") or 0.0))) if finite else None
    ties = [r["ply"] for r in finite
            if critical and r["cycles_to_failure"] == critical["cycles_to_failure"]] if finite else []

    shown = rows
    trunc = None
    if detail == "auto" and len(rows) > config.SUMMARY_PLY_LIMIT:
        ranked = sorted(finite, key=lambda r: r["cycles_to_failure"])[:config.SUMMARY_TOP_N]
        keep = {r["ply"] for r in ranked} | ({critical["ply"]} if critical else set())
        shown = [r for r in rows if r["ply"] in keep]
        trunc = (f"ply {len(rows)}개 중 수명 짧은 순 {len(shown)}개만 반환 — "
                 f'전체는 detail="full"')

    data = {
        "cycle": {"loads_max": loads_max, "loads_min": loads_min,
                  "note": "성분별로 큰 값을 max로 정렬하므로 두 인자의 순서는 결과에 영향을 주지 않습니다"},
        "plies": shown,
        **({"truncation": trunc} if trunc else {}),
        "critical_ply": critical,
        **({"tied_plies": ties} if len(ties) > 1 else {}),
        "life_cycles": critical["cycles_to_failure"] if critical else None,
        "meaning": ("life_cycles = 임계 ply의 반복 수명 추정 (성분별 부호 보존 진폭·평균 + Goodman + S-N). "
                    f"{FAT.N_CAP:.0e} 도달 시 at_cap=true. life_cycles=null이면 전 ply 진폭 0(무손상)"),
    }
    data["delta_T"] = dT if dT != 0.0 else None
    if skipped:
        data["excluded_plies"] = skipped
    if dT == 0.0 and any(p_.has_cte for p_ in si.plies):
        warnings.append(item("W130", field="delta_T",
                             detail="ply 에 CTE 가 있는데 delta_T 를 주지 않았다 — 경화 냉각 잔류가 "
                                    "평균응력을 이동시키는데 빠져 있다. Goodman 이 인장 평균에 벌점을 "
                                    "주므로 수명이 과대평가(비보수)일 수 있다"))
    hash_payload = {"laminate": payload, "loads_max": loads_max, "loads_min": loads_min,
                    "detail": detail, "delta_T": delta_t}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warnings,
                         payload=hash_payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warnings, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=[
                         "피로: 재료축 성분별(σ1/σ2/τ12) 부호 보존 진폭·평균 + 표준 Goodman(인장 평균만 감산) + S-N",
                         "등진폭·비례하중, 성분 독립(다축 상호작용 미고려) 1차 근사",
                         "S-N은 시험 데이터 범위(보통 1e4~1e7 사이클) 밖에서는 외삽 — 큰 N은 자릿수만 참고할 것",
                         "층간·박리 피로, 잔류강도 저하, 하중 순서·환경 효과는 미포함"],
                     include_debug=include_debug, t0=t0)


def _thermal_free_loads(si, dT: float, dC: float):
    """자유 열·흡습 변형 → (qbars, z, h, N_f, M_f). 물성 누락 시 (None, error)."""
    from app.solver import thermal as TH
    for need, prop, label in ((dT != 0.0, "has_cte", "CTE(alpha)"), (dC != 0.0, "has_cme", "CME(beta)")):
        if not need:
            continue
        missing = [k for k, p_ in enumerate(si.plies) if not getattr(p_, prop)]
        if missing:
            return None, item("E203", field="laminae",
                              detail=f"laminae{missing} 에 {label}가 없습니다")
    eps_free = []
    for p_ in si.plies:
        e = np.zeros(3)
        if dT != 0.0:
            e = e + TH.alpha_vector(p_.alpha1, p_.alpha2, p_.angle_deg) * dT
        if dC != 0.0:
            e = e + TH.alpha_vector(p_.beta1, p_.beta2, p_.angle_deg) * dC
        eps_free.append(e)
    A, B, D, z, h = _abd_of_plies(si.plies)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]
    N_f, M_f = TH.free_strain_loads(qbars, eps_free, z)
    return (A, B, D, h, N_f, M_f), None


def _offaxis_ratio(M: np.ndarray) -> float:
    """행렬의 16·26 성분이 전체 대비 차지하는 비 (비틀림 표현 한계 판정용)."""
    peak = float(np.max(np.abs(M)))
    if peak <= 0:
        return 0.0
    return max(abs(float(M[0, 2])), abs(float(M[1, 2]))) / peak


def run_bistable_shapes(payload, delta_t=None, panel=None, delta_c=None,
                        include_debug: bool = False) -> dict:
    """비대칭 적층의 경화 후 쌍안정 형상 — Hyer 모델 (compute_bistable_shapes, §18.2).

    선형 CLT는 판 크기와 무관하게 하나의 안장 형상만 준다. 실제로는 판이 임계 크기를
    넘으면 안장이 불안정해지고 서로 거울상인 **원통 형상 두 개**로 분기한다. 이 도구는
    그 분기와 두 안정 형상, 임계 판 크기를 준다.
    """
    from app.solver import nonlinear as NL
    from app.solver import thermal as TH
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    def _num(v):
        return (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))

    dT = float(delta_t) if _num(delta_t) else (None if delta_t is None else "bad")
    dC = float(delta_c) if _num(delta_c) else (None if delta_c is None else "bad")
    if dT == "bad" or dC == "bad" or ((dT in (None, 0.0)) and (dC in (None, 0.0))):
        return ENV.build(data=None, errors=[item("E100", field="delta_T/delta_C",
                                                 detail="delta_T [K] 또는 delta_C [%M] 중 최소 하나는 0이 아닌 숫자여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    dT, dC = dT or 0.0, dC or 0.0
    lx, ly = _panel_to_si(panel, si.unit_system) if panel is not None else (None, None)
    if lx is None:
        return ENV.build(data=None, errors=[item("E100", field="panel",
                                                 detail="panel {\"Lx\": >0, \"Ly\": >0} 은 필수입니다 — 쌍안정 분기는 판 크기에 의존합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    loads, err = _thermal_free_loads(si, dT, dC)
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    A, B, D, h, N_f, M_f = loads

    try:
        kap_lin = NL.linear_curvature_vector(A, B, D, N_f, M_f)
        kx_lin, ky_lin, kxy_lin = (float(kap_lin[0]), float(kap_lin[1]), float(kap_lin[2]))
        en = NL.HyerEnergy(A, B, D, N_f, M_f, lx, ly)
        span = NL.search_span(en, kx_lin, ky_lin)
        sols = NL.find_equilibria(en, span)
        crit = NL.critical_scale(A, B, D, N_f, M_f, lx, ly)
    except np.linalg.LinAlgError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    except (OverflowError, FloatingPointError) as exc:
        # 유한·양수 입력이라도 극단 판 크기·물성 조합은 수치 범위를 넘는다 — 내부 오류가 아니다
        return ENV.build(data=None, errors=[item("E100", field="panel",
                                                 detail=f"판 크기·물성 조합이 수치 범위를 넘습니다 "
                                                        f"({type(exc).__name__}). panel 단위를 확인하세요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    equilibria = []
    for s in sols:
        kap = np.array([s["a"], s["b"], 0.0])
        equilibria.append({
            "kappa_x": s["a"] * f["kappa"], "kappa_y": s["b"] * f["kappa"],
            "shape": NL.classify_shape(s["a"], s["b"], scale=span),
            "stable": s["stable"],
            "stability": s["stability"],
            "energy_per_area": s["energy"] / (lx * ly) * f["energy_area"],
            "warpage_range": TH.warpage_over_panel(kap, lx, ly)["warpage_range"] * f["z"],
        })
    stable = [e for e in equilibria if e["stability"] == "stable"]
    unstable = [e for e in equilibria if e["stability"] == "unstable"]
    marginal = [e for e in equilibria if e["stability"] == "marginal"]

    data: dict = {
        "linear_reference": {
            "kappa_x": kx_lin * f["kappa"], "kappa_y": ky_lin * f["kappa"],
            "kappa_xy": kxy_lin * f["kappa"],
            "shape": NL.classify_shape(kx_lin, ky_lin),
            "note": "선형 CLT 해 3성분 — 판 크기 무관. 임계 크기를 넘은 판에서는 실제로 실현되지 "
                    "않는다. kappa_xy 는 이 모델이 표현하지 못하는 성분이라 대조용으로 함께 싣는다",
        },
        "equilibria": equilibria,
        "stable_count": len(stable),
        "marginal_count": len(marginal),
        "bistable": len(stable) >= 2,
        "critical_panel": (
            {"Lx": crit["lx"] / units.TO_SI[si.unit_system]["length"],
             "Ly": crit["ly"] / units.TO_SI[si.unit_system]["length"],
             "scale_vs_input": crit["scale"],
             "definition": "종횡비를 유지한 채 판을 키울 때 안장 해가 안정성을 잃는 크기 (분기점)"}
            if crit["scale"] is not None else None),
    }
    if stable and unstable:
        # 스냅스루는 '어느 안정형상에서 빠져나오는가'에 따라 장벽이 다르다. 가장 깊은 우물만
        # 보고하면 얕은 우물의 실제 장벽을 크게 과대평가한다(적대 검증 HYER-2: 최대 19배).
        e_saddle = min(e["energy_per_area"] for e in unstable)
        per_stable = [{"kappa_x": e["kappa_x"], "kappa_y": e["kappa_y"],
                       "barrier": e_saddle - e["energy_per_area"]} for e in stable]
        data["energy_barrier"] = {
            "min_barrier": min(b["barrier"] for b in per_stable),
            "per_stable_shape": per_stable,
            "definition": "각 안정형상에서 불안정(안장) 해까지의 단위면적 에너지 차. "
                          "min_barrier 가 스냅스루를 지배하는 장벽이다 — 얕은 우물이 먼저 넘어간다",
            "note": "스냅 하중 자체는 하중 인가 평형 추적이 필요해 이 도구가 계산하지 않는다",
        }

    warn = list(warnings)
    # 이 모델은 κxy = 0 을 가정한다. 선형 CLT가 유의한 κxy 를 낸다면 형상 판정 자체가 틀린다.
    # A/B/D 의 16·26 성분비는 대리지표일 뿐이라 실제 κxy 로 판정한다 (적대 검증 HYER-1).
    tw = abs(kxy_lin) / max(abs(kx_lin), abs(ky_lin), 1e-300)
    if tw > 0.3:
        warn.append(item("W130", field="laminae",
                         detail=f"선형 CLT의 |κxy|/max(|κx|,|κy|) = {tw:.3g} — 비틀림이 지배적인데 "
                                f"이 모델은 κxy = 0 을 가정한다. **형상 판정을 신뢰하지 말 것.** "
                                f"compute_thermal_response 의 κ 3성분을 보라"))
    elif tw > 0.05:
        warn.append(item("W130", field="laminae",
                         detail=f"선형 CLT의 |κxy|/max(|κx|,|κy|) = {tw:.3g} — 이 모델이 표현하지 "
                                f"못하는 비틀림 성분이 있어 곡률·휨량이 과소평가된다"))
    if data["bistable"]:
        warn.append(item("W130", field="panel",
                         detail=f"판이 임계 크기를 넘어 안정 형상이 {len(stable)}개다 — "
                                f"선형 해석(compute_thermal_response)의 안장 곡률은 실현되지 않는다"))
    if data["bistable"] and crit["scale"] is None:
        warn.append(item("W130", field="critical_panel",
                         detail="이 판은 이미 쌍안정인데 임계 크기 추적에 실패했다(분기가 추적 "
                                "가지 밖에서 일어남). critical_panel 은 null 이지만 분기는 이미 "
                                "지난 상태다 — 모순으로 읽지 말 것"))
    if not stable:
        warn.append(item("W130", field="equilibria",
                         detail=f"이 판 크기에서 안정 평형해를 찾지 못했다(정지점 {len(sols)}개). "
                                f"결과를 신뢰하지 말고 판 크기를 바꿔 경향을 확인할 것"))
    if marginal:
        warn.append(item("W130", field="equilibria",
                         detail=f"판이 분기점 바로 근처라 정지점 {len(marginal)}개의 안정성 판정이 "
                                f"한계적이다(Hessian 최소 고윳값 ≈ 0). stable_count 를 단정하지 말 것"))
    for mat, nm in ((A, "A"), (B, "B"), (D, "D")):
        if _offaxis_ratio(mat) > 0.1:
            warn.append(item("W130", field="laminae",
                             detail=f"{nm}16/{nm}26 성분이 큽니다 — 이 모델은 γxy⁰=0·κxy=0 가정이라 "
                                    f"비틀림 형상([±θ] 반대칭의 실제 경화 형상)을 표현하지 못합니다"))
            break

    extra = [
        "Hyer 모델: w = −½(κx x² + κy y²), 면내 변형은 von Karman 적합조건 "
        "ε_x,yy + ε_y,xx = −κxκy 를 만족하는 최소 다항족 (Rayleigh–Ritz 3자유도)",
        "원통(κxκy=0)은 전개 가능면이라 막 벌점이 없고, 안장은 L⁴ 벌점을 받는다 — 이것이 분기의 원인",
        "면내 전단 γxy⁰ = k·x·y 자유도를 포함한다(4자유도) — 이걸 빼면 안장 가지의 막 벌점이 "
        "과대평가돼 없는 쌍안정을 만들어낸다",
        "한계: κxy = 0 이라 비틀림 형상은 표현하지 못한다. 자유 경계, 저차 근사 — 곡률 절대값은 "
        "FE 대비 오차가 있다",
        "정지점 탐색은 고정 격자 스캔 + 고정 반복 뉴턴 (결정론적, 난수 없음)",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "delta_T": delta_t, "delta_C": delta_c, "panel": panel}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


def run_large_deflection(payload, panel, pressure, edge_condition: str = "movable",
                         include_debug: bool = False) -> dict:
    """균일압력 하 SS 판의 기하 비선형 대처짐 (compute_large_deflection, §18.3)."""
    from app.solver import nonlinear as NL
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    lx, ly = _panel_to_si(panel, si.unit_system) if panel is not None else (None, None)
    err = None
    if lx is None:
        err = item("E100", field="panel", detail="panel {\"Lx\": >0, \"Ly\": >0} 은 필수입니다")
    elif not (isinstance(pressure, (int, float)) and not isinstance(pressure, bool)
              and math.isfinite(pressure)):
        err = item("E100", field="pressure", detail="pressure는 유한한 숫자여야 합니다 (압력 단위 = 탄성계수 단위)")
    elif edge_condition not in ("movable", "immovable"):
        err = item("E100", field="edge_condition",
                   detail="edge_condition은 'movable'(면내 이동 자유) 또는 'immovable'(면내 구속) 이어야 합니다")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    A, _B, _D, _z, h = _abd_of_plies(si.plies)
    D_use, appl, _h, _K = _bending_stiffness_for_navier(si, warnings)
    q_si = float(pressure) * units.TO_SI[si.unit_system]["modulus"]
    try:
        res = NL.large_deflection(D_use, A, lx, ly, q_si, edge_condition == "immovable")
    except (ValueError, OverflowError, FloatingPointError) as e:
        return ENV.build(data=None, errors=[item("E100", field="laminate", detail=str(e))],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    except np.linalg.LinAlgError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    ratio = abs(res["w_center"]) / h
    data = {
        "w_center": res["w_center"] * f["z"],
        "w_center_linear": res["w_linear"] * f["z"],
        "w_over_thickness": ratio,
        "stiffening_ratio": res["stiffening_ratio"],
        "membrane_dominant": ratio > 1.0,
        "edge_condition": edge_condition,
        "panel": {"Lx": panel["Lx"], "Ly": panel["Ly"]},
        "applicability": appl,
        "definition": ("αW + βW³ = 16q/π², w = W·sin(πx/Lx)·sin(πy/Ly) (SS, 1항 Galerkin). "
                       "stiffening_ratio = w_linear/w_center — 1보다 크면 막 효과로 뻣뻣해진 정도"),
    }
    warn = list(warnings)
    if ratio < 0.3:
        warn.append(item("W130", field="pressure",
                         detail=f"w/h = {ratio:.3g} < 0.3 — 기하 비선형이 거의 무의미하다. "
                                f"solve_load_response 의 선형 해로 충분하다"))
    if ratio > 3.0:
        warn.append(item("W130", field="pressure",
                         detail=f"w/h = {ratio:.3g} — 1항 Galerkin의 유효 범위를 크게 벗어났다. "
                                f"경향만 보고 정밀 판정은 FE로 할 것"))
    if edge_condition == "movable":
        warn.append(item("W130", field="edge_condition",
                         detail="면내 이동 자유 가정 — 실제 가장자리가 구속되면 처짐이 더 작다 "
                                "(등방 정사각에서 β가 약 3.9배). 두 극단을 모두 확인할 것"))
    aspect = max(lx, ly) / min(lx, ly)
    if aspect > 2.0:
        warn.append(item("W130", field="panel",
                         detail=f"종횡비 {aspect:.3g} — 1항 Galerkin은 세장 판에서 처짐을 "
                                f"**과소평가**한다(종횡비 10에서 약 1.6배). 여유가 빠듯하면 FE로 확인할 것"))
    off = _offaxis_ratio(A)
    if off > 0.1:
        warn.append(item("W130", field="laminae",
                         detail=f"A16/A26 성분비 {off:.3g} — 1항 해는 면내 직교이방(A16=A26=0)을 "
                                f"전제한다. 불균형 적층에서는 β 근사 오차가 커진다"))
    extra = [
        "SS(단순지지) 4변, 1항 Galerkin 근사 — 처짐 절대값은 FE 대비 오차가 있다 "
        "(선형 극한에서 등방 정사각 기준 +2.4%)",
        "면내 경계조건이 지배적 가정이다: movable(평균 막력 0) vs immovable(평균 면내변형 0)",
        "비대칭 적층은 축소 굽힘강성 D* = D − B·A⁻¹·B 로 근사 (막-굽힘 커플링 완전 반영 아님)",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "panel": panel, "pressure": pressure,
                    "edge_condition": edge_condition}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


def run_postbuckling(payload, panel, applied_Nx, load_ratio: float = 0.0,
                     include_debug: bool = False) -> dict:
    """좌굴 후 거동 — 진폭·면내 강성비·유효폭 (compute_postbuckling, §18.4)."""
    from app.solver import nonlinear as NL
    from app.solver import plate_navier as NAV
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    lx, ly = _panel_to_si(panel, si.unit_system) if panel is not None else (None, None)
    err = None
    if lx is None:
        err = item("E100", field="panel", detail="panel {\"Lx\": >0, \"Ly\": >0} 은 필수입니다")
    elif not (isinstance(applied_Nx, (int, float)) and not isinstance(applied_Nx, bool)
              and math.isfinite(applied_Nx) and applied_Nx > 0):
        err = item("E100", field="applied_Nx", detail="applied_Nx는 압축 크기(양수)여야 합니다")
    elif (not isinstance(load_ratio, (int, float)) or isinstance(load_ratio, bool)
          or not math.isfinite(load_ratio)):
        err = item("E100", field="load_ratio", detail="load_ratio = Ny/Nx 는 유한한 숫자여야 합니다 (압축 양수)")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    A, _B, _D, _z, h = _abd_of_plies(si.plies)
    D_use, appl, _h, _K = _bending_stiffness_for_navier(si, warnings)
    res = NAV.buckling_ncr(D_use, lx, ly, float(load_ratio))
    if res["N_cr"] is None:
        return ENV.build(data=None, errors=[item("E100", field="load_ratio",
                                                 detail=f"load_ratio={load_ratio}에서는 {res['reason']}")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    n_si = float(applied_Nx) * units.TO_SI[si.unit_system]["load_n"]
    try:
        pb = NL.postbuckling(D_use, A, lx, ly, res["mode_m"], res["mode_n"],
                             res["N_cr"], n_si, h, load_ratio=float(load_ratio))
    except np.linalg.LinAlgError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    except (OverflowError, FloatingPointError) as exc:
        return ENV.build(data=None, errors=[item("E100", field="panel",
                                                 detail=f"판 크기·하중 조합이 수치 범위를 넘습니다 "
                                                        f"({type(exc).__name__})")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    data = {
        "N_cr": res["N_cr"] * f["A"],
        "applied_Nx": float(applied_Nx),
        "load_ratio_Ny_over_Nx": float(load_ratio),
        "mode": {"m": res["mode_m"], "n": res["mode_n"]},
        "buckled": pb["buckled"],
        "load_over_critical": pb["load_ratio"],
        "stiffness_ratio": pb["stiffness_ratio"],
        "definition": pb["definition"],
        "applicability": appl,
    }
    if pb["buckled"]:
        data.update({
            "amplitude": pb["amplitude"] * f["z"],
            "amplitude_over_thickness": pb["amplitude_over_thickness"],
            "effective_width_ratio": pb["effective_width_ratio"],
        })
    else:
        data["note"] = pb["note"]

    warn = list(warnings)
    if res.get("boundary"):
        warn.append(item("W130", field="mode",
                         detail=f"임계 모드가 스캔 상한({res['mode_scan']})에 걸렸습니다 — N_cr 과대평가 가능"))
    warn.append(item("W130", field="edges",
                     detail="비재하 가장자리 면내 이동 자유 가정 — 구속되면 진폭은 작아지고 "
                            "강성비는 커진다(비보수 방향 아님)"))
    off_pb = _offaxis_ratio(A)
    if off_pb > 0.1:
        warn.append(item("W130", field="laminae",
                         detail=f"A16/A26 성분비 {off_pb:.3g} — 1항 해는 면내 직교이방을 전제한다. "
                                f"불균형 적층에서는 강성비·진폭 근사 오차가 커진다"))
    if pb["buckled"] and pb["amplitude_over_thickness"] > 3.0:
        warn.append(item("W130", field="amplitude",
                         detail=f"W/h = {pb['amplitude_over_thickness']:.3g} — 1항 근사의 유효 범위를 "
                                f"크게 벗어났다. 경향만 보고 정밀 판정은 FE로 할 것"))
    extra = [
        "SS 4변, 1항 Galerkin — 좌굴 모드 (m,n)은 compute_buckling과 동일한 스캔으로 결정",
        "강성비는 끝단 수축 e = a11·N + W²p²/8 의 미분에서 유도 (등방 정사각에서 정확히 0.5)",
        "b_eff/b = √(N_cr/N) 은 von Karman 유효폭 — 후좌굴 하중 재분배의 반경험 지표",
        "강성비는 2축 하중비 R = Ny/Nx 를 반영한다 — R>0(2축 압축)이면 남는 강성이 크게 줄어든다",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "panel": panel, "applied_Nx": applied_Nx,
                    "load_ratio": load_ratio}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


GAMMA12_VALID_LIMIT = 0.05      # 이 이상은 실제로 이미 전단 파손 영역 (§18.7)


def run_nonlinear_shear(payload, loads=None, include_debug: bool = False) -> dict:
    """재료 면내 전단 비선형 응답 — Hahn–Tsai 할선반복 (solve_nonlinear_shear_response, §18.7)."""
    from app.solver import shear_nonlinear as SNL
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    N_si, M_si, load_err = _loads_to_si(loads, si.unit_system)
    if load_err is not None:
        return ENV.build(data=None, errors=[load_err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    if not (np.all(np.isfinite(N_si)) and np.all(np.isfinite(M_si))):
        return ENV.build(data=None, errors=[item("E100", field="loads",
                                                 detail="loads의 모든 성분은 유한한 숫자여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    nl_idx = [k for k, p in enumerate(si.plies) if p.s6666 is not None]
    if not nl_idx:
        return ENV.build(data=None, errors=[item("E100", field="laminae",
                                                 detail="어느 ply에도 material.shear_nonlinear{S6666}가 없습니다 — "
                                                        "선형과 동일하므로 solve_load_response를 쓰세요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    linear_only = [k for k in range(len(si.plies)) if k not in nl_idx]
    if linear_only:
        warnings.append(item("W120", field="laminae",
                             detail=f"laminae{linear_only} 에 shear_nonlinear가 없어 선형 G12로 다룹니다"))

    plies = [{"E1": p.E1, "E2": p.E2, "G12": p.G12, "nu12": p.nu12,
              "angle_deg": p.angle_deg, "s6666": p.s6666} for p in si.plies]
    z = ABD.z_coordinates(si.thicknesses)
    h = si.total_thickness
    try:
        res = SNL.solve_secant(plies, z, N_si, M_si)
        lin = SNL.linear_reference(plies, z, N_si, M_si)
        a_s, _, d_s = RESP.compliance_blocks(res["A"], res["B"], res["D"])
        a_l, _, d_l = RESP.compliance_blocks(lin["A"], lin["B"], lin["D"])
        eff_s = RESP.effective_constants(a_s, d_s, h)
        eff_l = RESP.effective_constants(a_l, d_l, h)
    except RESP.SingularSystemError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    fm = f["modulus"]

    def _eff(e):
        return {"membrane": {k: (v * fm if k != "nu_xy" else v) for k, v in e["membrane"].items()},
                "bending": {k: v * fm for k, v in e["bending"].items()}}

    per_ply = [{"ply": r["ply"], "tau12": r["tau12"] * fm, "gamma12": r["gamma12"],
                "G12_secant": r["G12_secant"] * fm, "secant_ratio": r["secant_ratio"],
                "nonlinear": r["nonlinear"]} for r in res["per_ply"]]
    max_gamma = max(abs(r["gamma12"]) for r in res["per_ply"])
    worst = min(res["per_ply"], key=lambda r: r["secant_ratio"])

    trunc = None
    if len(per_ply) > config.SUMMARY_PLY_LIMIT:
        ranked = sorted(per_ply, key=lambda r: r["secant_ratio"])
        per_ply = sorted(ranked[:config.SUMMARY_TOP_N], key=lambda r: r["ply"])
        trunc = (f"ply {len(res['per_ply'])}개 중 연화가 큰 {len(per_ply)}개만 반환 (§6.6 토큰 예산). "
                 f"softening은 전체 기준")

    def _ratio(nl, l0):
        return float(l0 / nl) if abs(nl) > 0 else None

    data = {
        "response": {"epsilon0": res["eps0"].tolist(),
                     "kappa": (res["kappa"] * f["kappa"]).tolist()},
        "linear_response": {"epsilon0": lin["eps0"].tolist(),
                            "kappa": (lin["kappa"] * f["kappa"]).tolist(),
                            "note": "S6666을 무시한 선형 CLT 해 — 대조용"},
        "softening": {
            "Gxy_secant_over_linear": float(eff_s["membrane"]["Gxy"] / eff_l["membrane"]["Gxy"]),
            "Ex_secant_over_linear": float(eff_s["membrane"]["Ex"] / eff_l["membrane"]["Ex"]),
            "max_gamma12": float(max_gamma),
            "worst_ply": {"ply": worst["ply"], "secant_ratio": float(worst["secant_ratio"])},
            "definition": "할선/선형 유효상수 비. 1보다 작을수록 전단 비선형으로 무르다",
        },
        "per_ply": per_ply,
        **({"truncation": trunc} if trunc else {}),
        "effective_constants": _eff(eff_s),
        "linear_effective_constants": _eff(eff_l),
        "convergence": {
            "iterations": res["steps"],
            "constitutive_residual": res["residual"],
            "converged": res["converged"],
            "definition": "잔차 = |γ12 − (τ12/G12 + S6666·τ12³)| / |γ12| 의 ply 최대 (수렴의 진짜 척도)",
        },
        "definition": "γ12 = τ12/G12 + S6666·τ12³ (Hahn–Tsai). ply별 할선 G12를 고정 40회 반복으로 갱신",
    }

    warn = list(warnings)
    if not res["converged"]:
        warn.append(item("W130", field="convergence",
                         detail=f"고정 {res['steps']}회 반복 후 구성식 잔차 {res['residual']:.3g} — "
                                f"수렴하지 않았다. 하중을 낮춰 단조성을 확인할 것"))
    if max_gamma > GAMMA12_VALID_LIMIT:
        warn.append(item("W130", field="softening.max_gamma12",
                         detail=f"γ12 = {max_gamma:.3g} > {GAMMA12_VALID_LIMIT} — Hahn–Tsai 3차식에는 "
                                f"강도 한계가 없어 파손 이후에도 계속 답을 낸다. 실제로는 이미 전단 "
                                f"파손했을 가능성이 크다. recover_ply_stresses로 파손 판정을 병행할 것"))
    if any(p.strength is None for p in si.plies):
        warn.append(item("W120", field="laminae",
                         detail="강도 데이터가 없어 파손 여부를 함께 판정하지 못했다 — "
                                "전단 연화가 큰 결과는 파손 후 영역일 수 있다"))

    extra = [
        "Hahn–Tsai 1파라미터 전단 비선형. ply당 할선 G12 하나를 쓰므로 구성식은 그 ply의 "
        "|τ12| 최대점(하단·중앙·상단 3점)에서 정확히 만족한다 — 굽힘 지배 하중에서는 근사다",
        "|τ12| 최대점을 쓰는 것은 보수적 선택이다(할선 G가 더 작아 더 무르게 나온다)",
        "E1·E2·ν12는 선형으로 둔다 — 섬유 지배 방향의 비선형은 이 모델의 범위 밖",
        f"할선 반복은 저완화 {SNL.RELAXATION} 로 고정 {SNL.SECANT_STEPS} 회 (결정론 — 조기 종료 없음)",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "loads": loads}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


def _stability_gate(si, n_si, panel, warnings, r_strength=None):
    """압축 하중일 때 좌굴을 함께 보게 만드는 지배모드 게이트 (§19.6).

    적대 검증 실측: [0/90]s h=0.5mm 에 Nx=−60 N/mm 를 걸면 recover_ply_stresses 가
    Tsai-Wu R = 7.07 · 경고 0건 으로 "7배 여유"라고 답하는데, 같은 적층·같은 하중의
    150×150 판은 N_cr = 1.04 N/mm 라 **이미 58배 초과**다(모순 배율 410). 준등방 1mm
    판에서도 61배가 재현된다. 두 응답 어느 쪽도 상대를 언급하지 않았다 — 강도 도구가
    표준 진입점이라 압축 질문은 구조적으로 이 경로로 들어온다.

    반환: (governing 블록 또는 None). 경고는 warnings 에 덧붙인다.
    """
    from app.solver import plate_navier as NAV
    nx, ny = float(n_si[0]), float(n_si[1])
    if nx >= 0.0 and ny >= 0.0:
        return None                     # 압축 성분이 없으면 좌굴은 무관하다
    if panel is None:
        warnings.append(item("W130", field="loads.N",
                             detail="면내 **압축**이 걸려 있는데 panel 이 없어 좌굴을 확인할 수 없다. "
                                    "얇은 판은 강도 여유가 충분해도 좌굴이 먼저 온다(실측 최대 410배 "
                                    "모순) — panel={\"Lx\",\"Ly\"} 를 주거나 compute_buckling 을 "
                                    "반드시 함께 호출할 것"))
        return None
    lx, ly = _panel_to_si(panel, si.unit_system)
    if lx is None:
        return None
    if nx >= 0.0:
        warnings.append(item("W130", field="loads.N",
                             detail="y방향만 압축이다 — 이 게이트는 Nx 압축 기준이라 좌굴 순위를 "
                                    "내지 않았다. 축을 바꿔 compute_buckling 을 직접 호출할 것"))
        return None
    d_use, _appl, _h, _k = _bending_stiffness_for_navier(si, warnings)
    ratio = ny / nx if nx != 0.0 else 0.0
    res = NAV.buckling_ncr(d_use, lx, ly, ratio)
    if res["N_cr"] is None:
        return None
    r_buckling = res["N_cr"] / abs(nx)
    modes = {"buckling": r_buckling}
    if r_strength is not None:
        modes["strength"] = r_strength
    name = min(modes, key=lambda k: modes[k])
    gov = {
        "governing_mode": name,
        "margin": modes[name],
        "margins": modes,
        "buckling": {"N_cr": res["N_cr"] * units.FROM_SI[si.unit_system]["A"],
                     "mode": {"m": res["mode_m"], "n": res["mode_n"]},
                     "load_ratio_Ny_over_Nx": ratio,
                     "boundary": "simply_supported"},
        "definition": ("같은 배수 척도로 정렬한다 — strength = Tsai-Wu R, "
                       "buckling = N_cr/|Nx|. 최솟값이 지배 모드다"),
    }
    if r_strength is not None and r_buckling < r_strength:
        warnings.append(item("W130", field="governing_mode",
                             detail=f"**좌굴이 지배한다** — 강도 여유 {r_strength:.3g} 배 vs 좌굴 여유 "
                                    f"{r_buckling:.3g} 배 ({r_strength / max(r_buckling, 1e-300):.0f}배 차이). "
                                    f"강도 판정만 보고하면 안 된다"))
    warnings.append(item("W130", field="governing_mode.buckling",
                         detail="좌굴 여유는 4변 단순지지 기준이다 — 실제 경계가 고정단에 가까우면 "
                                "compute_buckling(boundary='clamped') 로 상계도 함께 볼 것"))
    return gov


def run_free_edge_delamination(payload, loads=None, fracture=None,
                               include_debug: bool = False) -> dict:
    """자유 가장자리 박리 — O'Brien ERR + 계면별 구동력 (assess_free_edge_delamination, §19.1)."""
    from app.solver import free_edge as FE
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    if len(si.plies) < 2:
        return ENV.build(data=None, errors=[item("E100", field="laminae",
                                                 detail="계면이 있어야 박리를 평가할 수 있습니다 (ply 2개 이상)")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    N_si, M_si, load_err = _loads_to_si(loads, si.unit_system)
    if load_err is not None:
        return ENV.build(data=None, errors=[load_err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)
    if not (np.all(np.isfinite(N_si)) and np.all(np.isfinite(M_si))):
        return ENV.build(data=None, errors=[item("E100", field="loads",
                                                 detail="loads의 모든 성분은 유한한 숫자여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    g_c = g_ic = g_iic = None
    bk_eta = FE.BK_ETA_DEFAULT
    if fracture is not None:
        if not isinstance(fracture, dict):
            return ENV.build(data=None, errors=[item("E100", field="fracture",
                                                     detail="fracture는 객체여야 합니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
        extra_keys = set(fracture) - {"G_c", "G_Ic", "G_IIc", "eta"}
        f_e = units.TO_SI[si.unit_system]["energy_area"]

        def _pos(key):
            v = fracture.get(key)
            return (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0)

        err_f = None
        if extra_keys:
            err_f = item("E100", field="fracture",
                         detail=f"fracture 에 지원하지 않는 키 {sorted(extra_keys)} 가 있습니다. "
                                f"{{\"G_c\"}} 또는 {{\"G_Ic\", \"G_IIc\", \"eta\"?}} 만 허용합니다")
        elif _pos("G_Ic") and _pos("G_IIc"):
            g_ic, g_iic = float(fracture["G_Ic"]) * f_e, float(fracture["G_IIc"]) * f_e
            if g_iic < g_ic:
                err_f = item("E100", field="fracture.G_IIc",
                             detail="G_IIc 는 보통 G_Ic 보다 크다 — 값을 바꿔 넣지 않았는지 확인하세요")
            if "eta" in fracture:
                if not _pos("eta"):
                    err_f = item("E100", field="fracture.eta", detail="eta 는 양수여야 합니다")
                else:
                    bk_eta = float(fracture["eta"])
        elif _pos("G_c"):
            g_c = float(fracture["G_c"]) * f_e
        elif fracture:
            err_f = item("E100", field="fracture",
                         detail="fracture 는 {\"G_c\": >0} 또는 {\"G_Ic\": >0, \"G_IIc\": >0, "
                                "\"eta\": >0(선택)} 이어야 합니다 (SI: J/m², SI_mm: N/mm)")
        if err_f is not None:
            return ENV.build(data=None, errors=[err_f], warnings=warnings, payload=payload,
                             unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]
    th = list(si.thicknesses)
    angles_all = [p_.angle_deg for p_ in si.plies]
    z = ABD.z_coordinates(th)
    A, B, D, _z, h = _abd_of_plies(si.plies)
    try:
        eps0, kappa = RESP.solve_response(A, B, D, N_si, M_si)
        e_lam, _ = FE.sublaminate_axial_modulus(qbars, th, 0, len(th))
    except RESP.SingularSystemError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    eps_x = float(eps0[0])
    sig = [qbars[k] @ eps0 for k in range(len(th))]      # 막 응력 (가장자리 구동력 판정용)
    stress_scale = max((float(np.max(np.abs(s_))) for s_ in sig), default=0.0)

    f = units.FROM_SI[si.unit_system]
    interfaces = []
    for k in range(1, len(th)):
        e1, t1 = FE.sublaminate_axial_modulus(qbars, th, 0, k)
        e2, t2 = FE.sublaminate_axial_modulus(qbars, th, k, len(th))
        e_star = (e1 * t1 + e2 * t2) / h
        d_e = e_lam - e_star
        g = FE.obrien_err(eps_x, h, e_lam, e_star)
        drive = FE.interface_driving(sig, th, z, k)
        row = {
            "interface": k,
            "between": [si.plies[k - 1].angle_deg, si.plies[k].angle_deg],
            "delta_E": d_e * f["modulus"],
            "model_valid": g is not None,
            "G": None if g is None else g * f["energy_area"],
            "driving": {
                "peel_moment": drive["peel_moment"] * f["load_m"],
                "transverse_force": drive["transverse_force"] * f["load_n"],
                "shear_force": drive["shear_force"] * f["load_n"],
            },
            "dominant_driver": FE.dominant_driver(drive, h, stress_scale),
        }
        # 혼합모드 (§19.9) — 거울 분할이면 대칭 논증으로 Mode II 가 정확히 0 이다
        mirror = FE.is_mirror_split(angles_all, th, k)
        row["mode_mix"] = {
            "mode_II_fraction": 0.0 if mirror else None,
            "basis": "mirror_symmetry" if mirror else "unknown",
            "note": ("두 부분적층이 거울상이라 상대 미끄러짐이 0 — 순수 Mode I 이다(대칭 논증)"
                     if mirror else
                     "부분적층 강성이 달라 Mode II 성분이 있다. 분할에는 3D/수치 해석이 필요해 "
                     "이 도구는 G_Ic~G_IIc 범위로만 답한다"),
        }
        if g_c is not None or g_ic is not None:
            # 인성이 주어지면 키 모양을 일정하게 유지한다 (모델 무효면 명시적 null)
            row["onset_strain"] = None
            row["margin"] = None
        if g is not None and (g_c is not None or g_ic is not None):
            if g_c is not None:
                ec = FE.onset_strain(g_c, h, d_e)
                row["onset_strain"] = ec
            elif mirror:
                gc_eff = FE.benzeggagh_kenane(g_ic, g_iic, 0.0, bk_eta)   # = G_Ic
                row["toughness_used"] = gc_eff * f["energy_area"]
                row["onset_strain"] = FE.onset_strain(gc_eff, h, d_e)
            else:
                lo = FE.onset_strain(g_ic, h, d_e)      # 순수 Mode I — 보수
                hi = FE.onset_strain(g_iic, h, d_e)     # 순수 Mode II — 낙관
                row["onset_strain"] = lo                # 보수값을 대표로 쓴다
                row["onset_strain_range"] = {"conservative_mode_I": lo, "optimistic_mode_II": hi}
            ec = row.get("onset_strain")
            row["margin"] = (ec / abs(eps_x)) if (ec is not None and eps_x != 0.0) else None
        interfaces.append(row)

    invalid = [r["interface"] for r in interfaces if not r["model_valid"]]
    if g_c is not None or g_ic is not None:
        ranked = [r for r in interfaces if r.get("onset_strain") is not None]
        governing = min(ranked, key=lambda r: (r["onset_strain"], r["interface"])) if ranked else None
    else:
        valid = [r for r in interfaces if r["G"] is not None]
        governing = max(valid, key=lambda r: (r["G"], -r["interface"])) if valid else None

    data: dict = {
        "applied_strain_x": eps_x,
        "laminate_modulus_Ex": e_lam * f["modulus"],
        "interfaces": interfaces,
        "governing_interface": (None if governing is None else {
            "interface": governing["interface"],
            "between": governing["between"],
            "G": governing["G"],
            "dominant_driver": governing["dominant_driver"],
            **({"onset_strain": governing["onset_strain"], "margin": governing["margin"]}
               if g_c is not None else {}),
        }),
        "definition": ("O'Brien: G = (ε_x²·h/2)·(E_LAM − E*), E* = ΣE_i·t_i/h (박리로 갈라진 "
                       "부분적층의 두께 가중 평균). 박리 길이에 무관한 정상상태 값이라 "
                       "임계 변형률 ε_c = √(2G_c/(h·ΔE)) 가 닫힌 형태로 나온다"),
        "driver_definition": ("구동력은 계면 위쪽 부분적층이 지고 있는 불균형 힘·모멘트다. "
                              "peel_moment→σz(개구), transverse_force→τyz, shear_force→τxz. "
                              "경계층 평형 논증에 근거한 크기 규모 지표이지 응력장이 아니다"),
    }

    warn = list(warnings)
    if g_c is None and g_ic is None:
        warn.append(item("W120", field="fracture",
                         detail="층간 파괴인성이 없어 개시 변형률·여유율을 내지 못했다. G(에너지방출률) "
                                "순위만 참고할 것 — fracture={\"G_Ic\":…, \"G_IIc\":…} 를 주면 "
                                "혼합모드(Benzeggagh–Kenane)로, {\"G_c\":…} 를 주면 단일 인성으로 판정한다"))
    if g_c is not None:
        warn.append(item("W130", field="fracture.G_c",
                         detail="단일 G_c 를 썼다 — 층간 인성은 모드 의존이 크다(보통 G_Ic ≪ G_IIc). "
                                "G_Ic·G_IIc 를 주면 계면별로 혼합모드를 판정한다"))
    unknown_mix = [r["interface"] for r in interfaces if r["mode_mix"]["basis"] == "unknown"]
    if unknown_mix and g_ic is not None:
        warn.append(item("W130", field="mode_mix",
                         detail=f"계면 {unknown_mix} 는 부분적층이 거울상이 아니라 Mode I/II 분할을 "
                                f"정할 수 없다(3D/수치 해석 필요). onset_strain 은 **보수적인 순수 "
                                f"Mode I 값**이고 onset_strain_range 가 실제 범위다"))
    mirror_ifs = [r["interface"] for r in interfaces if r["mode_mix"]["basis"] == "mirror_symmetry"]
    if mirror_ifs:
        warn.append(item("W120", field="mode_mix",
                         detail=f"계면 {mirror_ifs} 는 두 부분적층이 거울상이라 대칭 논증으로 "
                                f"**순수 Mode I** 이다 — 여기서 G_Ic 를 쓰는 것은 보수가 아니라 정확하다"))
    warn.append(item("W130", field="driving",
                     detail="구동력은 경계층 평형에서 나온 크기 규모 지표다. 실제 자유 가장자리 "
                            "응력은 계면에서 특이점을 갖는 3D 탄성 문제라 개시 계면이 다를 수 있다"))
    if invalid:
        warn.append(item("W130", field="interfaces",
                         detail=f"계면 {invalid} 에서 ΔE = E_LAM − E* 가 음수라 O'Brien 모델이 "
                                f"적용되지 않는다(G = null). **구동력이 없다는 뜻이 아니다** — "
                                f"이 계면들은 이 도구로 판정할 수 없으니 별도 해석이 필요하다"))
    A_hat, B_hat, _Dh, _Kh = ABD.normalized_stiffness(A, B, D, h)
    if float(np.linalg.norm(B_hat)) > 1e-6 * float(np.linalg.norm(A_hat)):
        warn.append(item("W130", field="laminate",
                         detail="비대칭 적층 — O'Brien 모델은 대칭 적층의 균일 축변형을 전제한다. "
                                "막-굽힘 커플링이 있으면 ε_x 가 두께 방향으로 일정하지 않아 G 가 근사다"))
    if float(np.linalg.norm(M_si)) > 0.0:
        warn.append(item("W130", field="loads.M",
                         detail="굽힘 모멘트가 걸려 있다 — O'Brien 모델은 축방향 인장을 전제한다. "
                                "ε_x 는 중앙면 값이며 G 는 참고값이다"))
    if eps_x <= 0.0:
        warn.append(item("W130", field="loads",
                         detail=f"축방향 변형률이 {eps_x:.3g} 로 인장이 아니다 — 가장자리 박리는 "
                                f"인장에서 평가한다. 압축이면 좌굴 유발 박리를 별도로 볼 것"))

    extra = [
        "O'Brien(1985) 가장자리 박리 ERR — 대칭 적층·균일 축인장·정상상태(박리 길이 무관) 가정",
        "E* 는 부분적층이 자유단에서 굽는 연화까지 포함한 막 유효탄성계수로 계산한다",
        "혼합모드: 거울 분할 계면은 대칭 논증으로 순수 Mode I(정확). 그 외는 분할 불가라 "
        "G_Ic~G_IIc 범위로 답한다 — Suo–Hutchinson 위상각은 수치표라 미탑재(§17.7.3 원칙)",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "loads": loads, "fracture": fracture}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system, assumptions_extra=extra,
                     include_debug=include_debug, t0=t0)


def _num_ok(v, positive=True):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and (v > 0 if positive else True))


def run_micromechanics(fiber, matrix, fiber_volume_fraction, model: str = "halpin_tsai",
                       xi_E2: float | None = None, xi_G12: float | None = None,
                       include_debug: bool = False) -> dict:
    """구성재 → lamina 직교이방 물성 (derive_lamina_from_constituents, §19.3).

    단위계 무관 — 출력 단위는 입력과 동일하다 (homogenize_layer 와 같은 관례).
    """
    from app.solver import micromechanics as MM
    t0 = time.perf_counter()
    hash_payload = {"fiber": fiber, "matrix": matrix, "Vf": fiber_volume_fraction,
                    "model": model, "xi_E2": xi_E2, "xi_G12": xi_G12}
    err = None
    f_req = ("E1", "E2", "G12", "nu12")
    if not (isinstance(fiber, dict) and all(_num_ok(fiber.get(k)) for k in f_req)):
        err = item("E100", field="fiber",
                   detail="fiber는 {E1>0, E2>0, G12>0, nu12} (횡등방 섬유)여야 합니다. "
                          "선택: alpha1, alpha2, rho")
    elif not (_num_ok(fiber.get("nu12"), positive=False) and -1.0 < fiber["nu12"] < 0.5):
        err = item("E100", field="fiber.nu12", detail="fiber.nu12는 (-1, 0.5) 범위여야 합니다")
    elif not (isinstance(matrix, dict) and _num_ok(matrix.get("E"))
              and _num_ok(matrix.get("nu"), positive=False) and -1.0 < matrix["nu"] < 0.5):
        err = item("E100", field="matrix",
                   detail="matrix는 {E>0, nu∈(-1,0.5)} (등방 수지)여야 합니다. 선택: alpha, rho")
    elif not (_num_ok(fiber_volume_fraction) and 0.0 < fiber_volume_fraction <= 1.0):
        err = item("E100", field="fiber_volume_fraction",
                   detail="fiber_volume_fraction은 (0, 1] 범위여야 합니다")
    elif model not in ("halpin_tsai", "chamis"):
        err = item("E100", field="model",
                   detail="model은 'halpin_tsai' 또는 'chamis' 여야 합니다")
    elif any(x is not None and not _num_ok(x) for x in (xi_E2, xi_G12)):
        err = item("E100", field="xi", detail="xi_E2 / xi_G12 는 양수여야 합니다")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=[], payload=hash_payload,
                         include_debug=include_debug, t0=t0)

    v_f = float(fiber_volume_fraction)
    res = MM.lamina_from_constituents(
        fiber, matrix, v_f, model=model,
        xi_e2=MM.XI_E2_DEFAULT if xi_E2 is None else float(xi_E2),
        xi_g12=MM.XI_G12_DEFAULT if xi_G12 is None else float(xi_G12))

    mat: dict = {"type": "orthotropic_2d", "E1": res["E1"], "E2": res["E2"],
                 "G12": res["G12"], "nu12": res["nu12"],
                 "source": {"type": "estimated",
                            "ref": f"micromechanics/{model} Vf={v_f:g}"}}
    for k in ("alpha1", "alpha2", "rho"):
        if k in res:
            mat[k] = res[k]

    warn = [item("W120", field="material.E2/G12",
                 detail="E2·G12는 **기지 지배**라 미시역학 추정의 불확실성이 크다(경계 참조). "
                        "실측이 있으면 materialtwin 에서 가져와 쓰고, 없으면 Halpin–Tsai의 "
                        "ξ를 실측으로 역보정할 것. E1·ν12는 섬유 지배라 ROM이 신뢰도가 높다")]
    if v_f > 0.7:
        warn.append(item("W130", field="fiber_volume_fraction",
                         detail=f"Vf = {v_f:g} — 실제 공정에서 도달하기 어려운 값이다(원형 섬유 "
                                f"정사각 배열 이론 한계 0.785). 물성이 낙관적으로 나온다"))
    for key in ("E2", "G12"):
        lo = res["bounds"][key]["reuss"]
        hi = res["bounds"][key]["voigt"]
        if not (lo * (1 - 1e-9) <= res[key] <= hi * (1 + 1e-9)):
            warn.append(item("W130", field=f"material.{key}",
                             detail=f"{key} 추정값이 Reuss–Voigt 경계를 벗어났다 — 입력을 확인할 것"))

    data = {
        "material": mat,
        "bounds": res["bounds"],
        "model": model,
        "fiber_volume_fraction": v_f,
        "xi": {"E2": MM.XI_E2_DEFAULT if xi_E2 is None else float(xi_E2),
               "G12": MM.XI_G12_DEFAULT if xi_G12 is None else float(xi_G12)},
        "definition": ("E1·ν12·ρ = 혼합법칙(ROM), E2·G12 = Halpin–Tsai(ξ) 또는 Chamis, "
                       "α = Schapery. Halpin–Tsai는 ξ→0에서 Reuss(하한), ξ→∞에서 Voigt(상한)로 "
                       "정확히 수렴하므로 bounds가 추정의 폭이다"),
        "usage": "material을 그대로 laminae[].material 에 넣어 쓸 수 있다",
    }
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     assumptions_extra=[
                         "횡등방 섬유 + 등방 수지, 완전 접착(계면 미끄러짐 없음), 공극 0 가정",
                         "단위계 무관 — 출력 단위는 입력과 동일",
                     ], include_debug=include_debug, t0=t0)


def run_moisture_uptake(payload, diffusion, time_s=None, mode: str = "absorption",
                        include_debug: bool = False) -> dict:
    """Fickian 수분 확산 동역학 (compute_moisture_uptake, §19.4)."""
    from app.solver import diffusion as DF
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    err = None
    if not isinstance(diffusion, dict):
        err = item("E100", field="diffusion",
                   detail="diffusion은 {\"D\": 확산계수(길이²/s), \"M_inf\": 포화 수분율[%M]} "
                          "또는 {\"D0\", \"Ed\", \"temperature_K\", \"M_inf\"} 여야 합니다")
    elif mode not in ("absorption", "desorption"):
        err = item("E100", field="mode",
                   detail="mode는 'absorption'(흡습) 또는 'desorption'(베이크/탈습) 이어야 합니다")
    if err is None:
        has_direct = _num_ok(diffusion.get("D"))
        has_arr = all(_num_ok(diffusion.get(k)) for k in ("D0", "Ed", "temperature_K"))
        if not (has_direct or has_arr):
            err = item("E100", field="diffusion.D",
                       detail="D(>0) 또는 Arrhenius 3종(D0>0, Ed>0, temperature_K>0)이 필요합니다")
        elif not _num_ok(diffusion.get("M_inf")):
            err = item("E100", field="diffusion.M_inf",
                       detail="M_inf(포화 수분율 [%M], >0)가 필요합니다")
        elif time_s is not None and not (_num_ok(time_s, positive=False) and time_s >= 0):
            err = item("E100", field="time_s", detail="time_s는 0 이상의 유한한 숫자(초)여야 합니다")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    f_d = units.TO_SI[si.unit_system]["diffusivity"]
    if _num_ok(diffusion.get("D")):
        d_si = float(diffusion["D"]) * f_d
        d_note = "직접 입력"
    else:
        d_si = DF.arrhenius_diffusivity(float(diffusion["D0"]) * f_d, float(diffusion["Ed"]),
                                        float(diffusion["temperature_K"]))
        d_note = (f"Arrhenius D = D0·exp(−Ed/RT), T = {diffusion['temperature_K']} K")
    m_inf = float(diffusion["M_inf"])
    h = si.total_thickness
    if d_si <= 0.0:
        return ENV.build(data=None, errors=[item("E100", field="diffusion",
                                                 detail="확산계수가 0 이하입니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    char = {}
    for frac, label in ((0.5, "t50"), (0.9, "t90"), (0.99, "t99")):
        tau = DF.tau_for_fraction(frac)
        secs = None if tau is None else tau * h * h / d_si
        char[label] = None if secs is None else {"seconds": secs, "hours": secs / 3600.0,
                                                 "days": secs / 86400.0}

    data: dict = {
        "thickness": h * f["z"],
        "diffusivity_SI_m2_per_s": d_si,
        "diffusivity_source": d_note,
        "M_inf": m_inf,
        "mode": mode,
        "characteristic_times": char,
        "definition": ("양면 노출 1D Fick 해석해(Shen–Springer). 무차원 시간 τ = D·t/h² 하나가 "
                       "전부를 지배한다 — **두께가 2배면 시간이 4배**다. "
                       "M/M∞ = 1 − (8/π²)Σ exp(−(2n+1)²π²τ)/(2n+1)²"),
    }
    if time_s is not None:
        tau = d_si * float(time_s) / (h * h)
        frac = DF.uptake_fraction(tau)
        if mode == "desorption":
            remaining = 1.0 - frac
            m_now = m_inf * remaining
            data["state"] = {"time_s": float(time_s), "tau": tau,
                             "remaining_fraction": remaining, "moisture_content": m_now,
                             "note": "mode=desorption — M_inf 를 초기 수분율로 보고 남은 양을 준다"}
        else:
            m_now = m_inf * frac
            data["state"] = {"time_s": float(time_s), "tau": tau,
                             "uptake_fraction": frac, "moisture_content": m_now}
        data["state"]["delta_C_for_thermal_tool"] = (
            m_now if mode == "absorption" else m_now - m_inf)
        data["state"]["chain"] = ("이 delta_C 를 compute_thermal_response(laminate, delta_C=...) 에 "
                                  "넣으면 흡습 변형·곡률·판 휨까지 이어진다")
        data["profile"] = [
            {"zeta": zz, "c_over_cinf": DF.concentration_profile(tau, zz)}
            for zz in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)]
        data["profile_note"] = ("ζ=(z+h/2)/h, 0·1 이 노출면. 두께 중앙(ζ=0.5)이 가장 늦게 젖는다 — "
                                "표면과 중앙의 차이가 클수록 흡습 구배로 인한 휨이 생긴다")

    warn = list(warnings)
    warn.append(item("W130", field="model",
                     detail="Fickian(단순 확산) 가정 — 실제 에폭시는 2단계 흡습·비Fickian 거동을 "
                            "보일 수 있고, D 는 온도·수분율에 의존한다. 장시간 외삽은 신중할 것"))
    warn.append(item("W120", field="diffusion.D",
                     detail="D·M_inf 는 재료·환경(온습도) 실측값이어야 한다. 문헌 대표값을 쓰면 "
                            "특성시간이 자릿수로 달라질 수 있다 — materialtwin 에서 실측을 확인할 것"))
    if len({(p_.name or "", p_.E1) for p_ in si.plies}) > 1:
        warn.append(item("W130", field="laminae",
                         detail="이종 재료 적층인데 단일 D·M_inf 를 쓰고 있다 — 층별 확산계수가 "
                                "다르면 실제 거동은 이 해와 다르다(등가 단일층 근사)"))
    if time_s is not None and data["state"]["tau"] > 5.0:
        warn.append(item("W130", field="time_s",
                         detail=f"τ = {data['state']['tau']:.3g} — 사실상 완전 포화/건조 상태다. "
                                f"특성시간(characteristic_times)으로 판단하는 것이 낫다"))

    hash_payload = {"laminate": payload, "diffusion": diffusion, "time_s": time_s, "mode": mode}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=[
                         "양면 노출·1D 두께방향 확산, D 상수, 초기 균일 분포 가정",
                         "τ = D·t/h² 가 유일한 지배 변수 — 두께 제곱에 비례해 시간이 늘어난다",
                         *_source_assumptions(si),
                     ], include_debug=include_debug, t0=t0)


def _compliant_core_gate(si, span_si, warnings, label):
    """순응 중간층이 있으면 CLT 굽힘강성이 과대평가임을 알린다 (§19.12 게이트)."""
    from app.solver import partial_composite as PC
    k = PC.detect_compliant_core(si.plies)
    if k is None or span_si is None or span_si <= 0:
        return None
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]
    th = list(si.thicknesses)
    A, B, D, _z, _h = _abd_of_plies(si.plies)
    ea1, ei1 = PC.sublaminate_ea_ei(qbars, th, 0, k)
    ea2, ei2 = PC.sublaminate_ea_ei(qbars, th, k + 1, len(th))
    _ea_c, ei_c = PC.sublaminate_ea_ei(qbars, th, k, k + 1)
    core = si.plies[k]
    g_core = core.g13 if core.g13 is not None else core.G12
    res = PC.composite_action(ea1, ei1, ea2, ei2, ei_c, float(D[0, 0]),
                              g_core, core.thickness, span_si)
    f = res.get("composite_action")
    if f is None or f >= 0.9:
        return res
    over = res["EI_full"] / res["EI_effective"] if res["EI_effective"] > 0 else None
    warnings.append(item("W130", field="laminae",
                         detail=f"laminae[{k}] 가 이웃보다 10배 이상 무른 순응층이다 — {label} 기준 "
                                f"합성도 {f * 100:.1f}% 로 **CLT 굽힘강성이 "
                                f"{over:.3g}배 과대평가**된다. assess_partial_composite_bending "
                                f"로 확인할 것"))
    return res


def run_partial_composite(payload, span, core_ply=None, include_debug: bool = False) -> dict:
    """순응층 부분합성 굽힘 (assess_partial_composite_bending, §19.12)."""
    from app.solver import partial_composite as PC
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    n = len(si.plies)
    err = None
    if n < 3:
        err = item("E100", field="laminae",
                   detail="부분합성은 면재-순응층-면재 3층 이상이어야 평가할 수 있습니다")
    elif not (_num_ok(span)):
        err = item("E100", field="span",
                   detail="span(굽힘 스팬, 길이 단위)은 유한한 양수여야 합니다 — 합성도가 스팬에 "
                          "강하게 의존합니다(실측 L=1mm 18.3배 vs L=200mm 1.03배)")
    elif core_ply is not None and (not isinstance(core_ply, int) or isinstance(core_ply, bool)
                                   or not (0 < core_ply < n - 1)):
        err = item("E100", field="core_ply",
                   detail=f"core_ply는 1..{n - 2} 정수여야 합니다 (면재 사이의 중간층)")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    auto = core_ply is None
    k = PC.detect_compliant_core(si.plies) if auto else int(core_ply)
    if k is None:
        return ENV.build(data=None, errors=[item("E100", field="core_ply",
                                                 detail="이웃보다 10배 이상 무른 중간층을 찾지 못했습니다 — "
                                                        "순응층이 없으면 CLT 가 이미 맞습니다. "
                                                        "특정 층을 보려면 core_ply 로 지정하세요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    span_si = float(span) * units.TO_SI[si.unit_system]["length"]
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]
    th = list(si.thicknesses)
    A, B, D, _z, h = _abd_of_plies(si.plies)
    ea1, ei1 = PC.sublaminate_ea_ei(qbars, th, 0, k)
    ea2, ei2 = PC.sublaminate_ea_ei(qbars, th, k + 1, len(th))
    _ea_c, ei_c = PC.sublaminate_ea_ei(qbars, th, k, k + 1)
    core = si.plies[k]
    g_core = core.g13 if core.g13 is not None else core.G12
    res = PC.composite_action(ea1, ei1, ea2, ei2, ei_c, float(D[0, 0]),
                              g_core, core.thickness, span_si)
    if res.get("composite_action") is None:
        return ENV.build(data=None, errors=[item("E100", field="laminate",
                                                 detail=f"부분합성을 계산할 수 없습니다: {res.get('reason')}")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    fac = res["composite_action"]
    over = res["EI_full"] / res["EI_effective"] if res["EI_effective"] > 0 else None
    data = {
        "core_ply": {"index": k, "angle_deg": core.angle_deg,
                     "thickness": core.thickness * f["z"],
                     "G_transverse": g_core * f["modulus"],
                     "detected": "auto" if auto else "explicit",
                     "G13_assumed_from_G12": core.g13 is None},
        "span": float(span),
        "composite_action": fac,
        "EI_layered": res["EI_layered"] * f["D"],
        "EI_full_CLT": res["EI_full"] * f["D"],
        "EI_effective": res["EI_effective"] * f["D"],
        "clt_overprediction": over,
        "shear_lag": {"alpha": res["alpha"] / f["z"] if res["alpha"] else None,
                      "alpha_L": res["alpha_L"],
                      "face_neutral_axis_distance": res["d"] * f["z"]},
        "definition": ("α² = (G_c/t_c)(1/EA₁ + 1/EA₂ + d²/EI_layered), "
                       "f = 1 − tanh(αL/2)/(αL/2), EI_eff = EI_layered + f·(EI_full − EI_layered). "
                       "f=0 은 면재가 각자 굽는 상태, f=1 은 CLT(완전합성)와 같다"),
        "meaning": ("clt_overprediction = CLT 굽힘강성 ÷ 실제 유효 굽힘강성. "
                    "1보다 크면 CLT 를 쓴 처짐·좌굴·진동수가 모두 낙관적이다"),
    }

    warn = list(warnings)
    if fac < 0.9:
        warn.append(item("W130", field="composite_action",
                         detail=f"합성도 {fac * 100:.1f}% — CLT 기반 도구(solve_load_response, "
                                f"compute_buckling, compute_natural_frequencies)의 굽힘 관련 값이 "
                                f"모두 {over:.3g}배 낙관적이다"))
    if core.g13 is None:
        warn.append(item("W120", field="core_ply.G_transverse",
                         detail="순응층에 G13 이 없어 면내 G12 로 대체했다 — 실제 횡전단 강성이 다르면 "
                                "합성도가 달라진다(α ∝ √G_c). 실측을 넣을 것"))
    if res["alpha_L"] is not None and res["alpha_L"] < 1.0:
        warn.append(item("W130", field="shear_lag.alpha_L",
                         detail=f"αL = {res['alpha_L']:.3g} < 1 — 전단 전달이 거의 일어나지 않는 "
                                f"영역이다. 면재가 사실상 따로 굽는다"))

    hash_payload = {"laminate": payload, "span": span, "core_ply": core_ply}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=[
                         "shear-lag 3층 보 모델 — 면재는 오일러 보, 순응층은 전단만 전달(축강성 무시)",
                         "단순지지 스팬 L 의 균일 굽힘 가정. 경계·하중 형태가 다르면 f 가 달라진다",
                         "d² 는 EI_full − EI_layered 항등식에서 역산해 두 값과 항상 정합한다",
                         *_source_assumptions(si),
                     ], include_debug=include_debug, t0=t0)


def run_prescribed_curvature(payload, kappa=None, bend_radius=None, bend_axis: str = "x",
                             width: str = "free", epsilon0=None, loads=None,
                             include_debug: bool = False) -> dict:
    """변위 제어 — 곡률·면내변형 지정 (solve_prescribed_curvature, §19.14)."""
    from app.solver import prescribed as PRE
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)

    def _vec3(v, label, dimensionless):
        """길이 3 리스트(None 허용) → SI 변환. (값, 오류)."""
        if v is None:
            return None, None
        if not (isinstance(v, list) and len(v) == 3):
            return None, item("E100", field=label, detail=f"{label}는 길이 3 리스트여야 합니다 "
                                                          f"(자유 성분은 null)")
        out = []
        for x in v:
            if x is None:
                out.append(None)
            elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
                out.append(float(x) * (1.0 if dimensionless else units.TO_SI[si.unit_system]["kappa"]))
            else:
                return None, item("E100", field=label,
                                  detail=f"{label}의 각 성분은 숫자 또는 null 이어야 합니다")
        return out, None

    err = None
    if bend_axis not in ("x", "y"):
        err = item("E100", field="bend_axis", detail="bend_axis는 'x' 또는 'y' 여야 합니다")
    elif width not in ("free", "constrained"):
        err = item("E100", field="width",
                   detail="width는 'free'(M_y=0, 자유 폭) 또는 'constrained'(κ_y=0, 구속 폭) "
                          "여야 합니다 — 두 경우 답이 다릅니다(실측 9.6% 차이)")
    elif kappa is not None and bend_radius is not None:
        err = item("E100", field="kappa/bend_radius",
                   detail="kappa 와 bend_radius 를 동시에 줄 수 없습니다")
    elif bend_radius is not None and not (_num_ok(bend_radius)):
        err = item("E100", field="bend_radius", detail="bend_radius는 유한한 양수여야 합니다")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    kap_si, err = _vec3(kappa, "kappa", dimensionless=False)
    if err is None:
        eps_si, err = _vec3(epsilon0, "epsilon0", dimensionless=True)
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    if bend_radius is not None:
        # 1/R 은 길이의 역수 — kappa 와 같은 차원이라 같은 계수로 변환한다
        k_val = 1.0 / (float(bend_radius) * units.TO_SI[si.unit_system]["length"])
        other = 0.0 if width == "constrained" else None
        kap_si = [k_val, other, None] if bend_axis == "x" else [other, k_val, None]

    if kap_si is None and eps_si is None:
        return ENV.build(data=None, errors=[item("E100", field="kappa/bend_radius/epsilon0",
                                                 detail="지정할 자유도가 없습니다 — kappa, bend_radius, "
                                                        "epsilon0 중 최소 하나는 주어야 합니다. "
                                                        "전부 힘 제어라면 solve_load_response 를 쓰세요")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    N_si, M_si, load_err = _loads_to_si(loads, si.unit_system) if loads is not None else (
        np.zeros(3), np.zeros(3), None)
    if load_err is not None:
        return ENV.build(data=None, errors=[load_err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    A, B, D, z, h = _abd_of_plies(si.plies)
    K = np.block([[A, B], [B, D]])
    presc = PRE.build_prescription(kap_si, eps_si)
    try:
        x, f_gen = PRE.partitioned_solve(K, presc, np.concatenate([N_si, M_si]))
    except PRE.SingularPartition as exc:
        return ENV.build(data=None, errors=[item("E400", field="laminate", detail=str(exc))],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    eps_res, kap_res = x[:3], x[3:]
    n_res, m_res = f_gen[:3], f_gen[3:]
    fu = units.FROM_SI[si.unit_system]
    naive = PRE.naive_moment(D, kap_res)

    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]
    from app.solver import failure as FAIL
    per_ply = []
    for k, p_ in enumerate(si.plies):
        z_lo, z_hi = float(z[k]), float(z[k + 1])
        row = {"ply": k, "angle_deg": p_.angle_deg}
        for label, zv in (("bottom", z_lo), ("mid", 0.5 * (z_lo + z_hi)), ("top", z_hi)):
            e_xyz = eps_res + zv * kap_res
            e12 = FAIL.stress_to_material_axes(np.array([e_xyz[0], e_xyz[1], 0.5 * e_xyz[2]]),
                                               p_.angle_deg)
            row[label] = {"epsilon_xyz": e_xyz.tolist(),
                          "epsilon_1": float(e12[0]), "epsilon_2": float(e12[1])}
        per_ply.append(row)
    trunc = None
    if len(per_ply) > config.SUMMARY_PLY_LIMIT:
        per_ply = per_ply[:config.SUMMARY_TOP_N]
        trunc = (f"ply {len(si.plies)}개 중 {len(per_ply)}개만 반환 (§6.6 토큰 예산)")

    fixed_names = [n for n, v in zip(("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y",
                                      "kappa_xy"), presc) if v is not None]
    data = {
        "prescribed_dof": fixed_names,
        "free_dof": [n for n in ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")
                     if n not in fixed_names],
        "response": {"epsilon0": eps_res.tolist(), "kappa": (kap_res * fu["kappa"]).tolist()},
        "equivalent_loads": {"N": (n_res * fu["A"]).tolist(), "M": (m_res * fu["load_m"]).tolist(),
                             "chain": "이 N·M 을 recover_ply_stresses·run_progressive_failure 에 "
                                      "그대로 넘기면 파손 판정까지 이어진다"},
        "naive_D_kappa": {
            "M": (naive * fu["load_m"]).tolist(),
            "overprediction_Mx": (float(naive[0] / m_res[0]) if abs(m_res[0]) > 0 else None),
            "note": "에이전트가 흔히 쓰는 지름길 M = D·κ 와의 대조. 비대칭 적층이나 자유 폭에서는 "
                    "크게 틀린다(실측 +244.8%)",
        },
        "surface_strain": {
            "bottom": (eps_res + float(z[0]) * kap_res).tolist(),
            "top": (eps_res + float(z[-1]) * kap_res).tolist(),
            "note": "assess_crack_shielding 의 applied_strain 에 넣을 값 — 손으로 (z−z_ns)/R 을 "
                    "계산하지 말 것",
        },
        "per_ply_strain": per_ply,
        **({"truncation": trunc} if trunc else {}),
        "definition": ("K[ε⁰;κ] = [N;M] 의 지정/미지 자유도 분할. 지정된 자유도는 일반변형률이 "
                       "알려지고 대응 일반력이 반력이 된다. 대칭 적층에서 κ 를 전부 지정하면 "
                       "M = D·κ 로, 비대칭이면 M = D*·κ 로 정확히 환원된다"),
    }

    warn = list(warnings)
    over = data["naive_D_kappa"]["overprediction_Mx"]
    if over is not None and abs(over - 1.0) > 0.05:
        warn.append(item("W130", field="naive_D_kappa",
                         detail=f"지름길 M = D·κ 는 이 조합에서 M_x 를 {over:.3g}배로 준다 — "
                                f"{100 * (over - 1):+.0f}% 어긋난다. 변위 제어를 힘 제어로 바꿔 "
                                f"암산하지 말 것"))
    if bend_radius is not None and width == "free":
        warn.append(item("W130", field="width",
                         detail="자유 폭(M_y=0) 가정이다 — 폭이 구속된 실제 지그라면 "
                                "width='constrained' 로 다시 보라. 실측 9.6% 차이가 났다"))
    A_hat, B_hat, _Dh, _Kh = ABD.normalized_stiffness(A, B, D, h)
    if float(np.linalg.norm(B_hat)) > 1e-6 * float(np.linalg.norm(A_hat)):
        warn.append(item("W130", field="laminate",
                         detail="비대칭 적층 — 곡률을 지정하면 막변형이 함께 생긴다(εx≠0). "
                                "equivalent_loads 의 N 이 0이 아니면 그 축력을 지그가 실제로 "
                                "받아줄 수 있는지 확인할 것"))

    hash_payload = {"laminate": payload, "kappa": kappa, "bend_radius": bend_radius,
                    "bend_axis": bend_axis, "width": width, "epsilon0": epsilon0, "loads": loads}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=[
                         "변위 제어 — 지정 자유도의 일반력은 지그가 제공하는 반력이다",
                         "자유 폭(M_y=0)과 구속 폭(κ_y=0)은 다른 문제다 — width 로 명시한다",
                         *_source_assumptions(si),
                     ], include_debug=include_debug, t0=t0)


# ── §19.16 파손 포락선 / §19.17 필요 두께 배율 ─────────────────────────────

ENVELOPE_DIRECTIONS = 72     # 고정 방향 격자 (5도 간격) — 결정론


def run_failure_envelope(payload, plane: str = "Nx-Ny", magnitude=None,
                         delta_t=None, include_debug: bool = False) -> dict:
    """하중 **방향** 축의 파손 포락선 (compute_failure_envelope, §19.16).

    지금까지 최소점을 찾는 유일한 방법이 비결정론적 반복 호출이었다. 고정 72방향
    격자에서 방향당 Tsai-Wu 강도비를 한 번씩 계산해 포락선과 최약 방향을 결정론적으로 준다.
    """
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    planes = {"Nx-Ny": (0, 1, "N"), "Nx-Nxy": (0, 2, "N"), "Mx-My": (0, 1, "M")}
    if plane not in planes:
        return ENV.build(data=None, errors=[item("E100", field="plane",
                                                 detail=f"plane은 {list(planes)} 중 하나여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    if magnitude is not None and not _num_ok(magnitude):
        return ENV.build(data=None, errors=[item("E100", field="magnitude",
                                                 detail="magnitude는 유한한 양수여야 합니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    with_strength = [k for k, p_ in enumerate(si.plies) if p_.strength is not None]
    if not with_strength:
        return ENV.build(data=None, errors=[item("E100", field="laminae",
                                                 detail="strength가 있는 ply가 없습니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

    i, j, kind = planes[plane]
    mag_disp = float(magnitude) if magnitude is not None else 1.0
    fac = units.TO_SI[si.unit_system]["load_n" if kind == "N" else "load_m"]
    mag_si = mag_disp * fac
    A, B, D, z, h = _abd_of_plies(si.plies)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p_.E1, p_.E2, p_.G12, p_.nu12), p_.angle_deg)
             for p_ in si.plies]

    from app.solver import failure as FAIL
    from app.solver import thermal as TH
    dT = float(delta_t) if _num_ok(delta_t, positive=False) else 0.0
    res_state = None
    if dT != 0.0:
        missing = [k for k, p_ in enumerate(si.plies) if not p_.has_cte]
        if missing:
            return ENV.build(data=None, errors=[item("E203", field="laminae",
                                                     detail=f"laminae{missing} 에 CTE가 없습니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
        eps_free = [TH.alpha_vector(p_.alpha1, p_.alpha2, p_.angle_deg) * dT for p_ in si.plies]
        n_th, m_th = TH.free_strain_loads(qbars, eps_free, z)
        e_r, k_r = RESP.solve_response(A, B, D, n_th, m_th)
        res_state = (e_r, k_r, eps_free)

    points = []
    for d in range(ENVELOPE_DIRECTIONS):
        theta = 2.0 * math.pi * d / ENVELOPE_DIRECTIONS
        vec = np.zeros(3)
        vec[i], vec[j] = math.cos(theta) * mag_si, math.sin(theta) * mag_si
        n_v = vec if kind == "N" else np.zeros(3)
        m_v = vec if kind == "M" else np.zeros(3)
        try:
            eps0, kappa = RESP.solve_response(A, B, D, n_v, m_v)
        except RESP.SingularSystemError:
            return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                             payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)
        worst_r, worst_k, worst_mode = None, None, None
        for k in with_strength:
            p_ = si.plies[k]
            Xt, Xc, Yt, Yc, S = p_.strength
            for zv in (float(z[k]), 0.5 * (z[k] + z[k + 1]), float(z[k + 1])):
                s_m = FAIL.stress_to_material_axes(
                    FAIL.ply_stresses_at(qbars[k], eps0, kappa, zv), p_.angle_deg)
                if res_state is None:
                    s_r = np.zeros(3)
                else:
                    e_r, k_r, eps_free = res_state
                    s_r = FAIL.stress_to_material_axes(
                        qbars[k] @ (e_r + zv * k_r - eps_free[k]), p_.angle_deg)
                r = FAIL.tsai_wu_with_offset(s_m, s_r, Xt, Xc, Yt, Yc, S).get("strength_ratio")
                if r is None or r < 0:
                    continue
                if worst_r is None or r < worst_r:
                    worst_r, worst_k = r, k
                    worst_mode = FAIL.max_stress(r * s_m + s_r, Xt, Xc, Yt, Yc, S)["mode"]
        if worst_r is None:
            continue
        points.append({"angle_deg": math.degrees(theta),
                       "direction": [float(vec[i] / mag_si), float(vec[j] / mag_si)],
                       "strength_ratio": float(worst_r),
                       "failure_load": [float(worst_r * vec[i] / fac),
                                        float(worst_r * vec[j] / fac)],
                       "critical_ply": worst_k, "mode": worst_mode})
    if not points:
        return ENV.build(data=None, errors=[item("E100", field="laminate",
                                                 detail="어느 방향에서도 파손면에 도달하지 못했습니다")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    weakest = min(points, key=lambda r: (r["strength_ratio"], r["angle_deg"]))
    strongest = max(points, key=lambda r: (r["strength_ratio"], -r["angle_deg"]))
    axes = plane.split("-")
    data = {
        "plane": plane, "axes": axes, "n_directions": ENVELOPE_DIRECTIONS,
        "probe_magnitude": mag_disp,
        "points": points,
        "weakest": {"angle_deg": weakest["angle_deg"], "failure_load": weakest["failure_load"],
                    "critical_ply": weakest["critical_ply"], "mode": weakest["mode"]},
        "strongest": {"angle_deg": strongest["angle_deg"],
                      "failure_load": strongest["failure_load"]},
        "anisotropy_ratio": strongest["strength_ratio"] / weakest["strength_ratio"],
        "delta_T": dT if dT != 0.0 else None,
        "definition": (f"{ENVELOPE_DIRECTIONS}방향 고정 격자(5도 간격). 방향마다 단위 하중을 걸고 "
                       f"Tsai-Wu 강도비 R 을 구해 failure_load = R×방향벡터 로 포락선을 만든다. "
                       f"열잔류가 있으면 R 은 2차식으로 푼다(§19.10)"),
    }
    warn = list(warnings)
    warn.append(item("W130", field="points",
                     detail=f"{ENVELOPE_DIRECTIONS}방향 격자라 각도 해상도가 5도다 — 뾰족한 "
                            f"포락선의 꼭짓점은 놓칠 수 있다"))
    if dT == 0.0 and any(p_.has_cte for p_ in si.plies):
        warn.append(item("W130", field="delta_T",
                         detail="CTE 가 있는데 delta_T 가 없다 — 경화 잔류가 빠진 포락선이다"))
    hash_payload = {"laminate": payload, "plane": plane, "magnitude": magnitude,
                    "delta_T": delta_t}
    if ENV.contains_nan_inf(data):
        return ENV.build(data=None, errors=[item("E403", field="data")], warnings=warn,
                         payload=hash_payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=["포락선은 Tsai-Wu 기준 first-ply-failure 다 — 진행성 파손 "
                                        "이후의 한계하중은 run_progressive_failure 로",
                                        *_source_assumptions(si)],
                     include_debug=include_debug, t0=t0)


def run_required_scale(payload, panel, applied_Nx, target_margin: float = 1.0,
                       load_ratio: float = 0.0, boundary: str = "simply_supported",
                       include_debug: bool = False) -> dict:
    """목표 좌굴 여유를 만족하는 최소 두께 배율 (solve_required_thickness_scale, §19.17).

    전 ply 두께를 s 배 하면 A ∝ s, D ∝ s³ 이므로 **N_cr ∝ s³** 이다(실측 전 자리 일치).
    따라서 s = (target·N/N_cr(1))^(1/3) 가 폐형해다 — 에이전트 이분법 5~10회를 1회로 바꾼다.
    """
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    a, b = _panel_to_si(panel, si.unit_system) if panel is not None else (None, None)
    err = None
    if a is None:
        err = item("E100", field="panel", detail="panel {\"Lx\": >0, \"Ly\": >0} 은 필수입니다")
    elif not _num_ok(applied_Nx):
        err = item("E100", field="applied_Nx", detail="applied_Nx는 압축 크기(양수)여야 합니다")
    elif not _num_ok(target_margin):
        err = item("E100", field="target_margin", detail="target_margin은 양수여야 합니다")
    elif not (_num_ok(load_ratio, positive=False)):
        err = item("E100", field="load_ratio", detail="load_ratio는 유한한 숫자여야 합니다")
    else:
        from app.solver import plate_navier as _NAV
        if _NAV.normalize_boundary(boundary) is None:
            err = item("E100", field="boundary",
                       detail=f"boundary 코드가 유효하지 않습니다. {_NAV.FREE_EDGE_NOTE}")
    if err is not None:
        return ENV.build(data=None, errors=[err], warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    from app.solver import plate_navier as NAV
    D_use, appl, h, _ = _bending_stiffness_for_navier(si, warnings)
    bp = NAV.normalize_boundary(boundary)
    if bp == ("SS", "SS"):
        res = NAV.buckling_ncr(D_use, a, b, float(load_ratio))
    else:
        res = NAV.scan_ritz_buckling(D_use, a, b, float(load_ratio), boundary,
                                     NAV.CLAMPED_MODE_LIMIT)
    if res["N_cr"] is None:
        return ENV.build(data=None, errors=[item("E100", field="load_ratio",
                                                 detail=res.get("reason", "압축 지배 모드 없음"))],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    n_si = float(applied_Nx) * units.TO_SI[si.unit_system]["load_n"]
    current = res["N_cr"] / n_si
    scale = (float(target_margin) / current) ** (1.0 / 3.0)
    f = units.FROM_SI[si.unit_system]
    data = {
        "current_margin": current,
        "target_margin": float(target_margin),
        "required_scale": scale,
        "required_total_thickness": h * scale * f["z"],
        "current_total_thickness": h * f["z"],
        "scaled_ply_thicknesses": [p_.thickness * scale * f["z"] for p_ in si.plies][:config.SUMMARY_TOP_N],
        "N_cr_current": res["N_cr"] * f["A"],
        "N_cr_scaled": res["N_cr"] * scale ** 3 * f["A"],
        "boundary": boundary,
        "mode": {"m": res["mode_m"], "n": res["mode_n"]},
        "definition": ("전 ply 를 균일 배율 s 로 키우면 D ∝ s³ 이므로 N_cr ∝ s³ 이다(정확). "
                       "s = (target/current)^(1/3) 폐형해 — 반복 탐색이 필요 없다"),
    }
    warn = list(warnings)
    warn.append(item("W130", field="required_scale",
                     detail="**균일 배율만** 유효하다 — 일부 ply 만 두껍게 하거나 적층 순서를 바꾸면 "
                            "이 지수 법칙이 깨진다. 그때는 compute_buckling 으로 직접 확인할 것"))
    if scale > 3.0:
        warn.append(item("W130", field="required_scale",
                         detail=f"필요 배율 {scale:.3g} 가 크다 — 두께로만 풀지 말고 적층 순서·"
                                f"경계조건·판 크기를 함께 검토할 것"))
    hash_payload = {"laminate": payload, "panel": panel, "applied_Nx": applied_Nx,
                    "target_margin": target_margin, "load_ratio": load_ratio,
                    "boundary": boundary}
    return ENV.build(data=data, errors=[], warnings=warn, payload=hash_payload,
                     unit_system=si.unit_system,
                     assumptions_extra=["균일 두께 배율 가정 — 물성·적층각·판 크기는 그대로",
                                        *_source_assumptions(si)],
                     include_debug=include_debug, t0=t0)
