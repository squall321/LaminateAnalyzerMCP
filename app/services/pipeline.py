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
                                                     detail="panel은 {\"Lx\": >0, \"Ly\": >0} (유한 양수, 길이 단위) 여야 합니다")],
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

    extra = [
        "자유변형 해석 가정: 선형 CTE/CME(온도·수분 무관 — Tg 이상 α 급변 미반영), 자유 경계·소변형",
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
    }
    return ENV.build(data=data, errors=[], warnings=[], payload={"components": components},
                     assumptions_extra=["균질화: Voigt(등변형) 상한 — 면내 강성·CTE 1차 근사"],
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
                     include_debug: bool = False) -> dict:
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

    extra = [
        "파손 판정: Tsai-Wu(F12=-0.5√(F11F22) 표준 상호작용) 강도비 R 주지표 + Max Stress 지배 모드",
        "응력은 각 ply의 bottom/mid/top 3점 (ply 내 z 선형)",
        *(["열하중 중첩: 선형 CTE, 자유 경계 (§17.1 가정 동일)"] if dT != 0.0 else []),
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "loads": loads, "delta_T": delta_t}
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


def _panel_to_si(panel, us):
    """panel {"Lx","Ly"} → SI 길이. 형식·유한성·양수 위반 시 (None, None) (EDGE-01)."""
    if not (isinstance(panel, dict)
            and all(isinstance(panel.get(k), (int, float)) and not isinstance(panel.get(k), bool)
                    and math.isfinite(panel.get(k)) and panel.get(k) > 0 for k in ("Lx", "Ly"))):
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
                 include_debug: bool = False) -> dict:
    """단순지지 직교이방 판 좌굴 임계 (compute_buckling Tool, §17.5.2)."""
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
                                                 detail="panel은 {\"Lx\": >0, \"Ly\": >0} (길이 단위) 여야 합니다")],
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

    D_use, appl, h, _ = _bending_stiffness_for_navier(si, warnings)
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
    f = units.FROM_SI[si.unit_system]
    data = {
        "N_cr": res["N_cr"] * f["A"],
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


def run_frequencies(payload, panel, n_modes: int = 5, include_debug: bool = False) -> dict:
    """단순지지 직교이방 판 고유진동수 (compute_natural_frequencies Tool, §17.5.2)."""
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
                                                 detail="panel은 {\"Lx\": >0, \"Ly\": >0} (길이 단위) 여야 합니다")],
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

    D_use, appl, h, _ = _bending_stiffness_for_navier(si, warnings)
    rho_areal = sum(p_.rho * p_.thickness for p_ in si.plies)   # SI kg/m²
    scan = max(NAV.MODE_SCAN, n_modes)      # f는 m·n 단조 증가 → scan ≥ n_modes면 상위 정확 (NAV-3)
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


def run_progressive(payload, loads, discount: float = 0.1, include_debug: bool = False) -> dict:
    """ply discount 진행성 파손 — FPF→한계하중 (run_progressive_failure Tool, §17.5.3)."""
    from app.solver import progressive as PROG
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

    res = PROG.run(si.plies, N_si, M_si, float(discount))
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
    if no_strength:
        data["note"] = f"laminae{no_strength}는 strength 미입력 — 탄성 유지(비파손) 가정"
    hash_payload = {"laminate": payload, "loads": loads, "discount": discount}
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
                                  "정량 예측은 3D 해석 필요")
        warnings.append(item("W130", field="free_edge_risk_ranking",
                             detail="자유단 박리는 CLT 평형법으로 정량화되지 않습니다 (순위는 정성 지표)"))

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


def run_fatigue(payload, loads_max, loads_min=None, detail: str = "auto",
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
    if skipped:
        data["excluded_plies"] = skipped
    hash_payload = {"laminate": payload, "loads_max": loads_max, "loads_min": loads_min,
                    "detail": detail}
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
        kx_lin, ky_lin = NL.linear_curvatures(A, B, D, N_f, M_f)
    except np.linalg.LinAlgError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    en = NL.HyerEnergy(A, B, D, N_f, M_f, lx, ly)
    span = NL.search_span(en, kx_lin, ky_lin)
    sols = NL.find_equilibria(en, span)
    crit = NL.critical_scale(A, B, D, N_f, M_f, lx, ly)

    f = units.FROM_SI[si.unit_system]
    equilibria = []
    for s in sols:
        kap = np.array([s["a"], s["b"], 0.0])
        equilibria.append({
            "kappa_x": s["a"] * f["kappa"], "kappa_y": s["b"] * f["kappa"],
            "shape": NL.classify_shape(s["a"], s["b"]),
            "stable": s["stable"],
            "energy_per_area": s["energy"] / (lx * ly) * f["energy_area"],
            "warpage_range": TH.warpage_over_panel(kap, lx, ly)["warpage_range"] * f["z"],
        })
    stable = [e for e in equilibria if e["stable"]]
    unstable = [e for e in equilibria if not e["stable"]]

    data: dict = {
        "linear_reference": {
            "kappa_x": kx_lin * f["kappa"], "kappa_y": ky_lin * f["kappa"],
            "shape": NL.classify_shape(kx_lin, ky_lin),
            "note": "선형 CLT 해 — 판 크기 무관. 임계 크기를 넘은 판에서는 실제로 실현되지 않는다",
        },
        "equilibria": equilibria,
        "stable_count": len(stable),
        "bistable": len(stable) >= 2,
        "critical_panel": (
            {"Lx": crit["lx"] / units.TO_SI[si.unit_system]["length"],
             "Ly": crit["ly"] / units.TO_SI[si.unit_system]["length"],
             "scale_vs_input": crit["scale"],
             "definition": "종횡비를 유지한 채 판을 키울 때 안장 해가 안정성을 잃는 크기 (분기점)"}
            if crit["scale"] is not None else None),
    }
    if stable and unstable:
        barrier = min(e["energy_per_area"] for e in unstable) - min(e["energy_per_area"] for e in stable)
        data["energy_barrier"] = {
            "value": barrier,
            "definition": "불안정(안장) 해와 안정 해의 단위면적 에너지 차 — 스냅스루를 막는 장벽",
            "note": "스냅 하중 자체는 하중 인가 평형 추적이 필요해 이 도구가 계산하지 않는다",
        }

    warn = list(warnings)
    if data["bistable"]:
        warn.append(item("W130", field="panel",
                         detail=f"판이 임계 크기를 넘어 안정 형상이 {len(stable)}개다 — "
                                f"선형 해석(compute_thermal_response)의 안장 곡률은 실현되지 않는다"))
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
        "한계: γxy⁰=0·κxy=0 (비틀림 형상 미표현), 자유 경계, 저차 근사 — 곡률 절대값은 FE 대비 오차가 있다",
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
    except ValueError as e:
        return ENV.build(data=None, errors=[item("E100", field="laminate", detail=str(e))],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)

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
    pb = NL.postbuckling(D_use, A, lx, ly, res["mode_m"], res["mode_n"],
                         res["N_cr"], n_si, h)

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
    if pb["buckled"] and pb["amplitude_over_thickness"] > 3.0:
        warn.append(item("W130", field="amplitude",
                         detail=f"W/h = {pb['amplitude_over_thickness']:.3g} — 1항 근사의 유효 범위를 "
                                f"크게 벗어났다. 경향만 보고 정밀 판정은 FE로 할 것"))
    extra = [
        "SS 4변, 1항 Galerkin — 좌굴 모드 (m,n)은 compute_buckling과 동일한 스캔으로 결정",
        "강성비는 끝단 수축 e = a11·N + W²p²/8 의 미분에서 유도 (등방 정사각에서 정확히 0.5)",
        "b_eff/b = √(N_cr/N) 은 von Karman 유효폭 — 후좌굴 하중 재분배의 반경험 지표",
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
