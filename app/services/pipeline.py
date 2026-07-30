# 요청 오케스트레이션 — 검증→SI 계산→표시 변환→envelope 조립 (계획서 §6.3, §7.2)
from __future__ import annotations

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


def run_thermal(payload, delta_t, panel=None, include_debug: bool = False) -> dict:
    """자유 열변형: 유효 CTE·열곡률·ply 잔류응력·판 휨 (compute_thermal_response Tool, §17.1)."""
    from app.solver import thermal as TH
    t0 = time.perf_counter()
    si, errors, warnings = VAL.validate_and_convert(payload)
    if errors:
        return ENV.build(data=None, errors=errors, warnings=warnings, payload=payload,
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None,
                         include_debug=include_debug, t0=t0)
    if not isinstance(delta_t, (int, float)) or isinstance(delta_t, bool) or delta_t == 0:
        return ENV.build(data=None, errors=[item("E100", field="delta_T",
                                                 detail="delta_T는 0이 아닌 숫자 [K]여야 합니다 (T_현재 − T_무응력기준)")],
                         warnings=warnings, payload=payload, unit_system=si.unit_system,
                         include_debug=include_debug, t0=t0)
    if panel is not None:
        if not (isinstance(panel, dict)
                and all(isinstance(panel.get(k), (int, float)) and not isinstance(panel.get(k), bool)
                        and panel.get(k) > 0 for k in ("Lx", "Ly"))):
            return ENV.build(data=None, errors=[item("E100", field="panel",
                                                     detail="panel은 {\"Lx\": >0, \"Ly\": >0} (길이 단위) 여야 합니다")],
                             warnings=warnings, payload=payload, unit_system=si.unit_system,
                             include_debug=include_debug, t0=t0)

    alphas, cte_err = _cte_vectors_or_error(si)
    if cte_err:
        return ENV.build(data=None, errors=cte_err, warnings=warnings, payload=payload,
                         unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    A, B, D, z, h = _abd_of_plies(si.plies)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg) for p in si.plies]
    N_th, M_th = TH.thermal_loads(qbars, alphas, z, float(delta_t))
    try:
        eps0, kappa = TH.thermal_response(A, B, D, N_th, M_th)
    except RESP.SingularSystemError:
        return ENV.build(data=None, errors=[item("E400", field="laminate")], warnings=warnings,
                         payload=payload, unit_system=si.unit_system, include_debug=include_debug, t0=t0)

    f = units.FROM_SI[si.unit_system]
    sig = TH.residual_ply_stresses(qbars, alphas, z, eps0, kappa, float(delta_t))
    per_ply = [{"ply": k, "sigma_xyz": (s * f["modulus"]).tolist()} for k, s in enumerate(sig)]
    worst = max(range(len(sig)), key=lambda k: float(np.max(np.abs(sig[k]))))

    data: dict = {
        "effective_cte": {
            "alpha_x": float(eps0[0] / delta_t), "alpha_y": float(eps0[1] / delta_t),
            "alpha_xy": float(eps0[2] / delta_t),
            "definition": "자유 열변형 막변형률/ΔT [1/K] (비대칭이면 곡률 커플링 포함 유효값)",
        },
        "response": {
            "delta_T": float(delta_t),
            "epsilon0": eps0.tolist(),
            "kappa": (kappa * f["kappa"]).tolist(),
            "kappa_per_K": (kappa * f["kappa"] / delta_t).tolist(),
        },
        "residual_stress": {
            "note": "ply 중앙면 값 σ = Q̄(ε0+z̄κ−αΔT). 힘 평형(Σσt=0)은 엔진 불변식",
            "per_ply": per_ply,
            "max_abs": {"ply": worst, "value": float(np.max(np.abs(sig[worst])) * f["modulus"])},
        },
    }
    if panel is not None:
        lx_si = panel["Lx"] * units.TO_SI[si.unit_system]["length"]
        ly_si = panel["Ly"] * units.TO_SI[si.unit_system]["length"]
        wp = TH.warpage_over_panel(kappa, lx_si, ly_si)
        data["warpage"] = {
            "panel": {"Lx": panel["Lx"], "Ly": panel["Ly"]},
            "range": wp["warpage_range"] * f["z"],
            "definition": "w(x,y)=−½(κx x²+κy y²+κxy xy)의 판 위 최대−최소 (coplanarity). 9점 평가",
        }

    extra = [
        "열해석 가정: 선형 CTE(온도 무관 — FR-4 등은 Tg 이상에서 α 급변, 미반영)",
        "delta_T = T_현재 − T_무응력기준(경화/리플로우 기준온도), 자유 경계·소변형",
        *_source_assumptions(si),
    ]
    hash_payload = {"laminate": payload, "delta_T": delta_t, "panel": panel}
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
