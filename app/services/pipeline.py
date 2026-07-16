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
