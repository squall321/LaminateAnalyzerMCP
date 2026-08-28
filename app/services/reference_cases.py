# 내장 기준 케이스 — 에이전트 few-shot 학습·자가 검증용, 기대값은 폐형해 수식으로 산출 (계획서 §6.3, §8.1)
from __future__ import annotations

import math

_T300_MPA = {"type": "orthotropic_2d", "E1": 181000.0, "E2": 10300.0, "G12": 7170.0,
             "nu12": 0.28, "name": "T300/5208급 흑연/에폭시"}


def _q_consts_mpa(E1, E2, G12, nu12):
    """평면응력 감축 강성 상수 (표시 단위 MPa 그대로) — 기대값 산출 전용 폐형해."""
    nu21 = nu12 * E2 / E1
    d = 1.0 - nu12 * nu21
    return E1 / d, nu12 * E2 / d, E2 / d, G12  # Q11, Q12, Q22, Q66


def _build_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}

    # R1 — 등방 단일층 (SI_mm)
    E, nu, h = 70000.0, 0.33, 2.0
    q11 = E / (1 - nu * nu)
    cases["single_isotropic_alu"] = {
        "description": "등방 알루미늄 단일층 2mm — 가장 단순한 검증 케이스. B=0, 중립면=midplane, 준등방성 점수 1",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "single_isotropic_alu",
            "laminae": [{"thickness": h, "angle_deg": 0.0,
                         "material": {"type": "isotropic", "E": E, "nu": nu, "name": "AL6061급"}}],
        }},
        "expected": {
            "closed_form": ["A11 = E*h/(1-nu^2)", "D11 = E*h^3/(12*(1-nu^2))", "B = 0", "z_ns = 0 (zeta=0.5)"],
            "A11_N_per_mm": q11 * h,
            "D11_N_mm": q11 * h**3 / 12.0,
            "B_zero": True,
            "zeta_x": 0.5,
            "quasi_isotropy_score": 1.0,
        },
        "tolerance": {"rtol": 1e-9},
    }

    # R4 — [0/90]s 대칭 크로스플라이 (SI_mm)
    t = 0.125
    Q11, Q12, Q22, Q66 = _q_consts_mpa(_T300_MPA["E1"], _T300_MPA["E2"], _T300_MPA["G12"], _T300_MPA["nu12"])
    cases["cross_ply_symmetric"] = {
        "description": "[0/90]s 대칭 크로스플라이 (t=0.125mm ×4) — B=0, A11=A22, D11>D22(외측 0° 지배)",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "cross_ply_symmetric",
            "laminae": [{"thickness": t, "angle_deg": a, "material": dict(_T300_MPA)}
                        for a in (0.0, 90.0, 90.0, 0.0)],
        }},
        "expected": {
            "closed_form": ["A11 = A22 = 2t(Q11+Q22)", "D11 = (14Q11+2Q22)t^3/3", "D22 = (2Q11+14Q22)t^3/3", "B = 0"],
            "A11_N_per_mm": 2 * t * (Q11 + Q22),
            "D11_N_mm": (14 * Q11 + 2 * Q22) * t**3 / 3.0,
            "D22_N_mm": (2 * Q11 + 14 * Q22) * t**3 / 3.0,
            "B_zero": True,
            "zeta_x": 0.5,
            "is_symmetric_stack": True,
        },
        "tolerance": {"rtol": 1e-9},
    }

    # R3 — [0/90] 비대칭 크로스플라이 (SI_mm)
    cases["cross_ply_unsymmetric"] = {
        "description": "[0/90] 비대칭 2층 — 커플링 B11=-B22≠0, 중립면이 강성 큰 0°(하단) 쪽으로 이동",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "cross_ply_unsymmetric",
            "laminae": [{"thickness": t, "angle_deg": 0.0, "material": dict(_T300_MPA)},
                        {"thickness": t, "angle_deg": 90.0, "material": dict(_T300_MPA)}],
        }},
        "expected": {
            "closed_form": ["B11 = (Q22-Q11)t^2/2", "z_ns_x = t(Q22-Q11)/(2(Q11+Q22)) < 0 (midplane 기준)"],
            "B11_N": (Q22 - Q11) * t**2 / 2.0,
            "z_ns_x_from_midplane_mm": t * (Q22 - Q11) / (2.0 * (Q11 + Q22)),
            "is_symmetric_stack": False,
        },
        "tolerance": {"rtol": 1e-9},
    }
    # R6 — 바이메탈 열 휨 (compute_thermal_response few-shot, §17.1)
    E1b, E2b, a1b, a2b, tb, dTb, Lb = 100000.0, 50000.0, 10e-6, 20e-6, 1.0, 100.0, 100.0
    nb = E1b / E2b
    hb = 2 * tb
    kappa_b = 24.0 * (a2b - a1b) * dTb / (hb * (14.0 + nb + 1.0 / nb))   # Timoshenko (동일 ν ⇒ 판=보)
    cases["bimetal_thermal_warpage"] = {
        "description": "등두께 바이메탈(하: 저CTE 강성, 상: 고CTE 연성) ΔT=+100K — Timoshenko 폐형해 열곡률과 100×100 판 휨",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "bimetal_thermal_warpage",
            "laminae": [
                {"thickness": tb, "angle_deg": 0.0,
                 "material": {"type": "isotropic", "E": E1b, "nu": 0.3, "alpha": a1b}},
                {"thickness": tb, "angle_deg": 0.0,
                 "material": {"type": "isotropic", "E": E2b, "nu": 0.3, "alpha": a2b}},
            ]},
            "tool": "compute_thermal_response", "delta_T": dTb, "panel": {"Lx": Lb, "Ly": Lb}},
        "expected": {
            "closed_form": ["|kappa_x| = 24*(a2-a1)*dT/(h*(14+n+1/n)), n=E1/E2 (동일 ν ⇒ 판=보)",
                            "warpage_range(정사각 판, 등2축) = kappa*L^2/4"],
            "kappa_x_per_mm": kappa_b,
            "warpage_range_mm": kappa_b * Lb * Lb / 4.0,
        },
        "tolerance": {"rtol": 1e-6},
    }

    # R7 — [0/90]s FPF (recover_ply_stresses few-shot, §17.2b)
    strength = {"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0}
    cases["cross_ply_fpf"] = {
        "description": "[0/90]s CFRP Nx 인장 — first-ply-failure는 90° ply 횡인장(Yt) (고전 결과). Tsai-Wu R = 파손까지 하중 배수",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "cross_ply_fpf",
            "laminae": [{"thickness": t, "angle_deg": a,
                         "material": dict(_T300_MPA, strength=dict(strength))}
                        for a in (0.0, 90.0, 90.0, 0.0)]},
            "tool": "recover_ply_stresses", "loads": {"N": [100.0, 0.0, 0.0]}},
        "expected": {
            "fpf_ply_in": [1, 2],
            "governing_mode": "transverse_tension",
            "property": "하중을 N×R로 재호출하면 R≈1 (강도비의 정의)",
        },
        "tolerance": {"note": "모드·ply는 정확 판정, R 수치는 엔진 계산값"},
    }
    # R8 — 등방 정사각 판 좌굴 (compute_buckling few-shot, §17.5.2)
    Eb, nub, tb, Lb = 70000.0, 0.3, 2.0, 200.0
    Db = Eb * tb**3 / (12.0 * (1.0 - nub * nub))
    cases["isotropic_square_buckling"] = {
        "description": "등방 정사각 SS 판(200×200, t=2mm) 단축 압축 — 고전 폐형해 N_cr = 4π²D/b², 모드 (1,1)",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "isotropic_square_buckling",
            "laminae": [{"thickness": tb, "angle_deg": 0.0,
                         "material": {"type": "isotropic", "E": Eb, "nu": nub, "name": "AL6061급"}}]},
            "tool": "compute_buckling", "panel": {"Lx": Lb, "Ly": Lb}},
        "expected": {
            "closed_form": ["N_cr = 4*pi^2*D/b^2, D = E*h^3/(12(1-nu^2))",
                            "정사각 단축 압축의 좌굴계수 k = 4, 모드 (m,n) = (1,1)"],
            "N_cr_N_per_mm": 4.0 * math.pi**2 * Db / (Lb * Lb),
            "mode": [1, 1],
        },
        "tolerance": {"rtol": 1e-9},
    }

    # R9 — 적층 설계 규칙 모범/위반 대조 (check_design_rules few-shot, §17.5.1)
    cases["design_rules_contrast"] = {
        "description": "규칙 검사 대조 예시 — good=[45/0/-45/90]s는 전 규칙 통과, bad=[0×5/90]은 대칭·10%·연속·인접각 위반",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "design_rules_good",
            "laminae": [{"thickness": t, "angle_deg": a, "material": dict(_T300_MPA)}
                        for a in (45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0)]},
            "tool": "check_design_rules",
            "contrast_bad_layup": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0]},
        "expected": {
            "good_all_pass": True,
            "bad_hard_fails_include": ["symmetry"],
            "note": "guideline 위반은 불합격이 아니라 검토 신호 — found/why_it_matters/fix_hint를 그대로 사용자에게 전달할 것",
        },
        "tolerance": {"note": "판정형 케이스 — 수치 허용오차 없음"},
    }
    # R10 — 피로 사이클 (estimate_fatigue_life few-shot, §17.7)
    fat_mat = dict(_T300_MPA, strength={"Xt": 1500.0, "Xc": 1200.0, "Yt": 40.0, "Yc": 246.0, "S": 68.0},
                   fatigue={"model_type": "log_linear", "k": 0.1})
    cases["fatigue_reversed_cycle"] = {
        "description": "[0/90]s 완전반복(R=−1) 사이클 — 피로에서 가장 가혹한 형태. "
                       "같은 최대하중의 영-인장(R=0)보다 수명이 짧아야 한다",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "fatigue_reversed_cycle",
            "laminae": [{"thickness": t, "angle_deg": a, "material": dict(fat_mat)}
                        for a in (0.0, 90.0, 90.0, 0.0)]},
            "tool": "estimate_fatigue_life",
            "loads_max": {"N": [100.0, 0.0, 0.0]},
            "loads_min": {"N": [-100.0, 0.0, 0.0]}},
        "expected": {
            "property": "life(R=−1) < life(R=0, loads_min 생략) — 진폭이 2배이므로",
            "governing_component_in": ["sigma_1", "sigma_2", "tau_12"],
            "note": "S-N은 log_linear k=0.1 (CFRP 관례: 10^7 사이클 강도 ≈ 정적의 30%)",
        },
        "tolerance": {"note": "판정형 — 순서 관계와 지배 성분을 확인"},
    }

    # R11 — 쌍안정 경화 형상 (compute_bistable_shapes few-shot, §18.2)
    # 기대값은 2층 [0/90] 등두께의 폐형해에서 산출한다 (엔진 경로와 독립, §8 규약).
    a1, a2, dT = 0.02e-6, 22.5e-6, -150.0
    q11, q12, q22, _ = _q_consts_mpa(181000.0, 10300.0, 7170.0, 0.28)
    a_11 = t * (q11 + q22)
    a_12 = 2.0 * t * q12
    b_11 = (t * t / 2.0) * (q22 - q11)
    b_22 = -b_11
    d_11 = (t ** 3 / 3.0) * (q11 + q22)
    d_12 = (2.0 * t ** 3 / 3.0) * q12
    e_lo = (a1 * dT, a2 * dT)          # 0° ply 자유변형
    e_hi = (a2 * dT, a1 * dT)          # 90° ply
    n_1 = t * ((q11 * e_lo[0] + q12 * e_lo[1]) + (q22 * e_hi[0] + q12 * e_hi[1]))
    n_2 = t * ((q12 * e_lo[0] + q22 * e_lo[1]) + (q12 * e_hi[0] + q11 * e_hi[1]))
    m_1 = (t * t / 2.0) * ((q22 * e_hi[0] + q12 * e_hi[1]) - (q11 * e_lo[0] + q12 * e_lo[1]))
    det_a = a_11 * a_11 - a_12 * a_12
    i11, i12 = a_11 / det_a, -a_12 / det_a     # A22 = A11 이므로 (A⁻¹)11 = A11/det
    ds11 = d_11 - b_11 * i11 * b_11            # D* = D − B·A⁻¹·B
    ds12 = d_12 - b_11 * i12 * b_22
    ms1 = m_1 - b_11 * (i11 * n_1 + i12 * n_2)  # M* = M_f − B·A⁻¹·N_f
    kappa_inf = ms1 / ds11
    l_crit = (5760.0 * ds11 ** 2 * (ds11 + ds12) / (a_11 * ms1 ** 2)) ** 0.25
    cases["bistable_cross_ply"] = {
        "description": "[0/90] 비대칭 2층 CFRP를 ΔT=−150K로 냉각 — 100×100mm 판은 임계 크기를 "
                       "넘어 서로 거울상인 원통 두 개로 분기한다(쌍안정). 선형 CLT가 주는 "
                       "안장 형상은 실현되지 않는다",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "bistable_cross_ply",
            "laminae": [{"thickness": t, "angle_deg": ang,
                         "material": dict(_T300_MPA, alpha1=a1, alpha2=a2)}
                        for ang in (0.0, 90.0)]},
            "tool": "compute_bistable_shapes",
            "panel": {"Lx": 100.0, "Ly": 100.0},
            "delta_T": dT},
        "expected": {
            "bistable": True,
            "stable_count": 2,
            "critical_panel_Lx_mm": l_crit,
            "kappa_inf_per_mm": kappa_inf,
            "kappa_crit_per_mm": kappa_inf / 2.0,
            "property": "안정해 2개는 원통형이고 (κx,κy)→(−κy,−κx) 거울쌍, 에너지 동일. "
                        "불안정 안장해 1개가 함께 나온다",
            "note": "폐형해 랜드마크는 D* = D−B·A⁻¹·B, M* = M_f−B·A⁻¹·N_f 로부터 "
                    "κ_∞ = M*x/D*11, κ_c = κ_∞/2, L_crit = [5760·D*11²(D*11+D*12)/(A11·M*x²)]^(1/4). "
                    "정사각 판 전용 폐형해다 — 직사각에는 쓰지 말 것",
        },
        "tolerance": {"critical_panel_Lx_mm": 1e-8,
                      "kappa_inf_per_mm": 5e-4,
                      "note": "κ는 L→∞ 극한값이라 유한 판(100mm)에서는 0.03% 정도 차이난다"},
    }

    # R12 — 재료 전단 비선형 (solve_nonlinear_shear_response few-shot, §18.7)
    s6666 = 4.0e-8            # 1/MPa^3
    nx, n_ply = 50.0, 4
    h_pm45 = n_ply * t
    tau_expect = nx / (2.0 * h_pm45)          # [±45] 단축인장의 폐형해: τ12 = σx/2 = Nx/(2h)
    g12 = 7170.0
    secant_expect = 1.0 / (1.0 + g12 * s6666 * tau_expect ** 2)
    cases["pm45_shear_nonlinear"] = {
        "description": "[±45]s CFRP 단축인장 — 전단 지배라 재료 전단 비선형이 가장 크게 드러난다. "
                       "선형 CLT는 Ex를 크게 과대평가한다",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "pm45_shear_nonlinear",
            "laminae": [{"thickness": t, "angle_deg": ang,
                         "material": dict(_T300_MPA, shear_nonlinear={"S6666": s6666})}
                        for ang in (45.0, -45.0, -45.0, 45.0)]},
            "tool": "solve_nonlinear_shear_response",
            "loads": {"N": [nx, 0.0, 0.0]}},
        "expected": {
            "per_ply_tau12_MPa": tau_expect,
            "secant_ratio": secant_expect,
            "Gxy_secant_over_linear": 1.0,
            "property": "적층 Gxy는 전혀 변하지 않고 Ex만 떨어진다 — 45°에서 Q̄66의 Q66 의존이 "
                        "정확히 상쇄되기 때문(Q̄66 = (Q11+Q22−2Q12)/4). 전단 비선형이 적층 수준에서 "
                        "어디로 나타나는지는 적층각에 따라 완전히 다르다",
            "note": "τ12 = Nx/(2h) 는 [±45] 단축인장의 정확한 폐형해. "
                    "할선비 = 1/(1 + G12·S6666·τ12²)",
        },
        "tolerance": {"per_ply_tau12_MPa": 1e-9, "secant_ratio": 1e-9,
                      "Gxy_secant_over_linear": 1e-9},
    }

    # R13 — von Karman 대처짐 (compute_large_deflection few-shot, §18.3)
    e_iso, nu_iso, h_iso, l_iso = 70000.0, 0.3, 2.0, 500.0
    d_iso = e_iso * h_iso ** 3 / (12.0 * (1.0 - nu_iso ** 2))
    press = 0.005                                   # MPa
    w_lin = 4.0 * press * l_iso ** 4 / (math.pi ** 6 * d_iso)   # 1항 Navier 폐형해
    cases["large_deflection_isotropic"] = {
        "description": "등방 정사각 SS 판(500×500, t=2mm)에 균일압력 — 선형 해는 두께의 수십 배 "
                       "처짐을 주지만 막 효과로 실제 처짐은 훨씬 작다",
        "input": {"laminate": {
            "unit_system": "SI_mm", "name": "large_deflection_isotropic",
            "laminae": [{"thickness": h_iso, "angle_deg": 0.0,
                         "material": {"type": "isotropic", "E": e_iso, "nu": nu_iso}}]},
            "tool": "compute_large_deflection",
            "panel": {"Lx": l_iso, "Ly": l_iso},
            "pressure": press,
            "edge_condition": "movable"},
        "expected": {
            "w_center_linear_mm": w_lin,
            "beta_ratio_immovable_over_movable": (3.0 - nu_iso) / (1.0 - nu_iso),
            "property": "w_center < w_center_linear (막 강성). edge_condition='immovable'로 "
                        "다시 부르면 β가 (3−ν)/(1−ν) 배 커져 처짐이 더 작다",
            "note": "선형 극한은 1항 Navier 폐형해 w = (4/π⁶)·qL⁴/D. Levy 정해(0.004062 qL⁴/D) "
                    "대비 +2.4%가 1항 근사의 알려진 오차다",
        },
        "tolerance": {"w_center_linear_mm": 1e-9,
                      "beta_ratio_immovable_over_movable": 1e-12},
    }
    return cases


CASES = _build_cases()


def get(case_id: str | None = None) -> dict:
    """case_id 없으면 목록, 있으면 해당 케이스의 입력·기대값 전체."""
    if case_id is None:
        return {"cases": [{"case_id": k, "description": v["description"]} for k, v in CASES.items()],
                "usage": "case_id를 지정하면 완전한 입력 payload와 폐형해 기대값을 반환합니다. "
                         "input.laminate를 그대로 analyze_laminate에 넣어 자가 검증할 수 있습니다."}
    if case_id not in CASES:
        return {"error": f"알 수 없는 case_id '{case_id}'. 사용 가능: {sorted(CASES)}"}
    return {"case_id": case_id, **CASES[case_id]}
