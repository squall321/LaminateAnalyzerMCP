# 내장 기준 케이스 — 에이전트 few-shot 학습·자가 검증용, 기대값은 폐형해 수식으로 산출 (계획서 §6.3, §8.1)
from __future__ import annotations

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
