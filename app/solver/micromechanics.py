# 미시역학 — 구성재(섬유+수지) → lamina 직교이방 물성 (계획서 §19.3)
"""materialtwin 이 수지 실측을 줘도 그걸 받아 ply 물성으로 바꿀 다리가 없어 체인이
끊겨 있었다. 기존 homogenize_layer 는 **등방 Voigt 병렬 혼합**이라 동박층 같은 혼합층에만
쓸 수 있고, 섬유/수지에서 직교이방 lamina 를 만들지 못한다.

모델:
- E1, ν12, ρ  : 혼합법칙(ROM) — 섬유 지배라 신뢰도가 높다
- E2, G12     : Halpin–Tsai(ξ 조정 가능) 또는 Chamis — **기지 지배라 불확실성이 크다**
- α1, α2      : Schapery

**Halpin–Tsai 의 ξ 극한이 곧 경계다** — ξ→0 이면 Reuss(하한), ξ→∞ 이면 Voigt(상한)로
정확히 수렴한다(테스트로 고정). 그래서 응답에 두 경계를 함께 실어 추정의 폭을 보여준다.
CFRP 처럼 강성비가 큰 계에서는 표준 ξ(E2:2, G12:1)가 문헌값보다 낮게 나오는 것이 알려져
있으므로, 실측이 있으면 ξ 를 역보정하거나 measured 물성을 쓰라고 안내한다.
"""
from __future__ import annotations

import math

XI_E2_DEFAULT = 2.0     # Halpin–Tsai 표준값 (원형 섬유 정사각 배열)
XI_G12_DEFAULT = 1.0


def voigt(p_f: float, p_m: float, v_f: float) -> float:
    """병렬(상한) 혼합법칙."""
    return v_f * p_f + (1.0 - v_f) * p_m


def reuss(p_f: float, p_m: float, v_f: float) -> float:
    """직렬(하한) 혼합법칙."""
    return 1.0 / (v_f / p_f + (1.0 - v_f) / p_m)


def halpin_tsai(p_f: float, p_m: float, v_f: float, xi: float) -> float:
    """Halpin–Tsai 반경험식. ξ→0 이면 Reuss, ξ→∞ 이면 Voigt 로 정확히 수렴한다."""
    r = p_f / p_m
    eta = (r - 1.0) / (r + xi)
    return p_m * (1.0 + xi * eta * v_f) / (1.0 - eta * v_f)


def chamis(p_f: float, p_m: float, v_f: float) -> float:
    """Chamis 식 — 기지 지배 물성에서 Halpin–Tsai 보다 다소 높게 나온다."""
    return p_m / (1.0 - math.sqrt(v_f) * (1.0 - p_m / p_f))


def lamina_from_constituents(fiber: dict, matrix: dict, v_f: float,
                             model: str = "halpin_tsai",
                             xi_e2: float = XI_E2_DEFAULT,
                             xi_g12: float = XI_G12_DEFAULT) -> dict:
    """구성재 → lamina 직교이방 물성 (SI 단위 입출력).

    fiber:  {E1, E2, G12, nu12, alpha1, alpha2, rho?}  (횡등방 섬유)
    matrix: {E, nu, alpha, rho?}                       (등방 수지)
    """
    e_m, nu_m = float(matrix["E"]), float(matrix["nu"])
    g_m = e_m / (2.0 * (1.0 + nu_m))
    ef1, ef2 = float(fiber["E1"]), float(fiber["E2"])
    gf12, nuf12 = float(fiber["G12"]), float(fiber["nu12"])

    e1 = voigt(ef1, e_m, v_f)                       # 섬유 지배 — ROM 이 정확에 가깝다
    nu12 = voigt(nuf12, nu_m, v_f)
    if model == "chamis":
        e2 = chamis(ef2, e_m, v_f)
        g12 = chamis(gf12, g_m, v_f)
    else:
        e2 = halpin_tsai(ef2, e_m, v_f, xi_e2)
        g12 = halpin_tsai(gf12, g_m, v_f, xi_g12)

    out: dict = {
        "E1": e1, "E2": e2, "G12": g12, "nu12": nu12,
        "bounds": {
            "E2": {"reuss": reuss(ef2, e_m, v_f), "voigt": voigt(ef2, e_m, v_f)},
            "G12": {"reuss": reuss(gf12, g_m, v_f), "voigt": voigt(gf12, g_m, v_f)},
        },
    }

    a_f1, a_f2 = fiber.get("alpha1"), fiber.get("alpha2")
    a_m = matrix.get("alpha")
    if a_f1 is not None and a_f2 is not None and a_m is not None:
        # Schapery — α1 은 강성 가중, α2 는 포아송 커플링을 뺀다
        num = v_f * ef1 * float(a_f1) + (1.0 - v_f) * e_m * float(a_m)
        alpha1 = num / (v_f * ef1 + (1.0 - v_f) * e_m)
        alpha2 = ((1.0 + nuf12) * float(a_f2) * v_f
                  + (1.0 + nu_m) * float(a_m) * (1.0 - v_f) - alpha1 * nu12)
        out["alpha1"], out["alpha2"] = alpha1, alpha2

    r_f, r_m = fiber.get("rho"), matrix.get("rho")
    if r_f is not None and r_m is not None:
        out["rho"] = voigt(float(r_f), float(r_m), v_f)
    return out
