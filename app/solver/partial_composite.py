# 순응 중간층의 부분합성 굽힘 (shear-lag) — CLT 가 원리적으로 못 보는 것 (계획서 §19.12)
"""CLT 는 모든 층이 **완전합성**으로 함께 굽는다고 가정한다(평면 유지). 그런데 폴더블
스택처럼 무른 중간층(OCA·PSA)이 끼면 두 면재가 서로 미끄러져 굽힘강성이 훨씬 낮다.

    α² = (G_c/t_c)·(1/EA₁ + 1/EA₂ + d²/EI_layered)
    f  = 1 − 2(1 − sech X)/X²,  X = αL/2            (합성도: 0=완전분리, 1=완전합성)
    EI_eff = EI_layered / (1 − r·f),  r = (EI_full − EI_layered)/EI_full

    **결합이 조화형이지 선형이 아니다.** 처음엔 EI_eff = EI_layered + f(EI_full − EI_layered)
    로 썼는데 두 극한(f=0, f=1)에서만 우연히 맞고 그 사이가 크게 틀렸다 — 적대 검증에서
    2.4~6.1배 과대(비보수)로 잡혔다. Newmark 부분상호작용 ODE
    N″ − α²N = −α²N_p 를 풀어 곡률 κ(x) = (M/EI₀)[(1−r) + r·cosh(αx)/cosh(X)] 을 얻고
    중앙처짐으로 적분하면 **1/EI 가 f 에 선형**임이 나온다(sympy 재유도로 확인).

**두 극한이 정확히 닫힌다** — G_c→0 이면 f=0(각자 굼), G_c→∞ 이면 f=1(CLT 와 일치).
실측: UTG(30µm)/OCA(50µm)/UTG, OCA G=0.3 MPa, L=10mm 에서 f=57.4% 로
**CLT 가 굽힘강성을 10.09배 과대평가**한다. 스팬 의존이 강해 L=1mm 면 22.1배가 된다.

결합은 **조화형** `EI_eff = EI_layered/(1 − r·f)` 다(r = ΔEI/EI_full). 선형결합
`EI_layered + f·ΔEI` 는 두 극한만 맞고 중간에서 최대 6배 비보수다(적대 검증 PC-01).

d² 를 직접 구하지 않는다. EI_full − EI_layered 가 정확히 평행축 항
(EA₁EA₂/(EA₁+EA₂))·d² 이므로 거기서 역산하면 두 값과 항상 정합한다 — 중립축을 따로
계산해 생기는 불일치가 없다.
"""
from __future__ import annotations

import math

import numpy as np

from app.solver import abd as ABD
from app.solver import material as MAT
from app.solver import plate_navier as NAV
from app.solver import response as RESP


def sublaminate_ea_ei(qbars: list[np.ndarray], thicknesses: list[float],
                      i0: int, i1: int) -> tuple[float, float]:
    """plies[i0:i1] 의 (축강성 EA, 자체 중립축 기준 굽힘강성 EI) — 단위 폭당.

    비대칭 부분적층은 자유단에서 굽으므로 **축소 굽힘강성 D* = D − B·A⁻¹·B** 가
    "혼자 굽을 때" 실제로 남는 강성이다.
    """
    sub_t = list(thicknesses[i0:i1])
    z = ABD.z_coordinates(sub_t)
    A, B, D = ABD.abd_matrices(list(qbars[i0:i1]), z)
    alpha, _, delta = RESP.compliance_blocks(A, B, D)
    h = float(sum(sub_t))
    ea = float(RESP.effective_constants(alpha, delta, h)["membrane"]["Ex"]) * h
    ei = float(NAV.reduced_bending_stiffness(A, B, D)[0, 0])
    return ea, ei


def composite_action(ea1: float, ei1: float, ea2: float, ei2: float,
                     ei_core: float, ei_full: float,
                     g_core: float, t_core: float, span: float) -> dict:
    """부분합성도 f 와 유효 굽힘강성."""
    ei_layered = ei1 + ei2 + ei_core
    delta_ei = ei_full - ei_layered
    if ea1 <= 0.0 or ea2 <= 0.0 or ei_layered <= 0.0 or span <= 0.0:
        return {"composite_action": None, "reason": "강성 또는 스팬이 0 이하"}
    if delta_ei <= 0.0:
        # 평행축 항이 없다 = 면재가 서로 떨어져 있지 않다(단층 등) → 부분합성이 무의미
        return {"composite_action": 1.0, "EI_layered": ei_layered, "EI_full": ei_full,
                "EI_effective": ei_full, "alpha": None, "alpha_L": None,
                "reason": "평행축 기여가 없어 부분합성 여지가 없다"}
    r_ratio = delta_ei / ei_full
    d2 = delta_ei * (ea1 + ea2) / (ea1 * ea2)      # 두 면재 중립축 거리² (항등식에서 역산)
    if g_core <= 0.0 or t_core <= 0.0:
        return {"composite_action": 0.0, "EI_layered": ei_layered, "EI_full": ei_full,
                "EI_effective": ei_layered, "alpha": 0.0, "alpha_L": 0.0,
                "d": math.sqrt(d2)}
    a2 = (g_core / t_core) * (1.0 / ea1 + 1.0 / ea2 + d2 / ei_layered)
    alpha_v = math.sqrt(a2)
    x = alpha_v * span / 2.0
    if x <= 1e-8:
        f = 0.0
    elif x > 350.0:
        f = 1.0 - 2.0 / (x * x)          # sech 언더플로 구간
    else:
        f = 1.0 - 2.0 * (1.0 - 1.0 / math.cosh(x)) / (x * x)
    denom = 1.0 - r_ratio * f
    ei_eff = ei_layered / denom if denom > 1e-300 else ei_full
    return {"composite_action": f, "EI_layered": ei_layered, "EI_full": ei_full,
            "EI_effective": min(ei_eff, ei_full),
            "alpha": alpha_v, "alpha_L": alpha_v * span, "d": math.sqrt(d2)}


CORE_STIFFNESS_RATIO = 10.0      # 이웃이 이만큼 뻣뻣해야 순응 코어로 인정한다


def detect_compliant_core(plies) -> tuple[int, int] | None:
    """가장 무른 중간층 **구간** (lo, hi) 를 찾는다 (양끝 포함).

    코어를 여러 ply 로 쪼개 모델링하는 것은 흔한 관례라(중앙 절점 배치 등) 인접한
    동등 순응층은 하나의 덩어리로 묶는다 — 단일 층만 보면 이웃이 같은 코어라
    "순응층 없음"으로 답해 버렸다(적대 검증 NC-07).
    """
    n = len(plies)
    if n < 3:
        return None
    g_of = [(p.g13 if p.g13 is not None else p.G12) for p in plies]
    inner = range(1, n - 1)
    k = min(inner, key=lambda i: (g_of[i], i))
    if g_of[k] <= 0.0:
        return None
    # 구간 경계는 **척도 불변 기하평균**으로 가른다 — 최연층과 확실한 면재(최외곽 ply)
    # 사이 기하평균보다 무른 인접 내부층은 같은 코어 덩어리다. 고정 배수를 쓰면
    # 등급형 코어(서로 다른 두 접착층)를 놓친다.
    face_g = min(g_of[0], g_of[n - 1])
    if face_g <= 0.0:
        return None
    thresh = math.sqrt(g_of[k] * face_g)
    lo = hi = k
    while lo - 1 >= 1 and g_of[lo - 1] <= thresh:
        lo -= 1
    while hi + 1 <= n - 2 and g_of[hi + 1] <= thresh:
        hi += 1
    neighbours = min(g_of[lo - 1], g_of[hi + 1])
    g_run = max(g_of[lo:hi + 1])
    return (lo, hi) if neighbours >= CORE_STIFFNESS_RATIO * g_run else None


def core_shear_and_thickness(plies, lo: int, hi: int) -> tuple[float, float]:
    """코어 구간의 등가 횡전단강성과 총 두께. **직렬 조화평균**이다.

    전단 유연성 t/G 가 층별로 더해지므로 G_eq = Σt_i / Σ(t_i/G_i) 다.
    """
    t_total = 0.0
    compliance = 0.0
    for i in range(lo, hi + 1):
        g = plies[i].g13 if plies[i].g13 is not None else plies[i].G12
        t_i = plies[i].thickness
        t_total += t_i
        compliance += t_i / g if g > 0 else math.inf
    g_eq = t_total / compliance if compliance > 0 and math.isfinite(compliance) else 0.0
    return g_eq, t_total
