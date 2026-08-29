# 순응 중간층의 부분합성 굽힘 (Newmark 부분상호작용) — CLT 가 원리적으로 못 보는 것 (계획서 §19.12)
"""CLT 는 모든 층이 **완전합성**으로 함께 굽는다고 가정한다(평면 유지). 그런데 폴더블
스택처럼 무른 중간층(OCA·PSA)이 끼면 부분적층들이 서로 미끄러져 굽힘강성이 훨씬 낮다.

**순응층이 여러 개일 수 있다.** 유리/OCA/유리/OCA/유리 같은 표준 폴더블 스택은 계면이
둘이다. 2층 폐형해로 하나만 모델링하면 나머지 미끄러짐이 통째로 사라져 실측 6.6배
(L=1mm 에서 15배) 비보수였다(적대 검증 PC2-01). 그래서 **N개 부분적층 + N−1개 계면**의
Newmark 정해를 푼다.

    T_j = 계면 j 까지 누적 전달력,  N_i = T_{i−1} − T_i   (∑N_i = 0 자동)
    T″ = R T + p·M,   R = diag(K)·S,   p_j = K_j d_j / EI₀
    S  = Bᵀ diag(1/EA) B + d dᵀ / EI₀        (대칭 양정)
    EI₀ = ∑EI_i,  d_j = z_{j+1} − z_j,  K_j = G_j / t_j

R = diag(K)·S 는 √K S √K 와 닮음이라 **대칭 고유분해(eigh)** 로 풀 수 있다 — 결정론적이다.
정상해 T_p = −R⁻¹pM 에 모드별 cosh 를 얹고 T(±L/2)=0 을 걸면 중앙처짐이 닫힌 형태로 나온다.

    δ = [(M + φT_p)L²/8 + φ·h(R)·(−T_p)] / EI₀,   φ = zᵀB,  h(λ) = (1 − sech(√λ L/2))/λ
    EI_eff = M L² / (8δ)                            (중앙처짐 등가 강성)

**2층으로 환원하면 기존 폐형해와 12자리까지 같다** — f = 1 − 2(1 − sech X)/X²,
EI_eff = EI₀/(1 − r·f). 독립 유한요소(층별 축변위 + 공유 처짐 + 분포 전단스프링)와
2·3·4 부분적층 전 구간에서 상대오차 ≤2e-5 로 일치하고, K→0 은 ∑EI_i, K→∞ 는 CLT 로 닫힌다.

**규약은 1D(A11·B11·D11) 로 통일한다** (적대 검증 PC2-02). 부분적층의 축강성은 A11,
축력이 작용하는 자체 중립축은 z_mid + B11/A11, 그 축 기준 굽힘강성은 D11 − B11²/A11 이다.
이 조합이라야 평행축 항등식이 전체 적층의 D11 − B11²/A11 과 **정확히** 닫힌다(실측 12자리,
비대칭 부분적층 포함). 전에는 EA 를 자유 횡수축 Ex·h 로, EI 를 3×3 D* 로 섞어 써서
항등식이 닫히지 않았고 d 가 최대 69% 부풀었다.

실측: UTG(30µm)/OCA(50µm)/UTG, OCA G=0.3 MPa, L=10mm 에서 f=56.1% 로 **CLT 가
굽힘강성을 10.35배 과대평가**한다. 유리/OCA/유리/OCA/유리 는 같은 스팬에서 **22.30배**다
(2층 폐형해로 하나만 보던 시절엔 3.40배라고 답했다).
"""
from __future__ import annotations

import math

import numpy as np

from app.solver import abd as ABD
from app.solver import plate_navier as NAV
from app.solver import response as RESP

# g(X) = (1 − sech X)/X² 를 급수로 바꾸는 지점. X 가 이보다 작으면 1 − sech X 가
# 상쇄되어 자릿수를 잃는다(K→0 극한에서 실측 58배 오차가 났다).
SMALL_X = 1.0e-3
BIG_X = 350.0                    # cosh 오버플로 구간 — sech = 0 으로 둔다


def _g_of_x(x: np.ndarray) -> np.ndarray:
    """(1 − sech X)/X² — 작은 X 는 급수, 큰 X 는 1/X²."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    series = 0.5 - 5.0 * x2 / 24.0 + 61.0 * x2 * x2 / 720.0
    safe = np.where(x > BIG_X, 1.0, np.maximum(x, SMALL_X))
    direct = np.where(x > BIG_X, 1.0 / x2, (1.0 - 1.0 / np.cosh(safe)) / (safe * safe))
    return np.where(x < SMALL_X, series, direct)


def sublaminate_ea_ei(qbars: list[np.ndarray], thicknesses: list[float],
                      i0: int, i1: int) -> tuple[float, float]:
    """plies[i0:i1] 의 (자유 횡수축 축강성 Ex·h, 축소 굽힘강성 D*11) — 단위 폭당.

    **부분상호작용 모델에는 쓰지 않는다** — 그쪽은 1D 규약(sublaminate_1d)이라야 평행축
    항등식이 닫힌다(적대 검증 PC2-02). 이 함수는 자유 폭 유효계수가 맞는 곳
    (면재 주름 응력의 E_f, 자유단 박리의 축탄성계수)에만 남는다.
    """
    sub_t = list(thicknesses[i0:i1])
    z = ABD.z_coordinates(sub_t)
    A, B, D = ABD.abd_matrices(list(qbars[i0:i1]), z)
    alpha, _, delta = RESP.compliance_blocks(A, B, D)
    h = float(sum(sub_t))
    ea = float(RESP.effective_constants(alpha, delta, h)["membrane"]["Ex"]) * h
    ei = float(NAV.reduced_bending_stiffness(A, B, D)[0, 0])
    return ea, ei


def sublaminate_1d(qbars: list[np.ndarray], thicknesses: list[float],
                   i0: int, i1: int, z_start: float) -> tuple[float, float, float]:
    """plies[i0:i1] 의 (축강성 A11, 자체 중립축 기준 굽힘강성, 전역 좌표계 중립축 위치).

    1D 규약이다. 축강성 A11, 중립축 오프셋 B11/A11, 굽힘강성 D11 − B11²/A11 —
    이 셋이라야 평행축 항등식이 전체 적층과 정확히 닫힌다(§19.12, 적대 검증 PC2-02).
    """
    sub_t = list(thicknesses[i0:i1])
    z_local = ABD.z_coordinates(sub_t)
    A, B, D = ABD.abd_matrices(list(qbars[i0:i1]), z_local)
    a11, b11, d11 = float(A[0, 0]), float(B[0, 0]), float(D[0, 0])
    if a11 <= 0.0:
        return 0.0, d11, z_start + 0.5 * float(sum(sub_t))
    ei = d11 - b11 * b11 / a11
    z_neutral = z_start + 0.5 * float(sum(sub_t)) + b11 / a11
    return a11, ei, z_neutral


def partial_interaction(ea: list[float], ei: list[float], z: list[float],
                        k_shear: list[float], span: float,
                        ei_core: float = 0.0, ei_full_clt: float | None = None) -> dict:
    """N개 부분적층 + N−1개 전단 계면의 Newmark 정해 (단위 폭당).

    ea/ei/z 는 부분적층별 축강성·자체 굽힘강성·중립축 위치. k_shear[j] 는 계면 j 의
    단위길이당 전단강성 G/t. ei_core 는 순응층 자신의 굽힘강성 합(공유 곡률로 함께 굽는다).
    """
    ea = np.asarray(ea, dtype=float)
    ei = np.asarray(ei, dtype=float)
    z = np.asarray(z, dtype=float)
    ks = np.asarray(k_shear, dtype=float)
    n = len(ea)
    if n < 2 or len(ks) != n - 1 or span <= 0.0:
        return {"composite_action": None, "reason": "부분적층 2개 이상과 스팬이 필요하다"}
    if not np.all(np.isfinite(ea)) or np.any(ea <= 0.0):
        return {"composite_action": None, "reason": "부분적층 축강성이 0 이하"}
    ei0 = float(ei.sum()) + float(ei_core)
    if ei0 <= 0.0:
        return {"composite_action": None, "reason": "굽힘강성이 0 이하"}

    z_bar = float((ea * z).sum() / ea.sum())
    ei_full = ei0 + float((ea * (z - z_bar) ** 2).sum())    # 완전합성(평행축) 상한
    delta_ei = ei_full - ei0
    if delta_ei <= 0.0:
        return {"composite_action": 1.0, "EI_layered": ei0, "EI_full": ei_full,
                "EI_effective": ei_full, "alpha": None, "alpha_L": None,
                "reason": "평행축 기여가 없어 부분합성 여지가 없다"}
    r_ratio = delta_ei / ei_full

    m = n - 1
    B = np.zeros((n, m))
    for j in range(m):
        B[j, j] = -1.0
        B[j + 1, j] = 1.0
    d = np.diff(z)
    S = B.T @ np.diag(1.0 / ea) @ B + np.outer(d, d) / ei0

    if np.all(ks <= 0.0):                       # 전단 결합이 전혀 없다 = 완전분리
        return {"composite_action": 0.0, "EI_layered": ei0, "EI_full": ei_full,
                "EI_effective": ei0, "alpha": 0.0, "alpha_L": 0.0,
                "d": float(np.sqrt(delta_ei * float((1.0 / ea).sum()))) if n == 2 else None,
                "interface_count": m, "alphas": [0.0] * m}
    ks = np.maximum(ks, 0.0)
    if np.any(ks <= 0.0):                       # 하나라도 0 이면 그 계면은 완전 미끄럼
        ks = np.maximum(ks, np.max(ks) * 1e-300)

    sk = np.sqrt(ks)
    W = (sk[:, None] * S) * sk[None, :]         # √K S √K — 대칭 양정, eigh 로 결정론적
    lam, U = np.linalg.eigh(W)
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        return {"composite_action": None, "reason": "부분상호작용 행렬이 양정이 아니다"}

    R = np.diag(ks) @ S
    p = ks * d / ei0
    try:
        t_p = -np.linalg.solve(R, p)            # M = 1 로 정규화 (모두 M 에 선형)
    except np.linalg.LinAlgError:
        return {"composite_action": None, "reason": "부분상호작용 행렬이 특이하다"}

    alphas = np.sqrt(lam)
    x_mode = alphas * span / 2.0
    h_lam = _g_of_x(x_mode) * span * span / 4.0          # h(λ) = (1 − sech X)/λ
    h_w = U @ np.diag(h_lam) @ U.T
    h_r = (sk[:, None] * h_w) / sk[None, :]              # √K h(W) √K⁻¹
    phi = z @ B                                          # = d
    deflection = ((1.0 + float(phi @ t_p)) * span * span / 8.0
                  + float(phi @ (h_r @ (-t_p)))) / ei0
    if not math.isfinite(deflection) or deflection <= 0.0:
        return {"composite_action": None, "reason": "처짐 해가 유효하지 않다"}
    ei_eff = span * span / (8.0 * deflection)
    ei_eff = min(max(ei_eff, ei0), ei_full)              # 두 물리 경계 안으로
    f = (1.0 - ei0 / ei_eff) / r_ratio if ei_eff > 0.0 else 0.0

    out = {
        "composite_action": float(min(max(f, 0.0), 1.0)),
        "EI_layered": ei0,
        "EI_full": ei_full,
        "EI_effective": ei_eff,
        "interface_count": m,
        "alphas": [float(a) for a in alphas],
        "alpha_L": [float(a * span) for a in alphas],
        "alpha": float(alphas.min()),                    # 가장 느리게 붙는 모드가 지배한다
        "d": float(np.sqrt(delta_ei * float((1.0 / ea).sum()))) if n == 2 else None,
        "face_distances": [float(v) for v in d],
        "neutral_axes": [float(v) for v in z],
    }
    if ei_full_clt is not None and ei_full_clt > 0.0:
        out["EI_full_CLT"] = float(ei_full_clt)
        out["assembly_residual"] = float(ei_full / ei_full_clt)
    return out


def composite_action(ea1: float, ei1: float, ea2: float, ei2: float,
                     ei_core: float, ei_full: float,
                     g_core: float, t_core: float, span: float) -> dict:
    """2층 전용 얇은 껍데기 — 일반해에 평행축으로 역산한 중립축 거리를 넘긴다.

    기존 호출 규약을 유지한다. ei_full 은 CLT 값(D11)으로 받아 보고용으로만 쓰고,
    모델 내부 상한은 평행축 조립값을 쓴다(두 값이 어긋나면 assembly_residual 로 드러난다).
    """
    ei_layered = ei1 + ei2 + ei_core
    if ea1 <= 0.0 or ea2 <= 0.0 or ei_layered <= 0.0 or span <= 0.0:
        return {"composite_action": None, "reason": "강성 또는 스팬이 0 이하"}
    delta_ei = ei_full - ei_layered
    if delta_ei <= 0.0:
        return {"composite_action": 1.0, "EI_layered": ei_layered, "EI_full": ei_full,
                "EI_effective": ei_full, "alpha": None, "alpha_L": None,
                "reason": "평행축 기여가 없어 부분합성 여지가 없다"}
    d = math.sqrt(delta_ei * (ea1 + ea2) / (ea1 * ea2))
    k = (g_core / t_core) if (g_core > 0.0 and t_core > 0.0) else 0.0
    res = partial_interaction([ea1, ea2], [ei1, ei2], [0.0, d], [k], span,
                              ei_core=ei_core, ei_full_clt=ei_full)
    if isinstance(res.get("alpha_L"), list):        # 계면 하나 — 스칼라로 되돌린다
        res["alpha_L"] = res["alpha_L"][0] if res["alpha_L"] else None
    return res


CORE_STIFFNESS_RATIO = 10.0      # 적층 최대 횡전단강성의 1/10 미만이면 순응층으로 본다


def transverse_shear_modulus(ply) -> float:
    """굽힘축 x 에 대한 ply 의 횡전단강성 G_xz = G13·cos²θ + G23·sin²θ.

    각도 변환을 빼먹으면 90° 로 배치한 이방 코어(허니콤 리본방향·이방 접착층)에서
    G23 대신 G13 을 써 실측 10배 과대평가했다(적대 검증 PC2-07). 변환식은
    interlaminar._qbar_transverse 와 같다.
    """
    g13 = ply.g13 if ply.g13 is not None else ply.G12
    g23 = ply.g23 if getattr(ply, "g23", None) is not None else g13
    c = math.cos(math.radians(ply.angle_deg)) ** 2
    return float(g13 * c + g23 * (1.0 - c))


def detect_compliant_cores(plies) -> list[tuple[int, int]]:
    """순응층 **구간들**을 전부 찾는다 (각 구간은 양끝 포함, 안쪽 순서대로).

    적층 전체 최대 횡전단강성의 1/CORE_STIFFNESS_RATIO 미만인 ply 를 순응층으로 보고,
    양옆이 강성층으로 막힌 연속 구간만 인정한다. **최외곽 ply 강성에 의존하지 않는다** —
    전에는 문턱을 sqrt(g_min·min(g[0],g[-1])) 로 잡아 바깥에 무른 보호코팅·PSA 가
    붙으면 문턱이 무너져 내부 코어를 통째로 놓쳤다(적대 검증 PC2-02).
    """
    n = len(plies)
    if n < 3:
        return []
    g_of = [transverse_shear_modulus(p) for p in plies]
    g_max = max(g_of)
    if g_max <= 0.0:
        return []
    thresh = g_max / CORE_STIFFNESS_RATIO
    runs: list[tuple[int, int]] = []
    i = 1
    while i <= n - 2:
        if g_of[i] < thresh:
            j = i
            while j + 1 <= n - 2 and g_of[j + 1] < thresh:
                j += 1
            runs.append((i, j))                  # 양옆(i−1, j+1)은 정의상 강성층이다
            i = j + 2
        else:
            i += 1
    return runs


def detect_compliant_core(plies) -> tuple[int, int] | None:
    """가장 무른 순응층 구간 하나 (역호환용). 여러 개면 가장 무른 것을 준다."""
    runs = detect_compliant_cores(plies)
    if not runs:
        return None
    g_of = [transverse_shear_modulus(p) for p in plies]
    return min(runs, key=lambda r: (min(g_of[r[0]:r[1] + 1]), r[0]))


def core_shear_and_thickness(plies, lo: int, hi: int,
                             shear_scale: list[float] | None = None) -> tuple[float, float]:
    """코어 구간의 등가 횡전단강성과 총 두께. **직렬 조화평균**이다.

    전단 유연성 t/G 가 층별로 더해지므로 G_eq = Σt_i / Σ(t_i/G_i) 다.
    shear_scale 이 있으면 ply 별 배수를 곱한다(점탄성 이완).
    """
    t_total = 0.0
    compliance = 0.0
    for i in range(lo, hi + 1):
        g = transverse_shear_modulus(plies[i])
        if shear_scale is not None:
            g *= float(shear_scale[i])
        t_i = plies[i].thickness
        t_total += t_i
        compliance += t_i / g if g > 0 else math.inf
    g_eq = t_total / compliance if compliance > 0 and math.isfinite(compliance) else 0.0
    return g_eq, t_total


def build_partial_model(plies, qbars, thicknesses, runs, span,
                        ei_full_clt: float | None = None,
                        shear_scale: list[float] | None = None) -> dict:
    """순응층 구간 목록에서 곧바로 부분상호작용 해를 만든다.

    강성 구간(면재)들이 부분적층이 되고, 각 순응층 구간이 그 사이 전단 계면이 된다.
    shear_scale 은 ply 별 횡전단강성 배수다(점탄성 이완 상태를 넘길 때 쓴다).
    """
    n = len(thicknesses)
    z_glob = ABD.z_coordinates(list(thicknesses))
    core = set()
    for lo, hi in runs:
        core |= set(range(lo, hi + 1))
    faces: list[tuple[int, int]] = []
    cur: list[int] = []
    for i in range(n):
        if i in core:
            if cur:
                faces.append((cur[0], cur[-1] + 1))
                cur = []
        else:
            cur.append(i)
    if cur:
        faces.append((cur[0], cur[-1] + 1))
    if len(faces) < 2:
        return {"composite_action": None, "reason": "순응층이 면재를 가르지 못한다"}
    ea, ei, zc = [], [], []
    for i0, i1 in faces:
        a, e, zz = sublaminate_1d(qbars, list(thicknesses), i0, i1, float(z_glob[i0]))
        ea.append(a)
        ei.append(e)
        zc.append(zz)
    ei_core = 0.0
    ks = []
    for lo, hi in runs:
        _a, e_c, _z = sublaminate_1d(qbars, list(thicknesses), lo, hi + 1, float(z_glob[lo]))
        ei_core += e_c
        g_eq, t_eq = core_shear_and_thickness(plies, lo, hi, shear_scale)
        ks.append(g_eq / t_eq if t_eq > 0.0 else 0.0)
    res = partial_interaction(ea, ei, zc, ks, span, ei_core=ei_core, ei_full_clt=ei_full_clt)
    res["faces"] = [[i0, i1 - 1] for i0, i1 in faces]
    res["cores"] = [[lo, hi] for lo, hi in runs]
    res["core_shear"] = [
        {"indices": list(range(lo, hi + 1)),
         "G_transverse": core_shear_and_thickness(plies, lo, hi, shear_scale)[0],
         "thickness": core_shear_and_thickness(plies, lo, hi, shear_scale)[1]}
        for lo, hi in runs]
    return res
