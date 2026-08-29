# 단순지지 직교이방 판 Navier 폐형해 — 좌굴 임계하중과 고유진동수, D* 축소강성 (계획서 §17.5.2)
"""가정: 4변 단순지지, specially orthotropic (D16=D26≈0).

좌굴은 모드 스캔이 경계에 닿으면 **적응 확장**한다 — 고정 상한(10)은 장판·직교이방에서
참 최소값을 놓쳐 N_cr을 비보수적으로 과대평가하기 때문(적대 검증 NAV-2). 확장은 결정론적
배가이며 MAX_SCAN에서 멈추고 boundary 플래그로 알린다.

비대칭 적층은 축소 굽힘강성 D* = D − B A⁻¹ B 로 근사(표준), D16/D26 유의 시 비보수적일 수
있음을 호출부(W130)에서 경고한다.
"""
from __future__ import annotations

import math

import numpy as np

MODE_SCAN = 10        # 초기 스캔 한계
MAX_SCAN = 160        # 적응 확장 상한 (결정론·유한 시간 보장)


def reduced_bending_stiffness(A: np.ndarray, B: np.ndarray, D: np.ndarray) -> np.ndarray:
    """D* = D − B A⁻¹ B (비대칭 적층의 굽힘 등가 강성, 계획서 §17.5.2)."""
    return D - B @ np.linalg.solve(A, B)


def _navier_term(Dm: np.ndarray, m: int, n: int, a: float, b: float) -> float:
    am, bn = m / a, n / b
    return (Dm[0, 0] * am**4 + 2.0 * (Dm[0, 1] + 2.0 * Dm[2, 2]) * am * am * bn * bn
            + Dm[1, 1] * bn**4)


def _scan_min(Dm: np.ndarray, a: float, b: float, load_ratio: float, scan: int):
    """(N_cr, m, n) 최소 — 스캔 내 유효 모드가 없으면 None."""
    best = None
    for m in range(1, scan + 1):
        for n in range(1, scan + 1):
            denom = (m / a) ** 2 + load_ratio * (n / b) ** 2
            if denom <= 0:            # 해당 모드에서 압축이 지배하지 않음 → 좌굴 정의 안 됨
                continue
            ncr = math.pi ** 2 * _navier_term(Dm, m, n, a, b) / denom
            if best is None or ncr < best[0]:
                best = (ncr, m, n)
    return best


MODE_HARD_CAP = 4096      # 다중해상도 탐색 상한 (세장 판은 m_opt 가 수백까지 간다)
MODE_DENSE = 64           # 이 이하는 전수 탐색
MODE_GEOM_RATIO = 1.15    # 그 위는 기하 격자
MODE_REFINE = 24          # 최선 후보 주변 국소 정밀화 반경
MODE_N_MAX = 24           # n 방향 상한 (n 은 낮은 차수에서 지배한다)


def _mode_grid(hard: int) -> list[int]:
    """고정 다중해상도 격자 — 저차 전수 + 기하 격자. 개수가 입력에 의존하지 않는다."""
    cands = set(range(1, min(MODE_DENSE, hard) + 1))
    v = float(MODE_DENSE)
    while v < hard:
        v *= MODE_GEOM_RATIO
        cands.add(min(int(v), hard))
    return sorted(cands)


def _min_over_modes(fn, hard: int = MODE_HARD_CAP, n_max: int = MODE_N_MAX):
    """fn(m, n) → 값 또는 None. 다중해상도 격자 + 국소 정밀화로 최소를 찾는다.

    세장 판은 임계 m 이 수백에 이른다(실측 [90]4 4000×40 에서 m=311). 정사각 격자
    전수 탐색을 상한 160 에서 끊으면 **N_cr 이 1.8배 과대**로 나온다(적대 검증
    PC2-02/PC2-03). 격자는 고정이라 결정론 계약을 지킨다.
    """
    grid = _mode_grid(hard)
    best = None
    for n in range(1, n_max + 1):
        local = None
        for m in grid:
            val = fn(m, n)
            if val is None:
                continue
            if local is None or val < local[0]:
                local = (val, m)
        if local is None:
            continue
        m0 = local[1]
        for m in range(max(1, m0 - MODE_REFINE), min(hard, m0 + MODE_REFINE) + 1):
            val = fn(m, n)
            if val is None:
                continue
            if val < local[0]:
                local = (val, m)
        if best is None or local[0] < best[0]:
            best = (local[0], local[1], n)
    return best


def buckling_ncr_fsdt(Dm: np.ndarray, a: float, b: float, load_ratio: float,
                      a55: float, a44: float, hard: int = MODE_HARD_CAP) -> dict:
    """횡전단 유연성을 **모드마다** 걸고 다시 최소화한 임계하중.

    N_fsdt(m,n) = N_cr(m,n)/(1 + R_s(m,n)) 의 전 모드 최소다. CLT 최소 모드에서만
    보정을 걸면 전단 유연한 샌드위치에서 실측 최대 1.53배(열 경로는 111배) 비보수다
    — FSDT 임계 모드가 고차 m 으로 옮겨가기 때문이다(적대 검증 PC2-01 계열).
    m→∞ 에서 N_fsdt → A55 로 포화하므로 **코어 전단 크림핑 한계가 자동으로 잡힌다**.
    """
    from app.solver.interlaminar import shear_flexibility_ratio

    def fn(m, n):
        denom = (m / a) ** 2 + load_ratio * (n / b) ** 2
        if denom <= 0:
            return None
        ncr = math.pi ** 2 * _navier_term(Dm, m, n, a, b) / denom
        rs = shear_flexibility_ratio(Dm, a55, a44, a, b, m, n)
        if not math.isfinite(rs) or rs < 0:
            return None
        return ncr / (1.0 + rs)

    best = _min_over_modes(fn, hard)
    if best is None:
        return {"N_cr": None, "mode_m": None, "mode_n": None, "mode_scan": hard,
                "boundary": False, "reason": "압축 지배 모드가 없습니다"}
    val, m, n = best
    denom = (m / a) ** 2 + load_ratio * (n / b) ** 2
    ncr_clt = math.pi ** 2 * _navier_term(Dm, m, n, a, b) / denom
    return {"N_cr": val, "N_cr_clt_at_mode": ncr_clt, "mode_m": m, "mode_n": n,
            "mode_scan": hard, "boundary": m >= hard or n >= MODE_N_MAX,
            "R_s_at_mode": ncr_clt / val - 1.0 if val > 0 else None}


def buckling_ncr(Dm: np.ndarray, a: float, b: float, load_ratio: float = 0.0,
                 mode_scan: int = MODE_SCAN, max_scan: int = MAX_SCAN) -> dict:
    """2축 압축(Nx, Ny=R·Nx — 압축 양수) 최소 임계 N_x,cr과 모드 (m,n).

    N_cr(m,n) = π²·navier(m,n) / [(m/a)² + R(n/b)²].
    최소가 스캔 경계에 있으면 스캔을 배가해 재탐색(내부 최소 확보). 유효 모드가 전혀 없으면
    N_cr=None (호출부가 입력 오류로 매핑 — 적대 검증 NAV-1).
    """
    def fn(m, n):
        denom = (m / a) ** 2 + load_ratio * (n / b) ** 2
        if denom <= 0:
            return None
        return math.pi ** 2 * _navier_term(Dm, m, n, a, b) / denom

    best = _min_over_modes(fn, MODE_HARD_CAP)
    if best is None:
        return {"N_cr": None, "mode_m": None, "mode_n": None,
                "mode_scan": MODE_HARD_CAP, "boundary": False,
                "reason": "스캔 범위에서 압축 지배 모드가 없습니다 (load_ratio가 과도한 음수)"}
    return {"N_cr": best[0], "mode_m": best[1], "mode_n": best[2],
            "mode_scan": MODE_HARD_CAP,
            "boundary": best[1] >= MODE_HARD_CAP or best[2] >= MODE_N_MAX}


def buckling_ncr_fsdt_scaled(Dm: np.ndarray, a: float, b: float, load_ratio: float,
                             a55: float, a44: float, scale: float,
                             hard: int = MODE_HARD_CAP) -> dict:
    """전 ply 를 배율 s 로 키운 적층의 FSDT 전모드 최소 임계하중.

    균일 배율에서 D ∝ s³·A55 ∝ s 는 **정확**하므로(실측 8자리) 적층을 다시 조립하지 않고
    N(m,n,s) = s³·N_cr(m,n)/(1 + s²·R_s(m,n)) 를 최소화하면 된다. 임계 모드가 배율에 따라
    옮겨가는 것까지 담는다(적대 검증 PC2-01: s=1 모드 고정은 크림핑 한계를 넘겼다).
    """
    from app.solver.interlaminar import shear_flexibility_ratio
    s2, s3 = scale * scale, scale ** 3

    def fn(m, n):
        denom = (m / a) ** 2 + load_ratio * (n / b) ** 2
        if denom <= 0:
            return None
        ncr = math.pi ** 2 * _navier_term(Dm, m, n, a, b) / denom
        rs = shear_flexibility_ratio(Dm, a55, a44, a, b, m, n)
        if not math.isfinite(rs) or rs < 0:
            return None
        return s3 * ncr / (1.0 + s2 * rs)

    best = _min_over_modes(fn, hard)
    if best is None:
        return {"N_cr": None, "mode_m": None, "mode_n": None}
    return {"N_cr": best[0], "mode_m": best[1], "mode_n": best[2],
            "boundary": best[1] >= hard or best[2] >= MODE_N_MAX}


def natural_frequencies(Dm: np.ndarray, rho_areal: float, a: float, b: float,
                        n_modes: int = 5, mode_scan: int = MODE_SCAN) -> list[dict]:
    """고유진동수 f_mn = (π²/2π)·√(navier(m,n)/ρ_areal) [Hz], 낮은 순 n_modes개.

    f는 m·n에 단조 증가하므로 scan ≥ n_modes 이면 상위 n_modes개가 정확하다
    (호출부가 mode_scan=max(MODE_SCAN, n_modes)를 전달 — 적대 검증 NAV-3).
    """
    scan = max(int(mode_scan), int(n_modes))
    modes = []
    for m in range(1, scan + 1):
        for n in range(1, scan + 1):
            omega = math.pi ** 2 * math.sqrt(_navier_term(Dm, m, n, a, b) / rho_areal)
            modes.append({"m": m, "n": n, "f_hz": omega / (2.0 * math.pi)})
    modes.sort(key=lambda x: (x["f_hz"], x["m"], x["n"]))
    return modes[:n_modes]


def bend_twist_significance(Dm: np.ndarray) -> float:
    """|D16|,|D26| 유의도 = max(|D16|,|D26|)/√(D11·D22) — 0.05 초과면 Navier 비보수 가능."""
    return float(max(abs(Dm[0, 2]), abs(Dm[1, 2])) / math.sqrt(Dm[0, 0] * Dm[1, 1]))


# ── 경계조건 확장: 고정단 (Rayleigh–Ritz, 계획서 §19.5) ─────────────────────
#
# Navier 해는 단순지지(SS) 전용이다. 실제 패널은 고정단에 가까운 경우가 많고 그러면
# N_cr 이 2.5배 이상 달라진다. 고정단에는 폐형해가 없으므로 고정-고정 보 고유함수를
# 시험함수로 쓰는 1항 Rayleigh–Ritz 를 쓴다.
#
# 모드별로 두 상수만 있으면 된다: p = ∫(φ')²/∫φ², q = ∫(φ'')²/∫φ².
#   SS:  p = (mπ)²,  q = (mπ)⁴
#   CC:  q = λ_n⁴ (해석적), p 는 고정 노드 Simpson 적분
# 이 표기로 쓰면 두 경계조건이 **같은 식**을 공유한다:
#   N_cr = [D11·(qφ/pφ)/a² + 2(D12+2D66)·pψ/b² + D22·a²·qψ/(pφ·b⁴)] / (1 + R·(a²/b²)·pψ/pφ)
#   ω²   = [D11·qφ/a⁴ + 2(D12+2D66)·pφ·pψ/(a²b²) + D22·qψ/b⁴] / ρ_areal
# SS 값을 넣으면 기존 Navier 식으로 정확히 환원된다(테스트로 고정).
#
# ⚠ 1항 Ritz 는 **상계**다 — 등방 정사각 고정단에서 k = 10.74 vs 정해 10.07 (+6.6%).
#   N_cr 을 과대평가하므로 비보수다. 호출부가 반드시 경고한다.

# 지원 경계조건 — S(단순지지)·C(고정) 의 변 쌍. 뒤집힌 표기는 정규화한다.
#
# **자유변(F)은 지원하지 않는다.** 1항 Ritz 는 보의 첫 **탄성** 모드를 쓰는데, 자유변이
# 있으면 강체 모드(lambda=0: 병진·회전)가 실제 변형을 지배한다. 그걸 빼면 판을 과도하게
# 구속해 좌굴하중이 크게 과대평가된다 — 실측으로 SSSF 정사각에서 k=8.96 이 나왔는데
# 문헌은 1.40 이다(**6.4배 비보수**). 자유변을 제대로 다루려면 다항식 기저의 다항 Ritz 가
# 필요하고 그건 이 커널의 범위 밖이다. 그럴듯한 오답 대신 정직하게 거부한다.
BEAM_PAIRS = ("SS", "CC", "CS")
_PAIR_ALIAS = {"SC": "CS"}
FREE_EDGE_NOTE = ("자유변(F)은 지원하지 않습니다 — 1항 Ritz 가 강체 모드를 놓쳐 좌굴하중을 "
                  "크게 과대평가합니다(실측 SSSF 정사각에서 6.4배 비보수). S·C 만 쓰세요")
# 하위호환 별칭 (기존 인자값)
BOUNDARY_ALIAS = {"simply_supported": "SSSS", "clamped": "CCCC"}
BOUNDARIES = ("simply_supported", "clamped")

CC_LAMBDA_STEPS = 200      # 고정 이분 반복수
CC_QUAD_NODES = 8001       # 고정 Simpson 노드수 (홀수)
CLAMPED_MODE_LIMIT = 40    # 세장 판은 임계 m 이 8 을 훌쩍 넘는다 (적대 검증 PC2-02)
# 상한 8 은 "고차 모드는 cosh 가 커져 수치가 나빠진다"는 우려에서 나왔는데, 모듈 자체
# 자가검증(q = λ⁴)이 n=60 에서도 상대오차 1e-9 로 성립한다. 8 에 묶어 두면 세장 패널에서
# 최소가 경계에 걸려 필요 두께 배율이 1.39배 과소 산출됐다.
_PQ_CACHE: dict[tuple[str, int], tuple[float, float, float]] = {}


def normalize_boundary(boundary: str) -> tuple[str, str] | None:
    """boundary 문자열 → (x변 쌍, y변 쌍). 알 수 없으면 None.

    "simply_supported"·"clamped" 는 SSSS·CCCC 로 확장한다. 4글자 표기는
    **앞 두 글자가 x 방향 두 변, 뒤 두 글자가 y 방향 두 변**이다
    (예: "CCSS" = x 변 고정-고정, y 변 단순-단순).
    """
    if not isinstance(boundary, str):
        return None
    code = BOUNDARY_ALIAS.get(boundary, boundary).upper()
    if len(code) != 4:
        return None
    px, py = code[:2], code[2:]
    px, py = _PAIR_ALIAS.get(px, px), _PAIR_ALIAS.get(py, py)
    if px not in BEAM_PAIRS or py not in BEAM_PAIRS:
        return None
    return px, py


def _char_eq(pair: str, lam: float) -> float:
    """보 특성방정식 f(λ)=0. 1/cosh 형태로 써서 오버플로를 피한다."""
    sech = 0.0 if lam > 700.0 else 1.0 / math.cosh(lam)
    if pair == "CC":
        return math.cos(lam) - sech            # cos·cosh = 1
    return math.tan(lam) - math.tanh(lam)      # CS: tan = tanh


def _eigenvalue(pair: str, n: int) -> float:
    """n번째 보 고유값 (고정 반복 이분법 — 결정론)."""
    if pair == "SS":
        return n * math.pi
    if pair == "CS":
        guess = (4 * n + 1) * math.pi / 4
    else:
        guess = (2 * n + 1) * math.pi / 2
    lo, hi = guess - 0.4, guess + 0.4
    f_lo_neg = _char_eq(pair, lo) < 0
    for _ in range(CC_LAMBDA_STEPS):
        mid = 0.5 * (lo + hi)
        if (_char_eq(pair, mid) < 0) == f_lo_neg:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _shape_funcs(pair: str, lam: float):
    """(phi, phi', phi'') — 각 경계조건의 정규화 전 고유함수."""
    ch, co, sh, si = math.cosh, math.cos, math.sinh, math.sin
    if pair == "SS":
        return (lambda x: si(lam * x), lambda x: lam * co(lam * x),
                lambda x: -lam * lam * si(lam * x))
    sg = (ch(lam) - co(lam)) / (sh(lam) - si(lam))      # CC·CS 공통 형상
    return (lambda x: ch(lam * x) - co(lam * x) - sg * (sh(lam * x) - si(lam * x)),
            lambda x: lam * (sh(lam * x) + si(lam * x) - sg * (ch(lam * x) - co(lam * x))),
            lambda x: lam ** 2 * (ch(lam * x) + co(lam * x) - sg * (sh(lam * x) + si(lam * x))))


def beam_pq(pair: str, n: int) -> tuple[float, float]:
    """(p, q) = (적분(phi')^2 / 적분 phi^2,  적분(phi'')^2 / 적분 phi^2).

    Rayleigh 몫의 유일한 재료다. **q = lambda^4 가 S·C·F 어느 조합에서도 성립**하는데,
    부분적분의 경계항이 S·C 양쪽에서 0이기 때문이다. 이 항등식이 수치적분의 독립 검증이 된다(테스트로 rel 1e-8 고정).
    """
    key = (pair, n)
    if key in _PQ_CACHE:
        _lam, p, q = _PQ_CACHE[key]
        return p, q
    lam = _eigenvalue(pair, n)
    if pair == "SS":
        p, q = lam * lam, lam ** 4
        _PQ_CACHE[key] = (lam, p, q)
        return p, q
    phi, d1, d2 = _shape_funcs(pair, lam)

    def simpson(fn):
        h = 1.0 / (CC_QUAD_NODES - 1)
        s = fn(0.0) + fn(1.0)
        for i in range(1, CC_QUAD_NODES - 1):
            s += (4.0 if i % 2 else 2.0) * fn(i * h)
        return s * h / 3.0

    i0 = simpson(lambda x: phi(x) ** 2)
    p, q = simpson(lambda x: d1(x) ** 2) / i0, simpson(lambda x: d2(x) ** 2) / i0
    _PQ_CACHE[key] = (lam, p, q)
    return p, q


def clamped_pq(n: int) -> tuple[float, float]:
    """고정-고정 보 (하위호환 별칭)."""
    return beam_pq("CC", n)


def _cc_eigenvalue(n: int) -> float:
    """고정-고정 고유값 (하위호환 별칭)."""
    return _eigenvalue("CC", n)


def mode_pq(boundary: str, n: int, axis: int = 0) -> tuple[float, float]:
    """경계조건·모드번호 → (p, q). 두 경계가 같은 Rayleigh 식을 공유하게 하는 다리.

    axis: 0 = x 방향(앞 두 글자), 1 = y 방향(뒤 두 글자).
    """
    pairs = normalize_boundary(boundary)
    if pairs is None:
        raise ValueError(f"지원하지 않는 boundary: {boundary!r}")
    return beam_pq(pairs[axis], n)


def ritz_ncr(Dm: np.ndarray, a: float, b: float, load_ratio: float,
             boundary: str, m: int, n: int) -> float | None:
    """(m,n) 모드의 임계 하중. 압축이 지배하지 않으면 None."""
    pf, qf = mode_pq(boundary, m, 0)
    pp, qq = mode_pq(boundary, n, 1)
    denom = 1.0 + load_ratio * (a * a / (b * b)) * (pp / pf)
    if denom <= 0.0:
        return None
    num = (Dm[0, 0] * (qf / pf) / (a * a)
           + 2.0 * (Dm[0, 1] + 2.0 * Dm[2, 2]) * pp / (b * b)
           + Dm[1, 1] * a * a * qq / (pf * b ** 4))
    return num / denom


def ritz_omega2(Dm: np.ndarray, rho_areal: float, a: float, b: float,
                boundary: str, m: int, n: int) -> float:
    """(m,n) 모드의 각진동수 제곱."""
    pf, qf = mode_pq(boundary, m, 0)
    pp, qq = mode_pq(boundary, n, 1)
    return (Dm[0, 0] * qf / a ** 4
            + 2.0 * (Dm[0, 1] + 2.0 * Dm[2, 2]) * pf * pp / (a * a * b * b)
            + Dm[1, 1] * qq / b ** 4) / rho_areal


def scan_ritz_buckling(Dm: np.ndarray, a: float, b: float, load_ratio: float,
                       boundary: str, scan: int) -> dict:
    """(m,n) 스캔 최소 임계하중."""
    best = None
    for m in range(1, scan + 1):
        for n in range(1, scan + 1):
            v = ritz_ncr(Dm, a, b, load_ratio, boundary, m, n)
            if v is None or v <= 0.0:
                continue
            if best is None or v < best[0]:
                best = (v, m, n)
    if best is None:
        return {"N_cr": None, "mode_m": None, "mode_n": None, "mode_scan": scan,
                "boundary": False,
                "reason": "스캔 범위에서 압축 지배 모드가 없습니다 (load_ratio가 과도한 음수)"}
    return {"N_cr": best[0], "mode_m": best[1], "mode_n": best[2], "mode_scan": scan,
            "boundary": best[1] >= scan or best[2] >= scan}


def ritz_frequencies(Dm: np.ndarray, rho_areal: float, a: float, b: float,
                     boundary: str, n_modes: int, mode_scan: int) -> list[dict]:
    """경계조건별 고유진동수 (낮은 순 n_modes개). SS 를 넣으면 Navier 와 정확히 일치한다."""
    scan = max(int(mode_scan), int(n_modes))
    if normalize_boundary(boundary) != ("SS", "SS"):
        scan = min(scan, CLAMPED_MODE_LIMIT)
        scan = max(scan, min(int(n_modes), CLAMPED_MODE_LIMIT))
    out = []
    for m in range(1, scan + 1):
        for n in range(1, scan + 1):
            w2 = ritz_omega2(Dm, rho_areal, a, b, boundary, m, n)
            out.append({"m": m, "n": n, "f_hz": math.sqrt(w2) / (2.0 * math.pi)})
    out.sort(key=lambda x: (x["f_hz"], x["m"], x["n"]))
    return out[:n_modes]
