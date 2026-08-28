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


def buckling_ncr(Dm: np.ndarray, a: float, b: float, load_ratio: float = 0.0,
                 mode_scan: int = MODE_SCAN, max_scan: int = MAX_SCAN) -> dict:
    """2축 압축(Nx, Ny=R·Nx — 압축 양수) 최소 임계 N_x,cr과 모드 (m,n).

    N_cr(m,n) = π²·navier(m,n) / [(m/a)² + R(n/b)²].
    최소가 스캔 경계에 있으면 스캔을 배가해 재탐색(내부 최소 확보). 유효 모드가 전혀 없으면
    N_cr=None (호출부가 입력 오류로 매핑 — 적대 검증 NAV-1).
    """
    scan = max(1, int(mode_scan))
    while True:
        best = _scan_min(Dm, a, b, load_ratio, scan)
        if best is None:
            if scan >= max_scan:
                return {"N_cr": None, "mode_m": None, "mode_n": None,
                        "mode_scan": scan, "boundary": False,
                        "reason": "스캔 범위에서 압축 지배 모드가 없습니다 (load_ratio가 과도한 음수)"}
            scan = min(scan * 2, max_scan)
            continue
        at_boundary = best[1] >= scan or best[2] >= scan
        if not at_boundary or scan >= max_scan:
            return {"N_cr": best[0], "mode_m": best[1], "mode_n": best[2],
                    "mode_scan": scan, "boundary": at_boundary}
        scan = min(scan * 2, max_scan)


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

BOUNDARIES = ("simply_supported", "clamped")
CC_LAMBDA_STEPS = 200      # 고정 이분 반복수
CC_QUAD_NODES = 8001       # 고정 Simpson 노드수 (홀수)
CLAMPED_MODE_LIMIT = 8     # 고차 모드는 cosh 가 커져 수치가 나빠진다
_CC_CACHE: dict[int, tuple[float, float, float]] = {}


def _cc_eigenvalue(n: int) -> float:
    """cos(λ)·cosh(λ) = 1 의 n번째 근. 1/cosh 형태로 써서 오버플로를 피한다."""
    guess = (2 * n + 1) * math.pi / 2.0
    lo, hi = guess - 0.4, guess + 0.4

    def f(x: float) -> float:
        return math.cos(x) - (1.0 / math.cosh(x) if x < 700.0 else 0.0)

    f_lo_neg = f(lo) < 0
    for _ in range(CC_LAMBDA_STEPS):
        mid = 0.5 * (lo + hi)
        if (f(mid) < 0) == f_lo_neg:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clamped_pq(n: int) -> tuple[float, float]:
    """고정-고정 보 n번째 모드의 (p, q) = (∫(φ')²/∫φ², ∫(φ'')²/∫φ²)."""
    if n in _CC_CACHE:
        lam, p, q = _CC_CACHE[n]
        return p, q
    lam = _cc_eigenvalue(n)
    sig = (math.cosh(lam) - math.cos(lam)) / (math.sinh(lam) - math.sin(lam))

    def phi(x):
        return math.cosh(lam * x) - math.cos(lam * x) - sig * (math.sinh(lam * x) - math.sin(lam * x))

    def d1(x):
        return lam * (math.sinh(lam * x) + math.sin(lam * x)
                      - sig * (math.cosh(lam * x) - math.cos(lam * x)))

    def d2(x):
        return lam * lam * (math.cosh(lam * x) + math.cos(lam * x)
                            - sig * (math.sinh(lam * x) + math.sin(lam * x)))

    def simpson(fn):
        h = 1.0 / (CC_QUAD_NODES - 1)
        s = fn(0.0) + fn(1.0)
        for i in range(1, CC_QUAD_NODES - 1):
            s += (4.0 if i % 2 else 2.0) * fn(i * h)
        return s * h / 3.0

    i0 = simpson(lambda x: phi(x) ** 2)
    p, q = simpson(lambda x: d1(x) ** 2) / i0, simpson(lambda x: d2(x) ** 2) / i0
    _CC_CACHE[n] = (lam, p, q)
    return p, q


def mode_pq(boundary: str, n: int) -> tuple[float, float]:
    """경계조건·모드번호 → (p, q). 두 경계가 같은 식을 공유하게 하는 다리."""
    if boundary == "clamped":
        return clamped_pq(n)
    a = n * math.pi
    return a * a, a ** 4


def ritz_ncr(Dm: np.ndarray, a: float, b: float, load_ratio: float,
             boundary: str, m: int, n: int) -> float | None:
    """(m,n) 모드의 임계 하중. 압축이 지배하지 않으면 None."""
    pf, qf = mode_pq(boundary, m)
    pp, qq = mode_pq(boundary, n)
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
    pf, qf = mode_pq(boundary, m)
    pp, qq = mode_pq(boundary, n)
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
    if boundary == "clamped":
        scan = min(scan, CLAMPED_MODE_LIMIT)
        scan = max(scan, min(int(n_modes), CLAMPED_MODE_LIMIT))
    out = []
    for m in range(1, scan + 1):
        for n in range(1, scan + 1):
            w2 = ritz_omega2(Dm, rho_areal, a, b, boundary, m, n)
            out.append({"m": m, "n": n, "f_hz": math.sqrt(w2) / (2.0 * math.pi)})
    out.sort(key=lambda x: (x["f_hz"], x["m"], x["n"]))
    return out[:n_modes]
