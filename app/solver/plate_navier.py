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
