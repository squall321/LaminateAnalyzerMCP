# 재료 면내 전단 비선형 (Hahn–Tsai) — 할선강성 고정반복 (계획서 §18.7)
"""UD 복합재의 면내 전단은 기지 지배라 뚜렷이 비선형이다. Hahn–Tsai(1973) 1파라미터 모델

    γ12 = τ12/G12 + S6666·τ12³        (S6666 ≥ 0)

을 쓰면 할선 전단강성이

    G_sec(τ12) = τ12/γ12 = 1 / (1/G12 + S6666·τ12²)

가 된다. τ가 커질수록 G가 떨어지므로 [±45] 인장처럼 전단 지배 적층은 선형 CLT보다
훨씬 무르다. 이 모듈은 ply별 G_sec를 갱신하며 ABD를 다시 조립하는 할선 반복을 돈다.

결정론: 수렴 판정으로 조기 종료하지 않고 **고정 반복수**만 돈다. 잔차를 함께 반환해
호출부가 수렴 여부를 사용자에게 알린다.
"""
from __future__ import annotations

import numpy as np

from app.solver import abd as ABD
from app.solver import failure as FAIL
from app.solver import material as MAT
from app.solver import response as RESP

SECANT_STEPS = 40      # 고정 반복수 (사양의 일부 — 바꾸면 응답이 바뀐다)
RELAXATION = 0.5       # 저완화 계수. 할선 반복은 진동할 수 있어 고정값으로 눌러 둔다
CONVERGED_TOL = 1.0e-6  # 잔차 판정 (상대)


def secant_g12(g12: float, s6666: float, tau12: float) -> float:
    """Hahn–Tsai 할선 전단강성. S6666=0 이면 선형 G12로 환원된다."""
    return 1.0 / (1.0 / g12 + s6666 * tau12 * tau12)


def _ply_peak_tau12(qbar: np.ndarray, eps0: np.ndarray, kappa: np.ndarray,
                    z_lo: float, z_hi: float, angle_deg: float) -> float:
    """ply 내 |τ12| 최대 (하단·중앙·상단 3점).

    중앙면 하나만 보면 굽힘 지배 하중에서 전단 연화를 과소평가한다 — 그쪽이 비보수라
    최대값을 쓴다(할선 G가 더 작아져 더 무르게, 즉 보수적으로 나온다).
    """
    peak = 0.0
    for z in (z_lo, 0.5 * (z_lo + z_hi), z_hi):
        sig = FAIL.ply_stresses_at(qbar, eps0, kappa, z)
        tau = abs(float(FAIL.stress_to_material_axes(sig, angle_deg)[2]))
        peak = max(peak, tau)
    return peak


def solve_secant(plies: list[dict], z: list[float], n_vec: np.ndarray, m_vec: np.ndarray,
                 steps: int = SECANT_STEPS) -> dict:
    """할선강성 고정반복.

    plies: [{E1,E2,G12,nu12,angle_deg,s6666(None이면 선형)}], z: ABD.z_coordinates 결과.
    """
    g_use = [float(p["G12"]) for p in plies]
    qbars: list[np.ndarray] = []
    eps0 = kappa = None
    tau_used = [0.0] * len(plies)

    for _ in range(steps):
        qbars = [MAT.qbar_matrix(MAT.q_matrix(p["E1"], p["E2"], g, p["nu12"]), p["angle_deg"])
                 for p, g in zip(plies, g_use)]
        A, B, D = ABD.abd_matrices(qbars, z)
        eps0, kappa = RESP.solve_response(A, B, D, n_vec, m_vec)
        tau_new = [_ply_peak_tau12(qbars[k], eps0, kappa, z[k], z[k + 1], p["angle_deg"])
                   for k, p in enumerate(plies)]
        tau_used = tau_new
        g_next = []
        for k, p in enumerate(plies):
            if p["s6666"] is None:
                g_next.append(float(p["G12"]))
                continue
            target = secant_g12(float(p["G12"]), float(p["s6666"]), tau_new[k])
            g_next.append((1.0 - RELAXATION) * g_use[k] + RELAXATION * target)
        g_use = g_next

    # 최종 상태를 마지막 g_use로 한 번 더 정리 (반환값과 per-ply를 정합시킨다)
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p["E1"], p["E2"], g, p["nu12"]), p["angle_deg"])
             for p, g in zip(plies, g_use)]
    A, B, D = ABD.abd_matrices(qbars, z)
    eps0, kappa = RESP.solve_response(A, B, D, n_vec, m_vec)
    per_ply, residual = [], 0.0
    for k, p in enumerate(plies):
        tau = _ply_peak_tau12(qbars[k], eps0, kappa, z[k], z[k + 1], p["angle_deg"])
        g_sec = g_use[k]
        gamma = tau / g_sec if g_sec > 0 else 0.0
        if p["s6666"] is not None:
            # 구성식 잔차 — γ12 가 정말 τ/G12 + S·τ³ 인가 (수렴의 진짜 척도)
            want = tau / float(p["G12"]) + float(p["s6666"]) * tau ** 3
            residual = max(residual, abs(gamma - want) / max(abs(gamma), 1e-300))
        per_ply.append({
            "ply": k,
            "tau12": tau,
            "gamma12": gamma,
            "G12_secant": g_sec,
            "secant_ratio": g_sec / float(p["G12"]),
            "nonlinear": p["s6666"] is not None,
        })
    return {"eps0": eps0, "kappa": kappa, "A": A, "B": B, "D": D,
            "per_ply": per_ply, "residual": residual,
            "converged": residual <= CONVERGED_TOL, "steps": steps}


def linear_reference(plies: list[dict], z: list[float], n_vec: np.ndarray,
                     m_vec: np.ndarray) -> dict:
    """S6666을 무시한 선형 CLT 해 (대조용)."""
    qbars = [MAT.qbar_matrix(MAT.q_matrix(p["E1"], p["E2"], p["G12"], p["nu12"]), p["angle_deg"])
             for p in plies]
    A, B, D = ABD.abd_matrices(qbars, z)
    eps0, kappa = RESP.solve_response(A, B, D, n_vec, m_vec)
    return {"eps0": eps0, "kappa": kappa, "A": A, "B": B, "D": D}
