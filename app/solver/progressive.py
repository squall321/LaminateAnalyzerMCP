# 진행성 파손 (ply discount) — FPF 이후 강성 저하 반복으로 한계하중까지 (계획서 §17.5.3)
"""하중 제어 quasi-static restart 표준 근사: 사건마다 강성을 깎고 처음부터 재해석한다.

각 ply는 bottom/mid/top 3점에서 평가한다 — 중앙만 보면 굽힘 하중에서 midplane 근처 ply의
파손이 통째로 누락되거나 σ≈0에서 거대 R 스퓨리어스 사건이 생긴다(적대 검증 PROG-1/PROG-2).
응력이 참조 스케일 대비 무시할 수준이면 그 ply는 '파손 불가'로 건너뛴다.

강도 없는 ply는 탄성 유지(비파손). 기지/전단 모드 → E2·G12·ν12 ×η,
섬유 모드 → 전 성분 ×η (ply 사망). 기지 파손된 ply의 후속 검사는 섬유 항만.
결정론: 사건 ≤ 2×n_plies, 동률은 낮은 ply 번호 우선.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from app.solver import abd as ABD
from app.solver import failure as FAIL
from app.solver import material as MAT
from app.solver import response as RESP

FIBER_MODES = ("fiber_tension", "fiber_compression")
# 응력이 참조 스케일(적층 전체 최대 |σ|)의 이 비율 미만이면 파손 판정에서 제외 —
# 자기평형 σ1≈0에서 R~1e18 스퓨리어스 사건을 막는다 (PROG-1).
STRESS_FLOOR_RATIO = 1.0e-6
# 하중 제어 붕괴 판정: 같은 하중 패턴에 대한 응답 크기가 초기의 이 배수를 넘으면
# 더 이상 하중을 지탱하지 못하는 것으로 본다. 이후 사건은 도달 불가능한 하중이므로
# ultimate_R 집계에서 제외된다 (PROG-1: max(R)이 도달 불가 사건에 오염되던 문제).
COLLAPSE_RESPONSE_RATIO = 10.0


def run(plies, N: np.ndarray, M: np.ndarray, discount: float = 0.1) -> dict:
    """plies: SiPly 목록(원본 불변), (N, M): SI 하중 패턴. 반환: events·ultimate 등 (§17.5.3)."""
    work = [dataclasses.replace(p) for p in plies]
    matrix_failed = [False] * len(work)
    fiber_failed = [False] * len(work)
    failable = [k for k in range(len(work)) if plies[k].strength is not None]
    events: list[dict] = []
    ex_curve: list[float] = []
    termination = "no_failable_plies"
    resp0 = None      # 초기 응답 크기 (붕괴 판정 기준)

    for _step in range(2 * len(work)):
        qbars = [MAT.qbar_matrix(MAT.q_matrix(p.E1, p.E2, p.G12, p.nu12), p.angle_deg)
                 for p in work]
        z = ABD.z_coordinates([p.thickness for p in work])
        A, B, D = ABD.abd_matrices(qbars, z)
        h = float(z[-1] - z[0])
        if not ABD.is_positive_definite(ABD.normalized_stiffness(A, B, D, h)[3]):
            termination = "stiffness_singular"
            break
        try:
            eps0, kappa = RESP.solve_response(A, B, D, N, M)
        except RESP.SingularSystemError:
            termination = "stiffness_singular"
            break

        # 하중 지지 능력 붕괴 판정 — 같은 하중에 응답이 초기의 COLLAPSE배를 넘으면 정지
        resp = float(np.linalg.norm(np.concatenate([eps0, kappa * h])))
        if resp0 is None:
            resp0 = resp
        elif resp0 > 0 and resp > COLLAPSE_RESPONSE_RATIO * resp0:
            termination = "load_carrying_collapse"
            break

        # ply×3점 응력을 먼저 모아 참조 스케일을 정한다 (상대 임계용)
        s12_all: dict[tuple[int, int], np.ndarray] = {}
        ref = 0.0
        for k, p in enumerate(work):
            for j, zv in enumerate((float(z[k]), float(z[k] + z[k + 1]) / 2.0, float(z[k + 1]))):
                s12 = FAIL.stress_to_material_axes(
                    FAIL.ply_stresses_at(qbars[k], eps0, kappa, zv), p.angle_deg)
                s12_all[(k, j)] = s12
                ref = max(ref, float(np.max(np.abs(s12))))
        floor = ref * STRESS_FLOOR_RATIO

        best = None      # (R, ply, mode)
        for k in failable:
            if fiber_failed[k]:
                continue
            Xt, Xc, Yt, Yc, S = plies[k].strength
            for j in range(3):
                s12 = s12_all[(k, j)]
                if float(np.max(np.abs(s12))) <= floor:
                    continue                      # 무시할 응력 — 스퓨리어스 사건 방지
                if matrix_failed[k]:
                    s1 = float(s12[0])
                    if abs(s1) <= floor:
                        continue
                    r = Xt / s1 if s1 > 0 else Xc / (-s1)
                    mode = "fiber_tension" if s1 > 0 else "fiber_compression"
                else:
                    r = FAIL.tsai_wu(s12, Xt, Xc, Yt, Yc, S).get("strength_ratio")
                    if r is None:
                        continue
                    mode = FAIL.max_stress(s12, Xt, Xc, Yt, Yc, S)["mode"]
                if r > 0 and (best is None or r < best[0]):
                    best = (r, k, mode)

        if best is None:
            termination = "no_failable_plies"
            break
        r, k, mode = best
        events.append({"step": len(events), "ply": k, "mode": mode, "R": r})

        # 강성 저하 (모드별)
        p = work[k]
        if mode in FIBER_MODES:
            work[k] = dataclasses.replace(p, E1=p.E1 * discount, E2=p.E2 * discount,
                                          G12=p.G12 * discount, nu12=p.nu12 * discount)
            fiber_failed[k] = True
            matrix_failed[k] = True
        else:
            work[k] = dataclasses.replace(p, E2=p.E2 * discount, G12=p.G12 * discount,
                                          nu12=p.nu12 * discount)
            matrix_failed[k] = True

        # 사건 후 유효 Ex (막 컴플라이언스 기준)
        qb2 = [MAT.qbar_matrix(MAT.q_matrix(q.E1, q.E2, q.G12, q.nu12), q.angle_deg) for q in work]
        A2, B2, D2 = ABD.abd_matrices(qb2, z)
        try:
            alpha2, _, _ = RESP.compliance_blocks(A2, B2, D2)
            ex_curve.append(1.0 / (h * alpha2[0, 0]))
        except RESP.SingularSystemError:
            ex_curve.append(0.0)
            termination = "stiffness_singular"
            break

        if all(fiber_failed[i] for i in failable):
            # strength 없는 ply는 탄성 유지 — 구분해 보고 (PROG-3)
            termination = ("all_plies_failed" if len(failable) == len(work)
                           else "all_failable_plies_failed")
            break
    else:
        termination = "event_limit"

    return {
        "events": events,
        "ultimate_R": max((e["R"] for e in events), default=None),
        "last_ply_R": events[-1]["R"] if events else None,
        "ex_eff_after_events": ex_curve,
        "termination": termination,
    }
