# 변위 제어 — 곡률·면내변형 지정과 자유도 분할 풀이 (계획서 §19.14)
"""서버의 모든 하중 입력이 힘 제어(N/M)였다. 그런데 폴더블·롤투롤·맨드릴 굽힘·척킹
평탄화의 실제 구속은 **변위 제어**다. 그래서 에이전트가 M = D·κ 라는 지름길을 쓰는데,
비대칭 스택에서 실측 **+244.8% 과대**다(참값 0.371 vs 지름길 1.278 N).
더 나쁜 것은 assess_crack_shielding 이 요구하는 applied_strain 스칼라를 만들려고
ε=(z−z_ns)/R 을 손으로 계산하게 된다는 점이다 — 서버가 "ABD 를 암산하지 말라"고
지시해 놓고 바로 그 행위를 강요한다.

새 물리가 아니라 K[ε⁰;κ] = [N;M] 의 **지정/미지 자유도 분할**이다. 지정된 자유도는
일반변형률이 알려지고 대응 일반력이 반력(미지)이 된다. 나머지는 반대다.

환원 항등식(테스트로 고정):
- 대칭 적층 + κ 전부 지정 → M = D·κ (정확)
- 비대칭 + κ 전부 지정, N 자유 → M = D*·κ = (D − B A⁻¹ B)κ (정확)
- 자유 폭(M_y=0)과 구속 폭(κ_y=0)은 **다른 답**이다 — 실측 9.6% 차이. 이 구분을 명시
  입력으로 받는 것이 이 도구 가치의 절반이다.
"""
from __future__ import annotations

import numpy as np


class SingularPartition(Exception):
    """자유 자유도 부분행렬이 특이 — 구속이 모자라거나 물성이 퇴화."""


def partitioned_solve(K: np.ndarray, prescribed: list[float | None],
                      applied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """K x = f 를 혼합 경계로 푼다.

    prescribed[i] 가 None 이 아니면 x[i] 가 그 값으로 고정되고 f[i] 는 반력이 된다.
    None 이면 f[i] = applied[i] 가 주어지고 x[i] 를 푼다.
    반환 (x, f) — 항상 6성분 전부.
    """
    fixed = [i for i in range(6) if prescribed[i] is not None]
    free = [i for i in range(6) if prescribed[i] is None]
    x = np.zeros(6)
    for i in fixed:
        x[i] = float(prescribed[i])
    if free:
        rhs = np.array([applied[i] - sum(K[i, j] * x[j] for j in fixed) for i in free])
        sub = np.array([[K[i, j] for j in free] for i in free])
        try:
            sol = np.linalg.solve(sub, rhs)
        except np.linalg.LinAlgError as exc:
            raise SingularPartition("자유 자유도 부분행렬이 특이합니다") from exc
        if not np.all(np.isfinite(sol)):
            raise SingularPartition("자유 자유도 해가 비유한값입니다")
        for i, v in zip(free, sol):
            x[i] = float(v)
    return x, K @ x


def build_prescription(kappa: list[float | None] | None,
                       epsilon0: list[float | None] | None) -> list[float | None]:
    """(ε⁰, κ) 6성분 지정 목록. None = 자유(대응 일반력이 주어짐)."""
    eps = list(epsilon0) if epsilon0 is not None else [None, None, None]
    kap = list(kappa) if kappa is not None else [None, None, None]
    return eps + kap


def naive_moment(D: np.ndarray, kappa_full: np.ndarray) -> np.ndarray:
    """에이전트가 쓰는 지름길 M = D·κ — 얼마나 틀리는지 보여주기 위한 대조값."""
    return D @ kappa_full
