# 적층 설계 규칙 검사기 — 업계 표준 휴리스틱의 코드화, 판정+이유+수정힌트 (계획서 §17.5.1)
"""숫자가 아니라 '판단의 언어'를 반환한다: 규칙 이름·위반 사실·물리적 이유·수정 방향.

전 규칙이 적층 정의(각도·두께·재료)만으로 판정 — 물성 데이터 불필요, 결정론.
balance는 각도뿐 아니라 두께·재료까지 짝을 확인하고(사양 §17.5.1), 10% 규칙은 두께 기준이다
(매수 기준은 두께 불균일 적층에서 물리적 의도와 어긋남 — 적대 검증 F1/F3).
"""
from __future__ import annotations

from collections import Counter

_ANGLE_TOL = 0.5          # 각도 매칭 허용 오차 [deg]
_THICK_RTOL = 1.0e-9      # 두께 동일 판정 상대 허용
_QUAD = (0.0, 45.0, -45.0, 90.0)


def _family(angle: float) -> float | None:
    """쿼드 방향군 매칭 (tol 내). 90/−90은 동일군."""
    for q in _QUAD:
        if abs(angle - q) <= _ANGLE_TOL or (q == 90.0 and abs(abs(angle) - 90.0) <= _ANGLE_TOL):
            return q
    return None


def _is_on_axis(angle: float) -> bool:
    return abs(angle) <= _ANGLE_TOL or abs(abs(angle) - 90.0) <= _ANGLE_TOL


def _mat_key(p) -> tuple:
    return (p.E1, p.E2, p.G12, p.nu12)


def _rule(rule: str, severity: str, ok: bool | None, found: str, why: str, fix: str) -> dict:
    return {"rule": rule, "severity": severity,
            "pass": ok, "found": found, "why_it_matters": why, "fix_hint": fix}


def _bridge_angle(t1: float, t2: float) -> float:
    """두 각도 사이 삽입 시 양쪽 차이를 45° 이하로 만드는 중간 각도 (없으면 중점)."""
    for cand in (0.0, 45.0, -45.0, 90.0):
        if (_angle_gap(t1, cand) <= 45.0 + _ANGLE_TOL
                and _angle_gap(cand, t2) <= 45.0 + _ANGLE_TOL):
            return cand
    return round((t1 + t2) / 2.0, 1)


def _angle_gap(a: float, b: float) -> float:
    """적층각 차이 (mod 180 — 89°와 −89°는 2°)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def check_rules(plies, fingerprint: list, coupling_ratio: float | None,
                a16_ratio: float | None, contiguity_limit: int = 4) -> list[dict]:
    """plies: SiPly 목록. 반환: 규칙별 판정 목록 (§17.5.1 표 순서)."""
    n = len(plies)
    angles = [p.angle_deg for p in plies]
    out: list[dict] = []

    # 1. symmetry (hard)
    sym = fingerprint == list(reversed(fingerprint))
    out.append(_rule(
        "symmetry", "hard", sym,
        f"기하 대칭 {'만족' if sym else '위반'}"
        + (f" (coupling_ratio={coupling_ratio:.3g})" if coupling_ratio is not None else ""),
        "비대칭이면 B≠0 — 막-굽힘 커플링으로 경화 후 뒤틀림·하중 커플링 발생",
        "midplane 기준 거울상 적층([...]s)으로 재배열"))

    # 2. balance (hard) — ±θ 짝을 각도(tol)·두께·재료까지 확인 (그리디 매칭)
    off = [(k, p) for k, p in enumerate(plies) if not _is_on_axis(p.angle_deg)]
    unmatched = list(off)
    unpaired: list[str] = []
    while unmatched:
        k, p = unmatched.pop(0)
        partner = None
        for i, (k2, p2) in enumerate(unmatched):
            same_mat = _mat_key(p2) == _mat_key(p)
            same_t = abs(p2.thickness - p.thickness) <= _THICK_RTOL * max(p.thickness, p2.thickness)
            if abs(p2.angle_deg + p.angle_deg) <= _ANGLE_TOL and same_mat and same_t:
                partner = i
                break
        if partner is None:
            unpaired.append(f"laminae[{k}]({p.angle_deg:g}°, t={p.thickness:.6g})")
        else:
            unmatched.pop(partner)
    balanced = len(unpaired) == 0
    out.append(_rule(
        "balance", "hard", balanced,
        ("±θ 짝 만족 (각도·두께·재료)" if balanced else f"미짝 ply {unpaired}")
        + (f" (|A16|/√(A11A22)={a16_ratio:.3g})" if a16_ratio is not None else ""),
        "±θ 짝이 없으면 A16/A26≠0 — 면내 인장이 전단으로 새는 커플링",
        "각 +θ ply마다 동일 두께·재료의 −θ ply를 추가"))

    # 3. ten_percent (guideline) — 쿼드 적층에서만, 두께 기준
    fams = [_family(a) for a in angles]
    if any(f is None for f in fams):
        out.append(_rule("ten_percent", "guideline", None,
                         "비쿼드 각도 포함 — 판정 생략(not_applicable)",
                         "10% 규칙은 0/±45/90 쿼드 적층 관례",
                         "쿼드 적층이라면 각 방향 두께 10% 이상 확보"))
    else:
        total_t = sum(p.thickness for p in plies)
        frac = {q: sum(p.thickness for p, f in zip(plies, fams) if f == q) / total_t
                for q in _QUAD}
        deficient = {q: f for q, f in frac.items() if f < 0.10 - 1e-9}
        out.append(_rule(
            "ten_percent", "guideline", len(deficient) == 0,
            "두께 기준 방향별 비율 " + ", ".join(f"{int(q)}°:{f:.1%}" for q, f in frac.items()),
            "특정 방향이 10% 미만이면 그 방향 하중·충격 후 잔류강도에 취약(매트릭스 지배)",
            f"부족 방향 {sorted(int(q) for q in deficient)}에 ply 추가" if deficient else "—"))

    # 4. contiguity (guideline)
    runs = []
    run_start, run_len = 0, 1
    for i in range(1, n):
        if abs(angles[i] - angles[i - 1]) <= _ANGLE_TOL:
            run_len += 1
        else:
            if run_len > contiguity_limit:
                runs.append((run_start, run_len))
            run_start, run_len = i, 1
    if run_len > contiguity_limit:
        runs.append((run_start, run_len))
    out.append(_rule(
        "contiguity", "guideline", len(runs) == 0,
        "동일 각도 연속 초과 없음" if not runs else
        f"연속 초과 {[(f'laminae[{s}]부터 {l}매') for s, l in runs]} (한계 {contiguity_limit})",
        "동일 각도 뭉침은 매트릭스 균열·자유단 박리를 조장",
        "다른 각도 ply를 사이에 분산 배치"))

    # 5. adjacent_angle (guideline) — 위반 계면별 삽입 각도를 계산해 실행 가능한 힌트로
    bad_if = [(i, _angle_gap(angles[i], angles[i + 1])) for i in range(n - 1)
              if _angle_gap(angles[i], angles[i + 1]) > 45.0 + _ANGLE_TOL]
    if bad_if:
        hints = [f"laminae[{i}]/[{i+1}] 사이에 {_bridge_angle(angles[i], angles[i+1]):g}° 삽입"
                 for i, _ in bad_if]
        fix = "; ".join(hints)
    else:
        fix = "—"
    out.append(_rule(
        "adjacent_angle", "guideline", len(bad_if) == 0,
        "인접 각도차 ≤45° 만족" if not bad_if else
        f"45° 초과 계면 {[(f'laminae[{i}]/[{i+1}]: {d:g}°') for i, d in bad_if]}",
        "인접 ply 각도차가 크면(예: 0/90 직접 접촉) 계면 층간응력 집중 → 박리 위험",
        fix))

    # 6. outer_protection (guideline) — 단일 ply도 그 각도로 자연 판정
    outer_ok = all(abs(abs(angles[i]) - 45.0) <= _ANGLE_TOL for i in (0, n - 1))
    out.append(_rule(
        "outer_protection", "guideline", outer_ok,
        f"최외층 각도 = 하{angles[0]:g}° / 상{angles[-1]:g}°",
        "최외층 ±45는 충격·긁힘·드릴링 손상에 강하고 0° 섬유 노출을 보호",
        "최외층을 ±45 ply로 배치"))

    # 7. 방향군 분포 요약 (info, 판정 아님 — 사양 표의 single_ply_angle_group)
    dist = Counter(round(a, 1) for a in angles)
    out.append({"rule": "single_ply_angle_group", "severity": "info", "pass": None,
                "found": ", ".join(f"{a:g}°×{c}" for a, c in sorted(dist.items())),
                "why_it_matters": "방향별 매수 분포 요약", "fix_hint": "—"})
    return out
