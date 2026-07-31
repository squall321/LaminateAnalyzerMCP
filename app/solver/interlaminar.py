# 층간 전단응력 (평형법 사후 복원)·횡전단 강성·모달 감쇠 (계획서 §17.6)
"""CLT는 τ_xz를 직접 주지 않으므로 3D 평형식으로 복원한다(표준 후처리).

**막-굽힘 연성 포함**: N_x=0, dM_x/dx=V 의 2×2 연성계를 풀어 응력 구배를 만든다.
  det = A·D − B², ε0' = −B·V/det, κ' = A·V/det → τ(z) = −∫Q̄(ζ)(ε0' + κ'ζ)dζ
전 두께 적분이 −(V/det)(ε0'·A... ) = 0 이 되어 **상면 자유표면 τ=0이 비대칭에서도 성립**한다
(적대 검증 IL-1/IL-01: midplane·생 D 기준 식은 비대칭에서 자유표면을 깨고 계면 τ를 수백 배
과소평가 — 비보수).

횡전단 강성은 (1) ply 각도 변환 Q̄55 = G13cos²θ + G23sin²θ 를 거치고,
(2) 에너지등가(Whitney) A55 = D11²/∫[g(z)]²/G13(z)dz 로 계산한다 — (5/6)ΣG·t 균질 공식은
샌드위치에서 코어 전단을 30배 과대평가해 이 기능의 목적(샌드위치 판정)을 무너뜨린다(TS-03/TS-05).
"""
from __future__ import annotations

import math

import numpy as np

SHEAR_CORRECTION = 5.0 / 6.0     # 균질 직사각 단면 참조계수 (에너지등가와의 대조용)
_GAUSS3 = ((-math.sqrt(3.0 / 5.0), 5.0 / 9.0), (0.0, 8.0 / 9.0), (math.sqrt(3.0 / 5.0), 5.0 / 9.0))


def coupled_gradients(a_kk: float, b_kk: float, d_kk: float, V: float) -> tuple[float, float]:
    """N=0, dM/dx=V 의 2×2 연성계 해 (ε0', κ') — 응력 구배 σ'(z) = Q̄(ε0' + κ'z)."""
    det = a_kk * d_kk - b_kk * b_kk
    if abs(det) < 1e-300:
        raise ZeroDivisionError("A·D − B² ≈ 0 (특이 연성계)")
    return -b_kk * V / det, a_kk * V / det


def transverse_shear_profile(qbars: list[np.ndarray], z: np.ndarray,
                             a_kk: float, b_kk: float, d_kk: float,
                             V: float, comp: int = 0) -> list[dict]:
    """ply 경계·내부 극값의 층간 전단응력 τ(z). comp=0이면 x방향(Q̄11), 1이면 y방향(Q̄22).

    τ(z) = −∫_{−h/2}^{z} Q̄(ζ)(ε0' + κ'ζ) dζ. 대칭(B=0)이면 ε0'=0이라 종전 식과 동일.
    """
    idx = (0, 0) if comp == 0 else (1, 1)
    e0p, kp = coupled_gradients(a_kk, b_kk, d_kk, V)
    out = [{"z": float(z[0]), "tau": 0.0, "interface": None}]
    acc = 0.0                      # ∫ Q̄(ε0'+κ'ζ) dζ 누적
    for k, qb in enumerate(qbars):
        q = float(qb[idx])
        z0, z1 = float(z[k]), float(z[k + 1])
        # dτ/dz = −Q̄(ε0'+κ'z) = 0 → z* = −ε0'/κ' (중립면). ply 안에 있으면 극값으로 추가.
        z_star = -e0p / kp if kp != 0.0 else None
        if z_star is not None and z0 < z_star < z1:
            acc_mid = acc + q * (e0p * (z_star - z0) + kp * (z_star * z_star - z0 * z0) / 2.0)
            out.append({"z": z_star, "tau": -acc_mid, "interface": None,
                        "note": "ply 내부 극값(중립면)"})
        acc += q * (e0p * (z1 - z0) + kp * (z1 * z1 - z0 * z0) / 2.0)
        out.append({"z": z1, "tau": -acc,
                    "interface": f"{k}/{k+1}" if k + 1 < len(qbars) else None})
    return out


def peak_shear(profile: list[dict]) -> dict:
    """|τ| 최대 지점 — 프로파일이 ply 경계와 내부 극값을 모두 담으므로 전역 최대가 보장된다."""
    peak = max(profile, key=lambda p: abs(p["tau"]))
    return {"z": peak["z"], "tau": peak["tau"], "interface": peak["interface"]}


def _qbar_transverse(g13: float, g23: float, angle_deg: float) -> tuple[float, float]:
    """적층각 변환: Q̄55 = G13c² + G23s², Q̄44 = G23c² + G13s² (TS-05)."""
    c = math.cos(math.radians(angle_deg)) ** 2
    s = 1.0 - c
    return g13 * c + g23 * s, g23 * c + g13 * s


def transverse_shear_stiffness(plies, g13_g23, qbars=None, z=None,
                               a_kk=None, b_kk=None, d_kk=None) -> tuple[float, float]:
    """횡전단 강성 (A55, A44).

    qbars·z·(A,B,D)가 주어지면 **에너지등가(Whitney)** A55 = D*²/∫[g(z)]²/Q̄55(z) dz 를 쓴다
    (g(z) = ∫Q̄11(ε0'+κ'ζ)dζ 의 −1배 = 단위 전단력의 τ 분포). 샌드위치·불균질 적층에서 정확.
    인자가 없으면 균질 근사 (5/6)ΣQ̄·t 로 폴백한다.
    """
    q55 = [_qbar_transverse(g[0], g[1], p.angle_deg) for p, g in zip(plies, g13_g23)]
    if qbars is None or z is None or a_kk is None:
        a55 = SHEAR_CORRECTION * sum(q[0] * p.thickness for p, q in zip(plies, q55))
        a44 = SHEAR_CORRECTION * sum(q[1] * p.thickness for p, q in zip(plies, q55))
        return a55, a44

    out = []
    for comp, gsel in ((0, 0), (1, 1)):
        idx = (0, 0) if comp == 0 else (1, 1)
        akk, bkk, dkk = a_kk[comp], b_kk[comp], d_kk[comp]
        try:
            prof = transverse_shear_profile(qbars, z, akk, bkk, dkk, 1.0, comp=comp)
        except ZeroDivisionError:
            out.append(0.0)
            continue
        # τ(z)를 ply별 2차식으로 다시 만들어 ∫τ²/Q̄5 dz 를 3점 Gauss로 정확 적분
        e0p, kp = coupled_gradients(akk, bkk, dkk, 1.0)
        integral = 0.0
        acc = 0.0
        for k, qb in enumerate(qbars):
            q = float(qb[idx])
            z0, z1 = float(z[k]), float(z[k + 1])
            gq = q55[k][gsel]
            if gq <= 0:
                continue
            half = (z1 - z0) / 2.0
            mid = (z1 + z0) / 2.0
            for xi, w in _GAUSS3:
                zz = mid + half * xi
                tau = -(acc + q * (e0p * (zz - z0) + kp * (zz * zz - z0 * z0) / 2.0))
                integral += w * half * tau * tau / gq
            acc += q * (e0p * (z1 - z0) + kp * (z1 * z1 - z0 * z0) / 2.0)
        out.append(1.0 / integral if integral > 0 else 0.0)
    return out[0], out[1]


def shear_flexibility_ratio(Dm: np.ndarray, a55: float, a44: float,
                            a: float, b: float, m: int = 1, n: int = 1) -> float:
    """모드 정합 전단 유연성 R_s(m,n) = π²·navier(m,n) / (A55·α'² + A44·β'²)·π²...

    정확히는 R_s = D_eff·k²/S_eff 형태: 분자 π²·navier(m,n)/((m/a)²+(n/b)²)가 유효 굽힘강성,
    분모가 모드 방향 유효 전단강성이다. 원통형(n→0) 극한에서 π²D11/(A55a²)로 환원된다
    (적대 검증 IL-2/TS-01/FLEX-01: Lx·D11 고정식은 좁은 판·고차 모드에서 400배 과소평가).
    """
    am, bn = m / a, n / b
    k2 = am * am + bn * bn
    if k2 <= 0:
        return float("inf")
    navier = (Dm[0, 0] * am**4 + 2.0 * (Dm[0, 1] + 2.0 * Dm[2, 2]) * am * am * bn * bn
              + Dm[1, 1] * bn**4)
    s_eff = (a55 * am * am + a44 * bn * bn) / k2
    if s_eff <= 0:
        return float("inf")
    return float(math.pi ** 2 * navier / (k2 * s_eff))


def modal_loss_factor(qbars: list[np.ndarray], z: np.ndarray, loss_factors: list[float],
                      a_kk=None, b_kk=None, d_kk=None, a: float = 1.0, b: float = 1.0,
                      m: int = 1, n: int = 1) -> float | None:
    """모드 정합 MSE 감쇠 η = navier(D_η, m, n)/navier(D, m, n).

    D_η = Σ η_k·Q̄_k·I_k (D와 같은 기저·같은 기준면)이므로 진동수 계산과 정확히 일치하는
    가중이 된다. 비대칭이면 중립면 기준으로 I_k를 잡는다 (MSE-01/MSE-02).
    """
    z_na = 0.0
    if a_kk is not None and b_kk is not None and a_kk > 0:
        z_na = b_kk / a_kk
    D = np.zeros((3, 3))
    D_eta = np.zeros((3, 3))
    for k, (qb, eta) in enumerate(zip(qbars, loss_factors)):
        i_k = ((float(z[k + 1]) - z_na) ** 3 - (float(z[k]) - z_na) ** 3) / 3.0
        D += qb * i_k
        D_eta += qb * (eta * i_k)
    am, bn = m / a, n / b
    def nav(M):
        return (M[0, 0] * am**4 + 2.0 * (M[0, 1] + 2.0 * M[2, 2]) * am * am * bn * bn
                + M[1, 1] * bn**4)
    den = nav(D)
    if den <= 0:
        return None
    return float(nav(D_eta) / den)
