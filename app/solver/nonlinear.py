# 기하 비선형 (von Karman) — Hyer bistability·대처짐·좌굴후 (계획서 §18)
"""§17까지의 폐형해와 달리 여기는 **Rayleigh–Ritz/Galerkin 저차 근사**다.
분기·임계값·경향은 신뢰하되 곡률·처짐 절대값은 FE 대비 오차가 있다(응답에 명시).

결정론: 난수·적응 종료 없이 고정 격자 스캔 + 고정 반복수 뉴턴만 쓴다.
"""
from __future__ import annotations

import math

import numpy as np

# 결정론 파라미터 (고정 — 바꾸면 응답이 바뀌므로 사양의 일부다)
GRID_N = 15              # a·b 각 방향 후보 격자
NEWTON_STEPS = 40        # 고정 반복수 (수렴 판정으로 조기 종료하지 않음)
DEDUP_RTOL = 1.0e-4      # 해 중복 판정 (탐색 폭 대비)
HESS_EPS_RATIO = 1.0e-5  # 수치 Hessian 스텝 (탐색 폭 대비)
MARGINAL_RATIO = 1.0e-8  # |최소고윳값|이 이 비율 이하면 안정성 판정을 marginal 로 둔다


# ── 1) Hyer bistability ─────────────────────────────────────────────────────

class HyerEnergy:
    """w = −½(a x² + b y²) 형상의 총 포텐셜 에너지 (§18.2).

    면내 변형은 von Karman **적합조건** ε_x,yy + ε_y,xx − γ_xy,xy = w,xy² − w,xx w,yy = −ab
    를 만족하는 최소 다항 족으로 잡는다::

        ε_x⁰ = c₁ − d₁ y²,   ε_y⁰ = c₂ + (d₁ + (k − ab)/2) x²,   γ_xy⁰ = k x y

    이 형태가 핵심이다. ab = 0(원통)이면 우변이 0이라 균일 변형이 적합해 **막 에너지
    벌점이 없고**(원통은 전개 가능면), ab ≠ 0(안장·구면)이면 변형이 2차로 변해야 해
    L⁴ 벌점이 붙는다. 판이 커질수록 안장이 불리해져 원통 두 개로 분기하는 것이 곧
    쌍안정성이다. c₁·c₂·d₁·k 는 에너지의 2차항이라 4×4 선형계로 정확히 소거한다.

    면내 전단 자유도 k 를 반드시 넣어야 한다. 중심 원점 사각영역에서 xy 와 {1, x², y²}
    의 교차적분이 모두 0이라 이 항은 **A66·k²·∫(xy)² 벌점으로만** 들어간다(A16/A26 커플링·
    B16/B26 모멘트항·N6 하중항은 적분이 0이라 소멸). γ_xy⁰ = 0 으로 묶어 두면 안장 가지의
    막 벌점이 과대평가돼 **없는 쌍안정을 만들어낸다** — 적대 검증에서 [0/0/90] 이 실측
    사례였다(3자유도는 안정해 2개, 4자유도는 1개).

    한계: κ_xy = 0 이라 **비틀림 형상은 여전히 표현하지 못한다**. [±θ] 반대칭 적층의 실제
    경화 형상은 비틀림이므로 이 모델을 쓰면 안 된다(파이프라인이 선형 CLT의 κ_xy 로 판정).
    """

    def __init__(self, A, B, D, N_f, M_f, lx: float, ly: float):
        self.A11, self.A12, self.A22 = float(A[0, 0]), float(A[0, 1]), float(A[1, 1])
        self.A66 = float(A[2, 2])
        self.B11, self.B12, self.B22 = float(B[0, 0]), float(B[0, 1]), float(B[1, 1])
        self.D11, self.D12, self.D22 = float(D[0, 0]), float(D[0, 1]), float(D[1, 1])
        self.N1, self.N2 = float(N_f[0]), float(N_f[1])
        self.M1, self.M2 = float(M_f[0]), float(M_f[1])
        self.lx, self.ly = float(lx), float(ly)
        s0 = lx * ly
        sx2, sy2 = lx ** 3 * ly / 12.0, lx * ly ** 3 / 12.0
        sx4, sy4 = lx ** 5 * ly / 80.0, lx * ly ** 5 / 80.0
        sxy = lx ** 3 * ly ** 3 / 144.0
        self.S0 = s0
        # 기저 f = [1, x², y²] 에 대한 ∫f_i f_j dA 와 ∫f_i dA
        self.G = np.array([[s0, sx2, sy2], [sx2, sx4, sxy], [sy2, sxy, sy4]])
        self.m = np.array([s0, sx2, sy2])
        self.Sxy2 = sxy                      # ∫(xy)² dA — γ_xy⁰ 벌점 계수
        # q = [c₁, c₂, d₁, k];  ε_x = Pu·q,  ε_y = Pv·q + v0(a,b),  γ_xy = (Pg·q)·x·y
        self.Pu = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0]])
        self.Pv = np.array([[0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.5],
                            [0.0, 0.0, 0.0, 0.0]])
        self.Pg = np.array([0.0, 0.0, 0.0, 1.0])
        self._Hq = (self.A11 * self.Pu.T @ self.G @ self.Pu
                    + self.A12 * (self.Pu.T @ self.G @ self.Pv + self.Pv.T @ self.G @ self.Pu)
                    + self.A22 * self.Pv.T @ self.G @ self.Pv
                    + self.A66 * np.outer(self.Pg, self.Pg) * self.Sxy2)

    # -- 면내 자유도 소거 ----------------------------------------------------
    def _v0(self, a: float, b: float) -> np.ndarray:
        return np.array([0.0, -0.5 * a * b, 0.0])

    def membrane_dofs(self, a: float, b: float) -> np.ndarray:
        """∂U/∂q = 0 의 해석적 해 q = [c₁, c₂, d₁]."""
        v0 = self._v0(a, b)
        Bu, Bv = self.B11 * a + self.B12 * b, self.B12 * a + self.B22 * b
        g = (self.A12 * self.Pu.T @ self.G @ v0 + self.A22 * self.Pv.T @ self.G @ v0
             + (Bu - self.N1) * self.Pu.T @ self.m + (Bv - self.N2) * self.Pv.T @ self.m)
        return np.linalg.solve(self._Hq, -g)

    def _fields(self, a: float, b: float):
        q = self.membrane_dofs(a, b)
        return self.Pu @ q, self.Pv @ q + self._v0(a, b), float(self.Pg @ q)

    # -- 에너지·도함수 -------------------------------------------------------
    def energy(self, a: float, b: float) -> float:
        u, v, k = self._fields(a, b)
        Iu, Iv = float(u @ self.m), float(v @ self.m)
        Bu, Bv = self.B11 * a + self.B12 * b, self.B12 * a + self.B22 * b
        return float(
            0.5 * (self.A11 * u @ self.G @ u + 2 * self.A12 * u @ self.G @ v
                   + self.A22 * v @ self.G @ v + self.A66 * k * k * self.Sxy2)
            + Bu * Iu + Bv * Iv
            + 0.5 * (self.D11 * a * a + 2 * self.D12 * a * b + self.D22 * b * b) * self.S0
            - self.N1 * Iu - self.N2 * Iv - (self.M1 * a + self.M2 * b) * self.S0)

    def gradient(self, a: float, b: float) -> np.ndarray:
        """포락선 정리로 q는 고정한 채 a,b 편미분 (∂U/∂q = 0 이므로 정확).

        a,b 는 v0 (적합조건 특수해) · B·D·M 항으로만 들어간다.
        """
        u, v, _k = self._fields(a, b)   # γ_xy 항은 a,b 에 직접 의존하지 않아 ∂/∂a 기여 0
        Iu, Iv = float(u @ self.m), float(v @ self.m)
        Bv = self.B12 * a + self.B22 * b
        out = []
        for dv, dBu, dBv, dD, mom in (
            (np.array([0.0, -0.5 * b, 0.0]), self.B11, self.B12,
             self.D11 * a + self.D12 * b, self.M1),
            (np.array([0.0, -0.5 * a, 0.0]), self.B12, self.B22,
             self.D12 * a + self.D22 * b, self.M2),
        ):
            Idv = float(dv @ self.m)
            out.append(self.A12 * float(u @ self.G @ dv) + self.A22 * float(v @ self.G @ dv)
                       + dBu * Iu + dBv * Iv + (Bv - self.N2) * Idv
                       + dD * self.S0 - mom * self.S0)
        return np.array(out)

    def hessian(self, a: float, b: float, span: float) -> np.ndarray:
        """수치 Hessian (고정 스텝 중앙차분 — 결정론)."""
        e = max(abs(span), 1e-12) * HESS_EPS_RATIO
        gx1, gx0 = self.gradient(a + e, b), self.gradient(a - e, b)
        gy1, gy0 = self.gradient(a, b + e), self.gradient(a, b - e)
        h11 = (gx1[0] - gx0[0]) / (2 * e)
        h22 = (gy1[1] - gy0[1]) / (2 * e)
        h12 = 0.5 * ((gx1[1] - gx0[1]) / (2 * e) + (gy1[0] - gy0[0]) / (2 * e))
        return np.array([[h11, h12], [h12, h22]])


def linear_curvature_vector(A, B, D, N_f, M_f) -> np.ndarray:
    """선형 CLT 곡률 3성분 (κx, κy, κxy).

    κxy 를 버리면 안 된다 — 이 모델은 κxy = 0 을 가정하므로, 선형 해가 유의한 κxy 를
    낸다면 그 적층에는 애초에 적용할 수 없다. 그 판정에 쓰려고 3성분을 모두 돌려준다
    (적대 검증 HYER-1: κxy 가 κx 의 59%인 적층이 아무 경고 없이 통과했다).
    """
    K = np.block([[A, B], [B, D]])
    sol = np.linalg.solve(K, np.concatenate([N_f, M_f]))
    return np.array([float(sol[3]), float(sol[4]), float(sol[5])])


def linear_curvatures(A, B, D, N_f, M_f) -> tuple[float, float]:
    """선형 CLT 면내 두 곡률 (비교·탐색 폭 기준)."""
    k = linear_curvature_vector(A, B, D, N_f, M_f)
    return float(k[0]), float(k[1])


def search_span(en: HyerEnergy, kx_lin: float, ky_lin: float) -> float:
    """격자 탐색 폭.

    b = 0(순수 원통)이면 v0 = 0 이라 ∂U/∂a 가 a 에 대해 **정확히 선형**이다 — 원통은
    막 벌점을 받지 않으므로 판 크기와 무관한 곡률을 갖는다. 두 점으로 그 근을 정확히
    잡아 선형 CLT 곡률과 함께 폭을 정한다.
    """
    cands = [abs(kx_lin), abs(ky_lin)]
    for idx, probe in ((0, lambda t: (t, 0.0)), (1, lambda t: (0.0, t))):
        ref = max(abs(kx_lin), abs(ky_lin), 1.0)
        g0 = en.gradient(*probe(0.0))[idx]
        g1 = en.gradient(*probe(ref))[idx]
        slope = g1 - g0
        if abs(slope) > 1e-300:
            cands.append(abs(-g0 / slope * ref))
    span = 1.5 * max(c for c in cands if math.isfinite(c))
    return span if span > 0 else 1.0


def find_equilibria(en: HyerEnergy, span: float) -> list[dict]:
    """고정 격자 스캔 + 고정 반복수 뉴턴으로 정지점을 모두 찾는다 (결정론)."""
    grid = np.linspace(-span, span, GRID_N)
    # 잔차 기준은 판 면적에 비례하는 M 항만으로는 부족하다 — 큰 판에서는 막 항이 L⁶로
    # 커져 참 평형해가 "수렴 실패"로 탈락한다. 격자 모서리에서 실제 gradient 크기를 재
    # 척도로 삼는다(고정 probe 점이라 결정론 유지).
    probe = max(float(np.linalg.norm(en.gradient(sx * span, sy * span)))
                for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)))
    ref = max(abs(en.M1), abs(en.M2), en.D11 * span, 1e-300) * en.S0
    ref = max(ref, probe)
    sols: list[dict] = []
    for a0 in grid:
        for b0 in grid:
            a, b = float(a0), float(b0)
            ok = True
            for _ in range(NEWTON_STEPS):
                try:
                    step = np.linalg.solve(en.hessian(a, b, span), en.gradient(a, b))
                except np.linalg.LinAlgError:
                    ok = False
                    break
                # 발산 억제: 한 스텝을 탐색 폭의 절반으로 제한 (결정론적 클리핑)
                step = np.clip(step, -0.5 * span, 0.5 * span)
                a -= float(step[0])
                b -= float(step[1])
                if not (math.isfinite(a) and math.isfinite(b)) or max(abs(a), abs(b)) > 100 * span:
                    ok = False
                    break
            if not ok or float(np.linalg.norm(en.gradient(a, b))) > 1e-8 * ref:
                continue
            tol = span * DEDUP_RTOL
            if any(abs(a - s["a"]) < tol and abs(b - s["b"]) < tol for s in sols):
                continue
            eig = np.linalg.eigvalsh(en.hessian(a, b, span))
            scale = max(abs(float(eig[0])), abs(float(eig[-1])), 1e-300)
            # 분기점 바로 위/아래에서는 고윳값이 0 근처라 stable/unstable 판정이 무의미하다.
            # bool 하나로 단정하지 말고 marginal 을 따로 표시해 호출부가 경고하게 한다.
            if float(eig[0]) > MARGINAL_RATIO * scale:
                state = "stable"
            elif float(eig[0]) < -MARGINAL_RATIO * scale:
                state = "unstable"
            else:
                state = "marginal"
            sols.append({"a": a, "b": b, "energy": en.energy(a, b),
                         "stable": state == "stable", "stability": state,
                         "min_eig": float(eig[0])})
    sols.sort(key=lambda s: (round(s["energy"], 12), round(s["a"], 12)))
    return sols


def classify_shape(a: float, b: float, tol_ratio: float = 0.1,
                   scale: float | None = None) -> str:
    """곡률 조합 → 형상 이름.

    부호 판정에 a*b 를 쓰면 곡률이 작을 때 곱이 언더플로해 안장이 spherical 로 뒤집힌다
    (적대 검증 SHAPE-1). 부호를 직접 비교한다. scale 을 주면 그 대비 문턱 아래를 flat 으로
    판정해 비정규수(subnormal) 곡률이 형상으로 둔갑하는 것을 막는다.
    """
    mag = max(abs(a), abs(b))
    if mag <= 0 or (scale is not None and mag <= 1e-12 * abs(scale)):
        return "flat"
    if min(abs(a), abs(b)) / mag < tol_ratio:
        return "cylindrical"
    return "saddle" if (a < 0.0) != (b < 0.0) else "spherical"


# ── 2) 대처짐 (von Karman 1항 Galerkin, Cardano 폐형해) ─────────────────────

def vk_coefficients(d_use: np.ndarray, A: np.ndarray, p: float, q: float,
                    immovable: bool) -> tuple[float, float]:
    """αW + βW³ = q̄ 의 계수 (§18.3). w = W·sin(px)·sin(qy), SS 경계.

    Airy 함수 F 의 적합조건 a22F,xxxx + (2a12+a66)F,xxyy + a11F,yyyy = w,xy² − w,xx w,yy
    를 풀어 얻은 특수해를 평형식에 Galerkin 투영한 결과 (sympy로 대수 검증).

    ``immovable`` — 면내 구속 가장자리는 균일 막력 N̄ 이 추가로 생겨 β 가 훨씬 커진다
    (등방 정사각에서 약 3.9배). 어느 쪽인지에 따라 대처짐이 크게 갈리므로 명시 입력이다.
    """
    p2, q2 = p * p, q * q
    alpha = (d_use[0, 0] * p2 * p2 + 2.0 * (d_use[0, 1] + 2.0 * d_use[2, 2]) * p2 * q2
             + d_use[1, 1] * q2 * q2)
    # Airy 적합조건의 컴플라이언스는 **3×3 A의 역행렬**의 좌상 2×2 이다. 2×2 블록만
    # 뒤집으면 A16/A26 이 있는 불균형 적층에서 β를 과대평가해 처짐·좌굴후 진폭을
    # 과소평가한다(적대 검증 VK-1: [30]₄ 에서 β 1.91배, 처짐 0.81배).
    a_in = np.linalg.inv(np.asarray(A))[:2, :2]
    beta = p2 * p2 / (16.0 * a_in[0, 0]) + q2 * q2 / (16.0 * a_in[1, 1])
    if immovable:
        # 면내 구속 항의 균일 막력은 그 컴플라이언스의 역이다 (N̄xy = 0 가정)
        a2 = np.linalg.inv(a_in)
        beta += (a2[0, 0] * p2 * p2 + 2.0 * a2[0, 1] * p2 * q2 + a2[1, 1] * q2 * q2) / 8.0
    return float(alpha), float(beta)


def large_deflection(d_use: np.ndarray, A: np.ndarray, lx: float, ly: float,
                     q_press: float, immovable: bool = False) -> dict:
    """균일압력 q를 받는 SS 판의 중앙 처짐 — αW + βW³ = 16q/π² (§18.3)."""
    p, q = math.pi / lx, math.pi / ly
    alpha, beta = vk_coefficients(d_use, A, p, q, immovable)
    if alpha <= 0:
        raise ValueError("굽힘 강성이 0 이하")
    q_eff = 16.0 * q_press / (math.pi ** 2)
    w_lin = q_eff / alpha
    if beta <= 0 or q_press == 0.0:
        return {"w_center": w_lin, "w_linear": w_lin, "stiffening_ratio": 1.0,
                "alpha": alpha, "beta": beta}
    # βW³ + αW − q_eff = 0 → Cardano (α,β>0 이므로 실근 1개)
    pc, rc = alpha / beta, -q_eff / beta
    try:
        sq = math.sqrt((rc / 2.0) ** 2 + (pc / 3.0) ** 3)
        w = float(np.cbrt(-rc / 2.0 + sq) + np.cbrt(-rc / 2.0 - sq))
    except (OverflowError, ValueError) as exc:   # 극단 판/압력에서 판별식이 넘친다
        raise ValueError(f"판 크기·압력 조합이 수치 범위를 넘습니다 ({type(exc).__name__})") from exc
    if not math.isfinite(w):
        raise ValueError("판 크기·압력 조합이 수치 범위를 넘습니다 (비유한 해)")
    return {"w_center": w, "w_linear": w_lin,
            "stiffening_ratio": w_lin / w if w != 0 else 1.0,
            "alpha": alpha, "beta": beta}


# ── 3) 좌굴 후 (postbuckling) ───────────────────────────────────────────────

def postbuckling(d_use: np.ndarray, A: np.ndarray, lx: float, ly: float,
                 mode_m: int, mode_n: int, n_cr: float, n_applied: float,
                 h: float, load_ratio: float = 0.0) -> dict:
    """N > N_cr 에서의 진폭·면내 강성비·유효폭 (§18.4).

    §18.3과 같은 1항 Galerkin에서 압력 대신 압축 막력을 넣으면
    αW − N(p² + R q²)W + βW³ = 0 → W² = (α/β)(N/N_cr − 1) 이 나온다.
    강성비는 끝단 수축 e = (a11 + R·a12)·N + W²p²/8 을 미분해 얻는다 (하드코딩 아님).
    **2축 하중비 R = Ny/Nx 를 반드시 넣어야 한다.** R 을 빼면 등방 정사각 R=0 에서만
    맞고 R>0(2축 압축)에서 남은 강성을 크게 과대평가한다 — 적대 검증 PB-1 실측으로
    R=1 에서 0.5 대 0.259(1.93배 비보수), R=2 에서 4.25배.
    가장자리 면내 이동 자유 가정 — 구속 시 진폭은 더 작고 강성비는 더 크다.
    """
    p, q = mode_m * math.pi / lx, mode_n * math.pi / ly
    alpha, beta = vk_coefficients(d_use, A, p, q, immovable=False)
    a_in = np.linalg.inv(np.asarray(A))[:2, :2]
    a_eff = float(a_in[0, 0] + load_ratio * a_in[0, 1])   # (a11 + R·a12)
    # S = (a11+R a12) / [(a11+R a12) + p²(p²+R q²)/(8β)] — 등방 정사각 R=0 에서 정확히 0.5
    denom = a_eff + p * p * (p * p + load_ratio * q * q) / (8.0 * beta) if beta > 0 else None
    stiff = (a_eff / denom) if (denom is not None and denom != 0.0) else 1.0
    lr = n_applied / n_cr if n_cr > 0 else float("inf")
    out = {"load_ratio": lr, "stiffness_ratio": float(stiff),
           "definition": "W=√((α/β)(N/N_cr−1)), "
                         "강성비=(a11+R·a12)/[(a11+R·a12)+p²(p²+R·q²)/(8β)], b_eff/b=√(N_cr/N)"}
    if lr <= 1.0:
        out.update({"buckled": False,
                    "note": "N ≤ N_cr — 좌굴 전. compute_buckling 의 여유율을 보라"})
        return out
    amp = math.sqrt(alpha / beta * (lr - 1.0)) if beta > 0 else float("inf")
    out.update({"buckled": True, "amplitude": amp, "amplitude_over_thickness": amp / h,
                "effective_width_ratio": math.sqrt(1.0 / lr)})
    return out


# ── 1b) 분기 임계 판 크기 ───────────────────────────────────────────────────

CRIT_SCAN = 81           # 로그 스캔 점수 (고정)
CRIT_BISECT = 30         # 이분 반복수 (고정)
CRIT_LO, CRIT_HI = 1.0e-4, 1.0e4   # 판 배율 탐색 범위


def _newton_from(en: HyerEnergy, a: float, b: float, span: float) -> tuple[float, float]:
    """주어진 시작점에서 고정 반복수 뉴턴 (분기 가지 추적용)."""
    for _ in range(NEWTON_STEPS):
        try:
            step = np.linalg.solve(en.hessian(a, b, span), en.gradient(a, b))
        except np.linalg.LinAlgError:
            return a, b
        step = np.clip(step, -0.5 * span, 0.5 * span)
        a -= float(step[0])
        b -= float(step[1])
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("nan"), float("nan")
    return a, b


def critical_scale(A, B, D, N_f, M_f, lx: float, ly: float) -> dict:
    """안장 가지가 안정성을 잃는 판 배율 s_crit (종횡비 고정, lx·ly 동시 배율).

    작은 판에서 출발해 배율을 키우며 **안장 가지를 연속적으로 추적**한다. 매 배율마다
    find_equilibria 를 다시 도는 대신(1초/회) 직전 해를 시작점으로 뉴턴 1회만 돌린다.
    Hessian 최소 고윳값의 부호가 바뀌는 구간을 잡아 고정 반복 이분법으로 좁힌다.
    """
    kx, ky = linear_curvatures(A, B, D, N_f, M_f)
    span = 1.5 * max(abs(kx), abs(ky), 1e-12)

    def min_eig(scale: float, start: tuple[float, float]):
        en = HyerEnergy(A, B, D, N_f, M_f, lx * scale, ly * scale)
        a, b = _newton_from(en, start[0], start[1], span)
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("nan"), start
        return float(np.linalg.eigvalsh(en.hessian(a, b, span))[0]), (a, b)

    prev, lo_s, lo_pt = (kx, ky), None, (kx, ky)
    for s in np.logspace(math.log10(CRIT_LO), math.log10(CRIT_HI), CRIT_SCAN):
        eig, pt = min_eig(float(s), prev)
        if not math.isfinite(eig):
            break
        if eig > 0:
            lo_s, lo_pt, prev = float(s), pt, pt
            continue
        if lo_s is None:
            break                      # 최소 배율에서 이미 불안정 — 괄호 없음
        lo, hi = lo_s, float(s)
        for _ in range(CRIT_BISECT):   # 고정 반복 이분법
            mid = math.sqrt(lo * hi)
            e_mid, pt_mid = min_eig(mid, lo_pt)
            if math.isfinite(e_mid) and e_mid > 0:
                lo, lo_pt = mid, pt_mid
            else:
                hi = mid
        s_crit = math.sqrt(lo * hi)
        return {"scale": s_crit, "lx": lx * s_crit, "ly": ly * s_crit}
    return {"scale": None, "lx": None, "ly": None}
