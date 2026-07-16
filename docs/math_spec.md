# 수학 사양 상세 (math_spec)

계획서 §4의 유도 상세와 구현 대응. 코드의 단일 소스는 `app/solver/`이며 본 문서는 그 근거를 남긴다.

## 1. 재료 강성 Q — `material.q_matrix`

평면응력 감축 강성. ν₂₁ = ν₁₂E₂/E₁, Δ = 1 − ν₁₂ν₂₁ 로

$$
Q_{11}=\frac{E_1}{\Delta},\quad Q_{22}=\frac{E_2}{\Delta},\quad
Q_{12}=\frac{\nu_{12}E_2}{\Delta},\quad Q_{66}=G_{12}
$$

열역학 안정 조건 Δ > 0 ⇔ |ν₁₂| < √(E₁/E₂). 등방성은 E₁=E₂=E, G=E/2(1+ν).

## 2. 변환 Q̄ — `material.qbar_matrix`

m=cosθ, n=sinθ. 공학 전단(Voigt, γ=2ε₁₂) 규약의 표준 전개식(계획서 §4.2)을 그대로 코드화.
**검증**: 독립 오라클(`tests/oracle/clt_sympy.py`)은 같은 결과를 전혀 다른 경로로 얻는다.

$$
\bar{Q} = T^{-1}\, Q\, R\, T\, R^{-1},\qquad
T=\begin{bmatrix} m^2 & n^2 & 2mn\\ n^2 & m^2 & -2mn\\ -mn & mn & m^2-n^2 \end{bmatrix},\;
R=\mathrm{diag}(1,1,2)
$$

무작위 50케이스에서 두 경로가 rtol 1e-9로 일치(`tests/test_oracle.py`) — 전개식 오타가 있었다면 여기서 걸린다.

## 3. z-좌표와 ABD — `abd.z_coordinates`, `abd.abd_matrices`

z₀ = −h/2, z_k = z_{k−1} + t_k (laminae[0] = 최하단, D1 규약).

$$
(A,B,D)_{ij}=\sum_k \bar{Q}_{ij}^{(k)}\left(\Delta z,\ \tfrac{\Delta z^2}{2},\ \tfrac{\Delta z^3}{3}\right)_k
$$

## 4. 정규화 K̂과 √12 계수 — `abd.normalized_stiffness`

A(N/m)·B(N)·D(N·m)는 단위가 달라 원시 6×6의 조건수는 무의미하다. 전 성분이 Pa가 되도록

$$
\hat A=\frac{A}{h},\qquad \hat B=\frac{\sqrt{12}\,B}{h^2},\qquad \hat D=\frac{12\,D}{h^3}
$$

**√12의 유도.** S = diag(s₁I₃, s₂I₃)에 대한 합동변환 S·[A B; B D]·S 가 위 정규화와 일치하려면
s₁² = 1/h, s₂² = 12/h³ 이고, 이때 비대각 블록 계수는 s₁s₂ = √(12/h⁴) = √12/h² 로 강제된다.
합동변환이므로 (Sylvester 관성 법칙) **양정치성이 원행렬과 정확히 동치**이고, 균질 단일재에서
Â = D̂ = Q̄, 두께 균등 스케일 t→αt에 대해 K̂ 불변이다(성질 테스트 P3에서 확인).
관례적으로 쓰이는 2B/h² 정규화는 합동변환이 아니어서 PD 판정에 쓸 수 없다 — 본 프로젝트는
모든 정규화 용도에 √12 규약 하나만 쓴다.

## 5. 중립면 — `neutral_axis`

**clt_weighted가 B₁₁/A₁₁과 동치인 증명.**
z_k² − z_{k−1}² = (z_k+z_{k−1})(z_k−z_{k−1}) = 2\bar z_k t_k 이므로

$$
\frac{B_{11}}{A_{11}}
=\frac{\tfrac12\sum \bar Q_{11}^{(k)}(z_k^2-z_{k-1}^2)}{\sum \bar Q_{11}^{(k)} t_k}
=\frac{\sum \bar Q_{11}^{(k)} t_k \bar z_k}{\sum \bar Q_{11}^{(k)} t_k}
$$

즉 Q̄₁₁-가중 도심과 정확히 같다. beam_equivalent는 가중치를 E_x(θ)=1/S̄₁₁로 바꾼 것으로,
1축 응력(측방 수축 자유) 가정의 보 해석에 해당한다. 대칭 적층이면 두 정의 모두 0.

## 6. 하중 응답 — `response`

[N; M] = K [ε₀; κ], K = [A B; B D]. 역행렬 대신 `linalg.solve`. 컴플라이언스 [α β; βᵀ δ] = K⁻¹.

$$
E_x^{\mathrm{eff}}=\frac{1}{h\,\alpha_{11}},\quad
\nu_{xy}=-\frac{\alpha_{12}}{\alpha_{11}},\quad
E_x^{b}=\frac{12}{h^3\,\delta_{11}}
$$

등방 단일층에서 E_x^eff = E_x^b = E, ν_xy = ν 가 복원되는 것이 단위 테스트다
(α = S/h, δ = 12S/h³ 이므로 자명). 비대칭 적층은 α·δ에 B 커플링이 포함된 유효값임을
응답 가정으로 명시한다.

- `membrane_bending_leakage` = (h/2)‖κ‖₂/‖ε₀‖₂ (단위 Nx). 대칭 ⇔ 0.
- `twist_under_bending` = |κ_xy|/|κ_x| (단위 Mx). D₁₆=0(크로스플라이)이면 0,
  대칭 앵글플라이 [θ/−θ/−θ/θ]는 D₁₆≠0이라 0이 아니다.
- 주강성 방향: E_x^eff(φ)를 φ∈[0°,180°) 1° 그리드 스캔(결정론, 난수 없음).

## 7. 회전 불변량 (성질 테스트 P4의 근거)

Q̄의 Tsai–Pagano 표현 Q̄₁₁+Q̄₂₂+2Q̄₁₂ = const, Q̄₆₆−Q̄₁₂ = U₅−U₄ = const 에서,
적분이 선형이므로 M ∈ {A, B, D} 각각에 대해

$$
I_1(M)=M_{11}+M_{22}+2M_{12},\qquad I_2(M)=M_{66}-M_{12}
$$

가 전 ply 공통 회전에 불변이다. (본 계획 수립 시 전개식으로 재확인 — I₂ 후보로 흔히 쓰는
M₆₆−(M₁₁+M₂₂−2M₁₂)/4 는 **불변이 아니므로** 쓰지 않는다.)

## 8. 수치 정책

- float64, 서버측 반올림 없음(D9). 직렬화는 shortest round-trip repr.
- 허용오차: 폐형해 rtol 1e-12 · 오라클 rtol 1e-9 · B≈0 판정 ‖B̂‖ ≤ 1e-9‖Â‖ (계획서 §8.5).
- PD 판정은 K̂의 Cholesky. 유효 물성 입력에서 K̂은 항상 PD(변형 에너지)이므로
  실패는 입력 오류 또는 내부 결함 신호다(E402).
