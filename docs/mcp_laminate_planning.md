# Laminate Neutral Surface & ABD 평가 MCP 서버 — 구현 계획서

| 항목 | 내용 |
|---|---|
| 문서 버전 | v2.5 (v2.4 + V1·Phase4·hwax 기동 실증 — MVP+V1 완료, 서버 0.2.0) |
| 작성일 | 2026-07-16 |
| 상태 | 구현 착수 가능 (미결 사항은 §15에 기본값과 함께 명시) |
| 대상 독자 | **이 계획서를 읽고 그대로 구현하는 코딩 에이전트**, 그리고 검토자(사람) |
| 최종 소비자 | **LLM 에이전트** — 완성된 MCP 서버의 Tool을 호출해 적층판 계산 결과를 얻는다 |

---

## 0. 이 문서의 사용법 (구현 에이전트에게)

- 이 문서는 브레인스토밍이 아니라 **실행 사양서**다. §3(확정 사양)과 §4(수학 사양)의 내용은 동결되어 있으므로 임의로 바꾸지 않는다.
- §15(미결 사항)의 항목은 사용자 확인이 없으면 **명시된 기본값으로 진행**한다.
- 구현 순서는 §12(단계 계획)를 따르고, 각 단계의 DoD(완료 정의)를 통과하기 전에 다음 단계로 넘어가지 않는다.
- 진행하면서 §13(체크리스트)의 체크박스를 직접 갱신한다.
- 설계 원칙 한 줄 요약. **이 서버의 사용자는 사람이 아니라 LLM 에이전트다.** Tool 이름, 스키마의 필드 설명, 오류 메시지, 응답 구조 전부가 "에이전트가 오해 없이 쓸 수 있는가"를 기준으로 판단된다.

---

## 1. 프로젝트 목표

### 1.1 핵심 목표
- 적층 복합재 입력(층 두께, 재료상수, 적층각)을 받아 다음을 계산해 반환하는 MCP 서버를 만든다.
  - 중립면 위치 `z_ns` (여러 정의 지원)
  - ABD 행렬 (A, B, D)
  - 엔지니어링 판단용 정량 지표(커플링, 이방성, 준등방성 등)
- LLM은 계산을 직접 하지 않는다. 수치는 전부 MCP Tool 호출로만 획득한다.
- 재현 가능한 결정론적 계산 파이프라인을 보장한다.
- **설계 반복(최적화 탐색)은 서버가 아니라 에이전트가 주도한다.** 서버는 빠르고 결정론적인 계산 커널만 제공하고, "각도를 바꿔가며 목표 강성에 맞추는" 식의 루프는 에이전트가 Tool을 반복 호출하며 수행한다. 이것이 Tool 범위를 정하는 기준선이다.

### 1.2 성공 기준 (측정 가능 형태)
| # | 기준 | 측정 방법 |
|---|---|---|
| S1 | 동일 입력 → 바이트 단위 동일 출력 | 같은 payload 2회 호출 후 응답 JSON 비교 테스트 |
| S2 | 단위/좌표계 오해의 시스템적 차단 | 모든 응답에 단위 명세 포함 + 단위 오입력 휴리스틱 경고(W110) 테스트 |
| S3 | 기준 케이스 20개 이상 통과 | 폐형해 5 + 성질 기반 6 + 오라클/문헌 벤치마크 9 이상, `pytest` 녹색 |
| S4 | 실패 시 명확한 오류 코드와 수정 가이드 | §6.4 오류 코드 전수에 대해 유발 테스트 존재 |
| S5 | 에이전트가 문서 없이 사용 가능 | Tool description과 스키마만 보고 Claude가 올바른 첫 호출을 구성하는 시나리오 테스트 (§12 Phase 5) |

---

## 2. 원안(v1.0) 검증 결과 요약

원안을 항목별로 검증한 결과다. 아래 조치는 모두 본문에 반영되어 있다.

| 원안 항목 | 판정 | 조치 |
|---|---|---|
| ABD 적분 공식 (§3.3) | ✅ 정확 | 유지. Q, Q̄ 전개식이 없었으므로 §4에 전량 명기 |
| 중립면 공식 (§3.2) | ⚠️ 모호 | beam 모드의 `E_eq` 미정의 → `E_x(θ)=1/S̄₁₁`로 확정. CLT 모드가 `z=B₁₁/A₁₁`과 동치임을 명시. 완전 커플링 정의(하중 기반)를 V1로 추가 |
| `find_available_port`를 MCP Tool로 노출 | ❌ 설계 오류 | Tool 목록에서 제거. LLM은 포트를 관리하지 않는다. stdio 전송이 기본이라 포트 자체가 불필요하고, HTTP 모드에서는 서버 기동 옵션(`--port 0` 자동 할당)으로 이동 (§6.1, §10) |
| `abd_condition_number` | ⚠️ 함정 | A/B/D는 단위가 서로 달라(N/m, N, N·m) 원시 6×6 조건수는 무의미. 합동변환 정규화(§4.6) 후 계산하도록 확정 |
| 적층 순서/각도 부호/전단 규약 | ❌ 누락 | 상호운용 버그 1순위. §3에 동결 규약으로 추가 (D1–D3) |
| 단위계 `SI \| ENG` | ⚠️ 모호 | `ENG`가 psi·in인지 MPa·mm인지 불명. `SI`(Pa·m)와 `SI_mm`(MPa·mm)로 확정, imperial은 V1로 연기 (§3 D4) |
| 결정론 요구 vs `metadata.timestamp` | ⚠️ 모순 | timestamp는 기본 응답에서 제외. 결정론 보장 범위를 §6.5에 명문화 |
| 기준 케이스 20개 | ⚠️ 정답 출처 미정 | 폐형해(수식 내장) + 성질 기반 테스트 + sympy 독립 오라클 + 문헌 벤치마크의 4원 전략으로 구체화 (§8) |
| `warpage_tendency_score` | ⚠️ 전제 누락 | 제조 후 뒤틀림은 열잔류변형(CTE, ΔT)이 지배 → 열 하중이 없는 MVP에서는 B 기반 간접 지표만 가능함을 명시하고 본 지표는 V2(열 확장 포함)로 이동 |
| 최적화 루프 (V2) | ⚠️ 범위 재검토 | 반복 탐색은 에이전트가 Tool 호출 루프로 수행하는 것이 MCP 설계에 부합 → 서버 내 최적화 Tool은 범위에서 제외 (§1.1) |
| 층별 응력/변형률 복원 | ➕ 누락 보완 | 하중 응답(V1)의 자연스러운 확장이자 V2 파손 판정의 전제 조건이므로 V2 범위에 명시 |
| Tool 구성 | ➕ 개선 | 에이전트 왕복 횟수를 줄이는 원샷 Tool `analyze_laminate` 추가, few-shot 자가 학습용 `get_reference_cases` 추가 (§6.2) |
| 아키텍처/일정 | ✅ 타당 | 날짜 기반 일정 → 에이전트 구현에 맞는 Phase + DoD 게이트로 재구성 (§12) |

---

## 3. 확정 사양 (Frozen Decisions)

구현 전 반드시 숙지할 동결 규약. 코드, 스키마, 문서, 테스트가 전부 이 표를 따른다.

| ID | 항목 | 결정 |
|---|---|---|
| D1 | 적층 순서 | `laminae[0]`이 **최하단(bottom) ply**. k = 1..n이 아래→위. ply k는 [z_{k-1}, z_k] 구간을 차지 |
| D2 | 좌표계와 각도 | 두께 방향 +z(위쪽), x–y는 적층판 면내. 적층각 θ는 **적층판 +x축에서 섬유 1축으로, +z축 기준 반시계(CCW) 회전이 양수**. 입력은 도(deg) 단위 `angle_deg`, 권장 범위 [-90, 90], (-360, 360) 입력은 정규화 |
| D3 | 전단 규약 | 공학 전단변형률(engineering shear, γ = 2ε₁₂) 기반 Voigt 표기. Q̄ 변환식은 §4.2의 것을 그대로 사용 |
| D4 | 단위계 | `unit_system` ∈ {`SI`, `SI_mm`}. `SI` = (Pa, m, kg/m³), `SI_mm` = (MPa, mm, t/mm³). 내부 계산은 항상 SI로 정규화. imperial(psi, in)은 V1 |
| D5 | 기준면 | 내부 계산은 항상 **midplane 기준** (z=0이 두께 중앙). 중립면 출력은 midplane 기준, bottom 기준, 무차원(ζ=(z−z_bot)/h) 세 가지를 항상 병기 |
| D6 | 전송(transport) | **stdio가 기본**. `--transport http`(streamable HTTP)는 운영 옵션. 포트 탐색은 HTTP 모드의 서버 기동 로직(§10)이며 MCP Tool이 아님 |
| D7 | 상태 관리 | **완전 무상태(stateless)**. 세션·캐시·서버측 저장 없음. 모든 Tool은 완결된 payload를 받아 완결된 응답을 반환 |
| D8 | 결정론 | 기본 응답은 바이트 단위 결정론적. 시각·소요시간 등 비결정 요소는 `include_debug=true`일 때만 `debug` 블록에 포함 (§6.5) |
| D9 | 수치 표현 | 내부 float64. 응답 직렬화는 Python `repr` 기반 shortest round-trip(전정밀도). 서버측 반올림 없음 |
| D10 | 언어 | ~~영어~~ → **한국어로 개정** (v2.4, Q8 채택). materialtwin이 한국어 오류 관례를 5라운드 적대 리뷰로 확정한 상태라, 같은 생태계에서 함께 쓰는 본 서버도 Tool description·오류·suggestion을 한국어로 정렬 |
| D11 | 명명 | Tool·필드 snake_case. 서버 이름 `laminate-analyzer`. 문서 용어는 "중립면(neutral surface)"이되 beam 모드 설명에서만 neutral axis 허용 |
| D12 | 구현 스택 | Python ≥ 3.11, 공식 `mcp` Python SDK(FastMCP), numpy ≥ 1.26, pydantic ≥ 2.7, pytest + hypothesis, sympy(테스트 전용 오라클). scipy 불필요 |

---

## 4. 수학 사양 (완전판)

### 4.1 재료 강성 Q (plane stress reduced stiffness)

직교이방성 ply (`E1, E2, G12, nu12`) 에 대해

$$
\nu_{21}=\nu_{12}\frac{E_2}{E_1},\qquad
\Delta = 1-\nu_{12}\nu_{21}
$$

$$
Q_{11}=\frac{E_1}{\Delta},\quad
Q_{22}=\frac{E_2}{\Delta},\quad
Q_{12}=\frac{\nu_{12}E_2}{\Delta},\quad
Q_{66}=G_{12},\quad
Q_{16}=Q_{26}=0
$$

등방성 ply (`E, nu`)는 E₁=E₂=E, ν₁₂=ν, G₁₂=E/(2(1+ν))로 위 식에 대입한다(각도 무관).

**물성 유효성 조건 (위반 시 오류)**
- E₁, E₂, G₁₂ > 0
- Δ = 1 − ν₁₂ν₂₁ > 0, 즉 |ν₁₂| < √(E₁/E₂)
- 등방성은 −1 < ν < 0.5

### 4.2 좌표 변환 Q̄

m = cos θ, n = sin θ 로 두면

$$
\begin{aligned}
\bar{Q}_{11}&=Q_{11}m^4+2(Q_{12}+2Q_{66})m^2n^2+Q_{22}n^4\\
\bar{Q}_{22}&=Q_{11}n^4+2(Q_{12}+2Q_{66})m^2n^2+Q_{22}m^4\\
\bar{Q}_{12}&=(Q_{11}+Q_{22}-4Q_{66})m^2n^2+Q_{12}(m^4+n^4)\\
\bar{Q}_{66}&=(Q_{11}+Q_{22}-2Q_{12}-2Q_{66})m^2n^2+Q_{66}(m^4+n^4)\\
\bar{Q}_{16}&=(Q_{11}-Q_{12}-2Q_{66})m^3n+(Q_{12}-Q_{22}+2Q_{66})mn^3\\
\bar{Q}_{26}&=(Q_{11}-Q_{12}-2Q_{66})mn^3+(Q_{12}-Q_{22}+2Q_{66})m^3n
\end{aligned}
$$

beam 모드용 방향별 공학 탄성계수는 컴플라이언스 변환으로 구한다.

$$
S_{11}=\tfrac{1}{E_1},\; S_{22}=\tfrac{1}{E_2},\; S_{12}=-\tfrac{\nu_{12}}{E_1},\; S_{66}=\tfrac{1}{G_{12}}
$$

$$
\bar{S}_{11}(\theta)=S_{11}m^4+(2S_{12}+S_{66})m^2n^2+S_{22}n^4,
\qquad E_x(\theta)=\frac{1}{\bar{S}_{11}(\theta)}
$$

### 4.3 z-좌표 생성

총 두께 h = Σtₖ. midplane 기준으로

$$
z_0=-\frac{h}{2},\qquad z_k=z_{k-1}+t_k\;(k=1..n),\qquad \bar z_k=\frac{z_{k-1}+z_k}{2}
$$

### 4.4 ABD 적분

$$
A_{ij}=\sum_k \bar{Q}_{ij}^{(k)}(z_k-z_{k-1}),\qquad
B_{ij}=\frac{1}{2}\sum_k \bar{Q}_{ij}^{(k)}(z_k^2-z_{k-1}^2),\qquad
D_{ij}=\frac{1}{3}\sum_k \bar{Q}_{ij}^{(k)}(z_k^3-z_{k-1}^3)
$$

구성 방정식 (N은 단위 폭당 힘 [N/m], M은 단위 폭당 모멘트 [N·m/m = N] — **단위 폭당 물리량**임을 스키마 description에 반드시 명시할 것).

$$
\begin{bmatrix}\mathbf{N}\\\mathbf{M}\end{bmatrix}
=\begin{bmatrix}\mathbf{A}&\mathbf{B}\\\mathbf{B}&\mathbf{D}\end{bmatrix}
\begin{bmatrix}\boldsymbol{\varepsilon}_0\\\boldsymbol{\kappa}\end{bmatrix}
$$

### 4.5 중립면 정의 (3종)

`mode` 파라미터로 선택하며, 서로 값이 다를 수 있음을 응답의 `axis_definition`으로 항상 밝힌다.

**(a) `beam_equivalent`** — 환산단면(transformed section) 중립축. ply별 공학 계수 E_x(θₖ) 가중 도심.

$$
z_{ns}=\frac{\sum_k E_x(\theta_k)\,t_k\,\bar z_k}{\sum_k E_x(\theta_k)\,t_k}
$$

가정. 1축 응력 상태(측방 수축 자유), 폭 방향 커플링 무시. 좁은 스트립/보 해석에 적합.

**(b) `clt_weighted`** — Q̄₁₁(x방향) 또는 Q̄₂₂(y방향) 가중 도심. ABD와 다음 관계로 동치다(구현은 이 식으로 검증).

$$
z_{ns,x}=\frac{\sum_k \bar Q_{11}^{(k)} t_k \bar z_k}{\sum_k \bar Q_{11}^{(k)} t_k}=\frac{B_{11}}{A_{11}},
\qquad z_{ns,y}=\frac{B_{22}}{A_{22}}
$$

**(c) `clt_full_coupled` (V1)** — 하중 기반 정의. N = 0, M = (M̂,0,0) 를 완전 ABD 역산으로 풀어 ε₀, κ를 얻고

$$
z_{ns,x}=-\frac{\varepsilon_{0x}}{\kappa_x}\quad(\kappa_x\neq 0)
$$

포아송·전단 커플링을 모두 반영한 값. **비대칭 적층에서는 중립면이 하중 조합에 의존**하므로 이 모드는 하중 케이스별로 보고한다.

공통 성질. 대칭 적층이면 세 정의 모두 z_ns = 0 (midplane). 이는 성질 기반 테스트 P5의 검증 항목이다.

### 4.6 정규화 ABD (지표·조건수 계산 전용)

A, B, D는 단위가 달라 그대로 섞어 쓸 수 없다. 전 항목이 Pa 차원이 되도록 다음 **프로젝트 표준 정규화**를 정의한다.

$$
\hat{\mathbf{A}}=\frac{\mathbf{A}}{h},\qquad
\hat{\mathbf{B}}=\frac{\sqrt{12}\,\mathbf{B}}{h^2},\qquad
\hat{\mathbf{D}}=\frac{12\,\mathbf{D}}{h^3},\qquad
\hat{\mathbf{K}}=\begin{bmatrix}\hat{\mathbf{A}}&\hat{\mathbf{B}}\\\hat{\mathbf{B}}&\hat{\mathbf{D}}\end{bmatrix}
$$

B의 계수가 2가 아니라 **√12**인 이유. K̂ = S·[A B; B D]·S, S = diag(h^{-1/2}I₃, (12/h³)^{1/2}I₃) 인 **합동변환(congruence)** 이 되려면 비대각 계수가 √(1/h)·√(12/h³) = √12/h² 이어야 한다. 합동변환이므로 양정치성이 원행렬과 정확히 보존되고, 균질 단일재 판에서 Â = D̂ = Q̄가 되어 해석이 직관적이며, 두께 균등 스케일링(t→αt)에 대해 K̂이 불변이라 조건수·지표가 두께 절대값에 오염되지 않는다.

- `abd_condition_number` := cond₂(K̂)
- `positive_definite` := K̂에 대한 Cholesky 성공 여부

### 4.7 하중 응답과 유효 공학 상수 (V1)

완전 역산으로 컴플라이언스를 얻는다.

$$
\begin{bmatrix}\boldsymbol{\varepsilon}_0\\\boldsymbol{\kappa}\end{bmatrix}
=\begin{bmatrix}\boldsymbol{\alpha}&\boldsymbol{\beta}\\\boldsymbol{\beta}^{T}&\boldsymbol{\delta}\end{bmatrix}
\begin{bmatrix}\mathbf{N}\\\mathbf{M}\end{bmatrix}
$$

역산은 역행렬 대신 `numpy.linalg.solve`(또는 Cholesky) 사용. 유효 막 공학 상수는

$$
E_x^{\mathrm{eff}}=\frac{1}{h\,\alpha_{11}},\quad
E_y^{\mathrm{eff}}=\frac{1}{h\,\alpha_{22}},\quad
G_{xy}^{\mathrm{eff}}=\frac{1}{h\,\alpha_{66}},\quad
\nu_{xy}^{\mathrm{eff}}=-\frac{\alpha_{12}}{\alpha_{11}}
$$

(비대칭 적층은 α에 B 커플링 효과가 포함된 값임을 응답 assumption으로 명시. 굽힘 유효 상수는 δ 기반으로 동일 패턴.)

### 4.8 모델 가정 (모든 응답의 `assumptions[]`로 노출)

- Kirchhoff–Love 판이론(CLT). 횡전단 변형 무시 → 두꺼운 판·샌드위치 코어에는 부정확
- 평면응력 상태의 선형 탄성, 완전 접착(층간 슬립 없음), 층 두께 균일
- 열·흡습 잔류변형 미포함 (V2)
- N, M은 단위 폭당 물리량

---

## 5. 평가 지표 명세

원안 §4의 브레인스토밍을 "정의식이 있는 지표"로 승격했다. 모든 지표는 무차원이고, 정의식·전형 범위·등급 기준을 함께 반환한다. 등급은 임의 점수화 대신 **원시값 + 등급 밴드** 로 보고한다(에이전트가 원시값으로 추가 판단 가능하도록).

### 5.1 MVP 지표

| 지표 | 정의 | 등급 밴드 (기본값) |
|---|---|---|
| `total_thickness` | h | — |
| `areal_mass` | Σρₖtₖ (밀도 입력 시에만) | — |
| `coupling_ratio` | ‖B̂‖_F / √(‖Â‖_F‖D̂‖_F) | <0.01 negligible / <0.05 low / <0.2 moderate / ≥0.2 high |
| `is_symmetric_stack` | 적층 정의의 기하학적 회문 검사 (재료·두께·각도) | bool |
| `membrane_anisotropy` | Â₁₁/Â₂₂ | 0.5~2 balanced / 그 외 directional |
| `bending_anisotropy` | D̂₁₁/D̂₂₂ | 동일 |
| `shear_extension_coupling` | max(\|Â₁₆\|, \|Â₂₆\|) / √(Â₁₁Â₂₂) | <0.01 balanced / <0.1 mild / ≥0.1 strong |
| `bend_twist_coupling` | max(\|D̂₁₆\|, \|D̂₂₆\|) / √(D̂₁₁D̂₂₂) | 동일 |
| `quasi_isotropy_score` | 1 − min(1, (e₁+e₂+e₃)/3), 아래 정의 | >0.95 quasi-isotropic / >0.8 near / ≤0.8 anisotropic |
| `ns_offset_ratio` | z_ns,x / h ∈ [−0.5, 0.5] | \|·\|<0.01 centered / <0.1 offset / ≥0.1 strongly offset |
| `abd_condition_number` | cond₂(K̂) (§4.6) | >1e8이면 W401 경고 |
| `positive_definite` | Cholesky(K̂) 성공 여부 | false면 E402 |

`quasi_isotropy_score`의 편차항 (Q_ref = (Â₁₁+Â₂₂)/2).

$$
e_1=\frac{|\hat A_{11}-\hat A_{22}|}{Q_{ref}},\quad
e_2=\frac{|\hat A_{16}|+|\hat A_{26}|}{Q_{ref}},\quad
e_3=\frac{|\hat A_{66}-\tfrac{1}{2}(\hat A_{11}-\hat A_{12})|}{Q_{ref}}
$$

(등방 재료 단일층에서 e₁=e₂=e₃=0 → score 1이 되는 것이 단위 테스트 항목이다.)

### 5.2 V1 지표

| 지표 | 정의 |
|---|---|
| `membrane_bending_leakage` | 단위 막하중 N=(N̂,0,0) (N̂=1 정규화), M=0 에서 완전 역산 후 (h/2)‖κ‖₂ / ‖ε₀‖₂. 대칭이면 0 |
| `twist_under_bending` | M=(M̂,0,0), N=0 에서 \|κ_xy\|/\|κ_x\| |
| `effective_constants` | §4.7의 E_x, E_y, G_xy, ν_xy (막/굽힘 각각) |
| `in_plane_principal_direction` | E_x^eff(φ)를 φ∈[0°,180°) 1° 간격 스캔한 최대 강성 방향(결정론적 그리드, 난수 없음) |
| `specific_stiffness` | E_x^eff / (areal_mass/h) — 밀도 입력 시에만 |
| `unit_load_response_library` | 단위 하중 6케이스 (N_x, N_y, N_xy, M_x, M_y, M_xy 각 1) 에 대한 ε₀, κ 표 |
| `dominant_coupling_terms` | B̂ 성분을 크기순 정렬한 상위 항 목록 (B11, B16 등 어느 커플링이 지배적인지) |

### 5.3 V2로 이동한 지표

- `warpage_tendency_score` — 제조 후 뒤틀림은 경화 냉각의 열잔류변형이 지배하므로 **CTE(α₁, α₂)와 ΔT 입력이 추가되는 V2에서** 열 모멘트 기반으로 계산한다. MVP에서는 `coupling_ratio`와 `dominant_coupling_terms`가 간접 지표 역할을 한다는 주석을 응답에 포함.
- `manufacturing_robustness_score` — V1 `run_sensitivity_analysis` 결과의 요약 통계로 정의(별도 지표가 아니라 민감도 Tool의 파생값).
- `target_stiffness_gap` — 목표 A/D 대비 거리. 목표 스키마가 필요하므로 V1 `evaluate_laminate`의 선택 입력 `criteria`로 흡수.

### 5.4 지표 공통 규칙

- 분모가 0에 접근하는 지표(비율류)는 |분모| < 1e-30(SI) 이면 값 대신 `null` + W402 경고를 반환한다. NaN/Inf는 어떤 경우에도 응답에 포함하지 않는다.
- 모든 지표는 `indices` 객체에 {value, grade, definition} 형태로 담는다. definition은 한 줄 영어 수식 문자열(에이전트가 재검산 가능하도록).

---

## 6. MCP 인터페이스 명세

### 6.1 서버 기동과 전송

- 기본 전송은 **stdio**. Claude Code/Desktop 로컬 등록 시 포트가 필요 없다.
- `--transport http --host 127.0.0.1 --port 0` 으로 streamable HTTP 모드 지원(공유 서버 운영용). port 0이면 OS 자동 할당, 선택된 포트는 stderr 기동 로그로 출력.
- **stdio 모드에서 stdout에는 JSON-RPC 외 어떤 바이트도 쓰지 않는다.** 로그는 전부 stderr. (`print()` 한 줄이 프로토콜을 깨는 대표 사고 지점이므로 lint 규칙으로 차단.)
- 기동 로그(stderr): server_version, engine_version, transport, port(HTTP일 때), pid.

### 6.2 Tool 카탈로그

| # | Tool | 단계 | 목적 |
|---|---|---|---|
| 1 | `analyze_laminate` | MVP | **에이전트 권장 진입점.** 검증→ABD→중립면→지표를 한 번에 수행하는 원샷 Tool. `include` 옵션으로 부분 선택 가능 |
| 2 | `validate_laminate_input` | MVP | 검증만 수행(계산 없음). 대화형 입력 구성 단계에서 사용 |
| 3 | `compute_abd_matrix` | MVP | ABD만 계산 |
| 4 | `compute_neutral_axis` | MVP | 중립면만 계산 (mode 선택) |
| 5 | `evaluate_laminate` | MVP | 지표 + 등급 + (선택) 사용자 기준 대비 pass/fail + 권고 |
| 6 | `get_reference_cases` | MVP | 내장 기준 케이스의 입력/기대출력 반환. **에이전트가 스키마를 few-shot으로 학습하고 서버를 자가 검증하는 용도** |
| 7 | `get_server_info` | MVP | 버전, 지원 단위계, 한계값(§11), 지표 목록 |
| 8 | `solve_load_response` | V1 | N, M 입력 → ε₀, κ, 유효 공학 상수, 단위하중 라이브러리 |
| 9 | `run_sensitivity_analysis` | V1 | 각도/두께/물성 섭동에 대한 주요 출력의 민감도(중앙차분, 결정론). 에이전트가 20회 호출할 것을 1회로 줄이는 배치 성격 |
| 10 | `generate_design_report` | V1 | 사람용 Markdown 리포트 + LLM용 요약 JSON |
| 11 | `batch_evaluate_laminates` | V1 | 최대 32개 적층안 일괄 평가(에이전트 주도 탐색 루프 가속용) |

원안 대비 변경. `get_server_health` → `get_server_info`로 통합(MCP에 ping이 내장되어 있어 health 전용 Tool은 불필요). `find_available_port` 삭제(§2). V2 후보는 §14 뒤 로드맵 참고.

### 6.3 대표 Tool 명세

공통 입력인 `laminate` 객체는 §7.1을 따른다. 아래는 개별 파라미터만 기술한다.

**`analyze_laminate`**
- 입력: `laminate` (필수), `include` (선택, 기본 ["abd","neutral_axis","indices"]), `neutral_axis_mode` (선택, 기본 "clt_weighted"), `include_debug` (선택, 기본 false)
- 출력: envelope의 `data`에 §7.2의 abd/neutral_surface/indices 블록
- 오류: 입력 검증 실패 시 E1xx/E2xx/E3xx, 수치 실패 시 E4xx

**`compute_neutral_axis`**
- 입력: `laminate`, `mode` ∈ {"beam_equivalent","clt_weighted","clt_full_coupled"(V1)}
- 출력: `neutral_surface` 블록(§7.2). clt_weighted는 x/y 두 값을 모두 반환
- 특기. mode별 정의와 적용 한계를 `axis_definition` 문자열로 응답에 포함(에이전트가 사용자에게 그대로 전달 가능한 한 문장)

**`evaluate_laminate`**
- 입력: `laminate`, `criteria` (선택) — 예: {"max_coupling_ratio": 0.05, "min_quasi_isotropy": 0.9, "target_D11": {...}}
- 출력: `indices` + `evaluation` {pass_fail[], recommendations[]}. criteria 미제공 시 pass_fail은 생략하고 등급만 반환

**`get_reference_cases`**
- 입력: `case_id` (선택, 생략 시 목록 반환)
- 출력: {case_id, description, input(완전한 laminate payload), expected(핵심 기대값과 허용오차)}

### 6.4 공통 응답 envelope와 오류 코드

```json
{
  "status": "ok | warning | error",
  "data": { ... } ,
  "errors":   [{"code": "E200", "message": "...", "field": "laminae[2].material.E1", "suggestion": "..."}],
  "warnings": [{"code": "W110", "message": "...", "field": "...", "suggestion": "..."}],
  "assumptions": ["Classical Lamination Theory (Kirchhoff-Love): transverse shear neglected", "..."],
  "metadata": {
    "server_version": "0.1.0", "engine_version": "0.1.0",
    "unit_system_in": "SI_mm", "units": {"E": "MPa", "thickness": "mm", "A": "N/mm", "B": "N", "D": "N*mm"},
    "method": "clt", "payload_hash": "sha256:..."
  }
}
```

- `status`는 errors가 있으면 error(이때 `data`는 null), warnings만 있으면 warning.
- 오류 코드 체계 (전수. 각 코드는 유발 테스트를 가져야 함 — 성공 기준 S4).

| 코드 | 이름 | 설명 / suggestion 방향 |
|---|---|---|
| E100 | SCHEMA_INVALID | pydantic 검증 실패. 실패 필드 경로와 기대 타입 |
| E101 | UNKNOWN_UNIT_SYSTEM | 지원 목록 제시 |
| E102 | EMPTY_LAMINATE | laminae ≥ 1 필요 |
| E103 | TOO_MANY_PLIES | 한계값(§11)과 분할 제안 |
| E104 | PAYLOAD_TOO_LARGE | 한계값 제시 |
| E200 | NON_POSITIVE_MODULUS | 해당 ply 인덱스, 필드 지목 |
| E201 | POISSON_UNSTABLE | Δ ≤ 0. \|ν₁₂\| < √(E₁/E₂) 조건 제시 |
| E202 | INVALID_MATERIAL_TYPE | isotropic/orthotropic_2d 중 택일 |
| E300 | NON_POSITIVE_THICKNESS | 해당 ply 인덱스 지목 |
| E301 | ANGLE_OUT_OF_RANGE | 정규화 범위 안내 |
| E400 | SINGULAR_SYSTEM | 완전 역산 불가. 적층/물성 재검토 제안 |
| E402 | NOT_POSITIVE_DEFINITE | 물리적으로 불가능한 강성. 물성 입력 오류 가능성 안내 |
| E403 | NAN_INF_DETECTED | 내부 방어선. 입력 echo와 함께 버그 리포트 안내 |
| E500 | COMPUTE_TIMEOUT | 한계값 제시 |
| E501 | INTERNAL_ERROR | stack trace 비노출, 오류 id만 |
| W110 | SUSPICIOUS_UNIT_MAGNITUDE | 단위 오입력 휴리스틱. 예) SI(Pa)인데 E₁=200 → "200 Pa is implausibly soft; did you mean 200e9 Pa or unit_system SI_mm?" |
| W111 | VERY_THIN_PLY | tₖ/h < 1e-6 |
| W112 | EXTREME_MODULUS_RATIO | E₁/E₂ > 1e4 등 |
| W401 | ILL_CONDITIONED | cond₂(K̂) > 1e8 |
| W402 | INDEX_UNDEFINED | 분모 소멸로 일부 지표 null (§5.4) |
| W200 | HIGH_COUPLING | coupling_ratio ≥ 0.2 정보성 경고 |

- W110은 성공 기준 S2의 핵심 장치다. 판정 휴리스틱(기본값). SI에서 1 ≤ E < 1e6 Pa, 또는 thickness > 1 m, SI_mm에서 E > 1e7 MPa 등이면 발동.

### 6.5 결정론 정책과 payload_hash

- 결정론 보장 범위. `data`, `errors`, `warnings`, `assumptions`, `metadata`(payload_hash 포함) 전부. 보장 제외. `include_debug=true`일 때의 `debug`{elapsed_ms, timestamp} 블록.
- `payload_hash` = sha256(canonical_json(입력 payload)). canonical_json은 UTF-8, 키 정렬, 구분자 `(",", ":")`, float는 repr 그대로. 동일 입력 재현·감사 추적용.
- 난수·시각·딕셔너리 순회 순서에 의존하는 코드 금지. 병렬화 없음(단일 요청 계산이 ms 단위라 불필요).

### 6.6 에이전트 UX 규칙 (Tool 작성 지침)

- **description이 곧 문서다.** 각 Tool description에는 (1) 한 줄 목적, (2) 단위 규약 요약("all per-unit-width"), (3) 대표 호출 예 1개, (4) 언제 다른 Tool을 쓸지 한 줄을 포함한다. 각 필드 description에는 단위와 유효 범위를 명시한다.
- 행렬은 3×3 중첩 배열(row-major)로 반환. 6×6 ABD 결합 행렬은 기본 미포함(`include_abd_6x6=true` 옵션) — 토큰 절약.
- 응답 크기 상한을 의식한다. `analyze_laminate` 전체 응답이 대략 4KB를 넘지 않도록 설계하고, `batch_evaluate`는 케이스당 요약(핵심 지표만)을 기본으로 한다.
- 오류의 `suggestion`은 "무엇을 어떻게 바꿔 재호출하라"는 실행 가능한 문장으로 쓴다(에이전트의 자가 복구 루프 전제).
- 서버 수준 `instructions` 필드(MCP 초기화 시 호스트에 전달)에 권장 호출 순서와 단위 규약을 3~5줄로 기술한다(§10.3 템플릿).

---

## 7. 데이터 스키마

### 7.1 입력 `laminate` 객체

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `unit_system` | "SI" \| "SI_mm" | ✔ | §3 D4 |
| `laminae` | array, 1..512 | ✔ | **index 0 = 최하단 ply** (§3 D1) |
| `laminae[].thickness` | number > 0 | ✔ | m 또는 mm |
| `laminae[].angle_deg` | number | ✔ | §3 D2 규약 |
| `laminae[].material` | object | ✔ | 아래 두 형태 중 하나 |
| `laminae[].material.type` | "isotropic" \| "orthotropic_2d" | ✔ | |
| (isotropic) `E`, `nu` | number | ✔ | |
| (orthotropic_2d) `E1`, `E2`, `G12`, `nu12` | number | ✔ | 1축 = 섬유 방향 |
| `laminae[].material.rho` | number > 0 | – | 밀도. 있으면 질량 지표 계산 |
| `laminae[].material.name` | string | – | 추적용 라벨 |
| `name` | string | – | 적층안 라벨 (batch 식별용) |
| `reference_output` | "midplane" \| "bottom" \| "top" | – | 기본 midplane. **출력 표기 선호일 뿐이며 어떤 값이든 세 표기 모두 반환** (§3 D5) |

원안과의 차이. `geometry{width,length}`는 ABD·중립면·지표 계산에 불필요하므로 MVP 입력에서 제거했다(beam 절대 강성 EI나 총질량이 필요한 V1 리포트에서 선택 입력으로 부활). `reference_z`는 입력 기하에 영향이 없고 출력 선호일 뿐이므로 `reference_output`으로 개명. `mode`는 laminate 객체에서 빼서 `compute_neutral_axis`의 파라미터로 이동(ABD 계산과 무관하므로).

V1 추가 예정. `materials` 레지스트리(중복 물성 참조용 `material_id`), `layup_notation`("[0/45/90]s" 문자열 파서), CTE 필드(V2).

### 7.2 출력 `data` 블록

```json
{
  "laminate_summary": {"n_plies": 4, "total_thickness": 0.0005, "areal_mass": 0.8, "is_symmetric_stack": true},
  "abd": {
    "A": [[...3x3...]], "B": [[...]], "D": [[...]],
    "A_hat": [[...]], "B_hat": [[...]], "D_hat": [[...]]
  },
  "neutral_surface": {
    "mode": "clt_weighted",
    "x": {"z_from_midplane": 0.0, "z_from_bottom": 0.00025, "zeta": 0.5},
    "y": {"z_from_midplane": 0.0, "z_from_bottom": 0.00025, "zeta": 0.5},
    "axis_definition": "Q11-weighted centroid, equals B11/A11; direction-decoupled approximation"
  },
  "indices": {
    "coupling_ratio": {"value": 0.0, "grade": "negligible", "definition": "||B_hat||_F / sqrt(||A_hat||_F * ||D_hat||_F)"},
    "quasi_isotropy_score": {"value": 0.62, "grade": "anisotropic", "definition": "..."}
  },
  "evaluation": {"pass_fail": [], "recommendations": []}
}
```

### 7.3 완전한 호출 예시 (문서·description·테스트 공용 픽스처)

입력 ([0/90]s, T300/5208급 물성, SI_mm).

```json
{
  "laminate": {
    "unit_system": "SI_mm",
    "name": "cross_ply_symmetric",
    "laminae": [
      {"thickness": 0.125, "angle_deg": 0,  "material": {"type": "orthotropic_2d", "E1": 181000, "E2": 10300, "G12": 7170, "nu12": 0.28, "name": "T300/5208"}},
      {"thickness": 0.125, "angle_deg": 90, "material": {"type": "orthotropic_2d", "E1": 181000, "E2": 10300, "G12": 7170, "nu12": 0.28}},
      {"thickness": 0.125, "angle_deg": 90, "material": {"type": "orthotropic_2d", "E1": 181000, "E2": 10300, "G12": 7170, "nu12": 0.28}},
      {"thickness": 0.125, "angle_deg": 0,  "material": {"type": "orthotropic_2d", "E1": 181000, "E2": 10300, "G12": 7170, "nu12": 0.28}}
    ]
  }
}
```

기대 성질(수치는 Phase 1에서 확정해 `get_reference_cases`와 회귀 테스트에 내장). B ≈ 0, z_ns = 0, A₁₆ = A₂₆ = 0, D₁₁ ≠ D₂₂(외측 0° ply가 굽힘 지배), status = "ok".

---

## 8. 검증 계획

정답의 출처를 4원화한다. **(1) 폐형해, (2) 성질 기반, (3) 독립 오라클, (4) 문헌 벤치마크.** 수치 정답을 사람이 손으로 박아 넣는 방식은 이 문서에서 금지한다(전사 오류가 그대로 "정답"이 되는 위험 차단).

### 8.1 폐형해 기준 케이스 (수식을 테스트에 내장)

| ID | 케이스 | 기대값 (기호 수식 그대로 코드화) |
|---|---|---|
| R1 | 등방 단일층 (E, ν, h) | A = hQ_iso, B = 0, D = (h³/12)Q_iso, z_ns = 0, quasi_isotropy_score = 1 |
| R2 | 직교이방 단일층 0° (h) | A = hQ, B = 0, D = (h³/12)Q, z_ns = 0 |
| R3 | [0/90] 비대칭 2층 (각 t) | A₁₁=A₂₂=(Q₁₁+Q₂₂)t, A₁₂=2Q₁₂t, A₆₆=2Q₆₆t, A₁₆=0, B₁₁=−B₂₂=(Q₂₂−Q₁₁)t²/2, B₁₂=B₆₆=B₁₆=B₂₆=0, D₁₁=D₂₂=(Q₁₁+Q₂₂)t³/3, z_ns,x=B₁₁/A₁₁=t(Q₂₂−Q₁₁)/(2(Q₁₁+Q₂₂)), beam 모드는 t(E₂−E₁)/(2(E₁+E₂)) |
| R4 | [0/90]s 대칭 4층 (각 t) | B = 0, A₁₁=A₂₂=2t(Q₁₁+Q₂₂), D₁₁=(14Q₁₁+2Q₂₂)t³/3, D₂₂=(2Q₁₁+14Q₂₂)t³/3, z_ns=0 |
| R5 | [+θ/−θ] 반대칭 2층 (각 t) | A₁₆=A₂₆=0, D₁₆=D₂₆=0, B₁₁=B₁₂=B₂₂=B₆₆=0, B₁₆=−t²Q̄₁₆(θ), B₂₆=−t²Q̄₂₆(θ) |

R3~R5는 D1(bottom-first) 규약 하의 도출이다. 구현이 규약을 어기면 부호가 뒤집혀 즉시 검출된다(규약 준수 테스트를 겸함).

### 8.2 성질 기반 테스트 (hypothesis로 무작위 적층 생성)

| ID | 성질 | 근거 |
|---|---|---|
| P1 | ply 순서를 임의로 섞어도 **A는 불변** | A는 각도 다중집합에만 의존 |
| P2 | 적층 상하 반전 시 A, D 불변, **B는 부호 반전** | z → −z 대칭 |
| P3 | 두께 균등 스케일 t→αt 시 A→αA, B→α²B, D→α³D, 그리고 K̂·조건수·모든 무차원 지표 불변 | 적분 차수 / §4.6 정규화 |
| P4 | 전 ply 공통 회전 φ에 대해 M∈{A,B,D} 각각 I₁(M)=M₁₁+M₂₂+2M₁₂ 와 I₂(M)=M₆₆−M₁₂ 불변 | 회전 불변량 (본 문서 작성 시 대수적으로 재확인 완료) |
| P5 | 대칭 적층 생성기 → ‖B̂‖_F ≤ tol, 세 중립면 정의 모두 z_ns ≈ 0 | 정의 |
| P6 | 유효 물성 범위의 임의 적층 → K̂ 양정치, Cholesky 성공 | 변형 에너지 양정치성 |
| P7 | balanced 적층(±θ 쌍) 생성기 → A₁₆=A₂₆≈0 | 정의 |
| P8 | 동일 payload 2회 호출 → 응답 바이트 동일 | 결정론 (S1) |

### 8.3 독립 오라클 (이중 구현)

- `tests/oracle/` 에 **sympy 기반 CLT를 별도로 구현**한다(운영 코드와 알고리즘·저자 분리, 유리수 연산). 무작위 케이스 N=50에 대해 numpy 엔진과 rtol 1e-9로 대조.
- 오라클 구현은 운영 코드를 import하지 않는다(복사-붙여넣기 오류 전파 차단).

### 8.4 문헌 벤치마크

- 재료. T300/5208급 흑연/에폭시 (E₁=181 GPa, E₂=10.3 GPa, G₁₂=7.17 GPa, ν₁₂=0.28) — Jones, *Mechanics of Composite Materials* 계열 교과서의 표준 예제 물성.
- Phase 4에서 교과서 예제([±45], [0/90], quasi-isotropic 등)의 공표값을 대조하고, 검증된 수치를 `get_reference_cases`와 회귀 테스트에 **그때** 고정한다(계획 단계에서 수치를 옮겨 적지 않는 이유는 §8 서두 참조).

### 8.5 허용오차 정책 (`tolerance_policy`로 문서화·응답 노출)

| 대조 대상 | 기준 |
|---|---|
| 엔진 vs 폐형해 | rtol 1e-12, atol 스케일 = 1e-12·‖A‖ 등 성분별 |
| 엔진 vs sympy 오라클 | rtol 1e-9 |
| 엔진 vs 문헌값 | rtol 5e-3 (공표값 반올림 감안) |
| B≈0 판정 (P5) | ‖B̂‖_F ≤ 1e-9·‖Â‖_F |

---

## 9. 아키텍처

```
LaminateAnalyzerMCP/
├─ pyproject.toml              # 패키징, [project.scripts] laminate-mcp
├─ server/
│  ├─ main.py                  # FastMCP 서버, Tool 등록, transport 선택 (엔진 로직 없음)
│  ├─ config.py                # 한계값(§11), 허용오차, 버전 상수
│  ├─ schemas.py               # pydantic v2 입력/출력 모델 (필드 description = 에이전트 문서)
│  ├─ errors.py                # 오류/경고 코드 enum + suggestion 템플릿 (§6.4의 단일 소스)
│  ├─ solver/                  # ★ 순수 함수 계층. MCP/pydantic을 import하지 않는다
│  │  ├─ material.py           # Q, Q̄, S̄, E_x(θ), 물성 유효성
│  │  ├─ abd.py                # z좌표, A/B/D, 정규화 K̂ (§4.6)
│  │  ├─ neutral_axis.py       # 3종 중립면 (§4.5)
│  │  └─ response.py           # V1: 하중 응답, 유효 상수
│  └─ services/
│     ├─ validation.py         # 스키마 이후의 물리 검증, W110 휴리스틱
│     ├─ evaluation.py         # §5 지표, 등급, 권고
│     └─ envelope.py           # 공통 응답 조립, canonical json, payload_hash
├─ tests/
│  ├─ test_material.py, test_abd.py, test_neutral_axis.py
│  ├─ test_properties.py       # §8.2 (hypothesis)
│  ├─ test_reference_cases.py  # §8.1, §8.4
│  ├─ test_mcp_stdio.py        # 프로토콜 스모크 (tools/list, 호출, 오류 envelope)
│  ├─ test_determinism.py      # P8, payload_hash
│  └─ oracle/clt_sympy.py      # §8.3 (운영 코드 import 금지)
└─ docs/
   ├─ mcp_laminate_planning.md # 본 문서
   ├─ math_spec.md             # §4의 유도 상세 (Phase 4)
   └─ agent_guide.md           # 에이전트용 사용 가이드 (Phase 5)
```

원칙.
- `solver/`는 numpy만 아는 순수 함수 계층으로 유지한다. MCP 없이 단독 테스트·재사용(다른 프로젝트에서 import) 가능해야 한다.
- 오류 코드와 suggestion 문구는 `errors.py` 한 곳에서만 정의한다(문서·테스트가 이를 소스로 참조).
- 의존성은 §3 D12로 고정. lock은 `uv`.

---

## 10. 운영 계획

- 로컬 등록(개발 기본). `claude mcp add laminate-analyzer -- uv run --directory /home/koopark/claude/LaminateAnalyzerMCP laminate-mcp`
- HTTP 모드(선택). `laminate-mcp --transport http --port 0` → stderr에 실제 포트 출력. 우선순위는 사용자 지정 포트 → 42000–42999 스캔 → OS 자동 할당(원안 정책 유지, 위치만 Tool에서 기동 옵션으로 이동).
- graceful shutdown(SIGTERM 처리), systemd/재기동 스크립트는 HTTP 운영 채택 시 Phase 6에서 작성.
- 버전 정책. `server_version`(인터페이스)과 `engine_version`(수치 커널)을 분리해 metadata로 노출(원안 유지).

### 10.3 서버 instructions 템플릿 (MCP 초기화 시 호스트로 전달)

```
Laminate analyzer: deterministic CLT calculations for composite laminates.
Start with analyze_laminate (one-shot). laminae[0] is the BOTTOM ply.
Angles in degrees, CCW from laminate x-axis. N/M are per-unit-width.
unit_system "SI" = Pa/m, "SI_mm" = MPa/mm. Never compute ABD yourself;
call get_reference_cases for worked examples.
```

---

## 11. 보안/안정성 한계값

| 항목 | 값 | 초과 시 |
|---|---|---|
| 최대 ply 수 | 512 | E103 |
| 최대 payload | 1 MiB | E104 |
| 계산 타임아웃 | 10 s | E500 |
| batch 크기 | 32 | E103 준용 |
| NaN/Inf 입력 | 금지 | E100 |
| NaN/Inf 출력 | 방어선 | E403 |
| stack trace | 응답 비노출, stderr 로그만 | E501 |
| 파일/네트워크 접근 | payload로부터 일절 없음 (경로·URL 필드 자체가 없음) | — |

---

## 12. 구현 단계 (Phase + DoD 게이트)

에이전트 구현 기준이므로 날짜 대신 게이트로 관리한다. 각 Phase는 DoD 통과 전 다음으로 진행 금지.

**Phase 0 — 사양 확정**
- 작업: §15 미결 사항을 사용자에게 확인(불가 시 기본값 채택 기록), 리포지토리 `git init` + 본 문서 커밋.
- DoD: §15 전 항목에 결정 기록 존재.

**Phase 1 — 계산 엔진 (solver/)**
- 작업: material.py, abd.py, neutral_axis.py 구현. §8.1 폐형해 + §8.2 P1~P7 테스트 작성.
- DoD: `pytest tests/ -x` 녹색, `pytest --cov=server/solver --cov-fail-under=90`.

**Phase 2 — MCP 래핑**
- 작업: schemas.py(필드 description 포함), errors.py, envelope.py, main.py(stdio). MVP Tool 7종 등록. 결정론 테스트(P8).
- DoD: `test_mcp_stdio.py` 녹색. MCP Inspector(`mcp dev server/main.py`)로 `analyze_laminate` 호출 데모. 모든 E/W 코드에 유발 테스트 존재(S4).

**Phase 3 — 평가 계층**
- 작업: evaluation.py(§5.1 지표 전부), W110 휴리스틱, `evaluate_laminate`·`get_reference_cases` 완성.
- DoD: 지표별 단위 테스트(등방 단일층 quasi_isotropy=1, 대칭 적층 coupling=0 등) 녹색.

**Phase 4 — 검증 강화**
- 작업: sympy 오라클(§8.3), 문헌 벤치마크 수치 확정·내장(§8.4), 강건성 테스트(극단 물성비·초박층·512 ply), math_spec.md.
- DoD: 기준 케이스 총 20개 이상 녹색(S3). 오라클 대조 rtol 1e-9 통과.

**Phase 5 — 에이전트 통합**
- 작업: agent_guide.md, 서버 instructions, Claude Code에 실제 등록 후 시나리오 테스트 — "이 적층 평가해줘" 한 문장으로 에이전트가 올바른 Tool 시퀀스를 밟는지, 오류 유발 입력에서 suggestion만으로 자가 복구하는지.
- DoD: S5 시나리오 통과 기록(대화 로그를 docs/에 보관).

**Phase 6 — 운영 옵션 (선택)**
- 작업: HTTP transport, 포트 자동 할당, 재기동 스크립트, 배포 가이드.
- DoD: HTTP 모드 스모크 테스트.

이후 V1(Tool 8~11, §5.2 지표), V2(열/흡습 CTE·ΔT와 warpage, 층별 응력 복원, Tsai-Wu/Tsai-Hill/Max Stress 파손 판정 — 강도 입력 Xt, Xc, Yt, Yc, S 스키마 추가)로 확장한다.

---

## 13. 구현 체크리스트

구현 에이전트가 직접 갱신할 것. (2026-07-16 구현 세션 반영 — 상세는 리포지토리 `context-notes.md`)

- [x] P0. §15 결정 기록(Q1~Q8 기본값 채택), git init, 초기 커밋
- [x] P1. material.py (Q, Q̄, S̄, E_x, 물성 검증)
- [x] P1. abd.py (z좌표, ABD, K̂ 정규화)
- [x] P1. neutral_axis.py (beam_equivalent, clt_weighted)
- [x] P1. 폐형해 테스트 R1~R5
- [x] P1. 성질 테스트 P1~P7 (hypothesis)
- [x] P1. solver 커버리지 ≥ 90% (실측 98%, services 포함)
- [x] P2. schemas.py + errors.py + envelope.py
- [x] P2. stdio 서버, MVP Tool 7종 (패키지는 `server/` 대신 `app/` — HEAXHub fastapi 스택 정합, 부록 C 14)
- [x] P2. 결정론 테스트 P8 + payload_hash
- [x] P2. 오류/경고 코드 유발 테스트 — E400(하중응답 V1)·E500(타임아웃 V1)만 트리거 유예, 카탈로그 전수 테스트로 대체
- [x] P2. 실프로세스 stdio 왕복 + uvicorn(root-path) 스모크로 Inspector 데모 대체
- [x] P3. evaluation.py 지표 + 등급 밴드 (9지표 + summary 3항목)
- [x] P3. W110 단위 휴리스틱 (+W111/W112/W120)
- [x] P3. get_reference_cases 내장 케이스 3종 (자가 검증 루프 테스트 포함)
- [x] P4. sympy 오라클(변환행렬 T·Reuter 독립 경로) + 무작위 50케이스 rtol 1e-9 대조
- [ ] P4. 문헌 벤치마크 수치 확정·내장 (유일한 잔여 — 교과서 공표값 대조)
- [x] P4. 강건성 테스트 (512 ply, 극단 물성비, 초박층, 각도 경계)
- [x] P4. math_spec.md (√12 합동변환 유도, B11/A11 동치 증명, 불변량 포함)
- [x] P5. 서버 instructions + `laminate://guide` 리소스 + agent_guide.md
- [x] P5. `.mcp.json` 상호 등록 + **S5 시나리오 로그** (docs/s5_scenario_log.md — 두 MCP 실 stdio 세션으로 materialtwin 실측 E→적층 해석→W110 복구→리포트 체인 기록)
- [x] P6. HTTP transport (streamable HTTP + /health) — hwax 등록 요구로 선행 구현
- [x] P6+. **HEAXHub 등록 완료**: integrations/ 심볼릭 링크(in-tree 통합) + 스캐너 실행 → 카탈로그 등록(v0.2.0)·빌드(6.1s)·서비스 기동(9117, /health 200)·Caddy 라우트 등록까지 실증. 포털 경유 접속은 authz 게이트(team) 통과용 토큰만 남음
- [x] V1. Tool 4종 (solve_load_response/run_sensitivity_analysis/batch_evaluate_laminates/generate_design_report) + §5.2 지표(누출/비틀림/유효상수/주강성방향/비강성/지배커플링항) + E400·E500 실트리거

---

## 14. 최종 산출물 정의

- 실행 가능한 MCP 서버 (stdio 기본, `uv run laminate-mcp`)
- Tool API 문서 = 스키마 description 그 자체 + agent_guide.md
- 검증 케이스/회귀 테스트 세트 (폐형해 + 성질 + 오라클 + 문헌, 20+)
- 수학 사양 문서 (math_spec.md)
- 운영/등록 가이드 (Claude Code 등록 명령 포함)

---

## 15. 미결 사항 (기본값 포함 — 결정 없이도 진행 가능)

| # | 질문 | 기본값 (미확인 시 채택) |
|---|---|---|
| Q1 | 기본 단위계를 SI_mm(MPa·mm)로 할까? (FEA 실무 관례) | 기본값 없음·필수 입력으로 강제 (unit_system 생략 시 E101 → 단위 실수 원천 차단) |
| Q2 | imperial(psi·in) 지원 필요? | V1로 연기 |
| Q3 | HTTP transport를 MVP에 포함? | 미포함 (stdio만, Phase 6 선택) |
| Q4 | `generate_design_report` 언어 | 한국어 기본 + 영어 옵션 |
| Q5 | V2 우선순위 (열 warpage vs 파손 판정) | 열 warpage 우선 (원안 4.4의 warpage 지표 의도 복원) |
| Q6 | materialtwin 소규모 증분(orientation 노출·입력, attributes 쓰기 — §16.4)의 진행 시점 | 본 프로젝트 V1 착수 시점에 MaterialTwinWeb 백로그로 함께 진행 (그 전까지는 시편 label 규약으로 우회) |
| Q7 | `estimate_ply_properties` (미시역학 추정 Tool) 시점 | V2 (열 warpage와 함께) |
| Q8 | 오류 메시지 언어 — materialtwin은 한국어 dict 관례 확정, 본 계획 D10은 영어 | ✅ **채택 완료(v2.4)** — 한국어로 정렬, D10 개정됨 |

---

## 16. 지식 축적과 materialtwin MCP 연계 (현황 실측 기반, MVP 범위 아님)

반복 사용으로 "특정 재료·물성 조합에 대한 이해"가 쌓이게 하되, D7(무상태)·D8(결정론)을 깨지 않는 설계. 본 절은 MaterialTwinWeb의 **실제 MCP 구현을 코드 레벨로 확인한 결과**(2026-07-16, `backend/mcp_server.py` 937줄, `docs/MCP.md`)에 기반한다.

### 16.1 연계 대상 현황 — materialtwin MCP는 이미 존재한다

| 항목 | 실측 내용 |
|---|---|
| 서버 | FastMCP `"materialtwin"`, stdio. `backend/mcp_server.py` 단일 파일. 웹과 동일 SQLite+Parquet 공유 |
| 도구 | **20종** — 조회 13 (`list_materials`, `get_material`, `get_curve`, `get_fits`, `get_mat_card`, `search_by_property`, `find_materials_in_property_range`, `database_summary`, `material_taxonomy`, `property_distribution`, `coverage_gaps`, `plot_curve`, `plot_ashby`) + 등록·수정·삭제 7 (`register_material`, `register_tensile_test`, `register_relaxation_test`, `update_material`, `delete_material`, `delete_test`, `recompute_properties`) |
| 리소스/프롬프트 | `materialtwin://guide`, `materialtwin://taxonomy` 리소스 + `find_material`, `register_test_data` 프롬프트 |
| 단위 관례 | 응답의 E는 **GPa(소수 3자리 반올림)**, 응력 MPa(2자리). 등록 입력은 strain 무차원 + stress MPa. LS-DYNA ton·mm·s 관례 |
| 오류 관례 | dict 도구는 `{"error": "한국어 사유"}` (MCP isError 아님, 한국어 확정 — 5라운드 적대 리뷰 거침) |
| 데이터 규칙 | `list_materials`/`search_by_property`는 **유효(valid=true) 시험만** 대표값으로 사용. `get_material`은 valid 플래그를 노출해 이상치 오인을 차단 |
| 등록 스코프 | `.mcp.json`이 **MaterialTwinWeb 프로젝트 스코프** — 본 프로젝트에서 함께 쓰려면 이쪽 `.mcp.json`(또는 user scope)에 materialtwin 등록을 추가해야 한다 |

따라서 v2.1에서 제안했던 "MCP 파사드 신설"은 **철회**한다. 필요한 것은 신설이 아니라 (a) 본 프로젝트에서의 등록, (b) 에이전트 조합 절차, (c) 아래 16.4의 소규모 갭 보완뿐이다.

### 16.2 원칙 — 지식은 3층으로 분리, 계산기는 끝까지 무상태

| 층 | 내용 | 저장소 | 기록 주체 |
|---|---|---|---|
| L1 재료 지식 | 실측 E·항복·UTS·구성방정식 피팅·점탄성(Prony) | **materialtwin DB** (조회·등록 모두 MCP로 이미 가능) | materialtwin MCP |
| L2 설계 이력 | 평가한 적층안의 payload_hash + 입력 요약 + 핵심 지표 | 사용 프로젝트의 `laminate_runs.jsonl` | **에이전트** (매 호출 후 1줄 append) |
| L3 설계 인사이트 | "유리/카본 하이브리드는 비대칭 시 coupling_ratio가 쉽게 0.2를 넘는다" 류 정성 지식 | 에이전트 메모리 / 프로젝트 CLAUDE.md | 에이전트 |

계산 서버에 지식을 넣지 않는 이유. 상태가 생기면 D7·D8이 무너지고, 재료 DB는 materialtwin과 중복 구축이 된다. L2 로그가 있으면 "전에 평가한 유사 적층 찾기"는 에이전트가 jsonl을 grep하는 것으로 충분하다.

### 16.3 재료 해석 체인 — "강성을 모를 때" 에이전트 표준 동작 (실제 도구명 기준)

1. 사용자 제공 인라인 물성
2. LaminateAnalyzer 내장 `materials` 레지스트리 (V1)
3. **materialtwin 조회** — `list_materials(query=...)` 또는 `search_by_property(prop="E_GPa", ...)` → `get_material(id)`로 상세 확인(**valid=true 시험만** 채택) → 필요 시 `get_curve`/`get_fits`로 근거 확인
4. **미시역학 추정** — `estimate_ply_properties` Tool (V2 신규, Q7). 섬유/기지 물성 + V_f → E₁(혼합법칙), E₂·G₁₂(Halpin-Tsai), ν₁₂(혼합법칙). 결정론적 순수 함수라 본 서버 철학에 부합
5. 문헌 대표값 + 명시적 경고, 그래도 없으면 사용자에게 질문

**단위 브리지 (에이전트 책임 + 서버 크로스체크).** materialtwin의 `E_GPa`를 본 서버 `SI_mm`(MPa)로 넘길 때 ×1000 변환은 에이전트가 수행한다. 실수하면 본 서버의 W110(자릿수 휴리스틱)이 잡아내는 이중 방어 구조다. E_GPa가 소수 3자리 반올림(=1 MPa 분해능)이라는 점은 적층 계산 입력으로 충분함을 명시해 둔다.

**provenance 스키마 확장 (V1, MVP 스키마 불변).**
- `laminae[].material.source` 선택 필드: `{"type": "measured | estimated | assumed | user", "ref": "materialtwin:material/42/test/7", "confidence": "high"}` — 응답 `assumptions[]`에 "E1 measured (materialtwin material 42, test 7), nu12 assumed 0.30" 형태로 전파
- 경고 코드 신설: **W120 ASSUMED_CONSTANT** (실측 없이 가정한 상수가 섞였을 때 어느 상수인지 명시)

**실측 데이터의 한계 (materialtwin 현황 기준 사실).**
- materialtwin에는 **ν(포아송비)와 G가 없다** — 단축 인장 기반이므로 당연하다. 등방 ply조차 ν는 가정값(금속 ~0.3, 폴리머 ~0.35)+W120으로 처리한다.
- 직교이방 ply 카드(E₁/E₂/G₁₂/ν₁₂)를 실측으로 구성하려면 0°/90°/±45° 쿠폰(ASTM D3518)이 필요한데, **웹 스키마의 `specimen.orientation`이 MCP 표면에는 노출·입력 모두 안 된다**(`register_tensile_test`에 orientation 파라미터 없음, `get_material`도 미반환). 갭 보완 전 임시 우회는 시편 label 명명 규약("L0-0deg" 등)이다.
- materialtwin의 E 신뢰도 정보(crosshead 저평가, R² confidence)는 `get_material` 상세에서 확인해 `source.confidence`로 넘겨받는다.

### 16.4 현황 대비 갭 — MaterialTwinWeb 백로그 제안 (소규모 증분)

파사드·신규 서버가 아니라 기존 `mcp_server.py`에 대한 작은 증분이다. 진행 시점은 Q6.

| 갭 (실측) | 최소 변경 제안 |
|---|---|
| orientation 미노출·미입력 | `register_tensile_test`에 `orientation` 선택 파라미터 추가 + `get_material` 시편 정보에 orientation 반환 (웹 스키마엔 이미 컬럼 존재 — 스키마 변경 0) |
| `attributes` 쓰기 불가 (`register_material`이 `{"source":"mcp"}` 하드코딩, `update_material`은 name/category/description/code만) | `update_material`에 `attributes` 병합 파라미터 추가 → 에이전트가 도출한 직교이방 ply 카드를 `attributes.ply_card`로 영속화하는 슬롯 확보 (읽기는 `get_material`이 이미 attributes를 반환하므로 즉시 왕복 성립) |
| 직교이방 카드 조합 도구 없음 | 위 두 갭이 닫히면 "orientation별 유효 시험 → E₁/E₂/G₁₂ 조합 → attributes 저장"은 에이전트 절차로 충분하므로 전용 `get_ply_card` 도구는 **보류** (과설계 방지) |
| 오류 관례 상이 (materialtwin 한국어 dict vs 본 계획 D10 영어) | 생태계 정합 관점의 결정 필요 — **Q8** |

### 16.5 Twin 검증 루프 (V2)와 전체 체인

- 순방향. 본 서버가 예측한 적층판 유효 강성 E_x^eff ↔ materialtwin에 등록된 **라미네이트 쿠폰 실측 E**(`get_material`) 대조. 오차 임계(예: 5%) 초과 시 ply 물성 재검토 신호.
- 역방향(캘리브레이션). 실측에 맞도록 ply 물성을 조정하는 역문제는 §1.1 원칙대로 **에이전트 주도 반복 호출**로 수행한다.
- 전체 체인이 이미 절반 존재한다. 재료 탐색(`find_material` 프롬프트, Ashby 검색) → ply 물성 확보(16.3 체인) → **적층 평가(본 서버)** → 해석용 카드 도출(`get_mat_card`, *MAT_024/*MAT_098) → 시험 등록으로 환류(`register_tensile_test`). 본 서버는 이 체인에서 비어 있는 "적층 이론 계산" 칸을 채우는 조각이며, 추적성은 test_id ↔ source.ref ↔ payload_hash로 연결된다.
- 정합 벤치마크. materialtwin의 guide 리소스·프롬프트 패턴은 본 서버 §6.6(instructions)·§10.3과 같은 사상이므로, 구현 시 `laminate://guide` 리소스를 동일 패턴으로 제공하는 것을 Phase 5에 포함한다.

### 16.6 hwax portal(HEAXHub) 기반 주소 체계 — IP 대신 슬러그 (현황 실측 기반)

서버 간·에이전트-서버 간 주소를 IP:port로 관리하지 않는다. HEAXHub 실측 결과(2026-07-16, `backend/app/services/integration_launcher.py`, `config/stacks.yaml`, `docs/API_REFERENCE.md`) 포털이 이미 슬러그 기반 주소 체계를 제공한다.

**실측으로 확인된 포털의 기존 능력**
- 런처가 앱에 `$PORT`(loopback 전용 바인드)와 `$ROOT_PATH=/apps/{slug}`를 주입하고, Caddy admin API로 `/apps/{slug}/* → 127.0.0.1:<port>` 라우트를 동적 등록한다. **포트는 런처만 알고, 클라이언트는 슬러그만 안다.**
- `proxy` 모드 — HEAXHub가 프로세스를 직접 돌리지 않아도 **외부 upstream URL을 `/apps/{slug}/*`에 매핑**할 수 있다(기존 서비스를 슬러그화하는 데 사용 가능).
- 카탈로그 API `GET /apps`, `GET /apps/{id}` — 에이전트가 슬러그를 **동적으로 발견**하는 레지스트리로 쓸 수 있다.
- 전체 경로 체인. `https://hwax.sec.samsung.net/heax-hub/apps/<slug>/...` → 상위 포털이 `/heax-hub/` strip → HEAXHub Caddy가 `/apps/<slug>` strip → 앱. 상위 포털은 websocket(`/ws`)도 이미 통과시키는 실적이 있다.

**본 프로젝트 적용 (Phase 6과 연동)**
- 로컬 개발(현 단계)은 stdio + `.mcp.json`이라 IP 문제 자체가 없다. 슬러그 주소는 **원격/공유 배포 단계**의 이야기다.
- 배포 시 본 서버를 `--transport http`(D6에 이미 존재)로 켜고 HEAXHub integration(fastapi 계열 service 스택 또는 proxy 모드)으로 등록 → 고정 주소 `…/apps/laminate-mcp/mcp`. 에이전트 등록은 `claude mcp add --transport http laminate <URL>` 한 줄.
- materialtwin MCP는 현재 stdio 전용이나 FastMCP transport 전환은 소규모(런 모드 인자 추가 수준) — MaterialTwinWeb 백로그(§16.4)에 병기.
- 서버끼리는 여전히 직접 통신하지 않으므로(§16.3 원칙), "상호 발견"은 결국 **에이전트가 두 슬러그를 아는 것**으로 축소된다. 정적(슬러그 관례 고정)이 기본, 동적(`GET /apps` 질의)은 선택.

**배포 전 검증 항목 (2026-07-16 PAT E2E로 완결 — 잔여 1건)**
1. ~~DNS rebinding Host 검증~~ ✅ 해소 — 포털 Host로 421 나던 것을 loopback 바인드+프록시 경계 전제로 비활성화, 포털 도메인 Host 헤더로 200/SSE 실증
2. ✅ **SSE의 Caddy 통과 실증** — Bearer PAT로 `:4180/apps/laminate_analyzer_mcp/mcp` initialize 호출 → 200 + `text/event-stream` 스트림 정상 수신
3. ✅ **포털 인증 완결** — HEAXHub에 PAT 기능을 정식 구현(commit 6221405: `POST/GET/DELETE /api/v1/auth/tokens`, sha256 해시 저장, 폐기·만료·audit). authz/deps가 `heax_pat_` 프리픽스로 분기. 익명 401 유지 확인. 접속: `claude mcp add --transport http laminate-analyzer <포털베이스>/apps/laminate_analyzer_mcp/mcp --header "Authorization: Bearer <PAT>"`
4. ✅ **`Mcp-Session-Id` 프록시 통과 실증** — initialize 응답 헤더 수신 → 동일 세션으로 notifications/initialized(202)·tools/list(Tool 11종 반환)까지 세션 지속 확인
5. cae00 오프라인 배포 제약(사전 빌드 산출물 배송 방식)에 MCP 서버 포함 — 미착수 (유일 잔여)
6. HEAXHub에 MCP 전용 스택 정의가 없음 — **fastapi 스택으로 등록·빌드·기동까지 실증 완료**, 전용 스택은 불필요로 판명
비고. 상위 HWAX 포털(hwax.sec.samsung.net) 1단이 추가된 운영 환경(cae00)에서는 5번 배송과 함께 재확인한다 — websocket 통과 실적상 리스크 낮음.

---

## 17. V2-1차: 열탄성 휨 + 크랙 차폐 해석해 (2026-07-16 밤 확정 사양, 서버 0.3.0)

사용자 확정 방향. (a) 동박률·층두께·판 크기로 고온 곡률/휨을 해석해로 계산, (b) 피보호층
주변 보호층의 점탄성과 크랙 발생·진전 메커니즘의 해석적 계산식을 넣을 수 있는 만큼 탑재.
"기능을 하나씩 넣어 경향을 설명"하는 목적 — 전 수식이 폐형해·결정론이다.

### 17.1 열탄성 확장 (compute_thermal_response)

- 재료에 선택 CTE 추가: isotropic `alpha`, orthotropic `alpha1, alpha2` [1/K].
  (|α|>1e-3 이면 ppm 입력 착오 의심 → W110 계열 경고. 탄소섬유 α₁<0 허용.)
- α 변환: αx = α₁m²+α₂n², αy = α₁n²+α₂m², αxy = 2mn(α₁−α₂) (공학 전단).
- 열하중: N_th = ΔT Σ Q̄α t_k, M_th = ΔT Σ Q̄α z̄_k t_k → K[ε₀;κ] = [N_th;M_th] (자유 열변형).
- 산출: 유효 CTE(αx,αy,αxy = ε₀/ΔT), 열곡률 κ, **ply별 잔류응력** σ = Q̄(ε₀+z̄κ−αΔT)
  (평형 검증: Σσt=0, Σσz̄t−적분 모멘트=0이 단위 테스트).
- **휨(warpage)**: 판 {Lx,Ly} 주어지면 w(x,y) = −½(κx x²+κy y²+κxy xy)의 9점(모서리4+변중4+중심)
  범위 = coplanarity. 순수 κx 실린더에서 w = κLx²/8 검증.
- ΔT 규약: T_현재 − T_무응력기준. 가정 명시: 선형 CTE(Tg 전이 미반영 — FR-4는 Tg 이상에서
  α 급변), 자유 경계, 소변형.
- **동박률 균질화 (homogenize_layer)**: 성분 [{material(iso, E·ν·α·ρ), volume_fraction}] →
  면내 병렬(Voigt) E=Σf·E, ν=Σf·ν, **α=Σf·E·α/Σf·E**(힘 평형), ρ=Σf·ρ. Σf=1 검증.
  동박층 = {Cu f=동박률, 수지 f=1−동박률}. 상한 모델임을 assumption으로 명시.
- 검증 폐형해: **Timoshenko 바이메탈** κ = 24Δα·ΔT/(h(14+n+1/n)) (등두께, n=E₁/E₂, 동일 ν
  이면 판=보 동일 — 유도 근거 math_spec) + 대칭 적층 κ_th=0 + 단일재 α_eff=α.

### 17.2 크랙 발생·차폐 해석해 (assess_crack_shielding)

대상 구도: 취성 피보호층(target ply)과 이웃 보호층. 전부 문헌 확립 폐형해, 가정 명시.

| 량 | 식 | 판단 |
|---|---|---|
| 평면변형 계수 | Ē = E/(1−ν²) | — |
| 터널(채널) 크랙 G_ss | G_ss = πσ_t²h_t/(4Ē_t) (균질 근사) | 크랙 진전 구동력 |
| 임계 채널링 응력/변형률 | σ_c = √(4Ē_tΓ_t/(πh_t)), ε_c ≈ σ_c/E_t | **크랙 발생 문턱** — h_t↓ ⇒ σ_c↑ (박층 유리) |
| 크랙 개구(COD) | δ_max = 2σ_t h_t/Ē_t (중앙크랙 프로파일) | 개구 크기 |
| Dundurs α,β (target/이웃) | α=(Ē_t−Ē_n)/(Ē_t+Ē_n), β 표준식 | α<0 = 이웃이 강성 → 차폐↑, α>0 = 유연 이웃 → 증폭 (부호·경향 지표로 보고, g(α) 수치보정은 미탑재 명시) |
| 계면 편향(He–Hutchinson) | α≈0에서 Γ_i/Γ_ℓ < 1/4 ⇒ 계면으로 편향(저지) | **보호층이 크랙을 계면에서 멈추는가** |
| shear-lag 전달길이 | ℓ = √(E_t h_t h_n / G_n) | ℓ↑ ⇒ 개구 구속 약화, 멀티플 크래킹 간격 척도 |
| **점탄성 차폐 이완** | 보호층 E(t)=E∞+(E₀−E∞)e^(−t/τ) (준탄성 근사) → ℓ(t→∞)/ℓ(0)=√(E₀/E∞), α(t) 이동 | **시간·고온에서 crack-opening 저지력이 얼마나 풀리는가** |

- 재료에 선택 `viscoelastic: {E0, Einf, tau_s}` 필드 추가 (materialtwin Prony와 단위 정합 — MPa·s).
- 입력 강도/인성(Γ_t, Γ_interface)은 선택 — 없으면 문턱값 대신 임계식과 필요 데이터 안내.
- 미탑재 명시(정직성): 채널링 g(α,β) 수치 보정표, 모드믹스 ψ 의존 계면인성, R-curve,
  피로 진전(Paris) — 문헌 보정계수의 수기 이식 리스크 때문. 경향 판단은 Dundurs 부호+상기
  폐형해로 충분하며, 정밀치가 필요해지면 표 데이터를 출처와 함께 별도 탑재.

### 17.2b 층별 응력 복원 + 파손 판정 (recover_ply_stresses, §17.4로도 참조)

- 재료에 선택 `strength: {Xt, Xc, Yt, Yc, S}` (압축 양수 관례, SI: Pa / SI_mm: MPa).
- 응력 복원: [ε₀;κ] = K⁻¹[N+N_th; M+M_th] (ΔT 선택 중첩) → ply별 bottom/mid/top에서
  σ_xyz = Q̄(ε₀+zκ−αΔT) → 재료축 σ₁/σ₂/τ₁₂ (표준 응력 변환).
- 판정: **Tsai-Wu 강도비 R**(aR²+bR=1의 양근 — 하중 R배에서 파손면, R>1 여유,
  F12=−½√(F11F22) 표준) 주지표 + **Max Stress 지배 모드**(섬유/횡/전단 인장·압축) 설명자.
- first_ply_failure = 전 ply·전 위치 최소 R. strength 없는 ply는 응력만(노트로 안내).
- 검증: 막 σ=N/h·굽힘 표면 σ=6M/h²(정역학 항등)·45° 변환 폐형해·막힘 평형 ΣσT=N·
  Tsai-Wu 단축 환원(R=Xt/σ 등)·[0/90]s FPF가 90°ply 횡인장이며 하중×R 재해석 시 R→1.

### 17.3 도구·검증

- 신규 Tool 3종: `compute_thermal_response(laminate, delta_T, panel?)`,
  `homogenize_layer(components)`, `assess_crack_shielding(laminate, target_ply, fracture?)`,
  `recover_ply_stresses(laminate, loads?, delta_T?)` → 총 15종.
- 오류: **E203 MISSING_THERMAL_PROPERTY** (CTE 없는 ply로 열해석 요청 시, ply 지목).
- 테스트: Timoshenko 대조, 대칭→κ_th=0, 단일재 α_eff=α·κ=0, 잔류응력 평형(ΣF=ΣM=0),
  warpage 실린더 기하, ROM 극한(f=1/0)·단조성, Griffith↔터널 G_ss 항등, He–H 문턱,
  shear-lag 극한(G_n→∞ ⇒ ℓ→0), 점탄성 단조성(E∞<E₀ ⇒ ℓ비>1), 단위 브리지, 도구 E2E.

---

## 17.5 V2-2차: 설계규칙·좌굴·진동·진행성 파손·흡습 (2026-07-31 확정 사양, 서버 0.5.0)

전부 데이터 불필요(강도는 진행성만)·폐형해·결정론. Tool 4종 신설 + 열해석 1종 확장 → 총 19종.

### 17.5.1 check_design_rules — 적층 설계 규칙 검사기

업계 표준 규칙을 코드화. 각 규칙은 {rule, severity, pass|not_applicable, found, why_it_matters, fix_hint} 반환.

| rule | severity | 판정 |
|---|---|---|
| symmetry | hard | 기하 회문(기존 fingerprint) — 위반 시 coupling_ratio 증거 병기 |
| balance | hard | 0/90 외 각도의 ±θ 짝 — 각도(tol 0.5°)·두께·재료까지 일치해야 짝. 미짝 ply 나열, A16/A26 병기 |
| ten_percent | guideline | 쿼드 적층일 때 각 방향 **두께 비율** ≥10% (매수 기준은 두께 불균일에서 오판 — F3). 비쿼드는 not_applicable |
| contiguity | guideline | 동일 각도 연속 ≤ N매 (기본 4, 파라미터) |
| adjacent_angle | guideline | 인접 ply 각도차 ≤45° (계면 층간응력 저감 — 0/90 직접 접촉 지적) |
| outer_protection | guideline | 최외층 ±45 권장 (충격·가공 손상 허용) |
| single_ply_angle_group | info | 방향별 매수 분포 요약 (판정 아님) |

### 17.5.2 compute_buckling / compute_natural_frequencies — Navier 폐형해

단순지지 직교이방 판(a=Lx, b=Ly), m,n = 1..10 결정론 스캔.

- 좌굴(2축 압축, R = Ny/Nx, 압축 양수): N_cr(m,n) = π²[D11(m/a)⁴+2(D12+2D66)(m/a)²(n/b)²+D22(n/b)⁴] / [(m/a)²+R(n/b)²], 최소값·모드(m,n)·(하중 주면) 여유율. Nxy 미지원(E100). **스캔은 경계 도달 시 배가 확장(상한 160)** — 고정 상한은 장판·직교이방에서 참 최소를 놓쳐 비보수적(적대 검증 NAV-2). 압축 지배 모드가 없으면 E100.
- 고유진동수: ω²(m,n) = π⁴[D11(m/a)⁴+2(D12+2D66)(m/a)²(n/b)²+D22(n/b)⁴]/ρ_areal, f=ω/2π [Hz]. 전 ply rho 필수. 스캔 = max(10, n_modes) (f는 m·n 단조 증가 — NAV-3).
- **비대칭 적층은 축소 굽힘강성 D\* = D − B A⁻¹ B 사용** (표준 근사) + W130 경고.
- D16/D26 유의(비율 > 0.05)면 W130 — Navier는 specially orthotropic 가정이라 비보수적일 수 있음을 명시.
- 검증 폐형해: 등방 정사각 SS 판 N_cr = 4π²D/b² (D=Eh³/12(1−ν²)), 장판 a/b=2에서 m=2 모드 전환(k=4 유지), 등방 진동 ω = π²[(m/a)²+(n/b)²]√(D/ρ_a).

### 17.5.3 run_progressive_failure — ply discount 진행성 파손

- 하중 패턴 P=(N,M) 고정, 강성 갱신 반복: ① 현 강성으로 단위응답→ply별 bottom/mid/top 3점 응력(중앙만 보면 굽힘에서 파손 누락 — 적대 검증 PROG-2) ② 미파손 ply의 Tsai-Wu R (기지 파손된 ply는 섬유 항만: R=Xt/σ₁ 또는 Xc/|σ₁|) ③ 최소 R에서 사건 기록 ④ 지배 모드별 강성 저하 — 기지/전단 모드: E2,G12,ν12 ×η, 섬유 모드: 전 성분 ×η (ply 사망). η 기본 0.1.
- 종료: **하중 지지 붕괴(같은 하중의 응답이 초기 10배 초과)** / 전 파손가능 ply 섬유 파손 / K̂ 비양정치 / 파손 가능 ply 없음. 사건 ≤ 2×n_plies (결정론).
- 스퓨리어스 방지: 응력이 참조 스케일의 1e-6 미만인 지점은 판정 제외 (σ1≈0에서 R~1e18 오염 차단 — PROG-1).
- 출력: events[{step, ply, mode, R}], **ultimate_R = 붕괴 전까지의 max R (하중 제어 용량)**, last_ply_R, 사건별 유효 Ex 저하 곡선, 종료 사유. strength 없는 ply는 탄성 유지(비파손, note).
- 한계 명시: 하중 제어 재시작(quasi-static restart) 표준 근사, 열잔류 미중첩(후속).

### 17.5.4 흡습 팽창 — compute_thermal_response 확장

- 재료 선택 필드: `beta`(등방)/`beta1,beta2`(직교) [1/%M], 입력 `delta_C` [%M 변화].
- 자유변형 ε_free = αΔT + βΔc — 열 기계 그대로 재사용(단일 자유변형 벡터). delta_T·delta_C 중 최소 하나.
- E203 재사용(detail로 α/β 구분). **등가 검증**: αΔT = βΔc 로 맞춘 두 해석이 동일 κ (바이메탈).

### 17.5.5 검증·경고

- W130 MODEL_APPLICABILITY 신설 — Navier 가정 이탈(비대칭 D\* 사용, D16/D26 유의) 명시용.
- 신규 few-shot: 등방 정사각 좌굴(4π²D/b² 기대값 내장) 케이스.
- 구현 후 멀티에이전트 적대 검증(수식 재유도 3렌즈 + 엣지 + 에이전트 UX) 수행.

---

## 부록 A. 기호표

| 기호 | 의미 | SI 단위 |
|---|---|---|
| tₖ, h | ply 두께, 총 두께 | m |
| zₖ | k번째 계면 좌표 (midplane 기준, z₀=−h/2) | m |
| θₖ | 적층각 (D2 규약) | deg 입력, 내부 rad |
| Q, Q̄ | (변환) 감축 강성 | Pa |
| A, B, D | 막/커플링/굽힘 강성 | N/m, N, N·m |
| Â, B̂, D̂, K̂ | 정규화 강성 (§4.6) | Pa |
| N, M | 단위 폭당 하중/모멘트 | N/m, N |
| ε₀, κ | 중립면 변형률, 곡률 | –, 1/m |
| z_ns, ζ | 중립면 위치, 무차원 위치 (z−z_bot)/h | m, – |

## 부록 B. 용어 대응

| 한국어 | 영어 (코드/응답 표기) |
|---|---|
| 중립면 | neutral surface (`neutral_surface`, beam 모드 한정 neutral axis) |
| 적층판 | laminate |
| 층(플라이) | ply / lamina (`laminae[]`) |
| 막-굽힘 커플링 | membrane-bending coupling (B matrix) |
| 준등방성 | quasi-isotropy |
| 대칭 적층 | symmetric layup |

## 부록 C. 원안(v1.0) 대비 변경 이력

1. `find_available_port` Tool 삭제 → HTTP 모드 기동 옵션으로 이동 (§2 판정 참조).
2. 조건수·양정치성 판정을 합동변환 정규화 K̂ 기반으로 재정의 (√12 계수 유도 포함).
3. 적층 순서(bottom-first)·각도 부호(CCW)·전단 규약(engineering) 동결 (D1~D3).
4. 단위계 `SI|ENG` → `SI|SI_mm` + 필수 입력화, imperial은 V1.
5. beam 모드 E_eq = 1/S̄₁₁ 확정, clt 모드 = B₁₁/A₁₁ 동치 명시, 완전 커플링 모드 V1 추가.
6. `analyze_laminate` 원샷 Tool과 `get_reference_cases` 신설. `get_server_health` → `get_server_info`.
7. 결정론 정책 명문화 (timestamp 분리, canonical JSON 해시).
8. 검증 정답 4원화 (폐형해 수식 내장 + 성질 + sympy 오라클 + 문헌), 수기 수치 금지.
9. `warpage_tendency_score` V2 이동 (CTE·ΔT 필요), 서버측 최적화 Tool 범위 제외 (에이전트 주도 원칙).
10. 일정을 날짜에서 Phase + DoD 게이트로 재구성, 구현 체크리스트 신설.
11. (v2.1) §16 신설 — 지식 축적 3층 구조(무상태 원칙 유지), 재료 해석 체인(강성 미상 시 MaterialTwinWeb 조회→미시역학 추정), MaterialTwinWeb MCP 파사드 제안, Twin 검증 루프. `source` provenance 필드(V1)와 W120 경고 코드 예약.
12. (v2.2) §16을 **materialtwin MCP 실측 현황 기반으로 재작성** — 파사드 신설 제안 철회(도구 20종 MCP 실존), 실제 도구명·단위(E_GPa)·오류 관례(한국어 dict)·valid 규칙 반영, orientation 미노출·attributes 쓰기 불가 갭을 소규모 증분 백로그로 정리, `.mcp.json` 스코프 이슈 명시, Q6 교체·Q8(오류 언어 정합) 신설.
13. (v2.3) §16.6 신설 — hwax portal(HEAXHub) 실측 결과 슬러그 기반 주소 체계(`/apps/{slug}/` Caddy 동적 라우트, proxy 모드, `GET /apps` 카탈로그)가 이미 존재함을 확인. 배포 시 IP:port 대신 슬러그 주소 채택, 원안 §10의 포트 스캔 정책은 로컬 HTTP 모드 한정으로 축소. 미실측 검증 5항목을 Phase 6 DoD로 이관.
14. (v2.4, 구현 세션) MVP 구현 완료를 반영한 개정 — (a) 패키지 레이아웃 `server/` → `app/` (HEAXHub fastapi 스택 기본 entrypoint `app.main:app` 정합, §9의 파일명은 app/ 하위로 읽을 것), (b) D10 언어 규약을 한국어로 개정(Q8 채택), (c) HTTP transport를 Phase 6에서 선행 구현(hwax 등록 요구) — MCP SDK의 DNS rebinding Host 검증은 loopback 바인드+프록시 경계 전제로 비활성화(§16.6 검증 항목 1건 해소), (d) E400·E500은 트리거 경로가 V1 기능(하중응답·타임아웃)에 있어 유발 테스트 유예, (e) StreamableHTTPSessionManager는 프로세스당 1회 기동 제약 확인(운영 영향 없음), (f) HEAXHub 스키마 v2 실검증 결과 — `health_check`/`restart_policy`는 `launch` 하위, `build.type` 필수, `source.ref` 불허 (materialtwin-web 오버레이는 이 세 가지를 위반한 채 스택 기본값으로 동작 중 — 해당 프로젝트에 전달 권장).
15. (v2.5, 완성 세션) V1 Tool 4종·§5.2 지표·sympy 오라클·강건성·math_spec/agent_guide/S5 로그 완료. hwax 등록을 심볼릭 링크 in-tree 통합으로 전환해 **카탈로그 등록(v0.2.0)·빌드·기동·Caddy 라우트까지 실증**(§16.6 검증 항목 갱신 — authz 게이트 확인, SSE 포털 통과는 토큰 확보 후). E400·E500 실트리거 확보로 v2.4(d) 유예 해소. 서버/엔진 0.2.0.
