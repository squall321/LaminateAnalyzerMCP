# 에이전트 사용 가이드 (agent_guide)

LLM 에이전트가 laminate-analyzer를 (materialtwin과 함께) 쓰는 표준 절차. 서버 instructions와
`laminate://guide` 리소스의 확장판이다.

## 0. 철칙

1. **ABD·중립면·지표를 암산하지 않는다.** 수치는 전부 Tool 호출로 얻는다.
2. 응답의 `warnings`(특히 W110 단위 의심)와 `assumptions`는 사용자 보고에 반드시 포함한다.
3. `payload_hash`를 보고서에 남기면 동일 입력 재현이 보장된다(결정론 서버).
4. 스키마가 헷갈리면 `get_reference_cases()` 예시를 그대로 따라 만든다.

## 1. 규약 3줄 요약

- `laminae[0]` = **최하단** ply. 각도는 deg, +x→섬유 1축 CCW 양수.
- `unit_system` 필수. `"SI"`(Pa·m·kg/m³) 또는 `"SI_mm"`(MPa·mm·t/mm³).
- 하중 N·M은 **단위 폭당** 물리량 (N: N/m|N/mm, M: 둘 다 N).

## 2. 표준 플로우

| 상황 | 호출 |
|---|---|
| "이 적층 평가해줘" | `analyze_laminate(laminate)` 원샷 |
| 기준 만족 판정 | `evaluate_laminate(laminate, criteria={"max_coupling_ratio":0.05, "min_quasi_isotropy_score":0.9})` |
| 하중 응답/유효 강성 | `solve_load_response(laminate, loads={"N":[100,0,0]})` |
| 후보 N개 탐색 | 후보 생성(에이전트) → `batch_evaluate_laminates(candidates)` → 상위만 `analyze_laminate` |
| 공차 강건성 | `run_sensitivity_analysis(laminate)` |
| 보고서 | `generate_design_report(laminate, criteria, language="ko")` → report_markdown 저장 |
| 적층안 건전성 1차 검토 | `check_design_rules(laminate)` — 대칭·밸런스·10%·인접각·연속·외층 관례 |
| 층별 응력·파손 | `recover_ply_stresses(laminate, loads, delta_T?)` — strength 있으면 Tsai-Wu R·FPF |
| 한계하중(첫 파손 이후) | `run_progressive_failure(laminate, loads)` → ultimate_R |
| 좌굴·공진 | `compute_buckling(laminate, panel, ...)` / `compute_natural_frequencies(laminate, panel)` |
| 열·흡습 휨 | `compute_thermal_response(laminate, delta_T=, delta_C=, panel=)` (+`homogenize_layer`) |

설계 최적화 루프는 서버가 아니라 **에이전트가 돈다**. 후보를 만들고 batch로 치고 선별을 반복한다.

## 3. materialtwin 연계 — 물성을 모를 때

재료 해석 체인(계획서 §16.3)의 실행 절차.

1. `materialtwin`의 `list_materials(query="AL6061")` 또는 `search_by_property("E_GPa", ...)`
   (참고: list 반환 tool은 MCP가 **원소별 content 항목**으로 직렬화한다 — 첫 항목만 읽지 말 것)
2. `get_material(id)` — **valid=true 시험만** 채택, confidence 확인
3. **단위 브리지: materialtwin의 `E_GPa` × 1000 = SI_mm의 MPa.** (×1000 실수하면
   laminate 쪽 W110이 잡아준다 — 경고가 뜨면 즉시 재검토)
4. 인장시험은 E만 준다. ν·G는 가정값을 쓰고 다음처럼 표기한다.

```json
{"type": "isotropic", "E": 68900.0, "nu": 0.33, "name": "AL6061-T6",
 "source": {"type": "assumed", "ref": "materialtwin:material/12/test/34 (E measured, nu assumed 0.33)",
            "confidence": "high"}}
```

→ 응답에 W120 경고와 assumptions 줄이 생겨 가정이 추적된다.

5. 해석 카드가 필요하면 다시 materialtwin의 `get_mat_card(test_id, units=...)`로 마무리
   (재료 실측 → ply 카드 → 적층 해석 → LS-DYNA 카드의 digital thread).

## 4. 오류 자가 복구

모든 오류는 `{code, message, field, suggestion}` — **suggestion을 그대로 수행하고 재호출**하면 된다.

| 자주 만나는 코드 | 대응 |
|---|---|
| E101 | unit_system을 "SI" 또는 "SI_mm"로 명시 |
| E100 (field 참조) | 해당 필드 타입/형식 수정. 예시는 get_reference_cases |
| E201 | \|ν₁₂\| < √(E₁/E₂) 위반 — 물성 재확인 |
| W110 | 단위 자릿수 의심 — GPa/MPa/Pa 혼동 여부를 사용자에게 확인 |
| E500 | ply 수·배치 크기를 줄여 분할 호출 |

## 5. 결과 해석 요령

- `coupling_ratio` ≥ 0.2 (W200): 경화 후 뒤틀림·해석 커플링 위험 — 대칭화 권고.
- `dominant_coupling_terms`: 어느 B 성분이 비대칭의 주범인지 — B16/B26이면 앵글플라이 반대칭성.
- `ns_offset_ratio`: 0에서 멀수록 강성 비대칭. beam 관점 중립축 이동량은 `compute_neutral_axis`.
- `quasi_isotropy_score` > 0.95: 면내 준등방 — 방향성 요구가 없으면 무난.
- 대칭 적층에서 B≈0, ζ=0.5는 **정상**이다(버그 아님).

## 6. HEAXHub(hwax) 경유 접속

로컬 stdio 대신 포털 배포본을 쓸 때.

```bash
claude mcp add --transport http laminate-analyzer \
  <포털베이스>/apps/laminate_analyzer_mcp/mcp
```

같은 방식으로 materialtwin도 슬러그 주소로 접속 가능해지면 두 서버 모두 IP 없이 연결된다.

## 7. 열 휨(PCB warpage)과 크랙 차폐 워크플로 (v0.3.0)

**PCB 리플로우 휨 경향** — 동박률이 층마다 다른 기판.

1. 각 동박층: `homogenize_layer([{Cu(E=110000, ν=0.34, α=17e-6), f=동박률}, {수지(E=3500, ν=0.35, α=60e-6), f=1−동박률}])`
2. 유전체 층은 α 포함 물성으로 직접 입력 → laminae 구성 (비대칭 동박 분포가 휨의 원인)
3. `compute_thermal_response(laminate, delta_T=+235(리플로우) 또는 −(경화 냉각), panel={"Lx":100,"Ly":60})`
   → `warpage.range`(coplanarity), `effective_cte`, ply 잔류응력. Tg 이상 α 급변은 미반영(가정 명시됨)
4. 경향 탐색: 동박률·두께를 바꿔가며 재호출 — 결정론이라 비교가 곧 감도

**크랙 차폐(보호층/피보호층)** — 예: PSA/UTG유리/PSA.

1. 보호층 material에 `viscoelastic: {E0, Einf, tau_s}` (materialtwin 완화시험 값 그대로)
2. `assess_crack_shielding(laminate, target_ply=유리, fracture={"applied_strain":..., "gamma_target":..., "gamma_interface":..., "gamma_next_layer":...})`
3. 해석 요령: `initiation_threshold.sigma_critical`(박층일수록↑ — h 절반이면 ×√2),
   Dundurs α>0이면 유연 이웃이라 개구 증폭 경향, `interface_deflection`(Γi/Γℓ<0.25 → 계면에서 저지),
   `viscoelastic.transfer_length_growth`(이완 후 구속 저하 √(E0/E∞)) — 시간·고온에서 차폐가 얼마나 풀리는지

## 8. 스프린트 A 도구 사용 요령 (v0.5.0)

**check_design_rules** — 아이디어 단계의 첫 관문. severity가 `hard`인 위반(대칭·밸런스)은
설계 결함으로 다루고, `guideline`은 검토 신호로만 다룬다(불합격 아님). 각 항목의
found/why_it_matters/fix_hint를 그대로 사용자에게 전달하면 근거 있는 설명이 된다.
`single_ply_angle_group`은 판정이 아니라 방향 분포 요약(info)이다.

**compute_buckling / compute_natural_frequencies** — `panel={"Lx","Ly"}` 필수(길이 단위).
좌굴은 **압축을 양수**로 보며 `load_ratio = Ny/Nx`, 전단(Nxy)은 미지원(E100).
`applied_Nx`를 주면 margin.factor = N_cr/Nx. 진동은 **전 ply에 rho** 필요.
경계조건은 4변 단순지지 고정 — 실제 경계가 고정단이면 보수적(낮게) 나온다.
응답의 `mode.at_scan_boundary`가 true면 스캔 상한에 걸린 것이니 결과를 과신하지 말 것(W130 동반).
비대칭 적층은 축소강성 D*로 근사되며 W130이 함께 온다.

**run_progressive_failure** — `ultimate_R`이 핵심: 입력 하중 패턴의 **최대 지지 배수**다
(loads×ultimate_R = 한계하중). `events`는 사건 순서(ply·모드·R)이고, `termination`이
`load_carrying_collapse`면 강성 붕괴로 정상 종료된 것이다. 첫 파손의 상세(위치·양 기준)는
`recover_ply_stresses`로 따로 본다. strength 없는 ply는 탄성 유지(비파손)로 가정된다.

**흡습(delta_C)** — `compute_thermal_response`에 `delta_C` [%M]와 재료 `beta`(또는 beta1/beta2)를
주면 열과 동일한 기계로 계산된다. ΔT와 ΔC를 함께 주면 총 응답만 반환한다(유효 계수는 분리 호출).

**공통 함정** — 이 도구들은 CLT 기반이라 두꺼운 판·샌드위치 코어(횡전단 지배)에서는 부정확하다.
응답의 `assumptions`와 W130 경고를 반드시 사용자에게 전달할 것.

## 9. 층간·감쇠·두께 한계 (v0.6.0)

**compute_interlaminar_stresses** — `shear={"Vx","Vy"}`는 **굽힘 모멘트의 공간 구배(전단력)**이지
면내 전단(Nxy)이 아니다. 반환의 `critical_location`이 최악 지점(계면 또는 ply 내부)이며
margin<1이면 박리 예상. `ilss_unevaluated`가 있으면 그 위치는 강도 미입력이라 평가에서 빠진
것이니 최악이 바뀔 수 있다고 보고할 것. 자유단 순위는 정성 지표(W130 동반).

**감쇠·횡전단** — `compute_natural_frequencies`가 전 ply에 loss_factor가 있으면 1차 모드 기준
모달 η와 Q를 준다(일부만 주면 W120으로 알림). `transverse_shear.R_s`는 **임계 모드 기준**이며
0.02를 넘으면 CLT가 비보수적이라는 뜻 — corrected_f1_hz·corrected_N_cr을 함께 보고할 것.
샌드위치는 A55가 코어 전단에 지배되어 R_s가 크게 나오는 것이 정상이다(CLT 한계의 정량 신호).

## 10. 피로 (v0.7.0)

`estimate_fatigue_life(laminate, loads_max, loads_min?)` — 각 ply에 `strength`와
`fatigue{model_type:"log_linear"|"basquin", k|b}`가 있어야 한다(둘 중 하나라도 없는 ply는
평가에서 빠지고 **W120으로 경고** — 그 ply가 임계면 수명이 과대평가이므로 반드시 사용자에게 전달).

- **사이클 표현**: `loads_min` 생략 = 영-인장(R=0). 완전반복(R=−1)은 `loads_min`에 부호 반대
  하중을 명시한다 — 이게 보통 가장 가혹하다. 압축 전용(0→−200)도 표현 가능하며,
  두 인자의 순서는 결과에 영향을 주지 않는다.
- **해석**: `governing_component`(σ1=섬유, σ2=횡/기지, τ12=전단)가 어느 손상 모드인지 알려준다.
  CFRP 인장 피로는 보통 σ2(기지) 지배 — 이건 정적 FPF와 같은 물리다.
- **한계 보고 필수**: 등진폭·비례하중 1차 근사이고, S-N은 시험 데이터 범위(1e4~1e7) 밖에서
  외삽이다. 큰 N은 자릿수만 참고하라고 전할 것. `at_cap=true`는 사실상 무한수명 신호.

## 11. 기하 비선형 — 선형 CLT가 깨지는 영역 (v0.8.0)

**언제 넘어가야 하는가.** `compute_thermal_response`가 panel을 받으면 `warpage.w_over_thickness`를
같이 준다. **이 값이 0.3을 넘으면 선형 결과를 그대로 보고하지 말 것** — W130으로도 알린다.
소변형 가정을 벗어나 실제 곡률은 그보다 작고, 비대칭 적층이면 형상 자체가 다를 수 있다.

**`compute_bistable_shapes(laminate, panel, delta_T)`** — 비대칭 적층의 경화 후 실제 형상.
선형 CLT는 판 크기와 무관하게 **안장 하나**만 준다. 실제로는 판이 임계 크기를 넘으면 안장이
불안정해지고 서로 거울상인 **원통 두 개**로 분기한다. 원통은 전개 가능면이라 막 에너지 벌점이
없고 안장은 L⁴ 벌점을 받는 것이 원인이다.

- 실측 예: [0/90] CFRP 0.25mm, ΔT=−150K → 임계 한 변 **18.9mm**. 100mm 판이면 이미 쌍안정이고,
  선형이 준 안장 κ=∓10.2 [1/m]는 실현되지 않는다(실제는 원통 κ=−9.90, +0.003).
- `critical_panel`은 종횡비를 유지한 채 판을 키울 때의 분기점이다. **두께에 정비례**하므로
  얇을수록 작은 판에서 쌍안정이 된다.
- `energy_barrier`는 스냅스루를 막는 장벽(단위면적). **스냅 하중 자체는 주지 않는다** — 그건
  하중 인가 평형 추적이 필요하다. 액추에이터 설계 질문이면 이 한계를 먼저 알릴 것.
- **한계**: κxy = 0 이라 **비틀림 형상을 표현하지 못한다**. 선형 CLT의 |κxy|/max(|κx|,|κy|) 로
  판정해 0.05 초과면 W130, **0.3 초과면 "형상 판정을 신뢰하지 말 것"** 이 뜬다 — 그때는
  `compute_thermal_response` 의 κ 3성분을 대신 보고할 것. `linear_reference.kappa_xy` 로 직접
  대조할 수 있다. (적대 검증 전에는 A/B/D 의 16·26 성분비라는 대리지표를 써서 [0/90/45/90]
  같은 케이스를 통째로 놓쳤다.)
- `energy_barrier.min_barrier` 가 스냅스루 지배 장벽이다 — **얕은 우물이 먼저 넘어간다**.
  `per_stable_shape` 로 형상별 장벽을 볼 수 있다. 가장 깊은 우물 기준 단일 값을 쓰면 최대
  19배 낙관적이 된다.
- `stability` 가 `marginal` 인 정지점이 있으면 판이 분기점 바로 근처라 안정성 단정이 무의미하다
  — `stable_count` 를 그대로 보고하지 말 것.

**`compute_large_deflection(laminate, panel, pressure, edge_condition)`** — w > h 영역의 처짐.
선형은 w ∝ q 지만 막 신장이 개입하면 실제 처짐이 훨씬 작아진다(`stiffening_ratio`가 그 배수).

- **`edge_condition`이 지배적 가정이다.** movable(면내 이동 자유, 기본·보수적) vs immovable
  (면내 구속)에서 β가 등방 정사각 기준 3.86배 차이난다. 실제 경계를 모르면 **두 극단을 모두
  돌려 범위로 보고할 것.**
- `w/h < 0.3` 이면 선형으로 충분하다고, `w/h > 3` 이면 1항 근사 범위를 벗어났다고 W130이 뜬다.

**`compute_postbuckling(laminate, panel, applied_Nx)`** — `compute_buckling`은 N_cr까지만 답한다.
박판은 좌굴 후에도 하중을 더 받으며, 면내 접선강성이 떨어지고 하중이 가장자리로 재분배된다.
`stiffness_ratio`, `effective_width_ratio`(b_eff/b=√(N_cr/N)), `amplitude_over_thickness`를 준다.
N_cr·모드는 `compute_buckling`과 동일하다.

- **강성비는 2축 하중비 R = Ny/Nx 에 크게 좌우된다.** 등방 정사각 R=0 에서 정확히 0.5 지만
  R=1(2축 등압축)에서 0.26, R=2 에서 0.12 로 떨어진다. R 을 안 넣고 0.5 를 가정하면 남은
  강성을 최대 4배 과대평가한다.
- 종횡비가 크거나(>2) A16/A26 이 유의한 불균형 적층이면 1항 근사 오차를 W130으로 알린다.

**`solve_nonlinear_shear_response(laminate, loads)`** — 위 셋은 **기하** 비선형(형상이 커서
생기는 것)이고 이것은 **재료** 비선형이다. UD 복합재의 면내 전단은 기지 지배라 뚜렷이
비선형이라(γ12 = τ12/G12 + S6666·τ12³), 전단 지배 적층에서는 이쪽이 훨씬 크게 작용한다.

- 실측 예: [±45]s CFRP에 Nx=50 N/mm → ply τ12 = 50 MPa, 할선 G12가 선형의 58%,
  적층 **Ex가 선형의 61%**. 선형 CLT만 믿으면 강성을 60% 과대평가한다.
- ply에 `material.shear_nonlinear = {"S6666": ...}` 필요. 단위 **1/응력³**
  (SI: 1/Pa³, SI_mm: 1/MPa³, CFRP는 1/MPa³ 기준 1e-8 자릿수). 없는 ply는 선형으로 두고 W120.
- **재미있는 불변식**: [±45]에서 적층 Gxy는 전혀 안 변하고 Ex만 떨어진다(45°에서 Q̄66의
  Q66 의존이 정확히 상쇄되기 때문). 전단 비선형이 적층 수준에서 어디로 나타나는지는
  적층각에 따라 완전히 다르다 — 직관으로 넘겨짚지 말 것.
- **필수 주의**: 3차식에는 강도 한계가 없어 **파손 이후에도 계속 답을 낸다**.
  γ12 > 0.05 면 W130이 뜬다 — 그때는 `recover_ply_stresses`로 파손 판정을 반드시 병행하고,
  사용자에게 "이 값은 이미 파손했을 영역"이라고 전할 것.
- `solve_load_response`에 shear_nonlinear 물성이 있는 적층을 넣으면 W130으로 이 도구를 가리킨다.

**공통 주의.** §18 도구는 §17까지와 달리 **폐형해가 아니라 Rayleigh–Ritz/Galerkin 저차 근사**다.
분기·임계값·경향은 신뢰하되 **절대값은 FE 대비 오차**가 있다(선형 극한에서도 등방 정사각 +2.4%).
응답의 assumptions를 반드시 함께 보고할 것.

## 11. REST 표면 (v0.7.2) — MCP가 없는 소비자용

같은 서버가 두 표면을 제공한다. 계산·검증·응답 envelope는 동일하다.

| 목적 | REST |
|---|---|
| 어떤 도구가 있나 | `GET /api/v1/tools` — 21종의 이름·설명·입력 JSON Schema |
| 한 도구 상세 | `GET /api/v1/tools/{name}` |
| 실행 | `POST /api/v1/tools/{name}` (본문 = 인자 객체) |
| 사용 규약 | `GET /api/v1/guide` (laminate://guide와 동일) |
| 서버 정보 | `GET /api/v1/info` · 대화형 문서 `/docs` |

HTTP 코드 규약: 계산 오류는 **200 + envelope**(status=error, errors[].suggestion으로 자가 복구),
없는 도구는 404, 인자가 스키마와 안 맞으면 422. 도구는 MCP 레지스트리를 단일 소스로 쓰므로
새 도구를 추가하면 REST에 자동 반영된다.

## 12. 자유 가장자리 박리 (v0.10.0) — 면내가 통과해도 여기서 먼저 뜯긴다

**언제 반드시 호출해야 하는가.** `recover_ply_stresses` 가 "여유 있음"이라고 답했을 때,
적층에 **각도 차가 큰 계면**이 있으면 그것만으로 안전을 보고하지 말 것.
실측: `[±30/90]s` 에 Nx=150 N/mm — 면내 Tsai-Wu R = 1.33("33% 여유")인데
현실적 G_Ic = 0.10 N/mm 기준 **가장자리 개시 여유는 0.91 로 이미 넘었다.**

`assess_free_edge_delamination(laminate, loads, fracture?)` — O'Brien 폐형해
G = (ε_x²·h/2)·(E_LAM − E*), ε_c = √(2G_c/(h·ΔE)). 박리 길이에 무관해 임계 변형률이
닫힌 형태로 나온다.

- `fracture={"G_c": ...}` 를 주면 계면별 **개시 변형률과 여유율**까지, 없으면 G 순위만이다
  (W120으로 알린다). G_c 단위는 SI: J/m², SI_mm: N/mm. 탄소/에폭시 G_Ic ≈ 0.1~0.3 N/mm.
- **`dominant_driver` 가 핵심 정보다.** `peel`(σz 개구) / `transverse_shear`(τyz) /
  `in_plane_shear`(τxz) / `none`. `[±45]` 계열은 σy 가 0이라 **전단 지배**,
  `[0/90]` 계열 중앙면은 **peel 지배**로 갈린다 — 직관으로 넘겨짚지 말 것.
- `G > 0` 인데 `dominant_driver: none` 이면 그 계면은 **재료 불연속이 없어 실제 개시가
  어렵다**는 뜻이다(예: `[±45]s` 의 −45/−45 중앙면). 에너지는 있지만 구동력이 없다.
- **`model_valid: false` + `G: null` 은 "안전"이 아니다.** 비대칭 적층에서 ΔE < 0 이면
  이 모델이 적용되지 않는다는 뜻이다. 사용자에게 "이 계면은 판정할 수 없다"고 전할 것.
- **혼합모드 분리를 하지 않는다.** 총 G 만 준다. G_Ic ≪ G_IIc 이므로 보수적으로
  G_Ic 를 넣어 판정하고, 그 사실을 보고에 반드시 포함할 것.
- 구동력은 경계층 평형 논증에서 나온 크기 규모 지표다 — 실제 자유 가장자리 응력은
  특이점을 갖는 3D 문제라 개시 계면이 다를 수 있다(W130).

## 13. 물성 체인과 게이트 (v0.11.0)

**ply 물성을 모를 때 — `derive_lamina_from_constituents(fiber, matrix, Vf)`**
섬유+수지에서 직교이방 lamina 를 만든다. materialtwin 실측 수지 → 이 도구 → 적층 해석으로
체인이 이어진다. **반드시 `bounds`(Reuss–Voigt)를 함께 보고할 것** — E2·G12 는 기지 지배라
불확실성이 크다(AS4/3501-6 에서 G12 가 문헌 대비 42% 낮게 나온다). 실측이 있으면 그걸 쓰고,
없으면 ξ 를 역보정하라.
⚠ **`homogenize_layer` 를 UD ply 생성에 쓰지 말 것.** 등방 Voigt 병렬이라 섬유/수지에 넣으면
E2 를 15배 과대, α2 를 260배 과소로 준다. 그 도구의 `scope` 필드가 용도를 명시한다.

**흡습에 시간이 필요할 때 — `compute_moisture_uptake(laminate, diffusion, time_s?, mode?)`**
τ = D·t/h² 하나가 지배한다 — **두께가 2배면 시간이 4배**다. `mode="desorption"` 이 베이크다.
결과의 `delta_C_for_thermal_tool` 을 `compute_thermal_response(delta_C=...)` 에 그대로 넘겨라.

**경계조건 — `boundary="clamped"`**
`compute_buckling`·`compute_natural_frequencies` 가 지원한다. 실제 패널이 고정단에 가까우면
N_cr 이 2.5배 달라진다. **단 고정단 좌굴값은 1항 Ritz 상계라 +6.6% 비보수다** — SS 값을 하한으로
함께 보고해 참값을 감싸라. 진동수는 정확도가 좋다(고정/SS 비 1.829 vs 문헌 1.83).
`panel` 에 `Lx`/`Ly` 외의 키를 넣으면 E100 이다 — 경계조건은 `boundary` 인자로 준다.

## 14. 반드시 함께 봐야 하는 조합 (조용한 오답 방지)

이 서버가 **자신 있게 답하는데 틀릴 수 있는** 자리들이다. 실측으로 확인된 것만 적는다.

| 상황 | 단독 호출이 주는 답 | 실제 | 대응 |
|---|---|---|---|
| **면내 압축** | `recover_ply_stresses` 가 R=7.07 "7배 여유" | 좌굴 여유 0.017 (**410배 모순**) | `panel` 을 함께 주면 `governing_mode` 가 정렬해 준다. 안 주면 W130 |
| **대칭 적층 가열** | `compute_thermal_response` warpage 7e-16 "무시 가능" | 면내 구속 시 **ΔT_cr = 24.9 K** | `panel` 을 주면 `restrained_buckling` 이 나온다 |
| **면내 인장 + 각도 차 큰 계면** | 면내 R=1.33 "33% 여유" | 가장자리 개시 여유 0.91 | `assess_free_edge_delamination` |
| **전단 지배 적층** | `solve_load_response` 선형 Ex | 실제 Ex 61% | `solve_nonlinear_shear_response` |
| **얇은 비대칭 적층 냉각** | 선형 안장 곡률 | 원통 두 개로 분기 | `compute_bistable_shapes` |

압축 하중을 받으면 **강도만 보고하지 말 것.** 열해석에서 "휨 없음"이 나오면 **구속 조건을 확인할 것.**

## 15. 열잔류와 혼합모드 (v0.12.0)

**경화 냉각 잔류를 빼면 파손·피로가 통째로 비보수다.**
`run_progressive_failure`·`estimate_fatigue_life` 에 `delta_T` 를 반드시 넘겨라.
실측 [0/90]s CFRP: ΔT=−150 K 에서 **FPF 하중이 81% 낮아지고**, 피로 수명은
1.23e8 → 1 회로 무너진다(90° 층 횡인장 잔류가 Yt 여유를 거의 다 먹는다).
CTE 가 있는데 `delta_T` 를 안 주면 W130 이 뜬다 — 그 경고를 무시하지 말 것.
`first_ply_failure_R = 0` 은 **기계 하중 없이 잔류만으로 이미 파손**이라는 뜻이다.

**박리 인성은 모드를 갈라 줘야 정확하다.**
`assess_free_edge_delamination` 의 `fracture` 에 `{G_Ic, G_IIc}` 를 주면 계면별로 판정한다.
- `mode_mix.basis == "mirror_symmetry"` → 두 부분적층이 거울상이라 **순수 Mode I** 이다.
  여기서 G_Ic 를 쓰는 것은 보수가 아니라 **정확**하다.
- `basis == "unknown"` → 분할에 3D 해석이 필요하다. `onset_strain` 은 **보수적인 Mode I 값**이고
  `onset_strain_range` 가 실제 범위다. 사용자에게 범위로 보고할 것.

## 16. 순응층이 있으면 CLT 굽힘강성을 믿지 말 것 (v0.13.0)

폴더블 스택(UTG/OCA/UTG)처럼 **무른 중간층**이 끼면 두 면재가 서로 미끄러져 실제 굽힘강성이
CLT 보다 훨씬 낮다. CLT 는 항상 완전합성(f=1)을 가정한다.

`assess_partial_composite_bending(laminate, span, core_ply?)` — **span 이 필수다.**
실측 UTG/OCA/UTG (OCA G=0.3 MPa): L=1mm 에서 CLT 가 **18.3배**, L=10mm 에서 **2.03배**
과대평가하고 L=200mm 면 1.03배로 수렴한다. 같은 스택이 스팬에 따라 완전히 다르다.

- `clt_overprediction` 이 헤드라인이다 — CLT 기반 처짐·좌굴·진동수가 모두 그만큼 낙관적이다.
- `compute_buckling`·`compute_natural_frequencies` 는 짧은 변을 스팬으로 써서 자동으로 W130 을
  띄운다. 그 경고가 뜨면 굽힘 관련 값을 그대로 보고하지 말 것.
- `αL < 1` 이면 전단 전달이 거의 없어 면재가 사실상 따로 굽는다.
- 순응층에 G13 이 없으면 G12 로 대체하고 W120 을 붙인다 — α ∝ √G_c 라 민감하다.

## 17. 판 경계조건 (v0.14.0)

`compute_buckling`·`compute_natural_frequencies` 의 `boundary` 가 **4글자 코드**를 받는다 —
앞 두 글자가 x 방향 두 변, 뒤 두 글자가 y 방향 두 변이다(`"CCSS"` = x변 고정-고정,
y변 단순-단순). SS/CC/CS 조합 9종. `"simply_supported"`·`"clamped"` 별칭도 그대로 쓴다.

- **SSSS 만 정해다.** 나머지는 1항 Rayleigh–Ritz 상계라 **비보수**다.
  실측 오차: CCCC +6.6%, SSCC +11.6% — **혼합 경계일수록 커진다.**
  `"SSSS"` 값을 하한으로 함께 보고해 참값을 감쌀 것.
- 진동수는 훨씬 정확하다(CCCC 비 1.829 vs 문헌 1.83). 좌굴만 주의하면 된다.
- **자유변(F)은 E100 으로 거부된다.** 1항 Ritz 가 강체 모드를 놓쳐 6.4배 비보수가 되기
  때문이다 — 근사값을 내주지 않는다. 자유변이 있는 패널은 FE 로 보내라.
- `panel` 에 경계조건을 넣으면 E100 이다. 반드시 `boundary` 인자로 준다.

