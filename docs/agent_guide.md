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
