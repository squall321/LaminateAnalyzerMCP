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
