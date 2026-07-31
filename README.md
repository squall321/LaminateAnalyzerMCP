# LaminateAnalyzerMCP

적층 복합재의 **중립면(neutral surface)·ABD 강성·평가 지표**를 결정론적으로 계산하는 MCP 서버.
LLM 에이전트가 계산을 직접 하지 않고 이 서버를 호출한다. 동일 입력 → 바이트 동일 응답(payload_hash 추적).

- 계획/사양: [docs/mcp_laminate_planning.md](docs/mcp_laminate_planning.md) (수학 사양·규약·오류코드의 단일 소스)
- 진행 상태: [checklist.md](checklist.md) · 결정 기록: [context-notes.md](context-notes.md)

## Tool 21종 (MVP 7 + V1 4 + V2 10)

`analyze_laminate`(원샷 진입점) · `validate_laminate_input` · `compute_abd_matrix` ·
`compute_neutral_axis` · `evaluate_laminate` · `get_reference_cases`(폐형해 자가검증) · `get_server_info` ·
`solve_load_response`(ε0/κ·유효 공학 상수) · `run_sensitivity_analysis` ·
`batch_evaluate_laminates`(≤32) · `generate_design_report`(ko/en) ·
`compute_thermal_response`(유효 CTE·열곡률·PCB 휨) · `homogenize_layer`(동박률) ·
`assess_crack_shielding`(크랙 문턱·보호층 차폐·점탄성 이완) · `recover_ply_stresses`(층별 응력·Tsai-Wu 파손) ·
`check_design_rules`(적층 관례 판정) · `compute_buckling` · `compute_natural_frequencies`(감쇠·횡전단 유연성 포함) ·
`run_progressive_failure`(한계하중) · `compute_interlaminar_stresses`(박리) · `estimate_fatigue_life`(반복 수명)

핵심 규약. `laminae[0]` = 최하단 ply, 각도 deg(CCW), `unit_system` 필수(`SI` | `SI_mm`), 하중은 단위 폭당.
에이전트용 상세 절차는 [docs/agent_guide.md](docs/agent_guide.md), 수식 유도는 [docs/math_spec.md](docs/math_spec.md),
두 MCP 협동 실 세션 기록은 [docs/s5_scenario_log.md](docs/s5_scenario_log.md).

## 두 표면 — MCP와 REST

같은 계산·검증·응답 envelope를 두 경로로 노출한다.

| 소비자 | 경로 | 발견 방법 |
|---|---|---|
| LLM 에이전트 (Claude 등) | `POST /mcp` (streamable HTTP) 또는 stdio | `tools/list` · `laminate://guide` 리소스 · `get_reference_cases` few-shot |
| 스크립트·서비스·사람 | `GET/POST /api/v1/tools[/{name}]` | `GET /api/v1/tools`(목록+JSON Schema) · `GET /api/v1/guide` · `/docs`(OpenAPI) |

```bash
curl -s localhost:8000/api/v1/tools | jq '.tools[].name'        # 도구 21종
curl -s -X POST localhost:8000/api/v1/tools/analyze_laminate \
     -H 'Content-Type: application/json' \
     -d '{"laminate":{"unit_system":"SI_mm","laminae":[...]}}'   # 실행
```

## 실행

```bash
# 로컬 stdio (Claude Code는 .mcp.json으로 자동 연결 — materialtwin과 함께 등록됨)
.venv/bin/laminate-mcp

# HTTP (HEAXHub fastapi 스택과 동일 형태)
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --root-path /apps/laminate_analyzer_mcp
# → GET /health · REST /api/v1/* · OpenAPI /docs · MCP POST /mcp
```

## hwax portal(HEAXHub) 등록 — 완료

`HEAXHub/integrations/laminate-analyzer-mcp`가 이 리포지토리의 **심볼릭 링크**(in-tree 통합)이고,
`.portal/manifest.yaml`(schema v2)이 정본이다. 카탈로그 등록·빌드·서비스 기동·Caddy 라우트까지
실증 완료(2026-07-16). Caddy 라우트는 `forward_auth(/api/v1/authz) → prefix strip → 127.0.0.1:<port>`
구조라 포털 경유 접속에는 **포털 인증**이 필요하다(visibility: team).

```bash
# 에이전트 연결 (포털 경유 — 인증 토큰 필요)
claude mcp add --transport http laminate-analyzer \
  <포털베이스>/apps/laminate_analyzer_mcp/mcp --header "Authorization: Bearer <token>"
```

## 개발

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest            # 216 tests
```
