# LaminateAnalyzerMCP

적층 복합재의 **중립면(neutral surface)·ABD 강성·평가 지표**를 결정론적으로 계산하는 MCP 서버.
LLM 에이전트가 계산을 직접 하지 않고 이 서버를 호출한다. 동일 입력 → 바이트 동일 응답(payload_hash 추적).

- 계획/사양: [docs/mcp_laminate_planning.md](docs/mcp_laminate_planning.md) (수학 사양·규약·오류코드의 단일 소스)
- 진행 상태: [checklist.md](checklist.md) · 결정 기록: [context-notes.md](context-notes.md)

## Tool 7종 (MVP)

`analyze_laminate`(원샷 진입점) · `validate_laminate_input` · `compute_abd_matrix` ·
`compute_neutral_axis` · `evaluate_laminate` · `get_reference_cases`(폐형해 자가검증) · `get_server_info`

핵심 규약. `laminae[0]` = 최하단 ply, 각도 deg(CCW), `unit_system` 필수(`SI` | `SI_mm`).

## 실행

```bash
# 로컬 stdio (Claude Code는 .mcp.json으로 자동 연결 — materialtwin과 함께 등록됨)
.venv/bin/laminate-mcp

# HTTP (HEAXHub fastapi 스택과 동일 형태)
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --root-path /apps/laminate_analyzer_mcp
# → GET /health, MCP는 POST /mcp (streamable HTTP)
```

## hwax portal(HEAXHub) 등록

`.portal/manifest.yaml`(schema v2)이 정본이고, `HEAXHub/integrations/laminate-analyzer-mcp/`에
오버레이가 배치되어 있다. HEAXHub 재기동 시 자동 스캔되어 `/apps/laminate_analyzer_mcp/`로 서빙된다.

```bash
# 에이전트 연결 (배포 후)
claude mcp add --transport http laminate-analyzer <포털베이스>/apps/laminate_analyzer_mcp/mcp
```

## 개발

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest            # 75 tests
```
