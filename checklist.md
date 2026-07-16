# LaminateAnalyzerMCP 체크리스트

> 정본 체크리스트는 [계획서 §13](docs/mcp_laminate_planning.md)에 있다. 여기는 세션 단위 실행 상태 요약.
> 결정 근거는 [context-notes.md](context-notes.md).

## 2026-07-16 구현 세션 (MVP + hwax 등록)

### 완료
- [x] Phase 0 — git init, Q1~Q8 기본값 채택 기록
- [x] Phase 1 — solver 엔진(material/abd/neutral_axis), 폐형해 R1~R5, 성질 P1~P7, 커버리지 98%
- [x] Phase 2 — MVP Tool 7종, envelope 결정론(P8), 오류코드 유발 테스트(E400/E500 유예), 실프로세스 stdio 스모크
- [x] Phase 3 — 지표 9종+요약 3항목, W110~W120, 내장 기준 케이스 3종(자가 검증 루프)
- [x] Phase 6 선행 — streamable HTTP + /health, uvicorn --root-path 스모크(포털 Host 헤더 포함)
- [x] hwax 등록 — `.portal/manifest.yaml` schema v2 검증 통과, `HEAXHub/integrations/laminate-analyzer-mcp/` 오버레이 배치
- [x] 상호 등록 — `.mcp.json`에 laminate-analyzer + materialtwin 두 서버 (로컬 stdio)
- [x] 테스트 75종 전부 통과 (`.venv/bin/pytest`)

### 남은 것 (다음 세션)
- [ ] HEAXHub 백엔드 재기동 시 카탈로그 등록 확인 (`GET /api/v1/apps?q=laminate`, Bearer 필요) — **사람 액션**
- [ ] GitHub push (`squall321/LaminateAnalyzerMCP`) 후 manifest source를 git으로 전환 — **사람 액션(gh 인증)**
- [ ] Phase 4 — sympy 오라클, 문헌 벤치마크 수치 내장, 강건성(512 ply), math_spec.md
- [ ] Phase 5 잔여 — S5 대화 시나리오 통과 로그, agent_guide.md
- [ ] §16.6 잔여 검증 — 포털 2단 프록시 통과(SSE) 실측, cae00 오프라인 배송
- [ ] V1 — solve_load_response(E400 실트리거), 타임아웃(E500), 민감도, 리포트, batch
