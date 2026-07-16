# LaminateAnalyzerMCP 체크리스트

> 정본 체크리스트는 [계획서 §13](docs/mcp_laminate_planning.md)에 있다. 여기는 세션 단위 실행 상태 요약.
> 결정 근거는 [context-notes.md](context-notes.md).

## 2026-07-16 세션 2 (완성: V1 + Phase 4/5 + hwax 기동 실증) — 서버 0.2.0

### 완료

- [x] V1 Tool 4종 — solve_load_response(ε0/κ·유효상수·누출/비틀림/주강성방향·비강성), run_sensitivity_analysis(중앙차분), batch_evaluate_laminates(≤32), generate_design_report(ko/en)
- [x] E400(특이계)·E500(타임아웃) 실트리거 + dominant_coupling_terms 지표
- [x] Phase 4 — sympy 독립 오라클(T·Reuter 변환행렬 경로) 50케이스 rtol 1e-9 대조, 강건성(512 ply·극단 물성·초박층·각도 경계), math_spec.md
- [x] Phase 5 — agent_guide.md, S5 시나리오 로그(두 MCP 실 stdio 세션: materialtwin 실측 E 200 GPa → 단위 브리지 → 하이브리드 적층 분석 → W110 복구 → 리포트)
- [x] hwax 등록 실증 — integrations 심볼릭 링크(in-tree) 전환 → 스캐너 실행(기동 동일 코드 경로) → 카탈로그 v0.2.0 · 빌드 6.1s · 서비스 기동(9117, /health 200) · Caddy 라우트 등록
- [x] Caddy 라우트 구조 실측 — forward_auth(/api/v1/authz) → strip → proxy. 익명 401은 팀 공개 앱의 정상 게이트
- [x] 테스트 103종 전부 통과

### 남은 것

- [x] ~~포털 경유 MCP 접속 마무리~~ — **HEAXHub에 PAT 기능 정식 구현으로 완결** (commit 6221405). Bearer PAT로 Caddy 경유 initialize→tools/list SSE 세션 관통 실증 (§16.6 항목 2·3·4 완료). E2E용 PAT(`laminate-mcp-e2e`, admin, 365d)는 scratchpad `pat.txt`에 1회 출력됨 — 개인 토큰은 `POST /api/v1/auth/tokens`로 직접 발급, 폐기는 `DELETE /api/v1/auth/tokens/{id}`
- [ ] GitHub push (`squall321/LaminateAnalyzerMCP` + HEAXHub PAT 커밋 2건) — **사람 액션(gh 인증)**
- [ ] P4 잔여 — 문헌 벤치마크 공표값 대조 1건
- [ ] cae00 오프라인 배송 방식에 포함 (§16.6 항목 5, 유일 잔여 검증)
- [ ] (선택) HEAXHub 프론트에 PAT 관리 UI (현재는 API 발급)
- [ ] V2 — 열/흡습(CTE·ΔT, warpage), 층별 응력 복원, 파손 판정(Tsai-Wu 등)

## 2026-07-16 세션 1 (MVP + hwax 등록 준비) — 서버 0.1.0

- [x] Phase 0~3 + HTTP transport 선행 + .mcp.json 상호 등록 + manifest schema v2 검증 (상세는 git 로그와 계획서 §13)
