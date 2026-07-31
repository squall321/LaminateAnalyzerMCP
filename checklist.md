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
- [x] ~~GitHub push~~ — 완료 (2026-07-31, v0.4.1·v0.5.0 반영)
- [x] ~~materialtwin HTTP transport~~ — **완료(2026-07-16 밤, 병행 협업)**: 웹앱 /mcp 마운트(SIF v1.1.0, bare-path 수정), 삭제 툴 HTTP 비노출, manifest opt-in → 게이트웨이 자동 합류(도구 147). 게이트웨이 한 세션에서 재료 실측 E→적층 해석 체인 실증(W120 추적, Â11=E/(1-ν²) 일치)
- [x] 계산도구×페르소나 스케일 설계를 **HWAXPortal에 전달** — `HWAXPortal/docs/CALC-TOOLS-FEDERATION-SCALING.md` (불변식 5 + 백로그 A1/A2/G1/G2). 실행 주체는 포털 관제 측
- [x] ~~V2 열/흡습·층별 응력 복원·파손 판정~~ — 완료 (v0.3.0 열휨+크랙차폐, v0.4.0 응력·Tsai-Wu, v0.5.0 흡습)

## 2026-07-30 세션 (V2-1차: 열 휨 + 크랙 차폐, 서버 0.3.0)

- [x] 열탄성: CTE 스키마(α/α1·α2, ppm 착오 W110) + compute_thermal_response(유효 CTE·열곡률·잔류응력·판 휨) + E203
- [x] 동박률 균질화: homogenize_layer (Voigt, α는 강성 가중)
- [x] 크랙 차폐: assess_crack_shielding (터널 G_ss·σ_c·Dundurs·He-Hutchinson 1/4·shear-lag·점탄성 이완) + material.viscoelastic
- [x] 검증: Timoshenko 바이메탈 rel 1e-6, 평형 불변식, Griffith 항등, 단위 브리지 — 테스트 122종 전부 통과
- [x] mcp<2 핀 (2.0이 FastMCP 제거 — materialtwin과 동일 결정), venv 재생성
- [x] ~~라이브 반영~~ — **push 완료(2026-07-31)**: v0.4.1 SIF 재빌드·재기동(/health 0.4.1), 게이트웨이 재기동으로 신규 4종 노출(188 도구). 카탈로그 라벨 0.4.1 동기화(HEAXHub 커밋)
- [x] 층별 기계 응력 복원 + 파손 판정 — recover_ply_stresses (v0.4.0): σ_xyz/σ1σ2τ12 3점 복원, Tsai-Wu R·Max Stress 모드·FPF, ΔT 중첩. [0/90]s FPF=90°횡인장·하중×R→R=1 검증, 테스트 132
- [x] ~~좌굴·check_design_rules~~ — 완료 (v0.5.0: 설계규칙·좌굴·진동·진행성파손·흡습, 적대 검증 19건 수정)

## 남은 것 (2026-07-31 기준)

### 물리 확장 (다음 스프린트 후보)
- [x] ~~층간(ILSS)~~ — v0.6.0 compute_interlaminar_stresses (평형법·2×2 연성계, 계면+내부 극값 여유율)
- [x] ~~샌드위치/FSDT~~ — v0.6.0 횡전단 유연성 R_s(임계모드 정합) + 에너지등가 A55/A44 + 1차 보정
- [x] ~~점탄성 감쇠 합성~~ — v0.6.0 모달 η·Q (MSE, 1차 모드·중립면 정합)
- [x] ~~간이 피로~~ — v0.7.0 estimate_fatigue_life (정규화 FI + Goodman + log_linear/basquin)
- [ ] (보류) 채널링 g(α,β) 보정표 — 수기 이식 리스크로 미탑재 유지 (계획서 §17.7.3)

### 품질·플랫폼
- [x] ~~P4 문헌 벤치마크~~ — v0.7.0 완결 (교과서 Q 공표값 + Tsai-Pagano 불변량 독립 교차검증)
- [ ] cae00 오프라인 배송 방식에 포함 (§16.6 항목 5)
- [ ] G3 게이트웨이 재집계 (포털 백로그) — 인스턴스 교체 시 도구 목록 미갱신
- [ ] 게이트웨이 프로세스 관리 (systemd 등) — 세션 종속 기동의 취약성 실측됨
- [x] ~~materialtwin 갭 2건~~ — 병행 작업으로 완료 확인 (orientation 파라미터·get_material 노출, update_material attributes 얕은 병합)
- [ ] (선택) HEAXHub 프론트 PAT 관리 UI

## 2026-07-16 세션 1 (MVP + hwax 등록 준비) — 서버 0.1.0

- [x] Phase 0~3 + HTTP transport 선행 + .mcp.json 상호 등록 + manifest schema v2 검증 (상세는 git 로그와 계획서 §13)
