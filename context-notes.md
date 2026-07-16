# Context Notes — 결정과 근거

> 다음 세션(사람/에이전트)이 재추론 없이 이어받기 위한 기록. 최신이 아래(세션 2가 하단).

## 2026-07-16 구현 세션

**패키지 이름 `app/` (계획서 §9는 `server/`였음).**
HEAXHub fastapi 스택의 entrypoint가 `uvicorn app.main:app --root-path $ROOT_PATH`로 고정되어 있어
launch.command 오버라이드 없이 그대로 타려면 `app/` 패키지가 필요했다. materialtwin(backend/app)과도 정합.
계획서 부록 C 14에 개정 기록.

**Q1~Q8 전부 기본값 채택 (Phase 0).**
특히 Q8(오류 언어)은 한국어 채택 → D10 개정. materialtwin이 한국어 오류 관례를 적대 리뷰로 확정한
생태계라 정합을 우선했다. Tool docstring도 한국어.

**DNS rebinding Host 검증 비활성화 (mcp_server.py).**
MCP SDK streamable HTTP의 기본 보호가 Host를 검증하는데, HEAXHub 배포 시 Host가 포털 도메인
(hwax.sec.samsung.net)으로 들어와 421이 난다(테스트로 실증). 서버는 127.0.0.1에만 바인드되고
신뢰 경계가 Caddy이므로 보호를 끄는 것이 옳다고 판단. 외부 노출 차단은 loopback 바인드가 담당.

**StreamableHTTPSessionManager는 프로세스당 1회 run 제약.**
FastAPI lifespan에서 mcp.session_manager.run()을 구동하는데 재기동(2회째 lifespan)이 불가.
운영(프로세스당 1기동)에는 영향 없음. 테스트는 module-scope TestClient로 공유.

**E400/E500 유발 테스트 유예.**
E400(SINGULAR_SYSTEM)은 완전 역산이 들어오는 V1 solve_load_response에서, E500(COMPUTE_TIMEOUT)은
타임아웃 집행 구현에서 트리거 경로가 생긴다. MVP 계산은 ms 단위 순수 numpy라 인위 트리거가 불가능.
카탈로그 전수 존재 테스트로 대체(test_catalog_complete).

**HEAXHub manifest 스키마 v2 실검증에서 얻은 규칙.**
`health_check`/`restart_policy`는 최상위가 아니라 `launch` 하위, restart는 `max_attempts`(not max_retries),
`build.type`(python_venv) 필수 — `build.stack`은 스키마 외 필드지만 스택 리졸버가 읽으므로 병기,
`source.ref` 불허. **materialtwin-web 오버레이는 status 누락 + 최상위 health_check/restart_policy +
source.ref + build.type 누락으로 스키마 위반 상태**(스택 기본값 덕에 동작 중) — MaterialTwinWeb 쪽에 전달할 것.

**manifest source는 당분간 local_path(symlink).**
GitHub push 전이라 즉시 동작을 위해 local_path 채택. push 후 git url로 전환(manifest 주석 참조).

**HEAXHub 등록 반영 시점.**
integrations_scanner.scan_integrations()는 uvicorn 기동 lifespan에서 1회 실행. 현재 :4040에 백엔드가
가동 중(API에 Bearer 인증)이며, 살아 있는 포털을 임의 재기동하지 않고 오버레이만 배치해 뒀다.
다음 재기동 또는 admin sync 때 카탈로그에 나타난다.

**단위 변환 검증 방식.**
같은 물리 적층을 SI/SI_mm로 넣어 A(×1e-3), B(×1), D(×1e3), ζ(불변) 관계를 직접 테스트
(test_units_bridge_si_vs_si_mm). areal_mass의 SI_mm 표기는 t/mm²(값이 매우 작음)로 일관성 우선.

**기준 케이스 기대값은 폐형해 수식으로 모듈 내 산출.**
수기 수치 금지 원칙(계획서 §8) 유지 — reference_cases.py가 Q 상수부터 수식으로 계산해 embed.
엔진 경로(qbar/ABD 적분)와는 독립.

## 2026-07-16 완성 세션 (서버 0.2.0)

**hwax 등록을 오버레이 복사 → 심볼릭 링크(in-tree)로 전환한 이유.**
스캐너의 SIF 경로는 `source.url`이 있을 때만 탄다. local_path(url 없음)는 레거시 in-tree 빌더로
떨어지는데, 이는 integration 디렉터리 안에 코드가 있어야 한다(첫 스캔에서
"integrations/.../.venv/bin/uvicorn not found"로 실증). 디렉터리 자체를 리포 심볼릭 링크로 만들면
데모 앱들과 동일한 in-tree 형태가 되고, 빌더가 리포의 .venv를 그대로 재사용한다.
manifest의 오버레이 사본이 사라져 정본이 하나가 되는 부수 이득도 있다.

**스캐너 out-of-band 실행 결정.**
가동 중인 백엔드(:4040)를 재기동하는 대신, 기동 lifespan과 동일한 코드
(`app.db.session.SessionLocal` + `scan_integrations`)를 backend venv로 1회 실행했다.
스캐너는 멱등 reconcile로 설계돼 있어(App/AppVersion 행 조정) 무중단 반영이 가능했다.
결과: 카탈로그 v0.2.0(updated), 빌드 6.1s, 기동 pid/port는 `var/integration_state/laminate_analyzer_mcp.json`.

**Caddy 라우트의 실제 구조(실측)와 401의 의미.**
`forward_auth(GET /api/v1/authz, 4040) → strip_path_prefix → reverse_proxy 127.0.0.1:9117`.
익명 요청 401은 visibility: team 앱의 정상 인증 게이트다. 포털 경유 MCP 접속은
`--header "Authorization: Bearer <token>"`로 인증을 함께 보내야 하며, SSE·Mcp-Session-Id의
포털 통과 확인은 토큰 확보 후 마무리(§16.6 항목 2·4). 앱 직결(9117)로는 /health 200·SSE 정상.

**프리픽스 헬스 프로브 404는 우리 결함이 아님.**
스캐너의 프로브는 `/apps/<id>/health`(프리픽스 포함)를 앱 포트에 직접 친다. heax-demo-fastapi도
동일하게 404가 난다(fastapi 스택 공통 — Caddy가 strip하므로 실경로는 문제없음). 라우트는 정상 등록됨.

**HEAXHub 빌더가 리포 .venv를 재구성함 — test extra가 삭제되는 트레이드오프 (실측 정정).**
빌더는 python3.12 venv를 새로 만들고 `pip install -e .`(본 의존성만)를 수행한다. 심볼릭 링크
방식에서는 워크스페이스 venv == 리포 .venv이므로 **재빌드 때마다 pytest/hypothesis/sympy 등
test extra가 사라진다**. 개발 재개 시 `pip install -e ".[test]"` 재실행 필요(이번 세션에서 실제 발생
→ 재설치 후 103 테스트 통과 재확인). `.mcp.json`의 stdio 경로와 laminate-mcp 스크립트는 유효 유지.

**교훈 — cd 지속성 사고.**
스캐너 실행 시 `cd HEAXHub/backend`가 세션에 지속되어 이후 pytest·git 커밋이 HEAXHub 리포에서
실행되는 사고가 있었다(HEAXHub에 잘못 생성된 커밋은 mixed reset으로 원상 복구 완료, 파일 무손실).
또 `pytest | tail` 파이프가 실패 종료코드를 가려 커밋 게이트가 뚫렸다. 이후 규칙: 타 리포 작업은
서브셸/절대경로로, 검증 게이트에 파이프 금지.

**S5 시나리오에서 얻은 에이전트 팁.**
materialtwin처럼 list를 반환하는 tool은 MCP가 **원소별 content 항목**으로 직렬화한다 —
content[0]만 파싱하면 첫 원소만 얻는다(agent_guide §3에 반영). 실측 SUS201 E=200 GPa로
digital thread(materialtwin:material/1/test/1 → ply source.ref → payload_hash)가 실제로 이어짐을 확인.

**E400·E500 실트리거 확보.**
E400은 solve_load_response의 특이계 경로(영행렬 직접 테스트 + 파이프라인 monkeypatch),
E500은 batch/sensitivity 루프의 시간 예산(config.COMPUTE_TIMEOUT_S) 집행으로 유발 테스트가 생겼다.
v2.4(d)의 유예가 해소됐고, 문헌 벤치마크 1건만 P4 잔여로 남는다.

## 2026-07-16 PAT 세션 (정식 인증 경로)

**HEAXHub에 PAT를 정식 구현한 이유와 설계.**
authz는 쿠키 세션(브라우저 SSO 승계)과 Bearer JWT(TTL 1h)만 받아 헤드리스 MCP 클라이언트가
설 자리가 없었다. GitHub 스타일 PAT를 추가: `heax_pat_` 프리픽스 + sha256 해시만 저장(평문 1회 노출),
만료(무기한 허용)·폐기·last_used(60s 스로틀 — authz가 /apps/* 요청마다 불리므로 쓰기 증폭 방지)·audit.
검증 단일 소스는 pat_service.resolve_user이고 deps.get_current_user와 authz._user_from_request가
프리픽스로 분기한다. 커밋 6221405(기능)·2b72d0c(laminate 통합 등록).

**라이브 반영 절차 기록.**
테이블은 테스트 모듈의 멱등 create_all이 라이브 PG에 생성(그들의 테스트는 실 PG 사용).
백엔드는 기존 기동 방식 그대로 재기동(`bash -c 'set -a; source .env; set +a; cd backend && uvicorn ...'`
nohup, 로그 var/logs/backend.log). 재기동 시 integrations reconcile이 앱들을 자동 복구했다
(materialtwin_web이 9118로 재기동 — 슬러그 주소라 클라이언트 영향 없음, 포트 불변 가정 금지 재확인).

**E2E 증거 (§16.6 2·3·4 완결).**
Bearer PAT로 Caddy(:4180) 경유 — /health 200, 익명 401 유지, initialize 200+SSE+Mcp-Session-Id,
동일 세션 notifications/initialized 202, tools/list가 Tool 11종 반환. E2E PAT는 admin 소유
"laminate-mcp-e2e"(365d) — scratchpad에 1회 출력, 필요 시 DELETE /api/v1/auth/tokens/{id}로 폐기.

**참고.** HEAXHub 작업 트리에 병행 작업으로 보이는 미커밋 `backend/app/api/v1/mcp.py`(untracked)와
`router.py` 수정이 있었다 — 내 커밋에는 포함하지 않음(명시적 파일 목록으로 커밋).

## 2026-07-16 게이트웨이 자동탐지 세션

**"포털 최상단 자동화"의 실체 — 이미 구축돼 있었고, 앱 쪽 opt-in만 비어 있었다.**
HWAXMcpGateway(:9110)에 heax_registry 자동탐지가 가동 중: HEAXHub `/api/v1/mcp/servers`
(병행 작업의 미커밋 mcp.py)를 60초마다 폴링해, manifest에 `mcp.expose: true`를 선언한
published 앱을 백엔드로 자동 합류시키고 **HEAX 서비스 PAT를 중앙 주입**해 Caddy authz를
통과한다(오늘 구현한 PAT가 이 용도로 이미 config에 들어가 있었음). 우리 manifest에 mcp
블록이 없어 레지스트리가 빈 목록이었고, opt-in 추가(0.2.1)+재스캔 후 60초 내
`heax-laminate_analyzer_mcp` 합류(도구 118→129), 게이트웨이 경유 analyze 실호출 ok.

**토큰 3층 구분 (혼동 주의).**
① HWAXPortal PAT(`POST /auth/pat`, RS256+JWKS) — rest_proxy `/api/<site>/*`용.
② GW_TOKEN — MCP 게이트웨이(:9110/mcp) 인바운드 인증, 에이전트 설정에 이것 하나만.
③ HEAXHub PAT(`heax_pat_`, 오늘 구현) — 게이트웨이→HEAX 앱 구간에 중앙 주입(클라이언트 무관).

**materialtwin 주의.** materialtwin MCP는 stdio 전용이라 레지스트리 opt-in 불가.
HTTP transport 추가(§16.4 백로그) 시 같은 manifest 블록으로 자동 합류 가능.

**manifest 스키마 v2에는 mcp 블록이 없다(additionalProperties: false 위반).**
스캐너가 스키마를 강제하지 않아 동작하지만, 병행 작업 정리 시 schema v2에 mcp 블록 추가 권장.
