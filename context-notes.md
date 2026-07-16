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
