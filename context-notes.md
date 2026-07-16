# Context Notes — 결정과 근거

> 다음 세션(사람/에이전트)이 재추론 없이 이어받기 위한 기록. 최신이 아래.

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
