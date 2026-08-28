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

## 2026-07-31 세션 3 (피로·문헌 벤치마크·체인 완결) — 서버 0.7.1

- [x] P4 문헌 벤치마크 완결 — 교과서 Q 공표값 + Tsai-Pagano 불변량 독립 교차검증(5종)
- [x] estimate_fatigue_life — **부호 보존 성분별 S-N + 표준 Goodman** (초기 FI 기반 설계는
      완전반복을 '무한수명'으로 뒤집는 비보수 결함이라 재설계). 적대 검증 발견 14종 반영
- [x] materialtwin 갭 2건 완료 확인 + **재료→적층 체인 E2E 실증**
      (0/90/45 쿠폰 → ply 카드 역산 → attributes 저장 → laminate 해석, 출처 추적)
- [x] 게이트웨이 운영 이슈 3건 포털 문서 전달(프로세스 수명·G3 재집계·버전 라벨)
- [x] 배포: SIF 0.7.1 · 카탈로그 · 게이트웨이 196 도구 · 완전반복 E2E 검증
- [x] 테스트 212종 통과
- **주의**: 이번 적대 검증은 verify 에이전트가 주간 한도로 전부 실패해 "확인 0건"으로
      보고됐다 — 미검증을 뜻하므로 finder 발견을 직접 재현·판정했다. 워크플로 결과의
      confirmed 수치는 verify 성공 여부와 함께 읽을 것

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
- [x] ~~자유 가장자리 박리~~ — v0.10.0 assess_free_edge_delamination (O'Brien ERR 폐형해 +
      계면별 지배 구동력 peel/τyz/τxz). 면내 R=1.33 통과 케이스가 가장자리 여유 0.91 로
      뒤집히는 것을 실측 확인. 도구 26종, 테스트 299
- [x] ~~미시역학~~ — v0.11.0 derive_lamina_from_constituents (ROM·Halpin–Tsai·Chamis·Schapery,
      ξ 극한이 Reuss/Voigt 와 정확히 일치). homogenize_layer 오용 역게이트 동반
- [x] ~~수분 확산 동역학~~ — v0.11.0 compute_moisture_uptake (Fickian 해석해, 흡습/베이크,
      두께방향 분포, delta_C 체인). τ_50 = 0.0492 문헌 일치
- [x] ~~경계조건 확장~~ — v0.11.0 boundary='clamped' (1항 Rayleigh–Ritz). SS 환원 rel 1e-12,
      등방 정사각 k=10.74(정해 10.07 대비 상계 +6.6%)를 W130으로 명시
- [x] ~~지배 파손모드 게이트~~ — v0.11.0 압축 하중 좌굴 침묵 봉쇄(실측 410배 모순),
      열 구속 좌굴 ΔT_cr, panel 미지 키 침묵 차단, 자유단 W130 이 신규 도구 지목
- [x] ~~박리 혼합모드~~ — v0.12.0 거울 분할은 대칭 논증으로 순수 Mode I(정확), 그 외는
      G_Ic~G_IIc 범위. Benzeggagh–Kenane. Suo–Hutchinson 위상각은 수치표라 미탑재
- [x] ~~열잔류 파손·피로 반영~~ — v0.12.0 tsai_wu_with_offset (R 2차식 정해). 진행성 파손은
      단계마다 잔류 재계산, 피로는 평균응력 이동. ΔT=−150K 에서 FPF −81%, 수명 8자릿수 감소
- [x] ~~순응층 부분합성 굽힘~~ — v0.13.0 assess_partial_composite_bending (shear-lag 폐형해).
      UTG/OCA/UTG L=10mm 에서 CLT 2.03배 과대. 좌굴·진동수에 교차 게이트
- [ ] (보류) 채널링 g(α,β) 보정표 — 수기 이식 리스크로 미탑재 유지 (계획서 §17.7.3)
- [x] ~~기하 비선형 (V3)~~ — v0.8.0: compute_bistable_shapes(Hyer 쌍안정·임계 판 크기)
      + compute_large_deflection(von Karman, movable/immovable) + compute_postbuckling
      + 선형 도구 유효범위 게이트(thermal warpage w/h>0.3 → W130). 도구 24종
- [x] ~~V3 적대 검증~~ — v0.9.1: 후보 23건 중 확정 13/반증 7. γxy⁰ 자유도 추가(허위 쌍안정 제거),
      κxy 게이트(대리지표→실제 곡률), 형상별 에너지 장벽, 좌굴후 강성비 R 반영,
      막 컴플라이언스 3×3 역행렬, 세장판·한계안정성 경고, E501 누출 차단. 테스트 281
- [x] ~~V3 배포~~ — 2026-08-28: push → SIF 재빌드 → 인스턴스 교체(9152→9236, 0.7.2→0.9.1)
      → Caddy 재등록 → 게이트웨이 자동 재집계(도구 구성 변경 0→25, 연합 총 260종).
      MCP·REST 양 표면 라이브 검증, 결정론 응답 해시 동일 확인.
      HEAXHub 오버레이 manifest 도 리포 정본으로 동기화(0.7.2·11종 → 0.9.1·25종)
- [x] ~~재료 전단 비선형~~ — v0.9.0 solve_nonlinear_shear_response (Hahn–Tsai 할선 고정반복,
      구성식 잔차 1e-11, [±45] τ=Nx/2h 폐형해 일치, 선형 도구 역게이트). 도구 25종, 테스트 269

### 품질·플랫폼
- [x] ~~P4 문헌 벤치마크~~ — v0.7.0 완결 (교과서 Q 공표값 + Tsai-Pagano 불변량 독립 교차검증)
- [ ] cae00 오프라인 배송 방식에 포함 (§16.6 항목 5)
- [ ] G3 게이트웨이 재집계 (포털 백로그) — 인스턴스 교체 시 도구 목록 미갱신
- [ ] 게이트웨이 프로세스 관리 (systemd 등) — 세션 종속 기동의 취약성 실측됨
- [x] ~~materialtwin 갭 2건~~ — 병행 작업으로 완료 확인 (orientation 파라미터·get_material 노출, update_material attributes 얕은 병합)
- [ ] (선택) HEAXHub 프론트 PAT 관리 UI

## 2026-07-16 세션 1 (MVP + hwax 등록 준비) — 서버 0.1.0

- [x] Phase 0~3 + HTTP transport 선행 + .mcp.json 상호 등록 + manifest schema v2 검증 (상세는 git 로그와 계획서 §13)
