# LaminateAnalyzer MCP 서버 — FastMCP tool 등록과 guide 리소스 (계획서 §6.2, §6.6)
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app import config, units
from app.errors import CATALOG, item
from app.services import envelope as ENV
from app.services import pipeline as PIPE
from app.services import reference_cases as REF

_INSTRUCTIONS = """적층 복합재 중립면·ABD·평가 지표를 결정론적으로 계산하는 서버.
- 진입점은 analyze_laminate (검증→ABD→중립면→지표 원샷). 완전한 입력 예시는 get_reference_cases로 확인.
- laminae[0]이 최하단(bottom) ply. 각도는 deg, +x→섬유1축 CCW 양수. N/M은 단위 폭당 물리량.
- unit_system 필수: "SI"(Pa·m·kg/m³) 또는 "SI_mm"(MPa·mm·t/mm³).
- ABD를 직접 암산하지 말고 반드시 이 서버로 계산할 것. 응답의 assumptions와 warnings를 사용자에게 전달할 것."""

# HTTP 모드는 loopback에만 바인드되고 신뢰 경계는 앞단 프록시(HEAXHub Caddy)다.
# 프록시 체인이 Host를 포털 도메인으로 전달하므로 Host 검증(DNS rebinding 보호)은
# 여기서 끄고, 외부 노출 차단은 127.0.0.1 바인드로 보장한다 (계획서 §16.6 검증 항목).
mcp = FastMCP(
    config.SERVER_NAME,
    instructions=_INSTRUCTIONS,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _guarded(fn, payload, **kwargs) -> dict:
    """모든 Tool의 공통 방어선 — 예외를 E501 envelope로 표준화 (stack trace 비노출, §11)."""
    try:
        return fn(payload, **kwargs)
    except Exception as e:  # noqa: BLE001
        return ENV.build(data=None,
                         errors=[item("E501", detail=type(e).__name__)],
                         warnings=[], payload=payload if isinstance(payload, dict) else {},
                         unit_system=payload.get("unit_system") if isinstance(payload, dict) else None)


@mcp.tool()
def analyze_laminate(laminate: dict, include: list[str] | None = None,
                     neutral_axis_mode: str = "clt_weighted",
                     include_abd_6x6: bool = False, include_debug: bool = False) -> dict:
    """적층 정의를 받아 검증→ABD→중립면→지표를 한 번에 계산한다 (권장 진입점).

    laminate: {unit_system: "SI"|"SI_mm", laminae: [{thickness, angle_deg, material}, ...]}.
    laminae[0]이 최하단 ply. material은 {type:"isotropic", E, nu} 또는
    {type:"orthotropic_2d", E1, E2, G12, nu12} (+선택 rho, name, source).
    include로 부분 선택(["abd","neutral_axis","indices"]), 기본은 전부.
    단계별로 따로 보려면 compute_abd_matrix / compute_neutral_axis / evaluate_laminate 사용.
    """
    return _guarded(PIPE.run_analysis, laminate, include=include,
                    neutral_axis_mode=neutral_axis_mode,
                    include_abd_6x6=include_abd_6x6, include_debug=include_debug)


@mcp.tool()
def validate_laminate_input(laminate: dict) -> dict:
    """입력 검증만 수행한다(계산 없음). 대화형으로 입력을 구성하는 중간 단계에서 사용.

    오류는 {code, message, field, suggestion} 목록으로 반환되며 suggestion만 따라도
    재호출로 자가 복구가 가능하도록 작성되어 있다.
    """
    return _guarded(PIPE.run_validation, laminate)


@mcp.tool()
def compute_abd_matrix(laminate: dict, include_abd_6x6: bool = False) -> dict:
    """ABD 강성 행렬만 계산한다. A[N/m|N/mm], B[N], D[N·m|N·mm] + 정규화 Â/B̂/D̂(Pa|MPa).

    정규화는 Â=A/h, B̂=√12·B/h², D̂=12D/h³ (합동변환 — 양정치성 보존, 두께 스케일 불변).
    """
    return _guarded(PIPE.run_analysis, laminate, include=("abd",), include_abd_6x6=include_abd_6x6)


@mcp.tool()
def compute_neutral_axis(laminate: dict, mode: str = "clt_weighted") -> dict:
    """중립면 위치를 계산한다. mode: clt_weighted(기본, x·y 각각 B11/A11·B22/A22) | beam_equivalent(E_x 가중 도심).

    출력은 midplane 기준·bottom 기준·무차원 ζ=(z-z_bot)/h 세 표기를 항상 병기한다.
    대칭 적층이면 모든 정의에서 0(midplane). 정의별 가정은 응답의 axis_definition 참조.
    """
    return _guarded(PIPE.run_analysis, laminate, include=("neutral_axis",), neutral_axis_mode=mode)


@mcp.tool()
def evaluate_laminate(laminate: dict, criteria: dict | None = None) -> dict:
    """평가 지표(커플링·이방성·준등방성·중립면 오프셋·조건수)와 등급, 권고를 반환한다.

    criteria(선택)로 기준 대비 pass/fail 판정: 지원 키 =
    max_coupling_ratio, min_quasi_isotropy_score, max_abs_ns_offset_ratio, max_condition_number.
    """
    return _guarded(PIPE.run_analysis, laminate, include=("indices",), criteria=criteria)


@mcp.tool()
def solve_load_response(laminate: dict, loads: dict,
                        scan_principal_direction: bool = True) -> dict:
    """단위 폭당 하중 N/M에 대한 응답 ε0·κ와 유효 공학 상수, V1 지표를 계산한다.

    loads = {"N": [Nx, Ny, Nxy], "M": [Mx, My, Mxy]} — N은 SI: N/m | SI_mm: N/mm,
    M은 두 단위계 모두 N (단위 폭당 모멘트). 생략된 벡터는 0.
    반환: response(ε0, κ), effective_constants(막 Ex/Ey/Gxy/ν_xy, 굽힘 Ex_b/Ey_b),
    v1_indices(membrane_bending_leakage, twist_under_bending, 주강성 방향 스캔, 비강성).
    """
    return _guarded(PIPE.run_load_response, laminate, loads=loads,
                    scan_principal_direction=scan_principal_direction)


@mcp.tool()
def run_sensitivity_analysis(laminate: dict, angle_delta_deg: float = 1.0,
                             thickness_rel: float = 0.01, modulus_rel: float = 0.01) -> dict:
    """ply별 각도/두께/탄성계수 섭동에 대한 D̂11·coupling_ratio·ζ_x의 민감도(중앙차분).

    제조 공차 강건성 판단용. 결정론적이며 에이전트가 2×3×n_plies회 반복 호출할 것을 1회로 줄인다.
    ply 수가 많으면 계산 시간 예산(10s) 초과 시 E500을 반환한다.
    """
    return _guarded(PIPE.run_sensitivity, laminate, angle_delta_deg=angle_delta_deg,
                    thickness_rel=thickness_rel, modulus_rel=modulus_rel)


@mcp.tool()
def batch_evaluate_laminates(laminates: list[dict], criteria: dict | None = None) -> dict:
    """최대 32개 적층안을 일괄 평가해 케이스당 핵심 지표 요약을 반환한다.

    에이전트 주도 설계 탐색 루프(후보 생성 → 일괄 평가 → 선별)의 가속용.
    criteria를 주면 케이스별 pass_all이 함께 반환된다. 상세가 필요한 후보만
    analyze_laminate로 재조회할 것.
    """
    try:
        return PIPE.run_batch(laminates, criteria=criteria)
    except Exception as e:  # noqa: BLE001
        return ENV.build(data=None, errors=[item("E501", detail=type(e).__name__)],
                         warnings=[], payload={"laminates": laminates if isinstance(laminates, list) else []})


@mcp.tool()
def generate_design_report(laminate: dict, criteria: dict | None = None,
                           language: str = "ko") -> dict:
    """사람용 Markdown 리포트(report_markdown)와 LLM용 요약(summary)을 생성한다.

    language: "ko"(기본) | "en". 리포트에는 입력·ABD·중립면·지표·권고·경고·가정이
    payload_hash와 함께 담겨 재현 가능한 보고 자료가 된다.
    """
    return _guarded(PIPE.run_report, laminate, criteria=criteria, language=language)


@mcp.tool()
def compute_thermal_response(laminate: dict, delta_T: float | None = None,
                             panel: dict | None = None, delta_C: float | None = None) -> dict:
    """온도(ΔT)·흡습(ΔC) 자유변형 — 유효 CTE/CME, 곡률, ply 잔류응력, (panel 주면) 판 휨.

    delta_T [K] 해석엔 ply별 alpha(또는 alpha1/2), delta_C [%M] 해석엔 beta(또는 beta1/2) 필요
    (없으면 E203). 둘 중 최소 하나. delta_T = T_현재 − T_무응력기준 (리플로우 +, 경화 냉각 −),
    delta_C = 수분함량 변화 [%M]. panel = {"Lx","Ly"} → warpage.range(coplanarity).
    동박층처럼 혼합층은 먼저 homogenize_layer로 등가 물성을 만들어 넣을 것.
    """
    return _guarded(PIPE.run_thermal, laminate, delta_t=delta_T, panel=panel, delta_c=delta_C)


@mcp.tool()
def homogenize_layer(components: list[dict]) -> dict:
    """혼합층(예: 동박률 있는 PCB 동박층)의 면내 등가 물성 — Voigt 병렬 균질화.

    components = [{"material": {"type":"isotropic","E":..,"nu":..,"alpha":..,"rho":..},
                   "volume_fraction": f}, ...] (Σf = 1).
    동박률 70% 동박층 = [{Cu, f:0.7}, {수지, f:0.3}]. 반환 material을 laminae에 그대로 사용.
    단위계 무관 — 출력 단위는 입력과 동일. E=Σf·E, α=Σf·E·α/Σf·E (힘 평형 가중).
    """
    try:
        return PIPE.run_homogenize(components)
    except Exception as e:  # noqa: BLE001
        return ENV.build(data=None, errors=[item("E501", detail=type(e).__name__)],
                         warnings=[], payload={"components": components if isinstance(components, list) else []})


@mcp.tool()
def assess_crack_shielding(laminate: dict, target_ply: int, fracture: dict | None = None) -> dict:
    """피보호층(취성 target ply)의 크랙 발생 문턱과 이웃 보호층의 개구 차폐를 평가한다.

    문헌 폐형해 기반: 터널크랙 G_ss=πσ²h/(4Ē), 임계 σ_c=√(4ĒΓ/πh)(박층 유리),
    Dundurs α(이웃 강성차 → 차폐/증폭 경향), He-Hutchinson 1/4 법칙(계면 편향 저지),
    shear-lag 전달길이, 보호층 점탄성 이완 시 차폐 저하(ℓ 성장 √(E0/E∞)).
    fracture(선택) = {"applied_strain":, "gamma_target":, "gamma_interface":, "gamma_next_layer":}
    (Γ 단위 SI: J/m², SI_mm: N/mm). 보호층 이완은 material.viscoelastic {E0,Einf,tau_s}로.
    """
    return _guarded(PIPE.run_crack_assessment, laminate, target_ply=target_ply, fracture=fracture)


@mcp.tool()
def recover_ply_stresses(laminate: dict, loads: dict | None = None,
                         delta_T: float | None = None, detail: str = "auto",
                         panel: dict | None = None) -> dict:
    """층별 응력 복원 + (강도 있으면) 파손 판정 — first-ply-failure와 여유율까지.

    loads = {"N":[Nx,Ny,Nxy], "M":[Mx,My,Mxy]} (단위 폭당), delta_T [K]를 주면 열하중 중첩
    (전 ply CTE 필요). 각 ply의 bottom/mid/top에서 적층판축 σ_xyz와 재료축 σ1/σ2/τ12 반환.
    material.strength {Xt,Xc,Yt,Yc,S}(압축 양수 관례)가 있는 ply는 Tsai-Wu 강도비 R
    (하중 R배에서 파손 — R>1 안전)과 Max Stress 지배 모드(섬유/횡/전단)를 함께 반환.
    first_ply_failure = 전 ply 최소 R. 강도는 materialtwin property registry에서 조달 가능.
    detail: "auto"(기본 — 32ply 초과 시 임계 상위 10개만+truncation note) | "full" | "summary".
    **압축 하중이면 panel={"Lx","Ly"} 를 함께 주라.** 얇은 판은 강도 여유가 충분해도 좌굴이
    먼저 온다 — 실측으로 [0/90]s 0.5mm 에 Nx=−60 N/mm 일 때 강도 R=7.07("7배 여유")인데
    같은 판의 좌굴 여유는 0.017 이었다(410배 모순). panel 을 주면 governing_mode 로
    강도·좌굴 여유를 같은 척도로 정렬해 지배 모드를 알려준다. 없으면 W130 으로 경고만 한다.
    """
    return _guarded(PIPE.run_ply_stresses, laminate, loads=loads, delta_t=delta_T,
                    detail=detail, panel=panel)


@mcp.tool()
def check_design_rules(laminate: dict, contiguity_limit: int = 4) -> dict:
    """적층 설계 규칙 검사 — 업계 관례(대칭/밸런스/10%/인접각/연속/외층 보호)를 판정한다.

    각 규칙은 {rule, severity(hard|guideline|info), pass, found, why_it_matters, fix_hint}로
    반환되어, 위반 사실뿐 아니라 물리적 이유와 수정 방향까지 설명 가능하다. 물성 데이터 불필요.
    적층안 아이디어를 낼 때 analyze_laminate와 함께 첫 검토 단계로 사용할 것.
    """
    return _guarded(PIPE.run_design_rules, laminate, contiguity_limit=contiguity_limit)


@mcp.tool()
def compute_buckling(laminate: dict, panel: dict, load_ratio: float = 0.0,
                     applied_Nx: float | None = None,
                     boundary: str = "simply_supported") -> dict:
    """직교이방 판의 좌굴 임계하중 N_cr.

    panel = {"Lx","Ly"} (길이 단위 — 다른 키를 넣으면 E100). 2축 압축: load_ratio = Ny/Nx
    (압축 양수), Nxy 미지원. applied_Nx(압축 크기)를 주면 margin.factor = N_cr/Nx.
    **boundary**: "simply_supported"(기본, Navier 폐형해 m·n=1..10 스캔) | "clamped" |
    **S·C 4글자 코드** — 앞 두 글자가 x 방향 두 변, 뒤 두 글자가 y 방향 두 변이다
    (예: "CCSS" = x변 고정-고정, y변 단순-단순). SS/CC/CS(=SC) 조합 9종을 지원한다.
    실제 패널은 고정단에 가까운 경우가 많고 N_cr 이 2.5배 이상 달라진다 — 다만 1항 Ritz 는
    **상계**라 비보수다. 등방 정사각 실측 오차 CCCC +6.6%, SSCC +11.6% (W130).
    'SSSS' 값을 하한으로 함께 보면 참값을 감싼다.
    **자유변(F)은 거부한다** — 1항 Ritz 가 강체 모드를 놓쳐 6.4배 비보수가 된다(E100).
    비대칭 적층은 축소강성 D*로 근사(W130), D16/D26 유의 시 경고(W130).
    횡전단 유연성이 유의하면(R_s>0.02) corrected_N_cr(1차 FSDT 보정)을 함께 반환한다.
    """
    return _guarded(PIPE.run_buckling, laminate, panel=panel, load_ratio=load_ratio,
                    boundary=boundary,
                    applied_Nx=applied_Nx)


@mcp.tool()
def compute_natural_frequencies(laminate: dict, panel: dict, n_modes: int = 5,
                                boundary: str = "simply_supported") -> dict:
    """직교이방 판의 고유진동수 [Hz] (낮은 순 n_modes개).

    전 ply에 밀도(rho)가 필요하다. panel = {"Lx","Ly"}. NVH·공진 회피 1차 판단용.
    **boundary**: "simply_supported"(기본) | "clamped" | S·C 4글자 코드(예: "CCSS" —
    앞 두 글자가 x변, 뒤 두 글자가 y변). 진동수는 정확도가 좋다 — 등방 정사각 1차 모드에서
    CCCC 비 1.829(문헌 1.83)이지만 CSCS 는 +9.0% 오차다. 자유변(F)은 거부한다.
    전 ply에 loss_factor(η)가 있으면 모달 감쇠(MSE법)와 Q factor를 함께 반환한다.
    횡전단 유연성 R_s = π²D11/(A55a²)도 함께 계산 — 0.02 초과면 CLT가 비보수적이므로
    W130과 1차 보정값(corrected_f1_hz)을 병기한다(두꺼운 판·샌드위치 판단 근거).
    비대칭 D* 근사·D16/D26 유의성은 W130으로 경고. 경계조건은 4변 단순지지 고정.
    """
    return _guarded(PIPE.run_frequencies, laminate, panel=panel, n_modes=n_modes,
                    boundary=boundary)


@mcp.tool()
def run_progressive_failure(laminate: dict, loads: dict, discount: float = 0.1,
                            delta_T: float | None = None) -> dict:
    """진행성 파손(ply discount) — FPF 이후 강성 저하를 반복해 한계하중까지 추적한다.

    loads = {"N":[...], "M":[...]} 하중 패턴. 반환: events(사건별 ply·모드·R),
    first_ply_failure_R, **ultimate_R(하중 제어 용량 = 최대 지지 배수)**, last_ply_R,
    사건별 유효 Ex 저하 곡선, 종료 사유. discount = 파손 ply 강성 잔존율 η (기본 0.1).
    strength 없는 ply는 탄성 유지. FPF 상세(위치·양기준)는 recover_ply_stresses 사용.
    """
    return _guarded(PIPE.run_progressive, laminate, loads=loads, discount=discount,
                    delta_t=delta_T)


@mcp.tool()
def compute_interlaminar_stresses(laminate: dict, shear: dict, detail: str = "auto") -> dict:
    """층간 전단응력 τxz·τyz 분포와 ILSS 여유 — 박리(delamination) 위험 판단.

    shear = {"Vx": , "Vy": } 단위 폭당 횡전단력(굽힘 모멘트의 공간 구배 = 전단력;
    SI: N/m, SI_mm: N/mm). CLT가 직접 주지 않는 τxz를 3D 평형식으로 사후 복원한다
    (등방 단일층이면 τmax = 1.5V/h 포물선, 상하 자유표면 0).
    material.ilss(층간 전단강도)가 있으면 계면별 margin = ILSS/|τ| (1 미만이면 박리 예상).
    자유단 박리는 정성 순위(free_edge_risk_ranking)로만 제공 — 정량은 3D 해석 필요(W130).
    detail: "auto"(기본 — 두꺼운 적층은 |τ| 상위 점만) | "full"(전 프로파일).
    """
    return _guarded(PIPE.run_interlaminar, laminate, shear=shear, detail=detail)


@mcp.tool()
def estimate_fatigue_life(laminate: dict, loads_max: dict, loads_min: dict | None = None,
                          detail: str = "auto", delta_T: float | None = None) -> dict:
    """하중 사이클에 대한 ply별 피로 수명(반복 횟수) 추정.

    loads_max/loads_min = {"N":[...], "M":[...]} (단위 폭당). loads_min 생략 시 0 → 영-인장(R=0).
    완전반복은 loads_min에 부호 반대 하중을 준다(예: max N=[100,0,0], min N=[-100,0,0]).
    두 인자의 순서는 결과에 영향을 주지 않는다(성분별로 정렬).
    각 ply에 strength{Xt,Xc,Yt,Yc,S}와 fatigue{model_type:"log_linear"|"basquin", k|b} 필요.
    재료축 성분(σ1/σ2/τ12)별 부호 보존 진폭·평균에 표준 Goodman + S-N을 적용하고, 최소 수명
    성분이 governing_component로 보고된다. life_cycles = 임계 ply 수명(등진폭·비례하중 1차 근사).
    strength/fatigue 없는 ply는 제외되며 W120으로 알린다(임계 ply가 빠지면 과대평가).
    """
    return _guarded(PIPE.run_fatigue, laminate, loads_max=loads_max, loads_min=loads_min,
                    detail=detail, delta_t=delta_T)


@mcp.tool()
def compute_bistable_shapes(laminate: dict, panel: dict, delta_T: float | None = None,
                            delta_C: float | None = None) -> dict:
    """비대칭 적층의 경화 후 실제 형상 — 쌍안정(원통 2개) 분기와 임계 판 크기.

    선형 CLT(compute_thermal_response)는 판 크기와 무관하게 안장 하나만 준다. 실제로는
    판이 임계 크기를 넘으면 안장이 불안정해지고 서로 거울상인 **원통 형상 두 개**로
    분기한다(Hyer). 원통은 전개 가능면이라 막 에너지 벌점이 없고 안장은 L⁴ 벌점을
    받는 것이 분기의 원인이다.
    panel = {"Lx","Ly"} 필수 — 분기 자체가 판 크기에 의존한다. delta_T [K] 또는
    delta_C [%M] 중 최소 하나(경화 냉각은 음수). ply별 alpha/beta 필요(없으면 E203).
    반환: equilibria(각 정지점의 곡률·형상·안정성·에너지), bistable, critical_panel,
    energy_barrier(스냅스루 장벽). 스냅 하중 자체는 계산하지 않는다.
    한계: γxy⁰=0·κxy=0 이라 [±θ] 반대칭의 비틀림 형상은 표현하지 못한다(W130).
    """
    return _guarded(PIPE.run_bistable_shapes, laminate, delta_t=delta_T, panel=panel,
                    delta_c=delta_C)


@mcp.tool()
def compute_large_deflection(laminate: dict, panel: dict, pressure: float,
                             edge_condition: str = "movable") -> dict:
    """균일압력 하 판의 기하 비선형(von Karman) 대처짐 — 막 효과로 인한 강성 증가.

    선형 해는 w ∝ q 지만, w가 두께 수준을 넘으면 막 신장이 개입해 실제 처짐이 훨씬
    작아진다. αW + βW³ = 16q/π² (SS 4변, 1항 Galerkin).
    pressure 단위 = 탄성계수 단위(SI: Pa, SI_mm: MPa). 양수/음수 모두 가능.
    edge_condition: "movable"(가장자리 면내 이동 자유, 평균 막력 0 — 기본·보수적) |
    "immovable"(면내 구속, 평균 면내변형 0). 등방 정사각에서 β가 약 3.9배 차이나므로
    두 극단을 모두 확인할 것.
    반환: w_center(비선형), w_center_linear(비교), stiffening_ratio, membrane_dominant.
    w/h < 0.3 이면 선형으로 충분하다고 W130으로 알린다.
    """
    return _guarded(PIPE.run_large_deflection, laminate, panel=panel, pressure=pressure,
                    edge_condition=edge_condition)


@mcp.tool()
def compute_postbuckling(laminate: dict, panel: dict, applied_Nx: float,
                         load_ratio: float = 0.0) -> dict:
    """좌굴 후 거동 — N > N_cr 에서도 판은 즉시 무너지지 않는다.

    compute_buckling은 N_cr까지만 답한다. 실제 박판은 좌굴 후에도 하중을 더 받으며,
    면내 접선강성이 떨어지고 하중이 가장자리로 재분배된다.
    applied_Nx = 가해진 압축 크기(양수, 단위 폭당), load_ratio = Ny/Nx.
    반환: load_over_critical, amplitude(W)·amplitude_over_thickness, stiffness_ratio
    (좌굴 후 면내 접선강성/좌굴 전 — 등방 정사각에서 정확히 0.5), effective_width_ratio
    (b_eff/b = √(N_cr/N), von Karman 유효폭).
    비재하 가장자리 면내 이동 자유 가정. W/h > 3 이면 1항 근사 범위를 벗어나 W130.
    """
    return _guarded(PIPE.run_postbuckling, laminate, panel=panel, applied_Nx=applied_Nx,
                    load_ratio=load_ratio)


@mcp.tool()
def solve_nonlinear_shear_response(laminate: dict, loads: dict) -> dict:
    """재료 면내 전단 비선형(Hahn–Tsai) 응답 — 전단 지배 적층이 선형보다 훨씬 무른 이유.

    UD 복합재의 면내 전단은 기지 지배라 뚜렷이 비선형이다: γ12 = τ12/G12 + S6666·τ12³.
    τ가 커질수록 할선 G가 떨어져 [±45] 인장처럼 전단 지배 적층은 선형 CLT가 강성을
    크게 과대평가한다(예: [±45]s에 Nx=50 N/mm → Ex가 선형의 61%).
    각 ply의 material.shear_nonlinear = {"S6666": ...} 필요 (단위 1/응력³ — SI: 1/Pa³,
    SI_mm: 1/MPa³. CFRP는 1/MPa³ 기준 1e-8 자릿수). 없는 ply는 선형으로 두고 W120으로 알린다.
    loads = {"N":[Nx,Ny,Nxy], "M":[Mx,My,Mxy]} (단위 폭당) — solve_load_response와 동일.
    반환: response(비선형)와 linear_response(대조), softening(할선/선형 유효상수 비),
    ply별 τ12·γ12·G12_secant, convergence(구성식 잔차).
    주의: 3차식에는 강도 한계가 없어 파손 이후에도 답을 낸다. γ12 > 0.05 면 W130이 뜨며
    recover_ply_stresses로 파손 판정을 반드시 병행할 것.
    """
    return _guarded(PIPE.run_nonlinear_shear, laminate, loads=loads)


@mcp.tool()
def assess_free_edge_delamination(laminate: dict, loads: dict,
                                  fracture: dict | None = None) -> dict:
    """자유 가장자리 박리 — 면내 강도로는 안전한데 가장자리에서 먼저 뜯기는 경우를 잡는다.

    recover_ply_stresses 가 "여유 있음"이라고 답해도 자유 가장자리에서는 각 ply 의
    σy·τxy 가 0이 되어야 해서 그 불균형이 층간 응력으로 넘어간다. 그 파손 모드를 보는
    유일한 도구다(compute_interlaminar_stresses 는 횡전단력 Vx/Vy 가 있어야 하고,
    순수 면내 인장에서는 0을 돌려준다).

    O'Brien 폐형해: G = (ε_x²·h/2)·(E_LAM − E*). **박리 길이에 무관**하므로 임계 변형률이
    닫힌 형태로 나온다 — ε_c = √(2·G_c/(h·ΔE)).
    loads = {"N":[Nx,Ny,Nxy], "M":[...]} (단위 폭당). 축인장이 전제라 M 이 있으면 W130.
    fracture(선택) = {"G_c": 층간 파괴인성} (SI: J/m², SI_mm: N/mm) — 주면 계면별 개시
    변형률과 여유율까지, 없으면 G 순위만.

    반환: 계면별 G·ΔE·개시변형률·여유율과 **dominant_driver**
    (peel=σz 개구 | transverse_shear=τyz | in_plane_shear=τxz | none).
    [±45] 계열은 σy 가 0이라 전단 지배, [0/90] 계열은 peel 지배로 갈린다.
    한계: 총 G 만 주고 G_I/G_II 혼합모드 분리를 하지 않는다(W130). 구동력은 경계층 평형에
    근거한 크기 규모 지표이지 응력장이 아니다 — 실제는 특이점을 갖는 3D 문제다.
    """
    return _guarded(PIPE.run_free_edge_delamination, laminate, loads=loads, fracture=fracture)


@mcp.tool()
def derive_lamina_from_constituents(fiber: dict, matrix: dict, fiber_volume_fraction: float,
                                    model: str = "halpin_tsai",
                                    xi_E2: float | None = None,
                                    xi_G12: float | None = None) -> dict:
    """섬유+수지 → lamina 직교이방 물성. ply 물성을 모를 때 체인의 시작점.

    homogenize_layer 는 등방 병렬 혼합이라 동박층 같은 혼합층 전용이다 — 이 도구가
    섬유/수지에서 E1·E2·G12·ν12(+α1·α2·ρ)를 만든다. materialtwin 에서 수지 실측을
    가져와 여기에 넣으면 적층 해석까지 체인이 이어진다.
    fiber = {E1, E2, G12, nu12, alpha1?, alpha2?, rho?} (횡등방 섬유),
    matrix = {E, nu, alpha?, rho?} (등방 수지), fiber_volume_fraction ∈ (0,1].
    model: "halpin_tsai"(기본) | "chamis". xi_E2(기본 2)·xi_G12(기본 1)로 보정 가능.
    **단위계 무관** — 출력 단위는 입력과 동일하다.
    반환 material 을 laminae[].material 에 그대로 넣어 쓴다(source=estimated 로 추적됨).
    주의: E1·ν12는 섬유 지배라 신뢰도가 높지만 **E2·G12는 기지 지배라 불확실성이 크다**
    (W120). 함께 반환되는 Reuss–Voigt bounds 가 추정의 폭이니 사용자에게 함께 전할 것.
    """
    try:
        return PIPE.run_micromechanics(fiber, matrix, fiber_volume_fraction, model=model,
                                       xi_E2=xi_E2, xi_G12=xi_G12)
    except Exception as e:  # noqa: BLE001
        return ENV.build(data=None, errors=[item("E501", detail=type(e).__name__)],
                         warnings=[], payload={"fiber": fiber if isinstance(fiber, dict) else {}})


@mcp.tool()
def compute_moisture_uptake(laminate: dict, diffusion: dict, time_s: float | None = None,
                            mode: str = "absorption") -> dict:
    """흡습에 **시간 축**을 준다 — "며칠이면 포화하나", "베이크 몇 시간" 에 답한다.

    compute_thermal_response 는 흡습 변형(delta_C)만 다루고 시간이 없다. 이 도구가
    Fickian 확산으로 시간↔수분율을 이어 준다 — 결과의 delta_C 를 그대로 넘기면 된다.
    diffusion = {"D": 확산계수(길이²/s — SI: m²/s, SI_mm: mm²/s), "M_inf": 포화 수분율[%M]}
    또는 Arrhenius로 {"D0","Ed"(J/mol),"temperature_K","M_inf"}.
    time_s 를 주면 그 시점의 수분율과 두께방향 분포까지, 없으면 특성시간(t50/t90/t99)만.
    mode: "absorption"(기본) | "desorption"(베이크 — M_inf 를 초기 수분율로 보고 남은 양).
    **τ = D·t/h² 하나가 지배한다 — 두께가 2배면 시간이 4배다.**
    주의: Fickian 단순 확산 가정이고 D·M_inf 는 실측이어야 한다(W120/W130).
    """
    return _guarded(PIPE.run_moisture_uptake, laminate, diffusion=diffusion,
                    time_s=time_s, mode=mode)


@mcp.tool()
def assess_partial_composite_bending(laminate: dict, span: float,
                                     core_ply: int | None = None) -> dict:
    """순응 중간층(OCA·PSA)이 끼면 CLT 굽힘강성이 과대평가된다 — 그 배수를 준다.

    CLT 는 모든 층이 **완전합성**으로 함께 굽는다고 가정한다(평면 유지). 폴더블 스택처럼
    무른 중간층이 있으면 두 면재가 서로 미끄러져 실제 굽힘강성이 훨씬 낮다.
    Newmark 부분상호작용 정해. 계면 하나면 α² = (G_c/t_c)(1/EA₁+1/EA₂+d²/EI₀),
    f = 1 − 2(1−sech(αL/2))/(αL/2)², **EI_eff = EI₀/(1 − r·f)** (조화형이지 선형결합이 아니다).
    순응층이 여러 개면 계면도 그만큼 두고 N층으로 함께 푼다.
    **span(굽힘 스팬)이 필수다** — 합성도가 스팬에 강하게 의존한다
    (실측 UTG/OCA/UTG, OCA G=0.3 MPa: L=1mm 에서 CLT 22.1배 과대, L=10mm 10.09배, L=200mm 1.03배).
    core_ply 생략 시 이웃보다 10배 이상 무른 중간층을 자동 탐지한다. 코어를 여러 ply 로
    쪼개 모델링해도(중앙 절점 배치 등) 하나로 묶어 같은 답을 낸다.
    반환: composite_action f(0=각자 굼, 1=CLT와 동일), EI_layered/EI_full_CLT/EI_effective,
    **clt_overprediction**(CLT ÷ 실제 — 1보다 크면 처짐·좌굴·진동수가 모두 낙관적).
    순응층의 G13 이 없으면 G12 로 대체하고 W120 으로 알린다(α ∝ √G_c 라 민감하다).
    """
    return _guarded(PIPE.run_partial_composite, laminate, span=span, core_ply=core_ply)


@mcp.tool()
def solve_prescribed_curvature(laminate: dict, kappa: list | None = None,
                               bend_radius: float | None = None, bend_axis: str = "x",
                               width: str = "free", epsilon0: list | None = None,
                               loads: dict | None = None) -> dict:
    """**변위 제어** — 곡률·굽힘반경을 지정한다. 폴더블·롤투롤·맨드릴 굽힘의 실제 구속이다.

    다른 도구는 전부 힘 제어(N/M)라 에이전트가 M = D·κ 지름길을 쓰게 되는데, 비대칭
    스택에서 실측 **+244.8% 과대**다. 이 도구는 K[ε⁰;κ]=[N;M] 를 지정/미지 자유도로
    분할해 정확히 푼다.
    bend_radius(굽힘반경, 길이 단위) + bend_axis("x"|"y") 또는 kappa=[κx,κy,κxy]
    (자유 성분은 null). epsilon0 로 면내변형도 지정할 수 있다. loads 는 자유 자유도의 힘.
    **width 가 중요하다**: "free"(M_y=0, 자유 폭 — 기본) vs "constrained"(κ_y=0, 구속 폭).
    두 경우 답이 다르다(실측 9.6% 차이) — 실제 지그가 어느 쪽인지 확인할 것.
    반환: 완전한 ε⁰·κ, **equivalent_loads**(recover_ply_stresses 에 그대로 넘겨 파손 판정까지
    이어진다), surface_strain(assess_crack_shielding 의 applied_strain 에 넣을 값 — 손으로
    (z−z_ns)/R 을 계산하지 말 것), ply별 재료축 변형률, 그리고 지름길 M=D·κ 와의 대조.
    """
    return _guarded(PIPE.run_prescribed_curvature, laminate, kappa=kappa,
                    bend_radius=bend_radius, bend_axis=bend_axis, width=width,
                    epsilon0=epsilon0, loads=loads)


@mcp.tool()
def compute_failure_envelope(laminate: dict, plane: str = "Nx-Ny",
                             magnitude: float | None = None,
                             delta_T: float | None = None) -> dict:
    """하중 **방향**에 대한 파손 포락선 — 어느 방향이 가장 약한지 결정론적으로 준다.

    지금까지 최소점을 찾는 유일한 방법이 비결정론적 반복 호출이었다. 고정 **72방향**
    격자(5도 간격)에서 방향당 Tsai-Wu 강도비를 한 번씩 계산한다.
    plane: "Nx-Ny"(기본) | "Nx-Nxy" | "Mx-My". delta_T 를 주면 열잔류를 포함해
    R 을 2차식으로 푼다(§19.10 — 잔류는 하중 배수와 함께 커지지 않는다).
    반환: 방향별 failure_load·임계 ply·지배 모드, **weakest**(가장 약한 방향),
    strongest, anisotropy_ratio(최강/최약 비 — 방향 민감도).
    first-ply-failure 기준이다. 한계하중은 run_progressive_failure 로.
    """
    return _guarded(PIPE.run_failure_envelope, laminate, plane=plane, magnitude=magnitude,
                    delta_t=delta_T)


@mcp.tool()
def solve_required_thickness_scale(laminate: dict, panel: dict, applied_Nx: float,
                                   target_margin: float = 1.0, load_ratio: float = 0.0,
                                   boundary: str = "simply_supported") -> dict:
    """목표 좌굴 여유를 만족하는 **최소 두께 배율**을 폐형해로 역산한다.

    전 ply 를 균일 배율 s 로 키우면 D ∝ s³ 이므로 **N_cr ∝ s³** 이다(실측 전 자리 일치).
    따라서 s = (target/current)^(1/3) 가 닫힌 형태로 나온다 — 에이전트가 compute_buckling 을
    이분법으로 5~10회 부르던 것을 1회로 바꾼다.
    applied_Nx(압축 크기), target_margin(목표 N_cr/N, 기본 1.0), boundary 는 compute_buckling 과 동일.
    반환: required_scale, 배율 적용 후 총두께·ply 두께, N_cr 전후.
    **균일 배율만 유효하다** — 일부 ply 만 두껍게 하거나 적층 순서를 바꾸면 지수 법칙이 깨진다(W130).
    """
    return _guarded(PIPE.run_required_scale, laminate, panel=panel, applied_Nx=applied_Nx,
                    target_margin=target_margin, load_ratio=load_ratio, boundary=boundary)


@mcp.tool()
def assess_sandwich_local_failure(laminate: dict, core: dict, applied: dict | None = None,
                                  shear: dict | None = None,
                                  core_ply: int | None = None) -> dict:
    """샌드위치 국부 파손 — 면재 강도가 통과해도 여기서 먼저 걸린다.

    compute_natural_frequencies·compute_buckling 이 R_s>0.02 로 "샌드위치는 CLT 밖"이라고
    경고해 왔지만 지목할 도구가 없었다. 세 모드를 대수식으로 판정한다:
    ① 면재 주름 Hoff–Mautner σ_wr = k(E_f·E_z·G_c)^(1/3) — **k 가 문헌마다 0.5~0.825 로
    갈려 1.65배 차이난다. 단일값을 내지 않고 범위로 준다**(보수값으로 판정할 것).
    ② 셀 내 딤플링 σ_d = 2E_f/(1−ν²)(t_f/s)² (cell_size 주면).
    ③ 코어 전단 τ_c = V/d (shear={"Vx"} 주면).
    core = {"Ez": 두께방향 압축탄성계수, "Gc": 코어 전단탄성계수, "cell_size"?, "shear_strength"?}.
    **허니콤은 면내 E 와 두께방향 E_z 가 크게 달라 ply 물성으로 대체할 수 없다** — 따로 받는다.
    applied = {"N","M"} 을 주면 면재 최대 압축응력 대비 여유와 **governing 모드**를 준다.
    실측: CFRP 1mm 면재 + 노멕스 코어에서 주름 376 MPa 가 재료 압축강도 600 MPa 보다 먼저 온다.
    """
    return _guarded(PIPE.run_sandwich_local, laminate, core=core, applied=applied,
                    shear=shear, core_ply=core_ply)


@mcp.tool()
def estimate_spectrum_life(laminate: dict, blocks: list, delta_T: float | None = None) -> dict:
    """변진폭 하중 스펙트럼 — Miner 누적손상. 손으로 Σn/N 을 합산하지 말 것.

    estimate_fatigue_life 는 등진폭 단일 블록만 받는다. 블록마다 지배 성분이 다르면
    (열사이클 σ2, 진동 τ12) 손합산은 ply·성분 정합을 놓쳐 틀린다.
    blocks = [{"loads_max": {"N":…,"M":…}, "loads_min"?: …, "cycles": >0, "name"?: …}, …].
    delta_T 를 주면 모든 블록에 경화 냉각 잔류를 반영한다(§19.10 — 평균응력 이동).
    반환: 블록별 손상·임계 ply·지배 성분, **total_damage**(D=Σn/N),
    **repeats_to_failure**(1/D — 이 스펙트럼을 몇 번 반복하면 파손하는가), ply별 손상 분해.
    한계: Miner 는 하중 **순서를 무시**한다 — 자릿수 판단용이다(W130).
    """
    return _guarded(PIPE.run_load_spectrum, laminate, blocks=blocks, delta_t=delta_T)


@mcp.tool()
def homogenize_patterned_layer(components: list[dict]) -> dict:
    """방향성 패턴층(동박 트레이스 등) → **직교이방** 물성. 패턴 방향을 살린다.

    homogenize_layer 는 등방 Voigt 라 방향이 사라진다 — 동박 패턴 방향만 다른 스택업
    A/B 에 서버가 **바이트 동일한 답**을 주고 있었다(실측). PCB 의 실린더/새들 형상은
    바로 그 방향 비대칭이 만드는 것이라 원리적으로 나올 수 없었다.
    평행 스트라이프 가정: E1=Σf·E(스트라이프 방향, 등변형·정확), E2=1/Σ(f/E)(가로, 등응력),
    α1 은 강성 가중, α2 는 Schapery. 동박 60% 실측으로 E1/E2 = 8.5배, α2/α1 = 2.24배.
    components 형식은 homogenize_layer 와 같다: [{"material": isotropic, "volume_fraction": f}].
    **반환 material 을 laminae 에 넣고 angle_deg 를 트레이스 방향으로 설정**하라 —
    그것이 패턴 방향이 휨 형상에 반영되는 경로다.
    등방 극한(전 상 동일 물성)에서는 homogenize_layer 와 정확히 일치한다.
    단위계 무관 — 출력 단위는 입력과 동일.
    """
    try:
        return PIPE.run_patterned_layer(components)
    except Exception as e:  # noqa: BLE001
        return ENV.build(data=None, errors=[item("E501", detail=type(e).__name__)],
                         warnings=[], payload={"components": components if isinstance(components, list) else []})


@mcp.tool()
def assess_bonded_lap_joint(laminate: dict, adhesive: dict, overlap: float, load: float,
                            laminate_2: dict | None = None,
                            joint_type: str = "double_lap") -> dict:
    """접착 겹치기 이음 — "겹침을 늘리면 강해진다"는 직관을 반박한다.

    Volkersen 전단지연: ω²=(G_a/t_a)(1/EA₁+1/EA₂), τ_peak/τ_avg=(ωL/2)coth(ωL/2).
    **ωL/2 ≫ 1 이면 τ_peak 가 겹침 길이에 무관해진다** — 실측(Al 2mm, G_a=1 GPa, t_a=0.2mm)으로
    L=25→50 mm 에서 평균은 절반이 되는데 **피크는 1.003배만 준다**.
    adhesive = {"G", "thickness", "shear_strength"?}. overlap·load 는 단위 폭당.
    laminate_2 생략 시 같은 피착재로 본다.
    **joint_type**: "double_lap"(기본, 편심 없음 — Volkersen 유효) |
    "single_lap"(편심 있음 — 굽힘 peel 이 보통 먼저 파손시키는데 Volkersen 은 peel 을
    모델링하지 않아 **비보수**다. W130 으로 강하게 알린다).
    반환: τ 분포, peak_over_average, **overlap_efficiency**(겹침 중 실효 비율),
    **saturation_overlap**(이걸 넘으면 더 늘려도 소용없는 길이), 여유율.
    """
    return _guarded(PIPE.run_lap_joint, laminate, adhesive=adhesive, overlap=overlap,
                    load=load, laminate_2=laminate_2, joint_type=joint_type)


@mcp.tool()
def assess_notched_strength(laminate: dict, hole_diameter: float,
                            unnotched_strength: float | None = None,
                            d0: float | None = None, a0: float | None = None) -> dict:
    """원공 노치 — **확정값 K_T** 와 **조건부값 σ_OH** 를 분리해서 준다.

    K_T∞ = 1 + √(2(√(Ex/Ey) − νxy) + Ex/Gxy) 는 **순수 폐형해**다(등방에서 정확히 3).
    적층에 따라 크게 갈린다 — 준등방 3.00, [0/90]s 4.92, UD 6.75.
    노치 강도는 Whitney–Nuismer 로 계산하는데 **d0(점응력)·a0(평균응력)는 재료·적층별
    시험 피팅 상수**이고 답이 여기에 강하게 민감하다 — 실측 [±45/0/90]s φ6.35 에서
    d0 = 0.5~2.0 mm 만 바꿔도 σ_OH 가 234~368 MPa 로 갈린다.
    **그래서 d0/a0 를 주지 않으면 σ_OH 를 아예 내지 않는다**(의도적 — 임의 기본값을 쓰면
    그럴듯한 오답이 된다). unnotched_strength 와 함께 주면 계산하되 W130 으로 종속을 명시한다.
    무한 판 가정이라 구멍이 판 폭의 1/5 를 넘으면 유한 폭 보정이 필요하다.
    """
    return _guarded(PIPE.run_notched_strength, laminate, hole_diameter=hole_diameter,
                    unnotched_strength=unnotched_strength, d0=d0, a0=a0)


@mcp.tool()
def solve_stress_relaxation(laminate: dict, times_s: list,
                            bend_radius: float | None = None, kappa: list | None = None,
                            bend_axis: str = "x", width: str = "free",
                            epsilon0: list | None = None, span: float | None = None) -> dict:
    """**곡률을 고정한 채** 점탄성층이 이완할 때 M(t)·ply 응력이 떨어지는 것을 본다.

    점탄성만 단독으로 보면 값이 작다 — 얇고 무른 층은 ABD 기여가 작아 자유 경계 곡률이
    몇 % 만 변한다. **진짜 질문은 변위를 고정했을 때다**: 폴더블을 접어 둔 채 시간이 지나면
    필요 모멘트와 ply 응력이 얼마나 떨어지는가 — 크리스(crease) 잔류의 실제 물리이고
    변위 제어(solve_prescribed_curvature)와 짝지어야만 물을 수 있다.
    준탄성: E(t) = E∞ + (E0 − E∞)exp(−t/τ). ply 에 material.viscoelastic {E0, Einf, tau_s} 필요.
    bend_radius(또는 kappa)로 곡률을 지정하고 times_s 에 시점들을 준다(최대 20개).
    width 는 §19.14 와 같은 규약("free"=M_y 0, "constrained"=κ_y 0).
    **span 을 주면 부분합성 이완도 함께 낸다 — 폴더블은 이쪽이 지배적이다.** 순응층의 진짜
    역할은 굽힘강성 기여가 아니라 **전단 결합**이라 CLT ABD 로는 이완 효과가 거의 안 보인다
    (실측 M 이완 0.0%). 같은 스택을 부분합성으로 보면 합성도가 62%→7%, 유효 굽힘강성이
    **0.66배**로 무너진다.
    반환: 시점별 M·N·최대 ply 응력·부분합성, **M_ratio·stress_ratio·EI_effective_ratio**.
    한계: 준탄성 근사(유전적분 없음)·단일 지수 완화 — 경향 판단용이다(W130).
    """
    return _guarded(PIPE.run_stress_relaxation, laminate, times_s=times_s, kappa=kappa,
                    bend_radius=bend_radius, bend_axis=bend_axis, width=width,
                    epsilon0=epsilon0, span=span)


@mcp.tool()
def get_reference_cases(case_id: str | None = None) -> dict:
    """내장 기준 케이스를 반환한다. case_id 생략 시 목록.

    각 케이스는 완전한 입력 payload(input.laminate)와 폐형해 기대값(expected)을 포함하므로,
    입력 스키마의 실전 예시(few-shot)이자 서버 자가 검증 수단으로 쓸 수 있다.
    """
    return REF.get(case_id)


@mcp.tool()
def get_server_info() -> dict:
    """서버·엔진 버전, 지원 단위계, 한계값, Tool·오류코드 목록을 반환한다."""
    return {
        "server": config.SERVER_NAME,
        "server_version": config.SERVER_VERSION,
        "engine_version": config.ENGINE_VERSION,
        "unit_systems": {k: units.UNIT_LABELS[k] for k in units.SUPPORTED_UNIT_SYSTEMS},
        "limits": {"max_plies": config.MAX_PLIES, "max_payload_bytes": config.MAX_PAYLOAD_BYTES,
                   "max_batch": config.MAX_BATCH},
        "conventions": {
            "stacking": "laminae[0] = 최하단(bottom) ply",
            "angle": "deg, 적층판 +x → 섬유 1축, +z 기준 CCW 양수",
            "loads": "N·M은 단위 폭당 물리량",
            "internal_units": "SI로 정규화 후 계산, 응답은 입력 단위계로 표시",
        },
        "error_codes": {code: name for code, (name, _, _) in CATALOG.items()},
        "determinism": "동일 payload → 바이트 동일 응답 (include_debug=true의 debug 블록 제외)",
    }


@mcp.resource("laminate://guide")
def guide() -> str:
    """LaminateAnalyzer 사용 가이드 — 규약·워크플로·materialtwin 연계."""
    return """# LaminateAnalyzer MCP 가이드

적층 복합재의 중립면·ABD·평가 지표를 결정론적으로 계산한다. LLM은 계산하지 않고 이 서버를 호출한다.

## 규약 (동결)
- laminae[0] = 최하단(bottom) ply, z축은 두께 방향 위쪽.
- 적층각 deg: 적층판 +x축 → 섬유 1축, +z 기준 반시계(CCW) 양수. (-360,360) 입력은 자동 정규화.
- unit_system 필수: "SI"(Pa·m·kg/m³) 또는 "SI_mm"(MPa·mm·t/mm³). 생략하면 E101.
- A[N/m|N/mm], B[N], D[N·m|N·mm]. 정규화 Â=A/h, B̂=√12B/h², D̂=12D/h³ (Pa|MPa).

## 전형적 워크플로
1) 처음이면: get_reference_cases()로 입력 예시 확인 → 그대로 analyze_laminate에 넣어 자가 검증.
2) 분석: analyze_laminate(laminate) 원샷 — 요약·ABD·중립면·지표·권고 반환.
3) 기준 판정: evaluate_laminate(laminate, criteria={"max_coupling_ratio": 0.05, ...}).
4) 단계별: validate_laminate_input → compute_abd_matrix → compute_neutral_axis.
5) 하중 응답: solve_load_response(laminate, loads={"N":[Nx,Ny,Nxy],"M":[Mx,My,Mxy]})
   — ε0/κ, 유효 공학 상수, 누출/비틀림 지표, 주강성 방향.
6) 설계 탐색 가속: 후보 여러 개를 batch_evaluate_laminates로 일괄 평가 → 유망 후보만 상세 분석.
7) 공차 강건성: run_sensitivity_analysis. 최종 보고: generate_design_report(ko|en).
8) 열 휨(PCB 등): 동박층은 homogenize_layer(동박률)로 등가 물성 →
   compute_thermal_response(laminate, delta_T, panel={"Lx","Ly"}) — 유효 CTE·열곡률·휨·잔류응력.
9) 크랙 차폐: assess_crack_shielding(laminate, target_ply, fracture) — 발생 문턱(σ_c),
   Dundurs 차폐 경향, 계면 편향(1/4 법칙), 보호층 점탄성 이완의 차폐 저하.
10) 층별 응력·파손: recover_ply_stresses(laminate, loads, delta_T?) — ply별 σ1/σ2/τ12와
    (strength 있으면) Tsai-Wu 여유율 R·지배 모드·first_ply_failure. 강도는 materialtwin에서.
11) 설계규칙: check_design_rules — 대칭/밸런스/10%/인접각/연속/외층 관례 판정+이유+수정힌트.
12) 좌굴/진동: compute_buckling·compute_natural_frequencies(panel 필수, SS Navier 폐형해).
13) 한계하중: run_progressive_failure — FPF 이후 ply discount로 ultimate_R까지.
14) 흡습: compute_thermal_response에 delta_C [%M] (재료 beta 필요) — 열과 동일 기계.
15) 층간·박리: compute_interlaminar_stresses(laminate, shear={"Vx","Vy"}) — τxz 분포·ILSS 여유.
16) 감쇠·두께 한계: compute_natural_frequencies가 loss_factor 있으면 모달 η·Q,
    G13/G23로 횡전단 유연성 R_s를 계산해 CLT가 부정확해지는 지점을 알려준다.
17) 피로: estimate_fatigue_life(laminate, loads_max, loads_min?) — Goodman+S-N 반복 수명.
18) 기하 비선형(V3): 선형 CLT가 깨지는 영역을 다룬다.
    - compute_bistable_shapes(laminate, panel, delta_T) — 비대칭 적층의 실제 경화 형상.
      선형 해가 주는 안장은 판이 임계 크기를 넘으면 실현되지 않고 원통 두 개로 분기한다.
    - compute_large_deflection(laminate, panel, pressure, edge_condition) — w>h 영역의 처짐.
    - compute_postbuckling(laminate, panel, applied_Nx) — N>N_cr 이후 진폭·강성비·유효폭.
    판단 기준: compute_thermal_response의 warpage.w_over_thickness 또는 대처짐의 w/h가
    0.3을 넘으면 선형 결과를 그대로 쓰지 말고 위 도구로 재확인할 것(W130으로도 알린다).
    - solve_nonlinear_shear_response(laminate, loads) — 재료 자체의 전단 비선형(Hahn–Tsai).
19) 자유 가장자리 박리: assess_free_edge_delamination(laminate, loads, fracture?) —
    면내 강도가 통과해도 가장자리에서 먼저 뜯기는지 본다. O'Brien G 와 계면별 지배 구동력
    (peel/transverse_shear/in_plane_shear). recover_ply_stresses 로 여유가 나왔더라도
    각도 차가 큰 계면이 있으면 이 도구를 반드시 함께 볼 것.
      위 셋은 기하 비선형(형상이 커서 생기는 것)이고 이것은 재료 비선형이다. 전단 지배
      적층([±45] 계열)에서는 이쪽이 훨씬 크게 작용한다. ply에 shear_nonlinear가 있는데
      solve_load_response를 쓰면 W130으로 알린다.

## materialtwin 연계 (물성을 모를 때)
- materialtwin MCP의 list_materials/search_by_property → get_material에서 실측 E를 얻는다.
  materialtwin의 E_GPa는 GPa이므로 SI_mm(MPa)로 넘길 때 ×1000 변환할 것.
- 단축 인장은 E만 준다: nu는 가정값(금속 ~0.3)을 쓰고 material.source={"type":"assumed"}로
  표기하면 W120 경고와 assumptions로 추적된다.

## REST로도 같은 계산이 가능하다
MCP 클라이언트가 없는 소비자(스크립트·서비스)는 GET /api/v1/tools(목록+스키마),
POST /api/v1/tools/{name}(실행), GET /api/v1/guide(이 문서)를 쓰면 된다 — 결과는 동일하다.

## 주의
- 오류 응답의 suggestion은 그대로 따라 수정→재호출 가능하게 작성돼 있다.
- warnings(특히 W110 단위 의심)와 assumptions를 사용자 보고에 반드시 포함할 것.
- 대칭 적층이면 B≈0·중립면=midplane이 정상이다. coupling_ratio ≥ 0.2는 W200 경고.
"""
