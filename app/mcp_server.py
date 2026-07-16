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

## materialtwin 연계 (물성을 모를 때)
- materialtwin MCP의 list_materials/search_by_property → get_material에서 실측 E를 얻는다.
  materialtwin의 E_GPa는 GPa이므로 SI_mm(MPa)로 넘길 때 ×1000 변환할 것.
- 단축 인장은 E만 준다: nu는 가정값(금속 ~0.3)을 쓰고 material.source={"type":"assumed"}로
  표기하면 W120 경고와 assumptions로 추적된다.

## 주의
- 오류 응답의 suggestion은 그대로 따라 수정→재호출 가능하게 작성돼 있다.
- warnings(특히 W110 단위 의심)와 assumptions를 사용자 보고에 반드시 포함할 것.
- 대칭 적층이면 B≈0·중립면=midplane이 정상이다. coupling_ratio ≥ 0.2는 W200 경고.
"""
