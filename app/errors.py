# 오류·경고 코드 카탈로그의 단일 소스 — 메시지·suggestion은 한국어 관례 (계획서 §6.4, Q8: materialtwin 정합)
from __future__ import annotations

# code -> (name, message_template, suggestion_template)
CATALOG: dict[str, tuple[str, str, str]] = {
    # 입력 구조 (E1xx)
    "E100": ("SCHEMA_INVALID",
             "입력 스키마가 유효하지 않습니다: {detail}",
             "필드 타입과 필수 항목을 확인해 다시 호출하세요. 완전한 입력 예시는 get_reference_cases로 확인할 수 있습니다."),
    "E101": ("UNKNOWN_UNIT_SYSTEM",
             "unit_system이 없거나 지원하지 않는 값입니다: {detail}",
             "unit_system을 'SI'(Pa·m·kg/m³) 또는 'SI_mm'(MPa·mm·t/mm³) 중 하나로 명시해 다시 호출하세요."),
    "E102": ("EMPTY_LAMINATE",
             "laminae가 비어 있습니다.",
             "최소 1개 ply를 포함해 다시 호출하세요. laminae[0]이 최하단 ply입니다."),
    "E103": ("TOO_MANY_PLIES",
             "ply 수({n})가 최대 {max}를 초과했습니다.",
             "적층을 {max}개 이하로 줄이거나 반복 구간을 나눠 호출하세요."),
    "E104": ("PAYLOAD_TOO_LARGE",
             "payload 크기({size} bytes)가 한계({max} bytes)를 초과했습니다.",
             "불필요한 필드를 제거하거나 요청을 분할해 다시 호출하세요."),
    # 재료 (E2xx)
    "E200": ("NON_POSITIVE_MODULUS",
             "탄성계수는 양수여야 합니다: {detail}",
             "{field}를 양수로 수정해 다시 호출하세요."),
    "E201": ("POISSON_UNSTABLE",
             "포아송비가 열역학적 안정 조건을 위반합니다: {detail}",
             "직교이방성은 |nu12| < sqrt(E1/E2), 등방성은 -1 < nu < 0.5 조건을 만족해야 합니다."),
    "E202": ("INVALID_MATERIAL_TYPE",
             "material.type이 유효하지 않습니다: {detail}",
             "material.type을 'isotropic'(E, nu) 또는 'orthotropic_2d'(E1, E2, G12, nu12)로 지정하세요."),
    "E203": ("MISSING_THERMAL_PROPERTY",
             "열해석에 필요한 CTE가 없는 ply가 있습니다: {detail}",
             "해당 ply의 material에 alpha(등방) 또는 alpha1/alpha2(직교이방) [1/K]를 넣어 다시 호출하세요. "
             "실측이 없으면 문헌 대표값 + source={{\"type\":\"assumed\"}}로 표기하세요."),
    # 기하 (E3xx)
    "E300": ("NON_POSITIVE_THICKNESS",
             "ply 두께는 양수여야 합니다: {detail}",
             "{field}를 양수로 수정해 다시 호출하세요."),
    "E301": ("ANGLE_OUT_OF_RANGE",
             "적층각이 허용 범위 (-360, 360)을 벗어났습니다: {detail}",
             "각도를 -90~90도 권장 범위로 바꿔 다시 호출하세요. (-360, 360) 입력은 자동 정규화됩니다."),
    # 수치 (E4xx)
    "E400": ("SINGULAR_SYSTEM",
             "강성 행렬 역산이 불가능합니다(특이 행렬).",
             "극단적 물성비나 0에 가까운 두께가 없는지 적층 구성을 재검토하세요."),
    "E402": ("NOT_POSITIVE_DEFINITE",
             "정규화 강성 행렬 K̂이 양정치가 아닙니다 — 물리적으로 불가능한 강성입니다.",
             "물성 입력 오류(단위 혼동, 부호 오류) 가능성을 확인하세요."),
    "E403": ("NAN_INF_DETECTED",
             "계산 결과에 NaN/Inf가 감지되었습니다(내부 방어선).",
             "payload_hash와 함께 입력을 관리자에게 보고하세요."),
    # 서버 (E5xx)
    "E500": ("COMPUTE_TIMEOUT",
             "계산이 제한 시간을 초과했습니다.",
             "ply 수를 줄이거나 요청을 분할해 다시 호출하세요."),
    "E501": ("INTERNAL_ERROR",
             "내부 오류가 발생했습니다: {detail}",
             "동일 입력으로 재시도 후에도 반복되면 관리자에게 보고하세요."),
    # 경고 (Wxxx)
    "W110": ("SUSPICIOUS_UNIT_MAGNITUDE",
             "값의 자릿수가 단위계와 어긋나 보입니다: {detail}",
             "unit_system과 값의 단위를 확인하세요. 예) SI(Pa)에서 E=200은 200e9(GPa 착오) 가능성이 큽니다."),
    "W111": ("VERY_THIN_PLY",
             "총 두께 대비 매우 얇은 ply가 있습니다: {detail}",
             "두께 입력 실수가 아닌지 확인하세요."),
    "W112": ("EXTREME_MODULUS_RATIO",
             "극단적인 물성비가 감지되었습니다: {detail}",
             "E1/E2 비가 물리적으로 타당한지 확인하세요."),
    "W130": ("MODEL_APPLICABILITY",
             "모델 적용 가정에서 벗어난 조건입니다: {detail}",
             "결과를 경향 판단용으로 사용하고, 정밀 판정이 필요하면 상세 해석(FE)을 병행하세요."),
    "W120": ("ASSUMED_CONSTANT",
             "실측 없이 가정된 상수가 포함되어 있습니다: {detail}",
             "결과 해석 시 해당 상수의 불확실성을 감안하세요."),
    "W200": ("HIGH_COUPLING",
             "막-굽힘 커플링이 큽니다: {detail}",
             "경화 후 뒤틀림·해석 커플링이 우려되면 대칭 적층을 검토하세요."),
    "W401": ("ILL_CONDITIONED",
             "정규화 강성 행렬의 조건수가 큽니다: {detail}",
             "수치 민감성이 있으니 결과 자릿수를 과신하지 마세요."),
    "W402": ("INDEX_UNDEFINED",
             "분모 소멸로 일부 지표를 정의할 수 없어 null로 반환했습니다: {detail}",
             "해당 지표는 이 적층에서 물리적 의미가 약합니다."),
}


def item(code: str, field: str | None = None, **fmt) -> dict:
    """코드 → {code, name, message, field, suggestion} 항목 생성. 템플릿 인자는 fmt로."""
    name, msg_t, sug_t = CATALOG[code]
    fmt.setdefault("field", field or "")
    out = {
        "code": code,
        "name": name,
        "message": msg_t.format(**fmt),
        "suggestion": sug_t.format(**fmt),
    }
    if field:
        out["field"] = field
    return out
