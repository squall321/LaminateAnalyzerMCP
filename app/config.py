# 서버 버전·한계값·허용오차 상수의 단일 소스 (계획서 §11, §3 D12)

SERVER_NAME = "laminate-analyzer"
SERVER_VERSION = "0.8.0"
ENGINE_VERSION = "0.8.0"

MAX_PLIES = 512
MAX_PAYLOAD_BYTES = 1 * 1024 * 1024  # 1 MiB
MAX_BATCH = 32
COMPUTE_TIMEOUT_S = 10.0  # 반복형 계산(batch/sensitivity/스캔)의 시간 예산 → 초과 시 E500
# ply별 상세 응답의 자동 요약 기준 (§6.6 토큰 예산): 초과 시 임계 상위 N개만 + note
SUMMARY_PLY_LIMIT = 32
SUMMARY_TOP_N = 10

# 경고 휴리스틱 임계값 (계획서 §6.4 W110~W112, §5.4)
COND_WARN_THRESHOLD = 1.0e8       # W401
THIN_PLY_RATIO = 1.0e-6           # W111: t_k/h
EXTREME_MODULUS_RATIO = 1.0e4     # W112: E1/E2
HIGH_COUPLING_RATIO = 0.2         # W200
DENOM_EPS = 1.0e-30               # W402: 지표 분모 소멸 판정 (SI 기준)

# W110 단위 오입력 휴리스틱 (계획서 §6.4)
SI_MODULUS_SUSPECT_LOW = 1.0      # SI에서 1 <= E < 1e6 Pa 면 의심
SI_MODULUS_SUSPECT_HIGH = 1.0e6
SI_THICKNESS_SUSPECT = 1.0        # SI에서 두께 > 1 m 면 의심
SI_MM_MODULUS_SUSPECT = 1.0e7     # SI_mm에서 E > 1e7 MPa 면 의심
