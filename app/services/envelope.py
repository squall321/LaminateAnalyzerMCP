# 공통 응답 envelope — canonical JSON, payload_hash, 결정론 보장 (계획서 §6.4, §6.5)
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone

from app import config, units

BASE_ASSUMPTIONS = [
    "고전 적층판 이론(CLT, Kirchhoff-Love): 횡전단 변형 무시 — 두꺼운 판·샌드위치 코어에는 부정확",
    "평면응력 선형탄성, 완전 접착(층간 슬립 없음), 층 두께 균일",
    "열·흡습 잔류변형 미포함 (V2 예정)",
    "N·M은 단위 폭당 물리량 (SI: N/m, N·m/m)",
]


def canonical_json(obj) -> str:
    """정렬 키·고정 구분자·전정밀도 float repr의 결정론적 직렬화 (D8, D9)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def payload_hash(payload) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def contains_nan_inf(obj) -> bool:
    """응답 데이터에 NaN/Inf가 섞였는지 재귀 검사 (E403 방어선)."""
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(contains_nan_inf(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_nan_inf(v) for v in obj)
    return False


def build(*, data, errors, warnings, payload, unit_system=None, method="clt",
          assumptions_extra=(), include_debug=False, t0=None) -> dict:
    """공통 envelope 조립. include_debug=True일 때만 비결정 요소(debug)가 포함된다 (D8)."""
    status = "error" if errors else ("warning" if warnings else "ok")
    try:
        ph = payload_hash(payload)
    except (TypeError, ValueError):
        ph = None
    env = {
        "status": status,
        "data": None if errors else data,
        "errors": errors,
        "warnings": warnings,
        "assumptions": [*BASE_ASSUMPTIONS, *assumptions_extra],
        "metadata": {
            "server_version": config.SERVER_VERSION,
            "engine_version": config.ENGINE_VERSION,
            "unit_system_in": unit_system,
            "units": units.UNIT_LABELS.get(unit_system),
            "method": method,
            "payload_hash": ph,
        },
    }
    if include_debug:
        env["debug"] = {
            "elapsed_ms": None if t0 is None else round((time.perf_counter() - t0) * 1000.0, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return env
