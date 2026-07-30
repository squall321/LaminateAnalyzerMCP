# 입력 검증(구조→물리)과 SI 정규화 — 오류/경고 코드 발급의 관문 (계획서 §6.4, §7.1)
from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from app import config, units
from app.errors import item
from app.schemas import Laminate, SourceInfo
from app.solver import material as MAT


@dataclass
class SiPly:
    thickness: float          # m
    angle_deg: float          # 정규화된 (-90, 90]
    E1: float                 # Pa
    E2: float
    G12: float
    nu12: float
    rho: float | None         # kg/m^3
    name: str | None
    source: SourceInfo | None
    is_isotropic: bool
    alpha1: float | None = None   # 1/K (열해석 선택)
    alpha2: float | None = None
    ve_E0: float | None = None    # Pa (점탄성 보호층, §17.2 선택)
    ve_Einf: float | None = None
    ve_tau_s: float | None = None
    strength: tuple | None = None  # (Xt,Xc,Yt,Yc,S) Pa (파손 판정, §17.4 선택)

    @property
    def has_cte(self) -> bool:
        return self.alpha1 is not None and self.alpha2 is not None


@dataclass
class SiLaminate:
    unit_system: str
    plies: list[SiPly]
    name: str | None
    reference_output: str
    fingerprint: list = field(default_factory=list)  # 대칭성 판정용 (t, angle, 물성) 튜플

    @property
    def thicknesses(self) -> list[float]:
        return [p.thickness for p in self.plies]

    @property
    def total_thickness(self) -> float:
        return float(sum(self.thicknesses))


def _loc_path(loc: tuple) -> str:
    parts = []
    for x in loc:
        parts.append(f"[{x}]" if isinstance(x, int) else ("." + str(x) if parts else str(x)))
    return "".join(parts)


def validate_and_convert(payload) -> tuple[SiLaminate | None, list[dict], list[dict]]:
    """payload(dict) → (SI 정규화 적층 | None, errors[], warnings[]). 오류가 있으면 None."""
    errors: list[dict] = []
    warnings: list[dict] = []

    if not isinstance(payload, dict):
        return None, [item("E100", field="laminate", detail="laminate는 객체(dict)여야 합니다")], warnings

    # 크기 한계 (E104)
    try:
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return None, [item("E100", field="laminate", detail="JSON 직렬화 불가능한 값이 포함되어 있습니다")], warnings
    if size > config.MAX_PAYLOAD_BYTES:
        return None, [item("E104", field="laminate", size=size, max=config.MAX_PAYLOAD_BYTES)], warnings

    # 단위계 (E101) — 필수 입력 (Q1 확정)
    us = payload.get("unit_system")
    if us not in units.SUPPORTED_UNIT_SYSTEMS:
        return None, [item("E101", field="unit_system", detail=repr(us))], warnings

    # ply 개수 (E102/E103)
    laminae = payload.get("laminae")
    if not isinstance(laminae, list) or len(laminae) == 0:
        return None, [item("E102", field="laminae")], warnings
    if len(laminae) > config.MAX_PLIES:
        return None, [item("E103", field="laminae", n=len(laminae), max=config.MAX_PLIES)], warnings

    # 구조 검증 (pydantic) — E100/E202 매핑
    try:
        model = Laminate.model_validate(payload)
    except ValidationError as e:
        for err in e.errors():
            loc = err.get("loc", ())
            path = _loc_path(loc)
            if "material" in loc and ("union_tag_invalid" in err["type"] or "union_tag_not_found" in err["type"]):
                errors.append(item("E202", field=path, detail=err["msg"]))
            else:
                errors.append(item("E100", field=path, detail=f"{path}: {err['msg']}"))
        return None, errors, warnings

    # 물리 검증 + SI 변환
    f = units.TO_SI[us]
    plies: list[SiPly] = []
    fingerprint = []
    for k, lam in enumerate(model.laminae):
        base = f"laminae[{k}]"
        if lam.thickness <= 0:
            errors.append(item("E300", field=f"{base}.thickness", detail=f"{base}.thickness = {lam.thickness}"))
        if abs(lam.angle_deg) >= 360.0:
            errors.append(item("E301", field=f"{base}.angle_deg", detail=f"{base}.angle_deg = {lam.angle_deg}"))

        m = lam.material
        if m.type == "isotropic":
            raw_moduli = {"E": m.E}
            if m.E <= 0:
                errors.append(item("E200", field=f"{base}.material.E", detail=f"E = {m.E}"))
            elif not (-1.0 < m.nu < 0.5):
                errors.append(item("E201", field=f"{base}.material.nu", detail=f"nu = {m.nu} (등방성은 -1 < nu < 0.5)"))
            if not errors:
                E1, E2, G12, nu12 = MAT.isotropic_to_orthotropic(m.E * f["modulus"], m.nu)
            else:
                E1 = E2 = G12 = nu12 = 0.0
            is_iso = True
        else:
            raw_moduli = {"E1": m.E1, "E2": m.E2, "G12": m.G12}
            bad = [(n, v) for n, v in raw_moduli.items() if v <= 0]
            if bad:
                for n, v in bad:
                    errors.append(item("E200", field=f"{base}.material.{n}", detail=f"{n} = {v}"))
                E1 = E2 = G12 = nu12 = 0.0
            else:
                reason = MAT.check_orthotropic_validity(m.E1, m.E2, m.G12, m.nu12)
                if reason is not None:
                    errors.append(item("E201", field=f"{base}.material.nu12", detail=reason))
                E1, E2 = m.E1 * f["modulus"], m.E2 * f["modulus"]
                G12, nu12 = m.G12 * f["modulus"], m.nu12
                if E2 > 0 and max(E1, E2) / min(E1, E2) > config.EXTREME_MODULUS_RATIO:
                    warnings.append(item("W112", field=f"{base}.material",
                                         detail=f"{base}: E1/E2 비 = {max(E1,E2)/min(E1,E2):.3g}"))
            is_iso = False

        # W110 단위 오입력 휴리스틱 (원시 입력값 기준, 계획서 §6.4)
        for n, v in raw_moduli.items():
            if v > 0:
                if us == "SI" and config.SI_MODULUS_SUSPECT_LOW <= v < config.SI_MODULUS_SUSPECT_HIGH:
                    warnings.append(item("W110", field=f"{base}.material.{n}",
                                         detail=f"{base}.material.{n} = {v} Pa — GPa/MPa 값을 SI(Pa)에 넣은 착오일 수 있습니다"))
                elif us == "SI_mm" and v > config.SI_MM_MODULUS_SUSPECT:
                    warnings.append(item("W110", field=f"{base}.material.{n}",
                                         detail=f"{base}.material.{n} = {v} MPa — Pa 값을 SI_mm(MPa)에 넣은 착오일 수 있습니다"))
        if us == "SI" and lam.thickness > config.SI_THICKNESS_SUSPECT:
            warnings.append(item("W110", field=f"{base}.thickness",
                                 detail=f"{base}.thickness = {lam.thickness} m — mm 값을 SI(m)에 넣은 착오일 수 있습니다"))

        # W120 가정 상수 (§16.3)
        if m.source is not None and m.source.type == "assumed":
            warnings.append(item("W120", field=f"{base}.material.source",
                                 detail=f"{base}.material({m.name or m.type})의 상수가 가정값(assumed)입니다"))

        # CTE (§17.1) — 값 자체는 단위계 무관 [1/K]. ppm 착오 휴리스틱.
        if m.type == "isotropic":
            a1 = a2 = m.alpha
        else:
            a1, a2 = m.alpha1, m.alpha2
            if (a1 is None) != (a2 is None):
                errors.append(item("E203", field=f"{base}.material",
                                   detail=f"{base}: alpha1/alpha2는 둘 다 있어야 합니다"))
        for nm, av in (("alpha", a1), ("alpha2", a2)):
            if av is not None and abs(av) > 1.0e-3:
                warnings.append(item("W110", field=f"{base}.material.{nm}",
                                     detail=f"{base}.material.{nm} = {av} 1/K — ppm/K 값을 그대로 넣은 착오일 수 있습니다 (예: 17 ppm/K는 17e-6)"))
                break

        # 점탄성 (§17.2) — 이완이므로 Einf <= E0 이어야 물리적.
        ve = m.viscoelastic
        if ve is not None and ve.Einf > ve.E0:
            errors.append(item("E100", field=f"{base}.material.viscoelastic",
                               detail=f"{base}: Einf({ve.Einf}) > E0({ve.E0}) — 이완 특성은 Einf <= E0 이어야 합니다"))

        angle_norm = MAT.normalize_angle_deg(lam.angle_deg) if abs(lam.angle_deg) < 360.0 else 0.0
        rho_si = m.rho * f["density"] if m.rho is not None else None
        plies.append(SiPly(lam.thickness * f["length"], angle_norm, E1, E2, G12, nu12,
                           rho_si, m.name, m.source, is_iso,
                           alpha1=a1, alpha2=a2,
                           ve_E0=ve.E0 * f["modulus"] if ve else None,
                           ve_Einf=ve.Einf * f["modulus"] if ve else None,
                           ve_tau_s=ve.tau_s if ve else None,
                           strength=(tuple(getattr(m.strength, k) * f["modulus"]
                                           for k in ("Xt", "Xc", "Yt", "Yc", "S"))
                                     if m.strength is not None else None)))
        fingerprint.append((lam.thickness, angle_norm, E1, E2, G12, nu12, rho_si, a1, a2))

    if errors:
        return None, errors, warnings

    si = SiLaminate(us, plies, model.name, model.reference_output, fingerprint)

    # W111 초박층 (총 두께 대비)
    h = si.total_thickness
    for k, p in enumerate(si.plies):
        if p.thickness / h < config.THIN_PLY_RATIO:
            warnings.append(item("W111", field=f"laminae[{k}].thickness",
                                 detail=f"laminae[{k}]: t/h = {p.thickness / h:.3g}"))

    return si, errors, warnings
