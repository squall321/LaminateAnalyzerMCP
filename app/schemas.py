# 입력 스키마(pydantic v2) — 필드 description이 곧 에이전트용 문서 (계획서 §7.1)
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class SourceInfo(BaseModel):
    """물성 출처 provenance (§16.3). 예: materialtwin 실측이면 ref='materialtwin:material/42/test/7'."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["measured", "estimated", "assumed", "user"] = "user"
    ref: str | None = Field(default=None, description="출처 식별자 (예: materialtwin:material/42)")
    confidence: str | None = Field(default=None, description="출처 신뢰도 라벨 (high/ok/low 등)")


class IsotropicMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["isotropic"]
    E: float = Field(description="영률 (SI: Pa, SI_mm: MPa)")
    nu: float = Field(description="포아송비 (-1 < nu < 0.5)")
    rho: float | None = Field(default=None, gt=0, description="밀도 (SI: kg/m^3, SI_mm: t/mm^3). 질량 지표 계산 시에만 필요")
    name: str | None = Field(default=None, description="재료 라벨 (추적용)")
    source: SourceInfo | None = None


class Orthotropic2DMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["orthotropic_2d"]
    E1: float = Field(description="섬유(1축) 방향 영률 (SI: Pa, SI_mm: MPa)")
    E2: float = Field(description="횡(2축) 방향 영률")
    G12: float = Field(description="면내 전단탄성계수")
    nu12: float = Field(description="주 포아송비 (|nu12| < sqrt(E1/E2))")
    rho: float | None = Field(default=None, gt=0, description="밀도 (선택)")
    name: str | None = Field(default=None, description="재료 라벨 (추적용)")
    source: SourceInfo | None = None


Material = Annotated[Union[IsotropicMaterial, Orthotropic2DMaterial], Field(discriminator="type")]


class Lamina(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thickness: float = Field(description="ply 두께 (SI: m, SI_mm: mm), 양수")
    angle_deg: float = Field(description="적층각 [deg]. 적층판 +x축에서 섬유 1축으로 +z 기준 CCW 양수")
    material: Material


class Laminate(BaseModel):
    """적층 정의. laminae[0]이 최하단(bottom) ply — z가 가장 작은 쪽 (D1 규약)."""
    model_config = ConfigDict(extra="forbid")
    unit_system: Literal["SI", "SI_mm"] = Field(
        description="단위계 (필수). SI = Pa·m·kg/m^3, SI_mm = MPa·mm·t/mm^3")
    laminae: list[Lamina] = Field(description="ply 목록, index 0 = 최하단(bottom)")
    name: str | None = Field(default=None, description="적층안 라벨")
    reference_output: Literal["midplane", "bottom", "top"] = Field(
        default="midplane",
        description="중립면 대표 표기 기준 (어떤 값이든 세 표기 모두 반환됨)")
