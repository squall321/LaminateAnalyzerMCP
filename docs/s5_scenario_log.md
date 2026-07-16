# S5 시나리오 로그 — 두 MCP 협동 (실제 stdio 세션, 2026-07-16)

> 스크립트 재현: 에이전트 표준 플로우(agent_guide.md §2·§3)를 실제 프로토콜로 실행한 기록.
> materialtwin 실측 조회 → 단위 브리지 → 하이브리드 적층 분석 → 하중 응답 → W110 복구 → 리포트.

### 0. 두 서버 동시 연결

```json
{
  "laminate": "laminate-analyzer",
  "materialtwin": "materialtwin"
}
```

### 1. materialtwin.list_materials(category='metal')

```json
[
  {
    "id": 1,
    "name": "SUS201_annealed Bilinear",
    "category": "metal",
    "mat_type": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "kind": "elastoplastic",
    "E_GPa": 200.0,
    "UTS_MPa": 376.84,
    "test_id": 1
  },
  {
    "id": 2,
    "name": "SUS301_annealed Bilinear",
    "category": "metal",
    "mat_type": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "kind": "elastoplastic",
    "E_GPa": 193.0,
    "UTS_MPa": 398.49,
    "test_id": 2
  },
  {
    "id": 3,
    "name": "SUS304_annealed Bilinear",
    "category": "metal",
    "mat_type": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "kind": "elastoplastic",
    "E_GPa": 193.0,
    "UTS_MPa": 311.71,
    "test_id": 3
  },
  {
    "id": 4,
    "name": "SUS304L_annealed Bilinear",
    "category": "metal",
    "mat_type": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "kind": "elastoplastic",
    "E_GPa": 193.0,
    "UTS_MPa": 297.24,
    "test_id": 4
  },
  {
    "id": 5,
    "name": "SUS316_annealed Bilinear",
    "category": "metal",
    "mat_type": "MAT_PIECEWISE_LINEAR_PLASTICITY",
    "kind": "elastoplastic",
    "E_GPa": 193.0,
    "UTS_MPa": 347.88,
    "test_id": 5
  }
]
```

### 2. 단위 브리지

```json
{
  "E_GPa(materialtwin)": 200.0,
  "E_MPa(SI_mm 입력)": 200000.0,
  "nu": "0.33 가정(단축 인장은 E만 제공)",
  "source.ref": "materialtwin:material/1/test/1"
}
```

### 3. laminate.analyze_laminate — 하이브리드 [금속/0/90]s

```json
{
  "status": "warning",
  "warnings": [
    "W120",
    "W120"
  ],
  "is_symmetric": true,
  "coupling_ratio": {
    "value": 9.074322098316848e-34,
    "grade": "negligible",
    "definition": "||B_hat||_F / sqrt(||A_hat||_F * ||D_hat||_F), B_hat = sqrt(12)*B/h^2"
  },
  "quasi_isotropy": 0.9640394776069617,
  "zeta_x": 0.5,
  "assumptions_tail": [
    "laminae[0].material(SUS201_annealed Bilinear): 물성 출처 assumed (materialtwin:material/1/test/1 (E measured, nu assumed 0.33))",
    "laminae[5].material(SUS201_annealed Bilinear): 물성 출처 assumed (materialtwin:material/1/test/1 (E measured, nu assumed 0.33))"
  ],
  "payload_hash": "sha256:c007b1ce0f2e12dd..."
}
```

### 4. laminate.solve_load_response (Nx=100 N/mm)

```json
{
  "epsilon0_x": 0.0005841832561521771,
  "membrane_Ex_MPa": 155617.41962253276,
  "leakage": 9.149818867908359e-34
}
```

### 5. 단위 실수 시나리오 (SI_mm 값을 SI로 제출)

```json
{
  "status": "warning",
  "w110": [
    "값의 자릿수가 단위계와 어긋나 보입니다: laminae[0].material.E = 200000.0 Pa — GPa/MPa 값을 SI(Pa)에 넣은 착오일 수 있습니다",
    "값의 자릿수가 단위계와 어긋나 보입니다: laminae[1].material.E1 = 181000.0 Pa — GPa/MPa 값을 SI(Pa)에 넣은 착오일 수 있습니다"
  ]
}
```

### 6. generate_design_report — 첫 12줄

```json
{
  "head": [
    "# 적층 평가 리포트 — hybrid_metal_cfrp",
    "",
    "## 요약",
    "",
    "- ply: 6개, h = 1.1 mm, 대칭 적층",
    "- coupling_ratio: 9.074e-34 (negligible)",
    "- quasi_isotropy: 0.964 (quasi_isotropic)",
    "- ζ(중립면, x): 0.5",
    "",
    "## 입력 적층",
    "",
    "| # | 각도 [deg] | 두께 [mm] | 재료 |"
  ]
}
```
