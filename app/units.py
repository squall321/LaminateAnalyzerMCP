# 단위계 정의와 SI 변환 계수 — 내부 계산은 항상 SI (계획서 §3 D4)

SUPPORTED_UNIT_SYSTEMS = ("SI", "SI_mm")

# 입력 → SI 변환 계수
TO_SI = {
    "SI": {"modulus": 1.0, "length": 1.0, "density": 1.0, "load_n": 1.0, "load_m": 1.0,
           "energy_area": 1.0, "shear_nl": 1.0},
    # MPa→Pa, mm→m, t/mm³→kg/m³, N/mm→N/m, M은 N·mm/mm = N·m/m 수치 동일,
    # 파괴인성 Γ: N/mm(=kJ/m²)→J/m²
    # S6666 [1/응력^3]: 1/MPa^3 → 1/Pa^3 = ×1e-18
    "SI_mm": {"modulus": 1.0e6, "length": 1.0e-3, "density": 1.0e12, "load_n": 1.0e3, "load_m": 1.0,
              "energy_area": 1.0e3, "shear_nl": 1.0e-18},
}

# SI 결과 → 입력 단위계 표시 변환 계수
FROM_SI = {
    "SI": {"A": 1.0, "B": 1.0, "D": 1.0, "z": 1.0, "hat": 1.0, "areal_mass": 1.0,
           "modulus": 1.0, "kappa": 1.0, "energy_area": 1.0, "shear_nl": 1.0,
           "load_n": 1.0, "load_m": 1.0},
    "SI_mm": {
        "A": 1.0e-3,          # N/m → N/mm
        "B": 1.0,             # N → N (동일 차원)
        "D": 1.0e3,           # N·m → N·mm
        "z": 1.0e3,           # m → mm
        "hat": 1.0e-6,        # Pa → MPa
        "areal_mass": 1.0e-9, # kg/m² → t/mm²
        "modulus": 1.0e-6,    # Pa → MPa
        "kappa": 1.0e-3,      # 1/m → 1/mm
        "energy_area": 1.0e-3,  # J/m² → N/mm(=kJ/m²)
        "shear_nl": 1.0e18,   # 1/Pa^3 → 1/MPa^3
        "load_n": 1.0e-3,     # N/m → N/mm (단위 폭당 힘)
        "load_m": 1.0,        # N·m/m → N·mm/mm (둘 다 N — 수치 동일)
    },
}

# 응답 metadata.units 에 노출할 단위 라벨 (계획서 §6.4)
UNIT_LABELS = {
    "SI": {
        "E": "Pa", "thickness": "m", "density": "kg/m^3",
        "A": "N/m", "B": "N", "D": "N*m", "z": "m",
        "A_hat/B_hat/D_hat": "Pa", "areal_mass": "kg/m^2", "angle": "deg",
        "alpha": "1/K", "delta_T": "K", "kappa": "1/m", "gamma": "J/m^2", "stress": "Pa",
        "S6666": "1/Pa^3",
    },
    "SI_mm": {
        "E": "MPa", "thickness": "mm", "density": "t/mm^3",
        "A": "N/mm", "B": "N", "D": "N*mm", "z": "mm",
        "A_hat/B_hat/D_hat": "MPa", "areal_mass": "t/mm^2", "angle": "deg",
        "alpha": "1/K", "delta_T": "K", "kappa": "1/mm", "gamma": "N/mm (=kJ/m^2)", "stress": "MPa",
        "S6666": "1/MPa^3",
    },
}
