"""
merge_to_csv.py
---------------
Run this from your project root directory:

    python3 merge_to_csv.py

Reads:
    - naman_observability.csv  (must be in same directory)
    - extract_cm.py
    - synthetic_data.py

Saves:
    - merged_tensor_65x103.csv  in the same directory as this script
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Allow imports from same directory ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_cm import extract_cm_all_years, C_M_COLUMNS, MARKET_TO_COUNTRY, YEARS
from synthetic_data import (
    make_rm_all_years,
    make_pm_all_years,
    make_dm_all_years,
    make_am_all_years,
    MARKETS,
    REGIME_NAMES,
    DIM_PER_REGIME,
)

# ── Dim constants ─────────────────────────────────────────────────────────────
DIM_R     = len(REGIME_NAMES) * DIM_PER_REGIME   # 72
DIM_C     = 10
DIM_P     = 6
DIM_D     = 9
DIM_A     = 6
DIM_TOTAL = DIM_R + DIM_C + DIM_P + DIM_D + DIM_A   # 103

# ── Column names for all 103 dims ─────────────────────────────────────────────
PHASE_NAMES = ["pre_formal", "formalizing", "platformizing",
               "fragmenting", "stagnant",   "collapse"]

R_COLS = [f"r_{regime}_d{d+1}" for regime in REGIME_NAMES
          for d in range(DIM_PER_REGIME)]        # 72 cols

C_COLS = C_M_COLUMNS                             # 10 cols

P_COLS = [f"pm_{ph}" for ph in PHASE_NAMES]      # 6 cols

D_COLS = ["vel_1",     "vel_2",     "vel_3",
          "inertia_1", "inertia_2", "inertia_3",
          "persist_1", "persist_2", "persist_3"]  # 9 cols

A_COLS = ["conc_1",    "conc_2",
          "choke_1",   "choke_2",
          "capture_1", "capture_2"]               # 6 cols

ALL_DIM_COLS = R_COLS + C_COLS + P_COLS + D_COLS + A_COLS
assert len(ALL_DIM_COLS) == DIM_TOTAL, f"Expected 103 cols, got {len(ALL_DIM_COLS)}"


# ── Missingness mask ──────────────────────────────────────────────────────────
def build_mask(market_id: str, cm_vector: np.ndarray) -> np.ndarray:
    mask = np.ones(DIM_TOTAL, dtype=np.float32)

    if market_id in ["IN-LOG", "NG-FIN"]:
        mask[24:28] = 0   # regulatory dims — OECD absent
        mask[32:36] = 0   # techprod dims   — OECD absent

    if market_id == "NG-FIN":
        mask[64:68] = 0   # temporal dims   — BIS absent

    for i, val in enumerate(cm_vector):
        if np.isnan(val):
            mask[DIM_R + i] = 0

    return mask


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    csv_in   = os.path.join("/home/gaian/Downloads/observability_final_v3_fixed.csv")
    csv_out  = os.path.join(root_dir, "merged_tensor.csv")

    if not os.path.exists(csv_in):
        print(f"ERROR: cannot find {csv_in}")
        print("Place naman_observability.csv in the same folder as this script.")
        sys.exit(1)

    print(f"Reading C_m from : {csv_in}")
    rng    = np.random.default_rng(42)
    cm_data = extract_cm_all_years(csv_in)
    rm_data = make_rm_all_years(rng)
    pm_data = make_pm_all_years(rng)
    dm_data = make_dm_all_years(rng)
    am_data = make_am_all_years(rng)

    rows = []
    for market_id in MARKETS:
        for year in YEARS:
            # Flatten R_m
            R = np.concatenate([rm_data[market_id][year][r]
                                 for r in REGIME_NAMES]).astype(np.float32)
            C = cm_data[market_id][year]          # (10,) may have NaN
            P = pm_data[market_id][year]          # (6,)
            D = dm_data[market_id][year]          # (9,)
            A = am_data[market_id][year]          # (6,)

            C_clean = np.nan_to_num(C, nan=0.0)
            x_m     = np.concatenate([R, C_clean, P, D, A])   # (103,)
            mask    = build_mask(market_id, C)                 # (103,)

            row = {
                "market" : market_id,
                "year"   : year,
                "obs_pct": round(float(mask.mean() * 100), 1),
            }
            row.update(dict(zip(ALL_DIM_COLS, x_m.tolist())))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)

    print(f"Saved            : {csv_out}")
    print(f"Shape            : {df.shape[0]} rows x {len(ALL_DIM_COLS)} dims  "
          f"(+3 meta cols = {df.shape[1]} total)")
    print(f"\nPreview (market / year / obs_pct / first 3 C_m dims):")
    print(df[["market", "year", "obs_pct"] + C_COLS[:3]]
          .to_string(index=False))


if __name__ == "__main__":
    main()