"""
extract_cm.py
-------------
Reads Naman's observability CSV.
Returns C_m for all 5 markets x 13 years (2010-2022).
"""

import pandas as pd
import numpy as np
import sys

MARKET_TO_COUNTRY = {
    "IN-LOG": "IND",
    "DE-HC":  "DEU",
    "NL-AG":  "NLD",
    "NG-FIN": "NGA",
    "US-ENR": "USA",
}

YEARS = list(range(2010, 2023))

# Columns we want from Naman's CSV.
# NOTE: Naman's CSV has extra columns (c_m_semantic_drift, c_m_hs_drift etc.)
# We only pull the 10 canonical constraint columns.
C_M_COLUMNS = [
    "c_m_observability",
    "c_m_incentive_to_report",
    "c_m_temporal_lag",
    "c_m_program_bias",
    "c_m_revision_lag",
    "c_m_informality_observ",
    "c_m_selection_bias",
    "c_m_methodology_drift",
    "c_m_greenwashing",
    "c_m_path_dependence",
]


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Robustly loads the CSV regardless of separator (tab, comma, auto).
    Strips whitespace from all column names.
    """
    for sep in ["\t", ",", None]:
        try:
            if sep is not None:
                df = pd.read_csv(csv_path, sep=sep)
            else:
                df = pd.read_csv(csv_path, sep=None, engine="python")

            df.columns = df.columns.str.strip()

            if "country" in df.columns and df.shape[1] > 5:
                return df

        except Exception:
            continue

    # Debug fallback
    df = pd.read_csv(csv_path, nrows=2)
    print(f"\nCould not detect separator. Raw columns: {list(df.columns)}")
    raise ValueError("Cannot parse CSV — check file encoding or separator")


def extract_cm_all_years(csv_path: str) -> dict:
    """
    Returns:
        cm[market_id][year] = np.array(10,)

    If a year is missing for a country, fills with NaN.
    merge_tensor.py sets mask=0 for those dims.
    """
    df = load_csv(csv_path)

    missing_cols = [c for c in C_M_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"Warning: columns not found in CSV (will be NaN): {missing_cols}")

    cm = {}
    for market_id, country_code in MARKET_TO_COUNTRY.items():
        country_df = df[df["country"] == country_code].copy()

        if country_df.empty:
            print(f"Warning: no rows found for {market_id} ({country_code})")

        rows = country_df.sort_values("year").set_index("year")

        cm[market_id] = {}
        for year in YEARS:
            vec = np.full(len(C_M_COLUMNS), np.nan, dtype=np.float32)

            if year in rows.index:
                for i, col in enumerate(C_M_COLUMNS):
                    if col in rows.columns:
                        val = rows.loc[year, col]
                        vec[i] = float(val) if pd.notna(val) else np.nan

            cm[market_id][year] = vec

    return cm


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/gaian/Downloads/observability_final_v3_fixed.csv"

    print(f"Reading: {path}\n")
    cm = extract_cm_all_years(path)

    # print(f"{'Market':<10} {'Year'}  {'observability':>15}  {'selection_bias':>15}  {'temporal_lag':>14}")
    # print("-" * 62)
    # for market in cm:
    #     for year in [2010, 2016, 2022]:
    #         row = cm[market][year]
    #         obs = f"{row[0]:.4f}" if not np.isnan(row[0]) else "NaN"
    #         sel = f"{row[6]:.4f}" if not np.isnan(row[6]) else "NaN"
    #         lag = f"{row[2]:.4f}" if not np.isnan(row[2]) else "NaN"
    #         print(f"{market:<10} {year}  {obs:>15}  {sel:>15}  {lag:>14}")
    #     print()
    print(cm)