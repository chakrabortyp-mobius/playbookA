"""
synthetic_data.py
-----------------
Synthetic R_m, P_m, D_m, A_m for all 5 markets x 13 years (2010-2022).

Replace each function with Tharun's real outputs when available.
The shapes and dict structure must stay the same so merge_tensor.py
does not need to change.

Shapes:
    R_m[market][year] = dict { regime_name: np.array(8,) }   9 regimes
    P_m[market][year] = np.array(6,)    soft phase membership, sums to 1
    D_m[market][year] = np.array(9,)    velocity(3) + inertia(3) + persistence(3)
    A_m[market][year] = np.array(6,)    concentration(2) + chokepoints(2) + capture(2)
"""

import numpy as np

MARKETS = ["IN-LOG", "DE-HC", "NL-AG", "NG-FIN", "US-ENR"]
YEARS   = list(range(2010, 2023))

REGIME_NAMES = [
    "political", "measurement", "coordination", "regulatory",
    "techprod",  "operational", "narrative",    "incentive", "temporal"
]
DIM_PER_REGIME = 8   # confirm with Tharun


# ── R_m ───────────────────────────────────────────────────────────────────────

def _base_directions(market_id: str, rng: np.random.Generator) -> dict:
    """
    Domain-grounded base directions per regime per market.
    Key structural signals from sprint doc:
      IN-LOG : coordination opposite operational → high CF
      NG-FIN : measurement opposite operational  → highest IA
      DE-HC  : regulatory aligned with techprod  → highest RP
      US-ENR : techprod dominant magnitude       → highest PESTLE_T
    """
    base = {r: rng.normal(0, 0.4, DIM_PER_REGIME).astype(np.float32)
            for r in REGIME_NAMES}

    if market_id == "IN-LOG":
        d = rng.normal(0, 1, DIM_PER_REGIME)
        d /= np.linalg.norm(d)
        base["coordination"] =  d.copy()
        base["operational"]  = -d.copy()                                      # opposite → high CF
        base["measurement"]  =  d * 0.6 + rng.normal(0, 0.15, DIM_PER_REGIME) # moderate IA

    elif market_id == "NG-FIN":
        d = rng.normal(0, 1, DIM_PER_REGIME)
        d /= np.linalg.norm(d)
        base["measurement"] =  d.copy()
        base["operational"] = -d.copy()                                       # opposite → highest IA

    elif market_id == "DE-HC":
        d = rng.normal(0, 1, DIM_PER_REGIME)
        d /= np.linalg.norm(d)
        base["regulatory"] = d.copy()
        base["techprod"]   = d + rng.normal(0, 0.05, DIM_PER_REGIME)         # aligned → highest RP

    elif market_id == "US-ENR":
        base["techprod"] = np.abs(rng.normal(0.9, 0.1, DIM_PER_REGIME))      # dominant

    return {k: v.astype(np.float32) for k, v in base.items()}


def make_rm_all_years(rng: np.random.Generator) -> dict:
    """
    Small Gaussian drift year-over-year on top of base direction.
    Simulates gradual structural change from Tharun's regime models.

    Returns:
        rm[market_id][year][regime_name] = np.array(8,)
    """
    rm = {}
    for market_id in MARKETS:
        base = _base_directions(market_id, rng)
        rm[market_id] = {}

        for i, year in enumerate(YEARS):
            t = i / (len(YEARS) - 1)          # 0.0 → 1.0
            rm[market_id][year] = {}
            for regime in REGIME_NAMES:
                drift = rng.normal(0, 0.03, DIM_PER_REGIME).astype(np.float32)
                trend = (base[regime] * 0.02 * t).astype(np.float32)
                rm[market_id][year][regime] = base[regime] + drift + trend

    return rm


# ── P_m ───────────────────────────────────────────────────────────────────────
# Phases: pre_formal, formalizing, platformizing, fragmenting, stagnant, collapse

_PM_2010 = {
    "IN-LOG": np.array([0.20, 0.50, 0.15, 0.10, 0.04, 0.01], dtype=np.float32),
    "DE-HC":  np.array([0.00, 0.18, 0.55, 0.20, 0.06, 0.01], dtype=np.float32),
    "NL-AG":  np.array([0.00, 0.14, 0.60, 0.20, 0.05, 0.01], dtype=np.float32),
    "NG-FIN": np.array([0.40, 0.40, 0.08, 0.08, 0.03, 0.01], dtype=np.float32),
    "US-ENR": np.array([0.00, 0.10, 0.65, 0.18, 0.06, 0.01], dtype=np.float32),
}
_PM_2022 = {
    "IN-LOG": np.array([0.05, 0.55, 0.22, 0.13, 0.04, 0.01], dtype=np.float32),
    "DE-HC":  np.array([0.00, 0.08, 0.65, 0.21, 0.05, 0.01], dtype=np.float32),
    "NL-AG":  np.array([0.00, 0.06, 0.68, 0.20, 0.05, 0.01], dtype=np.float32),
    "NG-FIN": np.array([0.22, 0.48, 0.11, 0.12, 0.05, 0.02], dtype=np.float32),
    "US-ENR": np.array([0.00, 0.04, 0.72, 0.18, 0.05, 0.01], dtype=np.float32),
}

def make_pm_all_years(rng: np.random.Generator) -> dict:
    """
    Linear interpolation between 2010 and 2022 anchors + tiny noise.
    Renormalised to sum=1 after noise.

    Returns:
        pm[market_id][year] = np.array(6,)
    """
    pm = {}
    for market_id in MARKETS:
        pm[market_id] = {}
        start = _PM_2010[market_id]
        end   = _PM_2022[market_id]

        for i, year in enumerate(YEARS):
            t   = i / (len(YEARS) - 1)
            vec = (1 - t) * start + t * end
            vec = np.clip(vec + rng.normal(0, 0.005, 6).astype(np.float32), 0, 1)
            vec /= vec.sum()
            pm[market_id][year] = vec

    return pm


# ── D_m ───────────────────────────────────────────────────────────────────────
# [v1,v2,v3 = velocity,  i1,i2,i3 = inertia,  p1,p2,p3 = persistence]

_DM_2010 = {
    "IN-LOG": np.array([ 0.08,  0.06, -0.02,  0.75, 0.70, 0.62,  0.85, 0.78, 0.65], dtype=np.float32),
    "DE-HC":  np.array([ 0.12,  0.10,  0.08,  0.40, 0.36, 0.42,  0.52, 0.48, 0.46], dtype=np.float32),
    "NL-AG":  np.array([ 0.10,  0.13,  0.07,  0.36, 0.32, 0.38,  0.48, 0.44, 0.42], dtype=np.float32),
    "NG-FIN": np.array([ 0.14,  0.05, -0.08,  0.82, 0.75, 0.71,  0.90, 0.84, 0.78], dtype=np.float32),
    "US-ENR": np.array([ 0.11,  0.15,  0.18,  0.33, 0.29, 0.27,  0.42, 0.39, 0.36], dtype=np.float32),
}
_DM_2022 = {
    "IN-LOG": np.array([ 0.18,  0.12, -0.04,  0.71, 0.65, 0.58,  0.82, 0.74, 0.61], dtype=np.float32),
    "DE-HC":  np.array([ 0.09,  0.07,  0.11,  0.42, 0.38, 0.45,  0.55, 0.51, 0.49], dtype=np.float32),
    "NL-AG":  np.array([ 0.07,  0.11,  0.09,  0.38, 0.35, 0.40,  0.50, 0.47, 0.44], dtype=np.float32),
    "NG-FIN": np.array([ 0.22,  0.08, -0.11,  0.79, 0.72, 0.68,  0.88, 0.81, 0.75], dtype=np.float32),
    "US-ENR": np.array([ 0.14,  0.19,  0.22,  0.35, 0.31, 0.29,  0.44, 0.41, 0.38], dtype=np.float32),
}

def make_dm_all_years(rng: np.random.Generator) -> dict:
    """
    Returns:
        dm[market_id][year] = np.array(9,)
    """
    dm = {}
    for market_id in MARKETS:
        dm[market_id] = {}
        start = _DM_2010[market_id]
        end   = _DM_2022[market_id]

        for i, year in enumerate(YEARS):
            t = i / (len(YEARS) - 1)
            vec = (1 - t) * start + t * end
            dm[market_id][year] = (vec + rng.normal(0, 0.01, 9).astype(np.float32))

    return dm


# ── A_m ───────────────────────────────────────────────────────────────────────
# [a1,a2 = concentration,  b1,b2 = chokepoints,  g1,g2 = capture]

_AM_2010 = {
    "IN-LOG": np.array([0.78, 0.72,  0.84, 0.79,  0.48, 0.43], dtype=np.float32),
    "DE-HC":  np.array([0.44, 0.41,  0.47, 0.43,  0.31, 0.27], dtype=np.float32),
    "NL-AG":  np.array([0.41, 0.38,  0.43, 0.40,  0.25, 0.22], dtype=np.float32),
    "NG-FIN": np.array([0.90, 0.85,  0.82, 0.75,  0.70, 0.65], dtype=np.float32),
    "US-ENR": np.array([0.55, 0.51,  0.52, 0.48,  0.34, 0.31], dtype=np.float32),
}
_AM_2022 = {
    "IN-LOG": np.array([0.74, 0.68,  0.81, 0.76,  0.44, 0.39], dtype=np.float32),
    "DE-HC":  np.array([0.41, 0.38,  0.44, 0.40,  0.28, 0.24], dtype=np.float32),
    "NL-AG":  np.array([0.38, 0.35,  0.40, 0.37,  0.22, 0.19], dtype=np.float32),
    "NG-FIN": np.array([0.88, 0.82,  0.79, 0.71,  0.67, 0.61], dtype=np.float32),
    "US-ENR": np.array([0.52, 0.48,  0.49, 0.45,  0.31, 0.28], dtype=np.float32),
}

def make_am_all_years(rng: np.random.Generator) -> dict:
    """
    Returns:
        am[market_id][year] = np.array(6,)
    """
    am = {}
    for market_id in MARKETS:
        am[market_id] = {}
        start = _AM_2010[market_id]
        end   = _AM_2022[market_id]

        for i, year in enumerate(YEARS):
            t = i / (len(YEARS) - 1)
            vec = (1 - t) * start + t * end
            am[market_id][year] = np.clip(
                vec + rng.normal(0, 0.008, 6).astype(np.float32), 0, 1
            )

    return am


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rm  = make_rm_all_years(rng)
    pm  = make_pm_all_years(rng)
    dm  = make_dm_all_years(rng)
    am  = make_am_all_years(rng)

    print("Shapes per market-year:")
    for market in MARKETS:
        r_flat = np.concatenate([rm[market][2022][r] for r in REGIME_NAMES])
        print(f"  {market} 2022 → R_m:{r_flat.shape} P_m:{pm[market][2022].shape} "
              f"D_m:{dm[market][2022].shape} A_m:{am[market][2022].shape}")
        print(f"    P_m sums to: {pm[market][2022].sum():.4f}")

    #print(rm)