"""
dae.py
======
Denoising Autoencoder — Stage 2 of the pipeline.

Input  : mae_outputs/mae_reconstructed.parquet   (x̂_m1 from MAE — N × 103)

Why 103-dim (not z_m^(1)):
    Playbook A Section 3.6 requires e_m to be sliceable by exact regime
    boundaries so Kumar can compute Component 2 for every T1 field:
        structural_normality = 1 − (||e_m[regime_dims]|| / ||e_m||)
    Examples:
        PESTLE_P  → e_m[0:8]    political
        PESTLE_L  → e_m[24:32]  regulatory
        CF (ALI)  → e_m[16:24]  coordination
        IA (ALI)  → e_m[8:16] + e_m[40:48]
        F  (ALI)  → e_m[97:103] agency
    These slices are meaningless if e_m is 32-dim.

Outputs: dae_outputs/dae_error_vectors.parquet     (e_m  — N × 103)
         dae_outputs/dae_error_fractions.json       (per-regime fractions)
         dae_outputs/dae_best_model.pth
         dae_outputs/dae_training_history.json

What it learns:
    Takes x̂_m1 + injected noise → reconstructs x̂_m1
    e_m = x̂_m1 − reconstruction(x̂_m1)
    Large e_m in a subspace = that regime is structurally abnormal for this market.

    Ticket formula: ||e_m[political_dims]|| / ||e_m|| etc.
    These are computed using EXACT regime boundaries from REGIME_SLICES.

NOTE: z_dae (bottleneck) is NOT saved. Readout layer uses e_m directly.

Run:
    python3 dae.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PARQUET  = "mae_outputs/mae_reconstructed.parquet"   # 103-dim
OUTPUT_DIR     = "dae_outputs"
EXPECTED_DIM   = 103

# Architecture
ENCODER_LAYERS = [80, 48]
BOTTLENECK_DIM = 32
DECODER_LAYERS = [48, 80]
ACTIVATION     = "relu"
DROPOUT        = 0.2
BATCH_NORM     = True

# Noise (train only)
NOISE_TYPE   = "gaussian"
NOISE_FACTOR = 0.3

# Training
EPOCHS         = 300
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 16
WEIGHT_DECAY   = 1e-5
GRAD_CLIP      = True
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING = True
ES_PATIENCE    = 20
ES_MIN_DELTA   = 1e-6
VAL_SPLIT      = 0.2

SCHEDULER_ENABLED  = True
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR   = 1e-6

WEIGHT_INIT = "xavier_uniform"
RANDOM_SEED = 42
PRINT_EVERY = 50


# ══════════════════════════════════════════════════════════════════════════════
# REGIME SUBSPACE BOUNDARIES  (103-dim tensor space)
# Exact boundaries from merged_tensor.csv column layout.
# Used to compute per-regime e_m fractions for readout Component 2.
#
# Verified against actual column names:
#   r_regulatory_d1..d8 → dims 24:32
#   r_techprod_d1..d8   → dims 32:40
#   r_temporal_d1..d8   → dims 64:72
# ══════════════════════════════════════════════════════════════════════════════

REGIME_SLICES = {
    "political"   : (0,   8),
    "measurement" : (8,  16),
    "coordination": (16, 24),
    "regulatory"  : (24, 32),
    "techprod"    : (32, 40),
    "operational" : (40, 48),
    "narrative"   : (48, 56),
    "incentive"   : (56, 64),
    "temporal"    : (64, 72),
    "C_m"         : (72, 82),
    "P_m"         : (82, 88),
    "D_m"         : (88, 97),
    "A_m"         : (97, 103),
}

R_M_REGIMES = [
    "political", "measurement", "coordination", "regulatory",
    "techprod",  "operational", "narrative",    "incentive", "temporal"
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_activation(name):
    return {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(), "tanh": nn.Tanh(), "gelu": nn.GELU()
            }.get(name.lower(), nn.ReLU())

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def add_noise(z, noise_type, noise_factor):
    if noise_factor == 0.0:
        return z
    if noise_type == "gaussian":
        return z + torch.randn_like(z) * noise_factor
    elif noise_type == "dropout":
        return z * (torch.rand_like(z) > noise_factor).float()
    elif noise_type == "uniform":
        return z + (torch.rand_like(z) - 0.5) * 2 * noise_factor
    raise ValueError(f"Unknown noise type: {noise_type}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DAEEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        prev, layers = input_dim, []
        for units in ENCODER_LAYERS:
            layers += [nn.Linear(prev, units)]
            if BATCH_NORM:
                layers += [nn.BatchNorm1d(units)]
            layers += [get_activation(ACTIVATION)]
            if DROPOUT > 0:
                layers += [nn.Dropout(DROPOUT)]
            prev = units
        layers += [nn.Linear(prev, BOTTLENECK_DIM)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DAEDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        prev, layers = BOTTLENECK_DIM, []
        for units in DECODER_LAYERS:
            layers += [nn.Linear(prev, units)]
            if BATCH_NORM:
                layers += [nn.BatchNorm1d(units)]
            layers += [get_activation(ACTIVATION)]
            if DROPOUT > 0:
                layers += [nn.Dropout(DROPOUT)]
            prev = units
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim      = input_dim
        self.bottleneck_dim = BOTTLENECK_DIM
        self.encoder        = DAEEncoder(input_dim)
        self.decoder        = DAEDecoder(input_dim)

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df           = pd.read_parquet(INPUT_PARQUET)
    meta_cols    = ["market", "year", "obs_pct"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    input_dim    = len(feature_cols)

    assert input_dim == EXPECTED_DIM, (
        f"Expected {EXPECTED_DIM}-dim from mae_reconstructed.parquet, got {input_dim}. "
        f"Make sure INPUT_PARQUET = mae_reconstructed.parquet (103-dim), "
        f"NOT mae_latents.parquet (32-dim)."
    )

    print(f"  Loaded x̂_m1 : {df.shape[0]} rows × {input_dim} dims  (103-dim ✓)")
    print(f"  Markets      : {df['market'].unique().tolist()}")

    markets  = df["market"].unique().tolist()
    rng      = np.random.default_rng(RANDOM_SEED)
    n_val    = max(1, round(len(markets) * VAL_SPLIT))
    val_mkts = rng.choice(markets, size=n_val, replace=False).tolist()
    trn_mkts = [m for m in markets if m not in val_mkts]

    print(f"  Train markets : {trn_mkts}")
    print(f"  Val markets   : {val_mkts}")

    trn_df = df[df["market"].isin(trn_mkts)].reset_index(drop=True)
    val_df = df[df["market"].isin(val_mkts)].reset_index(drop=True)

    X_trn = torch.tensor(trn_df[feature_cols].values, dtype=torch.float32)
    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)

    return TensorDataset(X_trn), TensorDataset(X_val), df, feature_cols, input_dim


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    criterion    = nn.MSELoss()
    optimizer    = optim.Adam(model.parameters(),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if SCHEDULER_ENABLED:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

    history       = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0
    best_state    = None

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ── Train: noisy input → clean target ─────────────────────────────────
        model.train()
        t_losses = []
        for (xb,) in train_loader:
            xb       = xb.to(device)
            xb_noisy = add_noise(xb, NOISE_TYPE, NOISE_FACTOR)
            x_hat, _ = model(xb_noisy)
            loss     = criterion(x_hat, xb)   # target = clean x̂_m1
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            t_losses.append(loss.item())

        # ── Validate: clean input, no noise ───────────────────────────────────
        model.eval()
        v_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb       = xb.to(device)
                x_hat, _ = model(xb)
                v_losses.append(criterion(x_hat, xb).item())

        avg_train = float(np.mean(t_losses))
        avg_val   = float(np.mean(v_losses))
        lr_now    = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if scheduler:
            scheduler.step(avg_val)

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4}/{EPOCHS}  "
                  f"train={avg_train:.6f}  val={avg_val:.6f}  "
                  f"lr={lr_now:.2e}  t={time.time()-t0:.2f}s")

        if avg_val < best_val_loss - ES_MIN_DELTA:
            best_val_loss = avg_val
            best_epoch    = epoch + 1
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ Best  epoch={best_epoch}  val={best_val_loss:.6f}")
        else:
            patience_ctr += 1

        if EARLY_STOPPING and patience_ctr >= ES_PATIENCE:
            print(f"\n  Early stop at epoch {epoch+1}. Best={best_epoch}")
            break

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        "epoch": best_epoch, "model_state_dict": best_state,
        "val_loss": best_val_loss,
        "model_config": {
            "input_dim": model.input_dim, "bottleneck_dim": BOTTLENECK_DIM,
            "encoder_dims": ENCODER_LAYERS, "decoder_dims": DECODER_LAYERS,
            "noise": {"type": NOISE_TYPE, "factor": NOISE_FACTOR,
                      "applied_at": "train_only"},
            "regime_slices": {k: list(v) for k, v in REGIME_SLICES.items()},
        },
        "architecture": "DenoisingAutoencoder", "version": "4.0",
    }, os.path.join(OUTPUT_DIR, "dae_best_model.pth"))

    with open(os.path.join(OUTPUT_DIR, "dae_training_history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss}, f, indent=2)

    print(f"\n  Model   → {OUTPUT_DIR}/dae_best_model.pth")
    print(f"  History → {OUTPUT_DIR}/dae_training_history.json")
    return best_state


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE + ERROR FRACTIONS
# e_m = x̂_m1 − reconstruction(x̂_m1)   shape: (N, 103)
# Ticket formula: ||e_m[regime_dims]|| / ||e_m||  per regime
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, best_state, full_df, feature_cols, device):
    model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    X_full = torch.tensor(full_df[feature_cols].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_full), batch_size=BATCH_SIZE, shuffle=False)

    all_x_hat = []
    with torch.no_grad():
        for (xb,) in loader:
            x_hat, _ = model(xb.to(device))
            all_x_hat.append(x_hat.cpu().numpy())

    X_np  = X_full.numpy()
    X_hat = np.vstack(all_x_hat)
    E_m   = X_np - X_hat   # e_m shape: (N, 103)

    print(f"  e_m shape : {E_m.shape}  (103-dim — regime boundaries exact ✓)")

    # Save error vectors
    e_df = pd.DataFrame(E_m, columns=[f"e_{i}" for i in range(E_m.shape[1])])
    e_df.insert(0, "market",  full_df["market"].values)
    e_df.insert(1, "year",    full_df["year"].values)
    e_df.insert(2, "obs_pct", full_df["obs_pct"].values)
    e_df.to_parquet(os.path.join(OUTPUT_DIR, "dae_error_vectors.parquet"), index=False)
    print(f"  e_m → {OUTPUT_DIR}/dae_error_vectors.parquet")

    # Per-market error fractions — exact regime boundaries
    print(f"\n  Structural misalignment (e_m) per market:")
    print(f"  {'Market':<10}  {'||e_m||':>9}  {'top regime':>14}  {'frac':>6}  {'2nd':>14}  {'frac':>6}")
    print(f"  {'-'*70}")

    summary = {}
    for market in full_df["market"].unique():
        idx    = full_df[full_df["market"] == market].index.tolist()
        e_mean = E_m[idx].mean(axis=0)
        total  = float(np.linalg.norm(e_mean)) + 1e-8

        # All subspace fractions
        all_fracs = {name: float(np.linalg.norm(e_mean[s:e])) / total
                     for name, (s, e) in REGIME_SLICES.items()}

        # R_m only for ranking/display
        rm_fracs = {r: all_fracs[r] for r in R_M_REGIMES}
        ranked   = sorted(rm_fracs.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = ranked[0], ranked[1] if len(ranked) > 1 else ("—", 0.0)

        print(f"  {market:<10}  {total:>9.4f}  {top1[0]:>14}  {top1[1]:>6.3f}"
              f"  {top2[0]:>14}  {top2[1]:>6.3f}")

        summary[market] = {
            "e_m_norm"  : float(total),
            "error_fracs": all_fracs,   # all subspaces — readout uses this
            "rm_fracs"  : rm_fracs,     # R_m only — for display
            "top_regime": top1[0],
            "top_frac"  : float(top1[1]),
        }

    with open(os.path.join(OUTPUT_DIR, "dae_error_fractions.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Error fractions → {OUTPUT_DIR}/dae_error_fractions.json")

    # Sanity: R_m fractions sum ≈ 0.7 (72 out of 103 dims are R_m)
    m0       = list(summary.keys())[0]
    rm_sum   = sum(summary[m0]["rm_fracs"].values())
    all_sum  = sum(summary[m0]["error_fracs"].values())
    print(f"  Sanity: R_m fraction sum for {m0} = {rm_sum:.4f}")
    print(f"  Sanity: all subspace fraction sum  = {all_sum:.4f}  (expect ≈ 1.0)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DENOISING AUTOENCODER  (Stage 2)")
print("=" * 60)
print(f"Device  : {device}")
print(f"Input   : {INPUT_PARQUET}  (103-dim x̂_m1)")
print(f"Noise   : {NOISE_TYPE}  factor={NOISE_FACTOR}  (train only)")
print()
print("Regime boundaries (exact — from merged_tensor.csv column layout):")
for name, (s, e) in REGIME_SLICES.items():
    print(f"  {name:<14} dims {s:>3}–{e-1:>3}  ({e-s} dims)")

print("\n[1/3] Loading x̂_m1 from MAE reconstructed output...")
train_ds, val_ds, full_df, feature_cols, input_dim = load_data()
print(f"  Train rows : {len(train_ds)}")
print(f"  Val rows   : {len(val_ds)}")
print(f"\n  Architecture:")
print(f"  [{input_dim}] → {ENCODER_LAYERS} → [{BOTTLENECK_DIM}] → {DECODER_LAYERS} → [{input_dim}]")

print("\n[2/3] Training...")
model = DenoisingAutoencoder(input_dim).to(device)
model.apply(init_weights)
print(f"  Total params : {sum(p.numel() for p in model.parameters()):,}")
best_state = train(model, train_ds, val_ds, device)

print("\n[3/3] Inference (clean input, no noise)...")
run_inference(model, best_state, full_df, feature_cols, device)

print("\n" + "=" * 60)
print("DONE")
print(f"  e_m (103-dim)    → {OUTPUT_DIR}/dae_error_vectors.parquet")
print(f"  regime fractions → {OUTPUT_DIR}/dae_error_fractions.json")
print(f"  Kumar Component 2 slices: e_m[0:8] political, e_m[16:24] coord,")
print(f"                            e_m[24:32] regulatory, e_m[40:48] operational")
print(f"  Next: run vae.py (also takes mae_reconstructed.parquet)")
print("=" * 60)