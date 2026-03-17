"""
dae.py
======
Denoising Autoencoder — Stage 2 of the pipeline.

Input  : mae_outputs/mae_reconstructed.parquet   (x̂ from MAE — 65 × 103)
         *** NOT mae_latents.parquet ***
         We use the reconstructed tensor (103-dim) so that e_m lives in
         the same space as the regime subspace boundaries defined in the
         original tensor. This means e_m[24:28] = regulatory dims,
         e_m[32:36] = techprod dims, etc. — required for Component 2
         (structural normality) in the readout layer.

Outputs: dae_outputs/dae_error_vectors.parquet   (e_m  — 65 × 103)
         dae_outputs/dae_best_model.pth
         dae_outputs/dae_training_history.json
         dae_outputs/dae_error_fractions.json

NOTE:    dae_latents (z_dae) is intentionally NOT saved. The VAE in
         Stage 3 takes z_m1 from MAE (mae_latents.parquet) as its input,
         NOT the DAE bottleneck. The DAE's only downstream artifact is
         e_m = x̂ − reconstruction(x̂), used by the readout layer for
         Component 2 (structural normality).

What it learns:
    Takes x̂_m1 (MAE reconstruction) + injected noise → reconstructs x̂_m1
    e_m = x̂_m1 − reconstruction  ← structural misalignment vector in 103-dim space
    Large e_m in a subspace = that regime is structurally abnormal for this market

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
# CONFIG — edit here
# ══════════════════════════════════════════════════════════════════════════════

# FIX 1: Input is mae_reconstructed (103-dim tensor space), NOT mae_latents
# This ensures e_m can be indexed by regime subspace boundaries
INPUT_PARQUET = "mae_outputs/mae_reconstructed.parquet"
OUTPUT_DIR    = "dae_outputs"

# Architecture
# input_dim is set automatically from MAE reconstructed dim (103)
ENCODER_LAYERS = [80, 48]     # hidden layers in encoder
LATENT_DIM     = 32           # DAE bottleneck
DECODER_LAYERS = [48, 80]     # hidden layers in decoder
ACTIVATION     = "relu"
DROPOUT        = 0.2
BATCH_NORM     = True

# Noise — added to x̂_m1 during training only
NOISE_TYPE     = "gaussian"   # gaussian | dropout | uniform | salt_pepper
NOISE_FACTOR   = 0.3          # std dev for gaussian; fraction for others

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

# Scheduler
SCHEDULER_ENABLED  = True
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR   = 1e-6

# Init
WEIGHT_INIT = "xavier_uniform"
RANDOM_SEED = 42
PRINT_EVERY = 50


# ══════════════════════════════════════════════════════════════════════════════
# REGIME SUBSPACE BOUNDARIES  (in 103-dim tensor space)
# These must match the column layout in merge_to_csv.py:
#   R_m:  dims   0–71  (72 dims = 9 regimes × 8 dims each)
#   C_m:  dims  72–81  (10 dims)
#   P_m:  dims  82–87  ( 6 dims)
#   D_m:  dims  88–96  ( 9 dims)
#   A_m:  dims  97–102 ( 6 dims)
#
# Within R_m, each regime occupies 8 consecutive dims (DIM_PER_REGIME=8):
#   political    :  0– 7
#   measurement  :  8–15
#   coordination : 16–23
#   regulatory   : 24–31
#   techprod     : 32–39
#   operational  : 40–47
#   narrative    : 48–55
#   incentive    : 56–63
#   temporal     : 64–71
# ══════════════════════════════════════════════════════════════════════════════

DIM_PER_REGIME = 8

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

# Regimes used for Component 2 attribution (R_m only)
R_M_REGIMES = [
    "political", "measurement", "coordination", "regulatory",
    "techprod",  "operational", "narrative",    "incentive", "temporal"
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_activation(name):
    return {
        "relu":       nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "elu":        nn.ELU(),
        "tanh":       nn.Tanh(),
        "gelu":       nn.GELU(),
    }.get(name.lower(), nn.ReLU())

def init_weights(module):
    if isinstance(module, nn.Linear):
        if WEIGHT_INIT == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif WEIGHT_INIT == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif WEIGHT_INIT == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif WEIGHT_INIT == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ══════════════════════════════════════════════════════════════════════════════
# NOISE  (training only — never at inference or validation)
# ══════════════════════════════════════════════════════════════════════════════

def add_noise(z: torch.Tensor, noise_type: str, noise_factor: float) -> torch.Tensor:
    """
    Corrupt x̂_m1 before feeding into DAE during training.
    The DAE must learn to reconstruct the clean x̂_m1 from noisy input.
    NOT applied at validation or inference — those use clean input.
    """
    if noise_factor == 0.0:
        return z

    if noise_type == "gaussian":
        return z + torch.randn_like(z) * noise_factor

    elif noise_type == "dropout":
        return z * (torch.rand_like(z) > noise_factor).float()

    elif noise_type == "uniform":
        return z + (torch.rand_like(z) - 0.5) * 2 * noise_factor

    elif noise_type == "salt_pepper":
        noisy = z.clone()
        mask  = torch.rand_like(z) < noise_factor
        n     = mask.sum()
        if n > 0:
            noisy[mask] = (torch.rand(n, device=z.device)
                           * (z.max() - z.min()) + z.min())
        return noisy

    raise ValueError(f"Unknown noise type: {noise_type}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DAEEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        prev   = input_dim
        layers = []
        for units in ENCODER_LAYERS:
            layers.append(nn.Linear(prev, units))
            if BATCH_NORM:
                layers.append(nn.BatchNorm1d(units))
            layers.append(get_activation(ACTIVATION))
            if DROPOUT > 0:
                layers.append(nn.Dropout(DROPOUT))
            prev = units
        layers.append(nn.Linear(prev, LATENT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DAEDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        prev   = LATENT_DIM
        layers = []
        for units in DECODER_LAYERS:
            layers.append(nn.Linear(prev, units))
            if BATCH_NORM:
                layers.append(nn.BatchNorm1d(units))
            layers.append(get_activation(ACTIVATION))
            if DROPOUT > 0:
                layers.append(nn.Dropout(DROPOUT))
            prev = units
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = LATENT_DIM
        self.encoder    = DAEEncoder(input_dim)
        self.decoder    = DAEDecoder(input_dim)

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_parquet(INPUT_PARQUET)

    # Feature columns: all except meta columns
    meta_cols    = ["market", "year", "obs_pct"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    input_dim    = len(feature_cols)

    assert input_dim == 103, (
        f"Expected 103 feature dims from mae_reconstructed.parquet, got {input_dim}. "
        f"Make sure you're using mae_reconstructed.parquet, NOT mae_latents.parquet."
    )

    print(f"  Loaded x̂_m1 : {df.shape[0]} rows × {input_dim} dims  (103-dim tensor space)")
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

    return (TensorDataset(X_trn),
            TensorDataset(X_val),
            df, feature_cols, input_dim)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
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
        train_losses = []
        for (xb,) in train_loader:
            xb       = xb.to(device)
            xb_noisy = add_noise(xb, NOISE_TYPE, NOISE_FACTOR)  # corrupt input
            x_hat, _ = model(xb_noisy)
            loss     = criterion(x_hat, xb)                     # target = CLEAN

            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate: CLEAN input → clean target (FIX 2: no noise at val) ────
        # Val loss measures clean reconstruction quality, not noisy reconstruction.
        # This gives a meaningful early-stopping signal.
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb       = xb.to(device)
                x_hat, _ = model(xb)              # clean input, no noise
                val_losses.append(criterion(x_hat, xb).item())

        avg_train = float(np.mean(train_losses))
        avg_val   = float(np.mean(val_losses))
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
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            print(f"  ✓ Best  epoch={best_epoch}  val={best_val_loss:.6f}")
        else:
            patience_ctr += 1

        if EARLY_STOPPING and patience_ctr >= ES_PATIENCE:
            print(f"\n  Early stop at epoch {epoch+1}. "
                  f"Best={best_epoch}, val={best_val_loss:.6f}")
            break

    # Save checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_config = {
        "input_dim":     model.input_dim,
        "latent_dim":    LATENT_DIM,
        "encoder_dims":  ENCODER_LAYERS,
        "decoder_dims":  DECODER_LAYERS,
        "activation":    ACTIVATION,
        "dropout":       DROPOUT,
        "batch_norm":    BATCH_NORM,
        "noise": {
            "type":         NOISE_TYPE,
            "factor":       NOISE_FACTOR,
            "applied_at":   "train_only",   # explicitly documented
        },
    }
    torch.save({
        "epoch":            best_epoch,
        "model_state_dict": best_state,
        "val_loss":         best_val_loss,
        "model_config":     model_config,
        "architecture":     "DenoisingAutoencoder",
        "version":          "2.0",
    }, os.path.join(OUTPUT_DIR, "dae_best_model.pth"))

    with open(os.path.join(OUTPUT_DIR, "dae_training_history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss}, f, indent=2)

    print(f"\n  Model   → {OUTPUT_DIR}/dae_best_model.pth")
    print(f"  History → {OUTPUT_DIR}/dae_training_history.json")
    return best_state, model_config


# ══════════════════════════════════════════════════════════════════════════════
# ERROR FRACTIONS  (Component 2 input for readout layer)
# ══════════════════════════════════════════════════════════════════════════════

def compute_error_fractions(e_m: np.ndarray) -> dict:
    """
    Per-regime contribution to ||e_m|| using EXACT subspace boundaries.

    Uses REGIME_SLICES defined at top of file — boundaries correspond to
    the actual regime dim layout in the 103-dim tensor. This replaces the
    previous equal-chunk approximation which was architecturally incorrect
    (32 dims / 9 regimes = unequal, unattributed trailing dims).

    Formula per Playbook A Component 2:
        structural_normality(regime) = 1 − (||e_m[regime_dims]|| / ||e_m||)

    Returns fraction of total ||e_m|| attributable to each subspace.
    All fractions sum to ≤ 1.0 (may be < 1 due to floating point, not < 0).
    """
    total_norm = float(np.linalg.norm(e_m)) + 1e-8
    fracs = {}
    for name, (start, end) in REGIME_SLICES.items():
        slice_norm     = float(np.linalg.norm(e_m[start:end]))
        fracs[name]    = slice_norm / total_norm
    return fracs


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE — compute e_m for all rows, clean input only
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
            xb       = xb.to(device)
            x_hat, _ = model(xb)           # clean input, no noise at inference
            all_x_hat.append(x_hat.cpu().numpy())

    X_hat = np.vstack(all_x_hat)           # (N, 103) reconstruction
    X_np  = X_full.numpy()                 # (N, 103) original x̂_m1

    # e_m = x̂_m1 − reconstruction(x̂_m1)   shape: (N, 103)
    # Lives in 103-dim space → regime subspace boundaries are meaningful
    E_m = X_np - X_hat

    print(f"  e_m shape : {E_m.shape}  (103-dim, regime boundaries preserved)")

    # ── Save error vectors ────────────────────────────────────────────────────
    e_cols = [f"e_{i}" for i in range(E_m.shape[1])]
    e_df   = pd.DataFrame(E_m, columns=e_cols)
    e_df.insert(0, "market",  full_df["market"].values)
    e_df.insert(1, "year",    full_df["year"].values)
    e_df.insert(2, "obs_pct", full_df["obs_pct"].values)
    e_df.to_parquet(os.path.join(OUTPUT_DIR, "dae_error_vectors.parquet"), index=False)
    print(f"  e_m → {OUTPUT_DIR}/dae_error_vectors.parquet")

    # ── Per-market summary ────────────────────────────────────────────────────
    print(f"\n  Structural misalignment (e_m) per market:")
    print(f"  {'Market':<10}  {'||e_m||':>9}  {'top regime':>14}  {'frac':>6}  {'2nd regime':>14}  {'frac':>6}")
    print(f"  {'-'*66}")

    summary = {}
    for market in full_df["market"].unique():
        idx    = full_df[full_df["market"] == market].index.tolist()
        e_mean = E_m[idx].mean(axis=0)          # avg e_m across years for this market
        norm   = float(np.linalg.norm(e_mean))
        fracs  = compute_error_fractions(e_mean)

        # Rank by R_m regimes only for interpretability
        rm_fracs = {k: v for k, v in fracs.items() if k in R_M_REGIMES}
        ranked   = sorted(rm_fracs.items(), key=lambda x: x[1], reverse=True)
        top1     = ranked[0] if len(ranked) > 0 else ("—", 0.0)
        top2     = ranked[1] if len(ranked) > 1 else ("—", 0.0)

        print(f"  {market:<10}  {norm:>9.4f}  {top1[0]:>14}  {top1[1]:>6.3f}  {top2[0]:>14}  {top2[1]:>6.3f}")

        summary[market] = {
            "e_m_norm"    : norm,
            "error_fracs" : fracs,          # all subspaces
            "rm_fracs"    : rm_fracs,       # R_m regimes only
            "top_regime"  : top1[0],
            "top_frac"    : top1[1],
        }

    # ── Save error fractions JSON for readout layer ───────────────────────────
    with open(os.path.join(OUTPUT_DIR, "dae_error_fractions.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Error fractions → {OUTPUT_DIR}/dae_error_fractions.json")

    # ── Sanity check: fractions should sum to ~1.0 ───────────────────────────
    market0      = list(summary.keys())[0]
    frac_sum     = sum(summary[market0]["error_fracs"].values())
    print(f"\n  Sanity check — error fraction sum for {market0}: {frac_sum:.4f}  (expect ≈ 1.0)")


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
print(f"Device      : {device}")
print(f"Input       : {INPUT_PARQUET}  (103-dim reconstructed tensor)")
print(f"Noise       : {NOISE_TYPE}  factor={NOISE_FACTOR}  (train only)")
print()
print("Regime subspace boundaries (for e_m attribution):")
for name, (s, e) in REGIME_SLICES.items():
    print(f"  {name:<14} dims {s:>3}–{e-1:>3}  ({e-s} dims)")

print("\n[1/3] Loading x̂_m1 from MAE reconstructed output...")
train_ds, val_ds, full_df, feature_cols, input_dim = load_data()
print(f"  Train rows  : {len(train_ds)}")
print(f"  Val rows    : {len(val_ds)}")
print(f"\n  Architecture:")
print(f"  [{input_dim}] → {ENCODER_LAYERS} → [{LATENT_DIM}] → {DECODER_LAYERS} → [{input_dim}]")

print("\n[2/3] Training...")
model = DenoisingAutoencoder(input_dim).to(device)
model.apply(init_weights)
print(f"  Total params : {sum(p.numel() for p in model.parameters()):,}")
best_state, model_config = train(model, train_ds, val_ds, device)

print("\n[3/3] Inference on all rows (clean input, no noise)...")
run_inference(model, best_state, full_df, feature_cols, device)

print("\n" + "=" * 60)
print("DONE")
print("  → dae_error_vectors.parquet   feeds readout layer (Component 2)")
print("  → dae_error_fractions.json    per-market regime attribution")
print("  → feed mae_latents.parquet    (NOT dae output) into VAE next")
print("=" * 60)