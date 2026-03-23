"""
vae.py
======
Variational Autoencoder — Stage 3 of the pipeline.

Input  : mae_outputs/mae_reconstructed.parquet  (x̂_m1 from MAE — N × 103)

Why 103-dim (not z_m^(1)):
    Playbook A Section 3.6 requires Σ_m to be sliceable by exact regime
    boundaries so Kumar can compute Component 4 for every T1 field:
        confidence = 1 − sigmoid(k × (mean(Σ_m[regime_dims]) − Σ_threshold))
    Examples from Section 3.6 and 3.7:
        PESTLE_P  → mean(Σ_m[0:8])    political
        PESTLE_L  → mean(Σ_m[24:32])  regulatory
        IA (ALI)  → mean(Σ_m[8:16] + Σ_m[40:48])
        CF (ALI)  → mean(Σ_m[16:24])  coordination
        RP (ALI)  → mean(Σ_m[24:32])  regulatory
        F  (ALI)  → mean(Σ_m[97:103]) agency
    These slices are meaningless if Σ_m is 16-dim.

Outputs: vae_outputs/vae_mu.parquet              (μ_m  — N × 103)
         vae_outputs/vae_sigma.parquet            (Σ_m  — N × 103)
         vae_outputs/vae_z.parquet               (z    — N × 103, = μ at eval)
         vae_outputs/vae_confidence.parquet       (per-market confidence + CI)
         vae_outputs/vae_confidence_summary.json  (Σ_threshold + per-market stats)
         vae_outputs/vae_best_model.pth
         vae_outputs/vae_training_history.json

Ticket AC:
    μ_m, Σ_m for all 5 L2s                              ✓
    Σ_threshold = median(Σ) across all 5 markets         ✓
    Regime slices meaningful for Kumar's readout layer   ✓

Component 4 formula (Playbook A Section 3.4):
    confidence = 1 − sigmoid(k × (mean(Σ_m[dims]) − Σ_threshold))
    k = 5.0

CI half-width (Section 3.5):
    CI_half_width ≈ 2.0 × mean(Σ_m[relevant_dims])

Run:
    python3 vae.py
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

INPUT_PARQUET  = "mae_outputs/mae_reconstructed.parquet"   # 103-dim x̂_m1
OUTPUT_DIR     = "vae_outputs"
EXPECTED_DIM   = 103

# LATENT_DIM = 103 so μ_m and Σ_m live in 103-dim regime-indexed space.
# fc_mu and fc_log_var each output 103 dims.
# This is intentional — not a standard VAE bottleneck design.
# The goal is uncertainty quantification per regime, not compression.
LATENT_DIM     = 103

# Architecture
ENCODER_LAYERS     = [128, 64]
DECODER_LAYERS     = [64, 128]
ENCODER_ACT        = "relu"
DECODER_ACT        = "relu"
ENCODER_DROPOUT    = 0.1
DECODER_DROPOUT    = 0.1
ENCODER_BATCH_NORM = True
DECODER_BATCH_NORM = True

# Noise (train only)
NOISE_TYPE   = "gaussian"
NOISE_FACTOR = 0.2

# VAE loss
KL_WEIGHT           = 1.0
RECON_LOSS_TYPE     = "mse"
KL_ANNEALING_EPOCHS = 10   # ramp KL 0 → KL_WEIGHT to prevent posterior collapse

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

# Confidence modulator (Playbook A Section 3.4)
CONFIDENCE_K      = 5.0
CI_SCALING_FACTOR = 2.0

WEIGHT_INIT = "xavier_uniform"
RANDOM_SEED = 42
PRINT_EVERY = 50


# ══════════════════════════════════════════════════════════════════════════════
# REGIME SUBSPACE BOUNDARIES  (103-dim — same as dae.py)
# Σ_m[start:end] for each regime is meaningful because LATENT_DIM = 103.
# Verified against actual column names in merged_tensor.csv.
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
    "techprod", "operational", "narrative", "incentive", "temporal"
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_activation(name):
    return {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(), "selu": nn.SELU(), "gelu": nn.GELU(),
            "tanh": nn.Tanh(), "swish": nn.SiLU()
            }.get(name.lower(), nn.ReLU())

def get_recon_loss_fn(loss_type):
    return {"mse": nn.MSELoss(reduction="sum"),
            "l1":  nn.L1Loss(reduction="sum"),
            "huber": nn.HuberLoss(reduction="sum", delta=1.0),
            }[loss_type.lower()]

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def inject_noise(z, noise_type, noise_factor):
    if noise_factor == 0.0:
        return z
    if noise_type == "gaussian":
        return z + torch.randn_like(z) * noise_factor
    elif noise_type == "dropout":
        return z * (torch.rand_like(z) > noise_factor).float()
    elif noise_type == "uniform":
        return z + (torch.rand_like(z) - 0.5) * 2 * noise_factor
    raise ValueError(f"Unknown noise_type: {noise_type}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# Encoder: 103 → shared trunk → fc_mu (103) + fc_log_var (103)
# Decoder: 103 → 103
# LATENT_DIM = 103 = input_dim intentionally.
# ══════════════════════════════════════════════════════════════════════════════

class VAEEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        prev, layers = input_dim, []
        for units in ENCODER_LAYERS:
            layers += [nn.Linear(prev, units)]
            if ENCODER_BATCH_NORM:
                layers += [nn.BatchNorm1d(units)]
            layers += [get_activation(ENCODER_ACT)]
            if ENCODER_DROPOUT > 0:
                layers += [nn.Dropout(ENCODER_DROPOUT)]
            prev = units
        self.shared     = nn.Sequential(*layers)
        self.fc_mu      = nn.Linear(prev, LATENT_DIM)   # 103-dim
        self.fc_log_var = nn.Linear(prev, LATENT_DIM)   # 103-dim

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


class VAEDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        prev, layers = LATENT_DIM, []
        for units in DECODER_LAYERS:
            layers += [nn.Linear(prev, units)]
            if DECODER_BATCH_NORM:
                layers += [nn.BatchNorm1d(units)]
            layers += [get_activation(DECODER_ACT)]
            if DECODER_DROPOUT > 0:
                layers += [nn.Dropout(DECODER_DROPOUT)]
            prev = units
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NoisyVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = LATENT_DIM   # 103
        self.encoder    = VAEEncoder(input_dim)
        self.decoder    = VAEDecoder(input_dim)

    def reparameterize(self, mu, log_var):
        """z = μ + ε·σ at train. Returns μ at eval (deterministic)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var, z


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

def vae_loss(recon, x_clean, mu, log_var, recon_loss_fn, kl_weight):
    batch_size = x_clean.size(0)
    recon_loss = recon_loss_fn(recon, x_clean) / batch_size
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


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
    print(f"  μ_m / Σ_m will be {LATENT_DIM}-dim — regime slicing meaningful ✓")

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
    train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    recon_loss_fn = get_recon_loss_fn(RECON_LOSS_TYPE)
    optimizer     = optim.Adam(model.parameters(),
                               lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if SCHEDULER_ENABLED:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

    history = {"train_loss": [], "val_loss": [],
               "train_recon": [], "val_recon": [],
               "train_kl": [], "val_kl": [],
               "kl_weights": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0
    best_state    = None

    for epoch in range(EPOCHS):
        t0   = time.time()
        kl_w = min(KL_WEIGHT,
                   KL_WEIGHT * (epoch + 1) / KL_ANNEALING_EPOCHS) \
               if KL_ANNEALING_EPOCHS > 0 else KL_WEIGHT

        # ── Train: noisy input → reconstruct clean ────────────────────────────
        model.train()
        t_losses, t_recons, t_kls = [], [], []
        for (xb,) in train_loader:
            xb       = xb.to(device)
            xb_noisy = inject_noise(xb, NOISE_TYPE, NOISE_FACTOR)
            recon, mu, log_var, z = model(xb_noisy)
            loss, rl, kl = vae_loss(recon, xb, mu, log_var, recon_loss_fn, kl_w)
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            t_losses.append(loss.item())
            t_recons.append(rl.item())
            t_kls.append(kl.item())

        # ── Validate: clean input, no noise ───────────────────────────────────
        model.eval()
        v_losses, v_recons, v_kls = [], [], []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, mu, log_var, z = model(xb)   # clean, no noise
                loss, rl, kl = vae_loss(recon, xb, mu, log_var, recon_loss_fn, kl_w)
                v_losses.append(loss.item())
                v_recons.append(rl.item())
                v_kls.append(kl.item())

        avg_train = float(np.mean(t_losses))
        avg_val   = float(np.mean(v_losses))
        lr_now    = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_recon"].append(float(np.mean(t_recons)))
        history["val_recon"].append(float(np.mean(v_recons)))
        history["train_kl"].append(float(np.mean(t_kls)))
        history["val_kl"].append(float(np.mean(v_kls)))
        history["kl_weights"].append(kl_w)
        history["lr"].append(lr_now)

        if scheduler:
            scheduler.step(avg_val)

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4}/{EPOCHS}  "
                  f"train={avg_train:.6f}  val={avg_val:.6f}  "
                  f"kl_w={kl_w:.4f}  lr={lr_now:.2e}  t={time.time()-t0:.2f}s")

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
            "input_dim": model.input_dim,
            "latent_dim": LATENT_DIM,
            "encoder": {"layer_sizes": ENCODER_LAYERS, "activation": ENCODER_ACT,
                        "dropout": ENCODER_DROPOUT, "batch_norm": ENCODER_BATCH_NORM},
            "decoder": {"layer_sizes": DECODER_LAYERS, "activation": DECODER_ACT,
                        "dropout": DECODER_DROPOUT, "batch_norm": DECODER_BATCH_NORM},
            "noise": {"type": NOISE_TYPE, "factor": NOISE_FACTOR,
                      "applied_at": "train_only"},
            "vae": {"kl_weight": KL_WEIGHT, "recon_loss_type": RECON_LOSS_TYPE,
                    "kl_annealing_epochs": KL_ANNEALING_EPOCHS},
            "regime_slices": {k: list(v) for k, v in REGIME_SLICES.items()},
        },
        "architecture": "NoisyVAE", "version": "4.0",
    }, os.path.join(OUTPUT_DIR, "vae_best_model.pth"))

    with open(os.path.join(OUTPUT_DIR, "vae_training_history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss}, f, indent=2)

    print(f"\n  Model   → {OUTPUT_DIR}/vae_best_model.pth")
    print(f"  History → {OUTPUT_DIR}/vae_training_history.json")
    return best_state


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE (Playbook A Section 3.4)
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(sigma_mean, sigma_threshold):
    return float(1.0 - torch.sigmoid(
        torch.tensor(CONFIDENCE_K * (sigma_mean - sigma_threshold))
    ).item())


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE — clean input, deterministic μ
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, best_state, full_df, feature_cols, device):
    model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    X_full = torch.tensor(full_df[feature_cols].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_full), batch_size=BATCH_SIZE, shuffle=False)

    all_mu, all_log_var, all_z = [], [], []
    with torch.no_grad():
        for (xb,) in loader:
            _, mu, log_var, z = model(xb.to(device))   # clean, no noise
            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
            all_z.append(z.cpu().numpy())

    MU    = np.vstack(all_mu)               # (N, 103)
    SIGMA = np.exp(np.vstack(all_log_var))  # (N, 103)  Σ_m = exp(log_var)
    Z     = np.vstack(all_z)               # (N, 103)  = MU at eval

    print(f"  μ_m  shape : {MU.shape}   (103-dim — regime boundaries ✓)")
    print(f"  Σ_m  shape : {SIGMA.shape}")
    print(f"  z    shape : {Z.shape}")

    def save_parquet(mat, prefix, fname):
        df_ = pd.DataFrame(mat, columns=[f"{prefix}_{i}" for i in range(mat.shape[1])])
        df_.insert(0, "market",  full_df["market"].values)
        df_.insert(1, "year",    full_df["year"].values)
        df_.insert(2, "obs_pct", full_df["obs_pct"].values)
        df_.to_parquet(os.path.join(OUTPUT_DIR, fname), index=False)
        print(f"  {prefix:<6} → {OUTPUT_DIR}/{fname}")

    save_parquet(MU,    "mu",    "vae_mu.parquet")
    save_parquet(SIGMA, "sigma", "vae_sigma.parquet")
    save_parquet(Z,     "z",     "vae_z.parquet")

    # ── Per-regime Σ_m (for Kumar's Component 4 reference) ───────────────────
    print(f"\n  Per-regime mean Σ_m per market:")
    print(f"  {'Market':<10}", end="")
    for r in R_M_REGIMES:
        print(f"  {r[:6]:>7}", end="")
    print()
    print(f"  {'-'*85}")
    for mkt in full_df["market"].unique():
        idx = full_df[full_df["market"] == mkt].index.tolist()
        sm  = SIGMA[idx].mean(axis=0)
        print(f"  {mkt:<10}", end="")
        for r in R_M_REGIMES:
            s, e = REGIME_SLICES[r]
            print(f"  {sm[s:e].mean():>7.4f}", end="")
        print()

    # ── Σ_threshold = median(Σ) across all 5 markets  (ticket AC) ────────────
    markets          = full_df["market"].unique()
    per_market_sigma = {mkt: float(SIGMA[full_df[full_df["market"] == mkt].index].mean())
                        for mkt in markets}
    sigma_threshold  = float(np.median(list(per_market_sigma.values())))
    print(f"\n  Σ_threshold = median(Σ) across all markets: {sigma_threshold:.6f}  ✓")

    # Save Σ_threshold as its own file — ticket AC requires it as explicit output
    sigma_thresh_path = os.path.join(OUTPUT_DIR, "vae_sigma_threshold.json")
    with open(sigma_thresh_path, "w") as f:
        json.dump({
            "sigma_threshold" : sigma_threshold,
            "description"     : "median(mean_Sigma) across all 5 markets",
            "formula"         : "confidence = 1 - sigmoid(k * (mean(Sigma_m[dims]) - sigma_threshold))",
            "k"               : CONFIDENCE_K,
            "per_market_sigma": per_market_sigma,
        }, f, indent=2)
    print(f"  Σ_threshold → {sigma_thresh_path}")

    # Quality checks
    active_dims = int(np.sum(np.var(MU, axis=0) > 0.01))
    print(f"  Active dims : {active_dims}/{LATENT_DIM}  ({100*active_dims/LATENT_DIM:.0f}%)")
    print(f"  Mean μ      : {MU.mean():.4f}  (want ≈ 0)")
    print(f"  Mean Σ      : {SIGMA.mean():.4f}  (want ≈ 1)")

    # Per-market confidence
    print(f"\n  Confidence (Component 4) per market:")
    print(f"  {'Market':<10}  {'mean_Σ':>8}  {'conf':>6}  {'CI_hw':>6}  {'data_poor':>9}")
    print(f"  {'-'*52}")

    rows = []
    for mkt in markets:
        sm        = per_market_sigma[mkt]
        conf      = compute_confidence(sm, sigma_threshold)
        ci_hw     = CI_SCALING_FACTOR * sm
        data_poor = mkt in ["IN-LOG", "NG-FIN"]
        print(f"  {mkt:<10}  {sm:>8.4f}  {conf:>6.3f}  {ci_hw:>6.3f}  "
              f"{'YES' if data_poor else 'no':>9}")
        rows.append({"market": mkt, "mean_sigma": sm,
                     "sigma_threshold": sigma_threshold,
                     "confidence_c4": conf, "ci_half_width": ci_hw,
                     "is_data_poor": data_poor})

    pd.DataFrame(rows).to_parquet(
        os.path.join(OUTPUT_DIR, "vae_confidence.parquet"), index=False)

    summary = {
        "sigma_threshold"  : sigma_threshold,
        "confidence_k"     : CONFIDENCE_K,
        "ci_scaling_factor": CI_SCALING_FACTOR,
        "active_dims"      : active_dims,
        "total_dims"       : LATENT_DIM,
        "mu_global_mean"   : float(MU.mean()),
        "sigma_global_mean": float(SIGMA.mean()),
        "per_market"       : {r["market"]: {"mean_sigma": r["mean_sigma"],
                                            "confidence_c4": r["confidence_c4"],
                                            "ci_half_width": r["ci_half_width"]}
                              for r in rows},
        "regime_slices"    : {k: list(v) for k, v in REGIME_SLICES.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "vae_confidence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Confidence → {OUTPUT_DIR}/vae_confidence.parquet")
    print(f"  Summary    → {OUTPUT_DIR}/vae_confidence_summary.json")

    # Sanity checks
    print(f"\n  Sanity checks:")
    dp_mkts   = [m for m in ["IN-LOG", "NG-FIN"] if m in per_market_sigma]
    rich_mkts = [m for m in markets if m not in dp_mkts]
    if dp_mkts and rich_mkts:
        dp_s   = np.mean([per_market_sigma[m] for m in dp_mkts])
        rich_s = np.mean([per_market_sigma[m] for m in rich_mkts])
        ok     = dp_s > rich_s
        print(f"  • Data-poor Σ_m > data-rich?  "
              f"poor={dp_s:.4f}  rich={rich_s:.4f}  "
              f"{'✓ CORRECT' if ok else '✗ expected with synthetic data'}")
    kl_ok = 0.3 < SIGMA.mean() < 3.0
    print(f"  • KL collapse (mean Σ in [0.3, 3.0]):  "
          f"mean_Σ={SIGMA.mean():.4f}  {'✓ OK' if kl_ok else '✗ check KL annealing'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("VARIATIONAL AUTOENCODER  (Stage 3)")
print("=" * 60)
print(f"Device       : {device}")
print(f"Input        : {INPUT_PARQUET}  (103-dim x̂_m1)")
print(f"LATENT_DIM   : {LATENT_DIM}  (= input_dim — μ_m/Σ_m regime-sliceable)")
print(f"Noise        : {NOISE_TYPE}  factor={NOISE_FACTOR}  (train only)")
print(f"KL annealing : {KL_ANNEALING_EPOCHS} epochs → KL_weight={KL_WEIGHT}")
print(f"Confidence   : k={CONFIDENCE_K}  CI_scale={CI_SCALING_FACTOR}")
print()
print("Regime boundaries (Σ_m slicing is meaningful — matches dae.py):")
for name, (s, e) in REGIME_SLICES.items():
    print(f"  {name:<14} dims {s:>3}–{e-1:>3}  ({e-s} dims)")

print("\n[1/3] Loading x̂_m1 from MAE reconstructed output...")
train_ds, val_ds, full_df, feature_cols, input_dim = load_data()
print(f"  Train rows : {len(train_ds)}")
print(f"  Val rows   : {len(val_ds)}")
print(f"\n  Architecture:")
print(f"  [{input_dim}] → {ENCODER_LAYERS} → [μ:{LATENT_DIM}, log_var:{LATENT_DIM}]")
print(f"  [{LATENT_DIM}] → {DECODER_LAYERS} → [{input_dim}]")

print("\n[2/3] Training...")
model = NoisyVAE(input_dim).to(device)
model.apply(init_weights)
print(f"  Total params : {sum(p.numel() for p in model.parameters()):,}")
best_state = train(model, train_ds, val_ds, device)

print("\n[3/3] Inference (clean input, deterministic μ)...")
run_inference(model, best_state, full_df, feature_cols, device)

print("\n" + "=" * 60)
print("DONE — readout layer inputs ready:")
print(f"  Stage 1 → mae_outputs/mae_latents.parquet            (z_m1, 32-dim)")
print(f"  Stage 2 → dae_outputs/dae_error_vectors.parquet      (e_m,  103-dim ✓)")
print(f"            dae_outputs/dae_error_fractions.json        (per-regime fracs ✓)")
print(f"  Stage 3 → vae_outputs/vae_mu.parquet                 (μ_m,  103-dim ✓)")
print(f"            vae_outputs/vae_sigma.parquet               (Σ_m,  103-dim ✓)")
print(f"            vae_outputs/vae_sigma_threshold.json        (Σ_threshold ✓ ticket AC)")
print(f"            vae_outputs/vae_confidence.parquet          (Component 4 per market)")
print()
print("Kumar's Component 2 slices (e_m):")
print("  PESTLE_P: e_m[0:8]    PESTLE_L: e_m[24:32]  CF: e_m[16:24]")
print("  PESTLE_E: e_m[8:16]   PESTLE_T: e_m[32:40]  IA: e_m[8:16]+e_m[40:48]")
print("Kumar's Component 4 slices (Σ_m) — same boundaries")
print("=" * 60)