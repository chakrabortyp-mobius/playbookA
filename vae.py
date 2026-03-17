"""
vae.py
======
Variational Autoencoder — Stage 3 of the pipeline.

Input  : mae_outputs/mae_latents.parquet   (z_m1 from MAE — N × 32)
         *** NOT dae output — VAE always takes z_m1 from Stage 1 ***

Outputs: vae_outputs/vae_mu.parquet           (μ_m  — N × LATENT_DIM)
         vae_outputs/vae_sigma.parquet         (Σ_m  — N × LATENT_DIM, exp(log_var))
         vae_outputs/vae_z.parquet             (z    — N × LATENT_DIM, sampled at train, = μ at eval)
         vae_outputs/vae_confidence.parquet    (per-market confidence scores + CI half-widths)
         vae_outputs/vae_best_model.pth
         vae_outputs/vae_training_history.json

What it produces:
    μ_m  — mean of the latent distribution q(z|x_m)
    Σ_m  — variance of the latent distribution  (exp(log_var))
    
    These are the ONLY outputs the readout layer (Stage C) needs from this stage.

    Component 4 formula (from Playbook A, Section 3.4):
        confidence = 1 − sigmoid(k × (mean(Σ_m[dims]) − Σ_threshold))
        Σ_threshold = median variance across all markets (computed here)
        k = 5.0

    CI half-width (Section 3.5):
        CI_half_width ≈ scaling_factor × mean(Σ_m[relevant_dims])
        scaling_factor = 2.0 (default, calibrate on expert assessments)

    Architectural note:
        Stage 3 VAE takes z_m1 (32-dim) from Stage 1 MAE.
        Stage 2 DAE is independent — it takes the MAE reconstruction (103-dim).
        The three stages are NOT sequential in data terms; DAE and VAE both
        consume MAE outputs but serve different downstream consumers.

        data-poor markets (IN-LOG, NG-FIN) WILL have higher Σ_m — this is
        CORRECT BEHAVIOR. The VAE is less confident about markets with
        missing source data. Do NOT try to suppress this.

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
# CONFIG — edit here
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PARQUET = "mae_outputs/mae_latents.parquet"   # z_m1 from Stage 1 MAE
OUTPUT_DIR    = "vae_outputs"

# Architecture
# input_dim is set automatically from MAE latent dim (32)
ENCODER_LAYERS = [24, 16]      # hidden layers before mu/log_var heads
LATENT_DIM     = 16            # VAE latent dimension (z, mu, log_var)
DECODER_LAYERS = [16, 24]      # hidden layers in decoder
ENCODER_ACT    = "relu"
DECODER_ACT    = "relu"
ENCODER_DROPOUT     = 0.1
DECODER_DROPOUT     = 0.1
ENCODER_BATCH_NORM  = True
DECODER_BATCH_NORM  = True

# Noise — applied to z_m1 input BEFORE encoding (same pattern as trainer YAML)
NOISE_TYPE   = "gaussian"      # gaussian | dropout | uniform | salt_pepper
NOISE_FACTOR = 0.2             # noise intensity

# VAE loss
KL_WEIGHT          = 1.0       # beta in beta-VAE; 1.0 = standard VAE
RECON_LOSS_TYPE    = "mse"     # mse | l1 | smooth_l1 | huber
KL_ANNEALING_EPOCHS = 10       # linearly ramp KL from 0 → KL_WEIGHT over N epochs
                                # prevents posterior collapse early in training

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

# Scheduler (ReduceLROnPlateau — same as trainer YAML)
SCHEDULER_ENABLED   = True
SCHEDULER_FACTOR    = 0.5
SCHEDULER_PATIENCE  = 5
SCHEDULER_MIN_LR    = 1e-6

# Confidence modulator parameters (Section 3.4, Playbook A)
# confidence = 1 − sigmoid(k × (mean(Σ_m[dims]) − Σ_threshold))
# Σ_threshold = median variance across all markets (computed at inference)
CONFIDENCE_K          = 5.0    # sensitivity; higher = more aggressive tempering
CI_SCALING_FACTOR     = 2.0    # CI_half_width ≈ scaling_factor × mean(Σ_m)

# Init & seed
WEIGHT_INIT = "xavier_uniform"
RANDOM_SEED = 42
PRINT_EVERY = 50


# ══════════════════════════════════════════════════════════════════════════════
# REGIME SUBSPACE SLICES  (same as dae.py — for per-regime Σ_m attribution)
# These index into the ORIGINAL 103-dim tensor space, not the VAE latent space.
# Used to compute per-regime confidence scores for the readout layer.
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

# NOTE: The VAE latent space (LATENT_DIM) does NOT map directly to regime
# boundaries. Regime-level Σ_m slicing is done by proportional mapping
# from the 32-dim z_m1 input space, not from LATENT_DIM.
# For full per-regime confidence, the readout layer should use
# mean(Σ_m) globally, or map proportionally as computed below.


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_activation(name):
    return {
        "relu":       nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "elu":        nn.ELU(),
        "selu":       nn.SELU(),
        "gelu":       nn.GELU(),
        "tanh":       nn.Tanh(),
        "sigmoid":    nn.Sigmoid(),
        "swish":      nn.SiLU(),
        "mish":       nn.Mish(),
    }.get(name.lower(), nn.ReLU())


def get_recon_loss_fn(loss_type):
    return {
        "mse":       nn.MSELoss(reduction="sum"),
        "l1":        nn.L1Loss(reduction="sum"),
        "smooth_l1": nn.SmoothL1Loss(reduction="sum"),
        "huber":     nn.HuberLoss(reduction="sum", delta=1.0),
    }[loss_type.lower()]


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
# NOISE  (applied to z_m1 input before encoding — training only)
# ══════════════════════════════════════════════════════════════════════════════

def inject_noise(z: torch.Tensor, noise_type: str, noise_factor: float) -> torch.Tensor:
    """
    Corrupt z_m1 before encoding during training.
    NOT applied at validation or inference — those use clean z_m1.
    Mirrors inject_noise in trainer YAML exactly.
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
            noisy[mask] = torch.rand(n, device=z.device) * (z.max() - z.min()) + z.min()
        return noisy
    raise ValueError(f"Unknown noise_type: {noise_type}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (mirrors NoisyVAE from builder/trainer YAML exactly)
# ══════════════════════════════════════════════════════════════════════════════

class VAEEncoder(nn.Module):
    """
    Shared trunk → two separate heads: fc_mu and fc_log_var.
    Produces the distribution q(z|x_m) = N(μ_m, exp(log_var_m)).
    """
    def __init__(self, input_dim):
        super().__init__()
        prev   = input_dim
        layers = []
        for units in ENCODER_LAYERS:
            layers.append(nn.Linear(prev, units))
            if ENCODER_BATCH_NORM:
                layers.append(nn.BatchNorm1d(units))
            layers.append(get_activation(ENCODER_ACT))
            if ENCODER_DROPOUT > 0:
                layers.append(nn.Dropout(ENCODER_DROPOUT))
            prev = units
        self.shared     = nn.Sequential(*layers)
        self.fc_mu      = nn.Linear(prev, LATENT_DIM)
        self.fc_log_var = nn.Linear(prev, LATENT_DIM)

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


class VAEDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        prev   = LATENT_DIM
        layers = []
        for units in DECODER_LAYERS:
            layers.append(nn.Linear(prev, units))
            if DECODER_BATCH_NORM:
                layers.append(nn.BatchNorm1d(units))
            layers.append(get_activation(DECODER_ACT))
            if DECODER_DROPOUT > 0:
                layers.append(nn.Dropout(DECODER_DROPOUT))
            prev = units
        layers.append(nn.Linear(prev, output_dim))   # no activation on output
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NoisyVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = LATENT_DIM
        self.encoder    = VAEEncoder(input_dim)
        self.decoder    = VAEDecoder(input_dim)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = μ + ε·σ  where ε ~ N(0,I).
        At eval: returns μ deterministically (no sampling).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at inference

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var, z

    def encode(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# total = recon_loss + kl_weight × kl_loss
# KL closed-form: -0.5 × sum(1 + log_var − μ² − exp(log_var))
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
    latent_cols  = [c for c in df.columns if c.startswith("latent_")]
    input_dim    = len(latent_cols)

    assert input_dim == 32, (
        f"Expected 32-dim z_m1 from mae_latents.parquet, got {input_dim}. "
        f"Make sure you're reading mae_latents.parquet, NOT mae_reconstructed.parquet "
        f"or dae output."
    )

    print(f"  Loaded z_m1 : {df.shape[0]} rows × {input_dim} dims")
    print(f"  Markets     : {df['market'].unique().tolist()}")

    markets  = df["market"].unique().tolist()
    rng      = np.random.default_rng(RANDOM_SEED)
    n_val    = max(1, round(len(markets) * VAL_SPLIT))
    val_mkts = rng.choice(markets, size=n_val, replace=False).tolist()
    trn_mkts = [m for m in markets if m not in val_mkts]

    print(f"  Train markets : {trn_mkts}")
    print(f"  Val markets   : {val_mkts}")

    trn_df = df[df["market"].isin(trn_mkts)].reset_index(drop=True)
    val_df = df[df["market"].isin(val_mkts)].reset_index(drop=True)

    Z_trn = torch.tensor(trn_df[latent_cols].values, dtype=torch.float32)
    Z_val = torch.tensor(val_df[latent_cols].values, dtype=torch.float32)

    return (TensorDataset(Z_trn),
            TensorDataset(Z_val),
            df, latent_cols, input_dim)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    recon_loss_fn = get_recon_loss_fn(RECON_LOSS_TYPE)
    optimizer     = optim.Adam(model.parameters(),
                               lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if SCHEDULER_ENABLED:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl":   [], "val_kl":   [],
        "kl_weights": [], "lr":       [],
    }
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0
    best_state    = None

    for epoch in range(EPOCHS):
        t0 = time.time()

        # KL annealing: ramp from 0 → KL_WEIGHT over KL_ANNEALING_EPOCHS
        if KL_ANNEALING_EPOCHS > 0:
            kl_w = min(KL_WEIGHT, KL_WEIGHT * (epoch + 1) / KL_ANNEALING_EPOCHS)
        else:
            kl_w = KL_WEIGHT

        # ── Train: noisy input → reconstruct clean z_m1 ───────────────────────
        model.train()
        t_losses, t_recons, t_kls = [], [], []
        for (zb,) in train_loader:
            zb       = zb.to(device)
            zb_noisy = inject_noise(zb, NOISE_TYPE, NOISE_FACTOR)

            recon, mu, log_var, z = model(zb_noisy)
            loss, rl, kl          = vae_loss(recon, zb, mu, log_var, recon_loss_fn, kl_w)

            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            t_losses.append(loss.item())
            t_recons.append(rl.item())
            t_kls.append(kl.item())

        # ── Validate: CLEAN input, no noise ───────────────────────────────────
        # Validation uses clean z_m1 to measure true reconstruction quality.
        # This gives a meaningful early-stopping signal uncorrupted by noise.
        model.eval()
        v_losses, v_recons, v_kls = [], [], []
        with torch.no_grad():
            for (zb,) in val_loader:
                zb            = zb.to(device)
                recon, mu, log_var, z = model(zb)    # clean input, no noise
                loss, rl, kl  = vae_loss(recon, zb, mu, log_var, recon_loss_fn, kl_w)
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
                  f"kl_w={kl_w:.4f}  lr={lr_now:.2e}  "
                  f"t={time.time()-t0:.2f}s")

        if avg_val < best_val_loss - ES_MIN_DELTA:
            best_val_loss = avg_val
            best_epoch    = epoch + 1
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ Best  epoch={best_epoch}  val={best_val_loss:.6f}")
        else:
            patience_ctr += 1

        if EARLY_STOPPING and patience_ctr >= ES_PATIENCE:
            print(f"\n  Early stop at epoch {epoch+1}. "
                  f"Best={best_epoch}, val={best_val_loss:.6f}")
            break

    # Save checkpoint — model_config travels with every checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_config = {
        "input_dim":  model.input_dim,
        "latent_dim": model.latent_dim,
        "encoder": {
            "layer_sizes": ENCODER_LAYERS,
            "activation":  ENCODER_ACT,
            "dropout":     ENCODER_DROPOUT,
            "batch_norm":  ENCODER_BATCH_NORM,
        },
        "decoder": {
            "layer_sizes": DECODER_LAYERS,
            "activation":  DECODER_ACT,
            "dropout":     DECODER_DROPOUT,
            "batch_norm":  DECODER_BATCH_NORM,
        },
        "noise": {
            "type":       NOISE_TYPE,
            "factor":     NOISE_FACTOR,
            "applied_at": "train_only",
        },
        "vae": {
            "kl_weight":          KL_WEIGHT,
            "recon_loss_type":    RECON_LOSS_TYPE,
            "kl_annealing_epochs": KL_ANNEALING_EPOCHS,
        },
        "confidence": {
            "k":              CONFIDENCE_K,
            "ci_scale":       CI_SCALING_FACTOR,
            "sigma_threshold": "computed_at_inference_as_median_variance",
        },
    }
    torch.save({
        "epoch":            best_epoch,
        "model_state_dict": best_state,
        "val_loss":         best_val_loss,
        "model_config":     model_config,
        "architecture":     "NoisyVAE",
        "version":          "1.0",
    }, os.path.join(OUTPUT_DIR, "vae_best_model.pth"))

    with open(os.path.join(OUTPUT_DIR, "vae_training_history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss}, f, indent=2)

    print(f"\n  Model   → {OUTPUT_DIR}/vae_best_model.pth")
    print(f"  History → {OUTPUT_DIR}/vae_training_history.json")
    return best_state, model_config


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE MODULATOR  (Section 3.4 + 3.5 of Playbook A)
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(sigma_mean: float, sigma_threshold: float) -> float:
    """
    Corrected confidence formula from Playbook A Section 3.4:
        confidence = 1 − sigmoid(k × (mean(Σ_m) − Σ_threshold))

    When Σ_m << threshold : confidence → ~0.8–1.0  (score preserved)
    When Σ_m == threshold : confidence → 0.5        (score halved)
    When Σ_m >> threshold : confidence → ~0.0       (score zeroed — too noisy)

    Σ_threshold = median variance across all markets (computed here,
    not hardcoded — data-poor markets naturally shift the median).
    """
    return float(1.0 - torch.sigmoid(
        torch.tensor(CONFIDENCE_K * (sigma_mean - sigma_threshold))
    ).item())


def compute_ci_half_width(sigma_mean: float) -> float:
    """
    Section 3.5: CI_half_width ≈ scaling_factor × mean(Σ_m)
    Start with scaling_factor = 2.0, calibrate on expert assessments.
    """
    return CI_SCALING_FACTOR * sigma_mean


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, best_state, full_df, latent_cols, device):
    model.load_state_dict(best_state)
    model.eval()    # reparameterize returns mu deterministically
    model.to(device)

    Z_full = torch.tensor(full_df[latent_cols].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Z_full), batch_size=BATCH_SIZE, shuffle=False)

    all_mu, all_log_var, all_z = [], [], []
    with torch.no_grad():
        for (zb,) in loader:
            zb = zb.to(device)
            _, mu, log_var, z = model(zb)   # clean input, no noise at inference
            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
            all_z.append(z.cpu().numpy())

    MU      = np.vstack(all_mu)       # (N, LATENT_DIM)
    LOG_VAR = np.vstack(all_log_var)  # (N, LATENT_DIM)
    SIGMA   = np.exp(LOG_VAR)         # Σ_m = exp(log_var) — what readout layer uses
    Z       = np.vstack(all_z)        # (N, LATENT_DIM)  = MU at eval

    print(f"  μ_m   shape : {MU.shape}")
    print(f"  Σ_m   shape : {SIGMA.shape}")
    print(f"  z     shape : {Z.shape}")

    meta = {"market": full_df["market"].values,
            "year":   full_df["year"].values,
            "obs_pct": full_df["obs_pct"].values}

    def save_parquet(mat, prefix, out_name):
        cols = [f"{prefix}_{i}" for i in range(mat.shape[1])]
        df_  = pd.DataFrame(mat, columns=cols)
        df_.insert(0, "market",  meta["market"])
        df_.insert(1, "year",    meta["year"])
        df_.insert(2, "obs_pct", meta["obs_pct"])
        path = os.path.join(OUTPUT_DIR, out_name)
        df_.to_parquet(path, index=False)
        print(f"  {prefix:<8} → {path}")
        return df_

    save_parquet(MU,    "mu",    "vae_mu.parquet")
    save_parquet(SIGMA, "sigma", "vae_sigma.parquet")
    save_parquet(Z,     "z",     "vae_z.parquet")

    # ── Confidence scores per market ─────────────────────────────────────────
    # Σ_threshold = median of mean(Σ_m) across all markets
    # Computed here so it reflects the actual data distribution.
    # Data-poor markets will have higher variance → lower confidence.
    # This is CORRECT BEHAVIOR per the spec — do not suppress it.

    markets          = full_df["market"].unique()
    per_market_sigma = {}
    for mkt in markets:
        idx               = full_df[full_df["market"] == mkt].index.tolist()
        sigma_mean        = float(SIGMA[idx].mean())   # mean Σ across all dims & years
        per_market_sigma[mkt] = sigma_mean

    sigma_threshold = float(np.median(list(per_market_sigma.values())))
    print(f"\n  Σ_threshold (median across markets): {sigma_threshold:.6f}")

    # Latent space quality checks
    active_dims = int(np.sum(np.var(MU, axis=0) > 0.01))
    print(f"  Active latent dims : {active_dims}/{LATENT_DIM}  "
          f"({100*active_dims/LATENT_DIM:.1f}%)  (want > 50% for good disentanglement)")
    print(f"  Mean μ             : {MU.mean():.4f}  (want ≈ 0)")
    print(f"  Mean Σ             : {SIGMA.mean():.4f}  (want ≈ 1)")

    # Build confidence output
    rows = []
    print(f"\n  Confidence (Component 4) per market:")
    print(f"  {'Market':<10}  {'mean_Σ':>8}  {'conf':>6}  {'CI_hw':>6}  {'data_poor':>9}")
    print(f"  {'-'*50}")

    for mkt in markets:
        sigma_mean   = per_market_sigma[mkt]
        confidence   = compute_confidence(sigma_mean, sigma_threshold)
        ci_hw        = compute_ci_half_width(sigma_mean)
        is_data_poor = mkt in ["IN-LOG", "NG-FIN"]

        print(f"  {mkt:<10}  {sigma_mean:>8.4f}  {confidence:>6.3f}  "
              f"{ci_hw:>6.3f}  {'YES' if is_data_poor else 'no':>9}")

        rows.append({
            "market":             mkt,
            "mean_sigma":         sigma_mean,
            "sigma_threshold":    sigma_threshold,
            "confidence_c4":      confidence,
            "ci_half_width":      ci_hw,
            "is_data_poor":       is_data_poor,
        })

    conf_df = pd.DataFrame(rows)
    conf_df.to_parquet(os.path.join(OUTPUT_DIR, "vae_confidence.parquet"), index=False)
    print(f"\n  Confidence → {OUTPUT_DIR}/vae_confidence.parquet")

    # Save summary JSON for readout layer
    summary = {
        "sigma_threshold":  sigma_threshold,
        "confidence_k":     CONFIDENCE_K,
        "ci_scaling_factor": CI_SCALING_FACTOR,
        "active_dims":      active_dims,
        "total_dims":       LATENT_DIM,
        "active_dims_ratio": active_dims / LATENT_DIM,
        "mu_global_mean":   float(MU.mean()),
        "sigma_global_mean": float(SIGMA.mean()),
        "per_market": {
            r["market"]: {
                "mean_sigma":    r["mean_sigma"],
                "confidence_c4": r["confidence_c4"],
                "ci_half_width": r["ci_half_width"],
            }
            for r in rows
        }
    }
    with open(os.path.join(OUTPUT_DIR, "vae_confidence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary    → {OUTPUT_DIR}/vae_confidence_summary.json")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"\n  Sanity checks:")
    print(f"  • Σ_m higher for data-poor markets (IN-LOG, NG-FIN)? ", end="")
    dp_sigma  = np.mean([per_market_sigma[m] for m in ["IN-LOG", "NG-FIN"]
                          if m in per_market_sigma])
    rich_mkts = [m for m in markets if m not in ["IN-LOG", "NG-FIN"]]
    rich_sigma = np.mean([per_market_sigma[m] for m in rich_mkts])
    print(f"data-poor={dp_sigma:.4f}  data-rich={rich_sigma:.4f}  "
          f"{'✓ CORRECT' if dp_sigma > rich_sigma else '✗ CHECK — unexpected'}")
    print(f"  • KL collapse check (mean Σ ≈ 1.0 means no collapse): "
          f"{'✓ OK' if 0.5 < SIGMA.mean() < 2.0 else '✗ Possible collapse — check KL annealing'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("VARIATIONAL AUTOENCODER  (Stage 3 — Uncertainty Quantification)")
print("=" * 60)
print(f"Device        : {device}")
print(f"Input         : {INPUT_PARQUET}  (z_m1 from Stage 1 MAE, 32-dim)")
print(f"Noise         : {NOISE_TYPE}  factor={NOISE_FACTOR}  (train only)")
print(f"KL annealing  : {KL_ANNEALING_EPOCHS} epochs  →  KL_weight={KL_WEIGHT}")
print(f"Confidence    : k={CONFIDENCE_K}  CI_scale={CI_SCALING_FACTOR}")
print()
print("Outputs:")
print("  vae_mu.parquet        →  μ_m  (readout Component 4 + Playbook B)")
print("  vae_sigma.parquet     →  Σ_m  (readout Component 4 + CI widths)")
print("  vae_z.parquet         →  z    (= μ_m at eval — deterministic)")
print("  vae_confidence.parquet→  per-market confidence scores")

print("\n[1/3] Loading z_m1 from MAE...")
train_ds, val_ds, full_df, latent_cols, input_dim = load_data()
print(f"  Train rows  : {len(train_ds)}")
print(f"  Val rows    : {len(val_ds)}")
print(f"\n  Architecture:")
print(f"  [{input_dim}] → {ENCODER_LAYERS} → [μ:{LATENT_DIM}, log_var:{LATENT_DIM}]")
print(f"  [{LATENT_DIM}] → {DECODER_LAYERS} → [{input_dim}]")

print("\n[2/3] Training...")
model = NoisyVAE(input_dim).to(device)
model.apply(init_weights)
print(f"  Total params : {sum(p.numel() for p in model.parameters()):,}")
best_state, model_config = train(model, train_ds, val_ds, device)

print("\n[3/3] Inference on all rows (clean input, deterministic μ)...")
run_inference(model, best_state, full_df, latent_cols, device)

print("\n" + "=" * 60)
print("DONE — Stage 3 complete. Readout layer inputs ready:")
print("  Stage 1 → mae_outputs/mae_latents.parquet      (z_m1)")
print("  Stage 2 → dae_outputs/dae_error_vectors.parquet (e_m, 103-dim)")
print("  Stage 3 → vae_outputs/vae_mu.parquet            (μ_m)")
print("            vae_outputs/vae_sigma.parquet          (Σ_m)")
print("            vae_outputs/vae_confidence.parquet     (Component 4)")
print("=" * 60)