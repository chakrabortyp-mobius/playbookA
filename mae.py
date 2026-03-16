"""
mae.py
======
Mask-Aware Autoencoder — no CLI arguments, edit config at top.

Input  : merged_tensor_65x103.csv
Outputs: mae_outputs/mae_latents.parquet
         mae_outputs/mae_reconstructed.parquet
         mae_outputs/mae_best_model.pth
         mae_outputs/mae_training_history.json

Run:
    python3 mae.py
"""

import os
import json
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit here
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH   = "merged_tensor.csv"
OUTPUT_DIR = "mae_outputs"

# Architecture
FEATURE_DIM      = 103
LATENT_DIM       = 32
ENCODER_LAYERS   = [64, 48]       # hidden layer sizes
DECODER_LAYERS   = [48, 64]
ACTIVATION       = "relu"
DROPOUT          = 0.0
BATCH_NORM       = False

# Training
EPOCHS           = 300
LEARNING_RATE    = 1e-3
BATCH_SIZE       = 16
WEIGHT_DECAY     = 1e-4
LOSS_EPS         = 1e-6
GRAD_CLIP        = True
GRAD_CLIP_NORM   = 1.0
EARLY_STOPPING   = True
ES_PATIENCE      = 30
ES_MIN_DELTA     = 1e-6
VAL_SPLIT        = 0.0           # fraction of markets held out for validation

# Init
WEIGHT_INIT      = "xavier_uniform"
BIAS_INIT        = "zeros"
RANDOM_SEED      = 42
PRINT_EVERY      = 50             # print summary every N epochs


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_activation(name):
    return {
        "relu":       nn.ReLU(),
        "tanh":       nn.Tanh(),
        "elu":        nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid":    nn.Sigmoid(),
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
            if BIAS_INIT == "zeros":
                nn.init.zeros_(module.bias)
            elif BIAS_INIT == "ones":
                nn.init.ones_(module.bias)
            elif BIAS_INIT == "uniform":
                nn.init.uniform_(module.bias, -0.1, 0.1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (mirrors Kumar's MaskAwareAutoencoder exactly)
# ══════════════════════════════════════════════════════════════════════════════

class MAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        prev  = 2 * FEATURE_DIM        # [x ‖ mask] concatenated → 206
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

    def forward(self, x, mask):
        return self.net(torch.cat([x, mask], dim=1))


class MAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        prev  = LATENT_DIM
        layers = []
        for units in DECODER_LAYERS:
            layers.append(nn.Linear(prev, units))
            if BATCH_NORM:
                layers.append(nn.BatchNorm1d(units))
            layers.append(get_activation(ACTIVATION))
            if DROPOUT > 0:
                layers.append(nn.Dropout(DROPOUT))
            prev = units
        layers.append(nn.Linear(prev, FEATURE_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class MaskAwareAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MAEEncoder()
        self.decoder = MAEDecoder()

    def forward(self, x, mask):
        z     = self.encoder(x, mask)
        x_hat = self.decoder(z)
        return x_hat, z


# ══════════════════════════════════════════════════════════════════════════════
# LOSS  (mirrors Kumar's mae_loss exactly)
# ══════════════════════════════════════════════════════════════════════════════

def mae_loss(x_hat, x_true, mask):
    err  = torch.square(x_hat - x_true) * mask
    num  = torch.sum(err, dim=1)
    den  = torch.clamp(torch.sum(mask, dim=1), min=LOSS_EPS)
    return torch.mean(num / den)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df       = pd.read_csv(CSV_PATH)
    meta     = ["market", "year", "obs_pct"]
    dim_cols = [c for c in df.columns if c not in meta]
    assert len(dim_cols) == 103, f"Expected 103 dims, got {len(dim_cols)}"

    # Train / val split by market
    markets   = df["market"].unique().tolist()
    rng       = np.random.default_rng(RANDOM_SEED)
    n_val     = max(1, round(len(markets) * VAL_SPLIT))
    val_mkts  = rng.choice(markets, size=n_val, replace=False).tolist()
    trn_mkts  = [m for m in markets if m not in val_mkts]

    print(f"  Train markets : {trn_mkts}")
    print(f"  Val markets   : {val_mkts}")

    def to_dataset(sub_df):
        X = torch.tensor(sub_df[dim_cols].values, dtype=torch.float32)
        M = torch.ones_like(X)
        for i, (_, row) in enumerate(sub_df.iterrows()):
            if row["obs_pct"] < 100.0:
                if row["market"] in ["IN-LOG", "NG-FIN"]:
                    M[i, 24:28] = 0
                    M[i, 32:36] = 0
                if row["market"] == "NG-FIN":
                    M[i, 64:68] = 0
        return TensorDataset(X, M)

    trn_df = df[df["market"].isin(trn_mkts)].reset_index(drop=True)
    val_df = df[df["market"].isin(val_mkts)].reset_index(drop=True)

    return to_dataset(trn_df), to_dataset(val_df), df, dim_cols


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    optimizer    = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)

    history          = {"train_loss": [], "val_loss": []}
    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    best_state       = None

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for xb, mb in train_loader:
            xb, mb   = xb.to(device), mb.to(device)
            x_hat, _ = model(xb, mb)
            loss     = mae_loss(x_hat, xb, mb)
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, mb in val_loader:
                xb, mb   = xb.to(device), mb.to(device)
                x_hat, _ = model(xb, mb)
                val_losses.append(mae_loss(x_hat, xb, mb).item())

        avg_train = float(np.mean(train_losses))
        avg_val   = float(np.mean(val_losses))

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4}/{EPOCHS}  "
                  f"train={avg_train:.6f}  val={avg_val:.6f}  "
                  f"t={time.time()-t0:.2f}s")

        # Best model
        if avg_val < best_val_loss - ES_MIN_DELTA:
            best_val_loss    = avg_val
            best_epoch       = epoch + 1
            patience_counter = 0
            best_state       = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            print(f"  ✓ Best  epoch={best_epoch}  val={best_val_loss:.6f}")
        else:
            patience_counter += 1

        if EARLY_STOPPING and patience_counter >= ES_PATIENCE:
            print(f"\n  Early stop at epoch {epoch+1}. "
                  f"Best={best_epoch}, val={best_val_loss:.6f}")
            break

    # Save checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_config = {
        "feature_dim": FEATURE_DIM, "latent_dim": LATENT_DIM,
        "encoder": {"layer_sizes": ENCODER_LAYERS, "activation": ACTIVATION,
                    "dropout": DROPOUT, "batch_norm": BATCH_NORM},
        "decoder": {"layer_sizes": DECODER_LAYERS, "activation": ACTIVATION,
                    "dropout": DROPOUT, "batch_norm": BATCH_NORM},
    }
    torch.save({
        "epoch": best_epoch, "model_state_dict": best_state,
        "val_loss": best_val_loss, "model_config": model_config,
        "architecture": "MaskAwareAutoencoder", "version": "2.0",
    }, os.path.join(OUTPUT_DIR, "mae_best_model.pth"))

    with open(os.path.join(OUTPUT_DIR, "mae_training_history.json"), "w") as f:
        json.dump({**history, "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss}, f, indent=2)

    print(f"\n  Model  → {OUTPUT_DIR}/mae_best_model.pth")
    print(f"  History→ {OUTPUT_DIR}/mae_training_history.json")

    return best_state


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, best_state, full_df, dim_cols, device):
    model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    X = torch.tensor(full_df[dim_cols].values, dtype=torch.float32)
    M = torch.ones_like(X)
    for i, (_, row) in enumerate(full_df.iterrows()):
        if row["obs_pct"] < 100.0:
            if row["market"] in ["IN-LOG", "NG-FIN"]:
                M[i, 24:28] = 0
                M[i, 32:36] = 0
            if row["market"] == "NG-FIN":
                M[i, 64:68] = 0

    loader = DataLoader(TensorDataset(X, M), batch_size=BATCH_SIZE, shuffle=False)

    all_z, all_xhat = [], []
    with torch.no_grad():
        for xb, mb in loader:
            xb, mb   = xb.to(device), mb.to(device)
            x_hat, z = model(xb, mb)
            all_z.append(z.cpu().numpy())
            all_xhat.append(x_hat.cpu().numpy())

    Z    = np.vstack(all_z)
    Xhat = np.vstack(all_xhat)

    # Save latents (z_m1)
    latent_cols = [f"latent_{i}" for i in range(Z.shape[1])]
    latent_df   = pd.DataFrame(Z, columns=latent_cols)
    latent_df.insert(0, "market",  full_df["market"].values)
    latent_df.insert(1, "year",    full_df["year"].values)
    latent_df.insert(2, "obs_pct", full_df["obs_pct"].values)
    latent_df.to_parquet(os.path.join(OUTPUT_DIR, "mae_latents.parquet"), index=False)

    # Save reconstructed
    recon_df = pd.DataFrame(Xhat, columns=dim_cols)
    recon_df.insert(0, "market",  full_df["market"].values)
    recon_df.insert(1, "year",    full_df["year"].values)
    recon_df.insert(2, "obs_pct", full_df["obs_pct"].values)
    recon_df.to_parquet(os.path.join(OUTPUT_DIR, "mae_reconstructed.parquet"), index=False)

    print(f"  z_m1 → {OUTPUT_DIR}/mae_latents.parquet        shape={Z.shape}")
    print(f"  x̂    → {OUTPUT_DIR}/mae_reconstructed.parquet  shape={Xhat.shape}")

    # Per-market reconstruction error
    print(f"\n  Reconstruction error (observed dims only):")
    print(f"  {'Market':<10}  {'MSE':>10}  {'obs%':>6}")
    print(f"  {'-'*30}")
    for market in full_df["market"].unique():
        idx = full_df[full_df["market"] == market].index.tolist()
        xb  = X[idx].numpy()
        mb  = M[idx].numpy()
        xhb = Xhat[idx]
        err = np.mean(np.sum((xhb - xb)**2 * mb, axis=1) / np.sum(mb, axis=1))
        obs = full_df.loc[idx[0], "obs_pct"]
        print(f"  {market:<10}  {err:>10.6f}  {obs:>5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MASK-AWARE AUTOENCODER")
print("=" * 60)
print(f"Device      : {device}")
print(f"Feature dim : {FEATURE_DIM}")
print(f"Latent dim  : {LATENT_DIM}")
print(f"Encoder     : {[2*FEATURE_DIM]} → {ENCODER_LAYERS} → [{LATENT_DIM}]")
print(f"Decoder     : [{LATENT_DIM}] → {DECODER_LAYERS} → [{FEATURE_DIM}]")

print("\n[1/3] Loading data...")
train_ds, val_ds, full_df, dim_cols = load_data()
print(f"  Train rows : {len(train_ds)}")
print(f"  Val rows   : {len(val_ds)}")

print("\n[2/3] Training...")
model = MaskAwareAutoencoder().to(device)
model.apply(init_weights)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params : {total_params:,}")
best_state = train(model, train_ds, val_ds, device)

print("\n[3/3] Inference on all 65 rows...")
run_inference(model, best_state, full_df, dim_cols, device)

print("\n" + "=" * 60)
print("DONE — feed mae_latents.parquet into DAE next")
print("=" * 60)