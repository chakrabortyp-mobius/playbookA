"""
visualize_embeddings.py
=======================
Produces exactly 3 visualization artifacts per Jira ticket PBA-018:

    1) viz_outputs/1_tsne_umap.png
       t-SNE + UMAP side-by-side — where do the 5 L2s sit in latent space?
       India should cluster differently from Germany.

    2) viz_outputs/2_error_heatmap.png
       5 L2s × 9 regimes — which regimes drive friction per market?
       India should light up on coordination + operational.

    3) viz_outputs/3_cosine_agreement_matrix.png
       9×9 cosine sim heatmap, one subplot per L2 — do regime embeddings agree?
       India should show low measurement vs operational (IA signal).

Inputs:
    mae_outputs/mae_latents.parquet          z_m1 (N × 32)
    dae_outputs/dae_error_fractions.json     per-market regime e_m fractions
    merged_tensor.csv                        original tensor (R_m dims 0–71)

Install:
    pip install umap-learn --break-system-packages

Run:
    python3 visualize_embeddings.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "viz_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MARKET_META = {
    "DE-HC" : {"label": "DE-HC\n(Germany Healthcare)", "color": "#2196F3", "marker": "o"},
    "IN-LOG": {"label": "IN-LOG\n(India Logistics)",   "color": "#FF5722", "marker": "s"},
    "NL-AG" : {"label": "NL-AG\n(Netherlands Agri)",   "color": "#4CAF50", "marker": "^"},
    "NG-FIN": {"label": "NG-FIN\n(Nigeria Finance)",   "color": "#FF9800", "marker": "D"},
    "US-ENR": {"label": "US-ENR\n(US Energy/Carbon)",  "color": "#9C27B0", "marker": "P"},
}

REGIMES = [
    "political", "measurement", "coordination", "regulatory",
    "techprod", "operational", "narrative", "incentive", "temporal"
]

REGIME_LABELS = [
    "Political", "Measurement", "Coordination", "Regulatory",
    "Tech Prod", "Operational", "Narrative", "Incentive/Rent", "Temporal/Shock"
]

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
}

DARK_BG   = "#0F1117"
DARK_CARD = "#1E1E2E"
GRID_COL  = "#2A2A3E"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: draw one 2-D scatter panel into a given Axes
# ══════════════════════════════════════════════════════════════════════════════

def _draw_scatter(ax, Z2d, markets, title, caption):
    ax.set_facecolor(DARK_BG)
    legend_handles = []

    for mkt, meta in MARKET_META.items():
        idx = np.where(markets == mkt)[0]
        if len(idx) == 0:
            continue

        ax.scatter(Z2d[idx, 0], Z2d[idx, 1],
                   c=meta["color"], marker=meta["marker"],
                   s=130, alpha=0.85, edgecolors="white",
                   linewidths=0.5, zorder=3)

        if len(idx) >= 3:
            from matplotlib.patches import Ellipse
            cx = Z2d[idx, 0].mean()
            cy = Z2d[idx, 1].mean()
            sx = Z2d[idx, 0].std() * 2.2 + 0.4
            sy = Z2d[idx, 1].std() * 2.2 + 0.4
            ax.add_patch(Ellipse(
                (cx, cy), width=sx * 2, height=sy * 2,
                edgecolor=meta["color"], facecolor=meta["color"],
                alpha=0.10, linewidth=1.8, linestyle="--", zorder=2
            ))
            ax.annotate(mkt, (cx, cy), fontsize=10, color=meta["color"],
                        fontweight="bold", ha="center", va="center", zorder=4)

        legend_handles.append(
            mpatches.Patch(color=meta["color"], label=meta["label"])
        )

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, -0.07, caption, transform=ax.transAxes,
            fontsize=7.5, color="#AAAAAA", ha="center")
    ax.set_xlabel("Dim 1", color="#AAAAAA", fontsize=9)
    ax.set_ylabel("Dim 2", color="#AAAAAA", fontsize=9)
    ax.tick_params(colors="#AAAAAA")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(True, alpha=0.12, color=GRID_COL)

    return legend_handles


# ══════════════════════════════════════════════════════════════════════════════
# VIZ 1 — t-SNE + UMAP side-by-side (single PNG)
# ══════════════════════════════════════════════════════════════════════════════

def viz1_tsne_umap(latents_path: str) -> str:
    print("[1/3] t-SNE + UMAP...")

    df          = pd.read_parquet(latents_path)
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    Z           = df[latent_cols].values
    markets     = df["market"].values

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    perplexity = min(15, max(5, len(Z) // 5))
    Z_tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=42,
        max_iter=1500, learning_rate="auto", init="pca", metric="cosine"
    ).fit_transform(Z)

    # ── UMAP ──────────────────────────────────────────────────────────────────
    Z_umap = None
    try:
        import umap as umap_lib
        n_neighbors = min(12, max(5, len(Z) // 6))
        Z_umap = umap_lib.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.2,
            metric="cosine", random_state=42, n_epochs=500
        ).fit_transform(Z)
    except ImportError:
        print("    umap-learn not found — showing t-SNE only")
        print("    pip install umap-learn --break-system-packages")

    n_panels = 2 if Z_umap is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 8))
    fig.patch.set_facecolor(DARK_BG)
    if n_panels == 1:
        axes = [axes]

    handles = _draw_scatter(
        axes[0], Z_tsne, markets,
        "t-SNE",
        f"perplexity={perplexity}, cosine  |  "
        "Local structure only — inter-cluster distances NOT meaningful"
    )

    if Z_umap is not None:
        _draw_scatter(
            axes[1], Z_umap, markets,
            "UMAP",
            f"n_neighbors={n_neighbors}, min_dist=0.2, cosine  |  "
            "Local + global structure — inter-cluster distances meaningful"
        )

    fig.suptitle(
        "Market Positions in Latent Embedding Space (z_m1)\n"
        "Color by L2 market  |  Each point = 1 market-year  |  "
        "Clusters = structural similarity",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )
    fig.legend(handles=handles, loc="lower center", ncol=5,
               facecolor=DARK_CARD, edgecolor="#444444",
               labelcolor="white", fontsize=8.5,
               bbox_to_anchor=(0.5, -0.07), framealpha=0.92)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "1_tsne_umap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# VIZ 2 — Reconstruction error heatmap: 5 L2s × 9 regimes
# ══════════════════════════════════════════════════════════════════════════════

def viz2_error_heatmap(error_fracs_path: str) -> str:
    print("[2/3] Error heatmap...")

    with open(error_fracs_path) as f:
        data = json.load(f)

    mkts   = [m for m in MARKET_META if m in data]
    matrix = np.zeros((len(mkts), len(REGIMES)))
    for i, mkt in enumerate(mkts):
        src = data[mkt].get("rm_fracs", data[mkt].get("error_fracs", {}))
        for j, r in enumerate(REGIMES):
            matrix[i, j] = src.get(r, 0.0)

    cmap = LinearSegmentedColormap.from_list(
        "friction", [DARK_BG, "#1A237E", "#FF6F00", "#FF1744"], N=256
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=0, vmax=matrix.max())

    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels(REGIME_LABELS, rotation=35, ha="right",
                       color="white", fontsize=9)
    ax.set_yticks(range(len(mkts)))
    ax.set_yticklabels([MARKET_META[m]["label"].replace("\n", " ")
                        for m in mkts], color="white", fontsize=9)

    for i in range(len(mkts)):
        for j in range(len(REGIMES)):
            val = matrix[i, j]
            tc  = "white" if val < matrix.max() * 0.6 else "#111111"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of ||e_m||", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(
        "Structural Friction Heatmap: Which Regime Drives Friction Per Market?\n"
        "e_m decomposition by regime subspace  |  Higher = more structural friction",
        color="white", fontsize=12, fontweight="bold", pad=12
    )

    # Highlight expected patterns per domain knowledge
    highlights = {
        "IN-LOG": (["coordination", "operational"], "#00E676"),
        "NG-FIN": (["measurement", "temporal"],     "#FFD600"),
        "DE-HC" : (["regulatory"],                  "#2196F3"),
        "US-ENR": (["techprod"],                    "#9C27B0"),
    }
    legend_patches = []
    for mkt_name, (regime_list, color) in highlights.items():
        if mkt_name not in mkts:
            continue
        i = mkts.index(mkt_name)
        for r in regime_list:
            if r in REGIMES:
                j = REGIMES.index(r)
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=color, linewidth=2.5, zorder=5
                ))
        legend_patches.append(mpatches.Patch(
            facecolor="none", edgecolor=color, linewidth=2,
            label=f"{mkt_name}: expected high-friction regimes"
        ))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right",
                  facecolor=DARK_CARD, edgecolor="#444444",
                  labelcolor="white", fontsize=7.5, framealpha=0.92)

    ax.text(0.01, -0.18,
            "Source: DAE Stage 2 e_m decomposed by subspace  |  "
            "Borders = directionally expected per domain knowledge",
            transform=ax.transAxes, fontsize=7, color="#666666")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "2_error_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# VIZ 3 — Cross-regime agreement matrix: 9×9 cosine sim, one subplot per L2
# ══════════════════════════════════════════════════════════════════════════════

def viz3_cosine_matrix(tensor_csv_path: str) -> str:
    print("[3/3] Cross-regime cosine agreement matrices...")

    df       = pd.read_csv(tensor_csv_path)
    meta_c   = ["market", "year", "obs_pct"]
    dim_cols = [c for c in df.columns if c not in meta_c]
    mkts     = [m for m in MARKET_META if m in df["market"].values]

    fig, axes = plt.subplots(1, len(mkts), figsize=(4.5 * len(mkts), 5))
    fig.patch.set_facecolor(DARK_BG)
    if len(mkts) == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list(
        "agreement", ["#1565C0", DARK_BG, "#C62828"], N=256
    )

    for ax, mkt in zip(axes, mkts):
        ax.set_facecolor(DARK_BG)

        # Average regime vectors across all years for this market
        X_mean = df[df["market"] == mkt][dim_cols].values.mean(axis=0)

        # Extract each regime's 8-dim vector, unit-normalise, then cosine sim
        vecs = []
        for r in REGIMES:
            s, e = REGIME_SLICES[r]
            v    = X_mean[s:e]
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
        cos_mat = cosine_similarity(np.vstack(vecs))   # (9, 9)

        ax.imshow(cos_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        short = [lb[:5] for lb in REGIME_LABELS]
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(short, rotation=45, ha="right",
                           color="white", fontsize=7)
        ax.set_yticklabels(short, color="white", fontsize=7)

        for i in range(9):
            for j in range(9):
                if i == j:
                    continue
                val = cos_mat[i, j]
                tc  = "white" if abs(val) < 0.45 else (
                    "#111111" if val > 0 else "white"
                )
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=tc)

        # Highlight key diagnostic pairs
        pairs = []
        if mkt in ("IN-LOG", "NG-FIN"):
            # measurement(1) vs operational(5) — low = high IA signal
            pairs += [((1, 5), "#FF5722"), ((5, 1), "#FF5722")]
        if mkt == "IN-LOG":
            # coordination(2) vs operational(5) — low = high CF signal
            pairs += [((2, 5), "#FF9800"), ((5, 2), "#FF9800")]
        if mkt == "DE-HC":
            # regulatory(3) vs techprod(4) — high = RP signal
            pairs += [((3, 4), "#2196F3"), ((4, 3), "#2196F3")]

        for (i, j), col in pairs:
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor=col, linewidth=2.5, zorder=5
            ))

        ax.set_title(mkt, color=MARKET_META[mkt]["color"],
                     fontsize=11, fontweight="bold", pad=6)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.013, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Cosine Similarity", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        "Cross-Regime Agreement Matrix (9 × 9 per L2)\n"
        "Low = regime embeddings disagree = structural friction signal",
        color="white", fontsize=12, fontweight="bold", y=1.03
    )
    fig.legend(handles=[
        mpatches.Patch(facecolor="none", edgecolor="#FF5722", linewidth=2,
                       label="Measurement vs Operational — IA signal (low for IN-LOG, NG-FIN)"),
        mpatches.Patch(facecolor="none", edgecolor="#FF9800", linewidth=2,
                       label="Coordination vs Operational — CF signal (low for IN-LOG)"),
        mpatches.Patch(facecolor="none", edgecolor="#2196F3", linewidth=2,
                       label="Regulatory vs TechProd — RP signal (high for DE-HC)"),
    ], loc="lower center", ncol=3, facecolor=DARK_CARD, edgecolor="#444444",
       labelcolor="white", fontsize=7.5,
       bbox_to_anchor=(0.46, -0.10), framealpha=0.92)

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    out = os.path.join(OUTPUT_DIR, "3_cosine_agreement_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    LATENTS_PATH     = "mae_outputs/mae_latents.parquet"
    ERROR_FRACS_PATH = "dae_outputs/dae_error_fractions.json"
    TENSOR_CSV_PATH  = "merged_tensor.csv"

    missing = [p for p in [LATENTS_PATH, ERROR_FRACS_PATH, TENSOR_CSV_PATH]
               if not os.path.exists(p)]
    if missing:
        print("ERROR — missing input files:")
        for p in missing:
            print(f"  ✗  {p}")
        print("\nRun mae.py then dae.py first.")
        exit(1)

    print("=" * 60)
    print("CEO DEMO — 3 VISUALIZATION ARTIFACTS")
    print("=" * 60)

    p1 = viz1_tsne_umap(LATENTS_PATH)
    p2 = viz2_error_heatmap(ERROR_FRACS_PATH)
    p3 = viz3_cosine_matrix(TENSOR_CSV_PATH)

    print()
    print("=" * 60)
    print("DONE")
    print(f"  ✓  {p1}")
    print(f"       → 'Where do 5 L2s sit in latent space?'")
    print(f"  ✓  {p2}")
    print(f"       → 'Which regime drives friction per market?'")
    print(f"  ✓  {p3}")
    print(f"       → 'Do regime embeddings agree with each other?'")
    print()
    print("Validation checks (hand to Kumar with these confirmed):")
    print("  Viz 1: IN-LOG clusters away from DE-HC in both panels")
    print("  Viz 2: IN-LOG coordination + operational = highest fractions")
    print("  Viz 2: NG-FIN measurement + temporal = highlighted")
    print("  Viz 3: IN-LOG measurement vs operational cell = BLUE (low)")
    print("  Viz 3: DE-HC regulatory vs techprod cell = RED (high)")
    print("=" * 60)