#!/usr/bin/env python3
"""
Flexible "head-property -> adaptation-effect" plotting script for CLIP ViT
attention heads (Layers 8-11 x Heads 0-11 = 48 heads).

================================================================================
WHAT IS PLOTTED
================================================================================
INPUTS  (predictors / head properties):
  * SPEC      : Act_Eff_Dim          <- head_spec_metrics.json   (model-fixed)
  * REL_INPUT : top_k_mean           <- new_task_rel_org_clip.json (per-DATASET)

OUTPUTS (observations / adaptation effects, per dataset x method) -- 7 total:
  CHANGE metrics (the "other 6"):
    * CKA_Linear                     <- act_change_metrics_<method>.json
    * Subspace_Alignment             <- act_change_metrics_<method>.json
    * qk_relative_frobenius          <- change_metrics_<method>.json
    * qk_weighted_subspace_overlap   <- change_metrics_<method>.json
    * ov_relative_frobenius          <- change_metrics_<method>.json
    * ov_weighted_subspace_overlap   <- change_metrics_<method>.json
  RELEVANCE metric (the "7th"):
    * top_k_mean                     <- new_task_rel_<method>.json

The 5 plot groups produced per pool:
  [1] SPEC (Act_Eff_Dim)        vs each of the 7 outputs.
  [2] REL_INPUT (orig top_k)    vs each of the 7 outputs.
  [3] SPEC & REL_INPUT JOINTLY  -> each of the 7 outputs
        (colored scatter: x=SPEC, y=REL_INPUT, color=output  +  binned heatmap).
  [4] RELEVANCE output          vs each of the 6 change metrics (corr scatters + bar).
  [5] SPEC & REL_INPUT JOINTLY  -> each of the 6 change metrics, conditioned on
        the RELEVANCE output (faceted into relevance tertiles; scatter + heatmap).

================================================================================
FLEXIBILITY (edit the USER CONFIG block below)
================================================================================
* Swap metrics : change SPEC_METRIC / REL_INPUT_METRIC / the OUTPUTS list keys.
* Pooling      : POOL_BY = any subset of ["dataset", "method", "layer"].
      []                          -> plots/general/                  (everything)
      ["method"]                  -> plots/general_method/{fft,lora4,lora16}/
      ["dataset"]                 -> plots/general_dataset/{imagenet1k,sun,text}/
      ["layer"]                   -> plots/general_layer/<layer-groups>/
      ["dataset","method"]        -> plots/general_dataset_method/<ds>_<m>/
  Unspecified dims are pooled (combined) together.
* Layer groups : LAYER_GROUPS controls how layers are split when "layer" is
      pooled, e.g. [[8,9],[10,11]] or [[8,9]] (multiple layers per group).
* Global pre-filters: INCLUDE_DATASETS / INCLUDE_METHODS / INCLUDE_LAYERS
      restrict what data is loaded at all (None = keep everything).

NOTE: head_spec_metrics (Act_Eff_Dim) is model-fixed; new_task_rel_org_clip
(REL_INPUT) is per-dataset -> the correct per-dataset file is always used for
each row, so pooling across datasets keeps each head's correct REL_INPUT.

This script only *creates* plots. Run it yourself.
"""

import json
import math
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ============================================================================
# ============================  USER CONFIG  =================================
# ============================================================================
BASE = Path("/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real")
PLOTS_ROOT = BASE / "plots"          # group folder = PLOTS_ROOT / ("general" + pooling-ext)

# ---- pooling -------------------------------------------------------------
# Subset (any order) of {"dataset", "method", "layer"}. Empty = pool everything.
POOL_BY = ["layer"]
# How layers are grouped when "layer" is in POOL_BY (each inner list = 1 group).
#   each layer separate  : [[8], [9], [10], [11]]
#   two groups           : [[8, 9], [10, 11]]
#   single chosen group  : [[8, 9]]
# LAYER_GROUPS = [[8], [9], [10], [11]]
LAYER_GROUPS = [[8, 9], [10, 11]]

# ---- global pre-filters (None = keep all) --------------------------------
INCLUDE_DATASETS = None              # e.g. ["imagenet1k", "sun"]
INCLUDE_METHODS = None               # e.g. ["fft"]
INCLUDE_LAYERS = None                # e.g. [8, 9, 10, 11]

# ---- input (predictor) metrics -------------------------------------------
SPEC_METRIC = "Act_Eff_Dim"          # key in head_spec_metrics.json
SPEC_LABEL = "Act. effective dim"
# available in head_spec_metrics.json:
#   wSCI_Score, Act_Eff_Dim, Text_Eff_Dim, Act_Entropy_bits,
#   Group_Entropy_bits, Act_Mag_Eff_Dim

REL_INPUT_METRIC = "top_k_mean"      # key in new_task_rel_org_clip.json
REL_INPUT_LABEL = "Orig-CLIP relevance (top-k mean)"
# available in new_task_rel_org_clip.json:
#   rank_mean, rank_consistency, median_disc, top_k_mean, gini

# ---- output (observation) metrics ----------------------------------------
# source: "act"    -> act_change_metrics_<method>.json
#         "weight" -> change_metrics_<method>.json   (qk_*/ov_* circuit split)
#         "rel"    -> new_task_rel_<method>.json
# kind  : "change" (one of the 6)  |  "relevance" (the 7th)
# 'col' is the internal dataframe column name (keep unique). 'label' = pretty.
#
# --- swap freely. Available keys: -----------------------------------------
#   act    : CKA_Linear, Subspace_Alignment, Principal_Angle_Deg, Eff_Dim_Orig,
#            Eff_Dim_Finetuned, Eff_Dim_Change, Total_Var_Orig,
#            Total_Var_Finetuned, Total_Var_Ratio, Spectral_Entropy_Orig_bits,
#            Spectral_Entropy_Finetuned_bits, Spectral_Entropy_Change,
#            Centroid_Shift_Cosine, Centroid_Shift_RelNorm, Became_More_Specific,
#            N_Active_Orig, N_Active_Finetuned, Top5_Group_Match_Count,
#            Top5_Group_Match_Score, Group_Dist_JS_Similarity, Group_Dist_Cosine
#   weight : <qk|ov>_relative_frobenius, _sv_correlation, _energy_ratio_change,
#            _spectral_distance, _mean_cos_angle, _weighted_subspace_overlap,
#            _effective_rank_change, _new_direction_fraction,
#            _within_subspace_fraction
#   rel    : rank_mean, rank_consistency, median_disc, top_k_mean, gini
# --------------------------------------------------------------------------
OUTPUTS = [
    dict(col="cka",        source="act",    key="CKA_Linear",
         label="Linear CKA",              kind="change"),
    dict(col="subspace",   source="act",    key="Subspace_Alignment",
         label="Subspace alignment",      kind="change"),
    dict(col="qk_frob",    source="weight", key="qk_relative_frobenius",
         label="QK rel. Frobenius",       kind="change"),
    dict(col="qk_overlap", source="weight", key="qk_weighted_subspace_overlap",
         label="QK wgt. subspace overlap", kind="change"),
    dict(col="ov_frob",    source="weight", key="ov_relative_frobenius",
         label="OV rel. Frobenius",       kind="change"),
    dict(col="ov_overlap", source="weight", key="ov_weighted_subspace_overlap",
         label="OV wgt. subspace overlap", kind="change"),
    # ---- the 7th (relevance) output; keep kind="relevance" -----------------
    dict(col="relevance",  source="rel",    key="top_k_mean",
         label="New-task relevance (top-k mean)", kind="relevance"),
]

# ---- file-name templates (per dataset folder) ----------------------------
F_ACT = "act_change_metrics_{method}.json"     # source == "act"
F_WEIGHT = "change_metrics_{method}.json"       # source == "weight"
F_REL = "new_task_rel_{method}.json"            # source == "rel" (OUTPUT relevance)
F_REL_INPUT = "new_task_rel_org_clip.json"      # REL_INPUT predictor (per dataset)
F_SPEC = "head_spec_metrics.json"               # SPEC predictor (model-fixed, top dir)

# ---- datasets / methods --------------------------------------------------
DATASETS = {"imagenet1k": "ImageNet-1k", "sun": "SUN", "text": "Synthetic Text"}
DATASET_ORDER = ["imagenet1k", "sun", "text"]
METHODS = {"fft": "FFT (full)", "lora4": "LoRA r=4", "lora16": "LoRA r=16"}
METHOD_ORDER = ["fft", "lora4", "lora16"]

# ---- styling / behaviour -------------------------------------------------
DPI = 150
POINT_SIZE = 44
POINT_ALPHA = 0.78
POINT_COLOR = "#4C72B0"
FIT_LINE = True
FIT_COLOR = "#C44E52"
CMAP_SCATTER = "viridis"      # value-colored scatters (reqs 3 & 5)
CMAP_HEATMAP = "magma"        # binned-mean heatmaps (reqs 3 & 5)
HEATMAP_BINS = 5              # target #bins per axis (auto-reduced if sparse)
REL_TERTILE_LABELS = ["Low rel.", "Med rel.", "High rel."]   # req 5 facets

PER_METRIC_FILES = False      # also dump one file per metric (reqs 1-5)
SAVE_CORR_CSV = True          # write correlations.csv per pool

sns.set_theme(style="whitegrid", context="talk")
# ============================================================================
# ==========================  END USER CONFIG  ===============================
# ============================================================================

SPEC_COL = "spec_input"
REL_INPUT_COL = "rel_input"
CHANGE_OUTPUTS = [o for o in OUTPUTS if o["kind"] == "change"]
REL_OUTPUT = next((o for o in OUTPUTS if o["kind"] == "relevance"), None)
CANON_DIMS = ["dataset", "method", "layer"]
_HEAD_RE = re.compile(r"Layer\s+(\d+)\s+Head\s+(\d+)")


# ============================================================================
# Loading
# ============================================================================
def _load_json(path):
    if not Path(path).exists():
        print(f"  WARNING missing file: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def load_master():
    """One row per (dataset, method, layer, head); SPEC fixed, REL_INPUT per
    dataset, outputs per (dataset, method)."""
    spec = _load_json(BASE / F_SPEC)
    head_names = sorted(
        [n for n in spec if _HEAD_RE.match(n)],
        key=lambda n: tuple(int(x) for x in _HEAD_RE.match(n).groups()),
    )
    if not head_names:
        raise RuntimeError(f"No 'Layer L Head H' keys found in {BASE / F_SPEC}")

    rows = []
    for ds in DATASET_ORDER:
        rel_in = _load_json(BASE / ds / F_REL_INPUT)
        for mth in METHOD_ORDER:
            src = {
                "act": _load_json(BASE / ds / F_ACT.format(method=mth)),
                "weight": _load_json(BASE / ds / F_WEIGHT.format(method=mth)),
                "rel": _load_json(BASE / ds / F_REL.format(method=mth)),
            }
            for name in head_names:
                layer, head = (int(x) for x in _HEAD_RE.match(name).groups())
                row = dict(dataset=ds, method=mth, layer=layer, head=head,
                           head_name=name)
                row[SPEC_COL] = spec.get(name, {}).get(SPEC_METRIC, np.nan)
                row[REL_INPUT_COL] = rel_in.get(name, {}).get(REL_INPUT_METRIC, np.nan)
                for o in OUTPUTS:
                    row[o["col"]] = src[o["source"]].get(name, {}).get(o["key"], np.nan)
                rows.append(row)

    df = pd.DataFrame(rows)
    if INCLUDE_DATASETS:
        df = df[df["dataset"].isin(INCLUDE_DATASETS)]
    if INCLUDE_METHODS:
        df = df[df["method"].isin(INCLUDE_METHODS)]
    if INCLUDE_LAYERS:
        df = df[df["layer"].isin(INCLUDE_LAYERS)]
    df = df.sort_values(["dataset", "method", "layer", "head"]).reset_index(drop=True)
    print(f"Loaded {len(df)} head-rows "
          f"({df[['dataset', 'method']].drop_duplicates().shape[0]} dataset x method files).")
    return df


# ============================================================================
# Pooling
# ============================================================================
def layer_label(group):
    return "L" + "-".join(str(l) for l in group)


def group_folder_name(pool_by):
    canon = [d for d in CANON_DIMS if d in pool_by]
    return "general" + ("_" + "_".join(canon) if canon else "")


def build_pools(df, pool_by):
    """Return list of (combo_label, sub_df). combo_label == '' when not pooling."""
    canon = [d for d in CANON_DIMS if d in pool_by]
    dims = []
    for d in canon:
        if d == "dataset":
            dims.append([(ds, df["dataset"] == ds) for ds in DATASET_ORDER
                         if (df["dataset"] == ds).any()])
        elif d == "method":
            dims.append([(m, df["method"] == m) for m in METHOD_ORDER
                         if (df["method"] == m).any()])
        elif d == "layer":
            dims.append([(layer_label(g), df["layer"].isin(g)) for g in LAYER_GROUPS
                         if df["layer"].isin(g).any()])
    if not dims:
        return [("", df)]
    combos = []
    for combo in product(*dims):
        labels = [c[0] for c in combo]
        mask = np.logical_and.reduce([c[1].values for c in combo])
        sub = df[mask]
        if len(sub):
            combos.append(("_".join(labels), sub.reset_index(drop=True)))
    return combos


def context_label(sub):
    ds = [d for d in DATASET_ORDER if d in set(sub["dataset"])]
    ms = [m for m in METHOD_ORDER if m in set(sub["method"])]
    ls = sorted(set(sub["layer"]))
    dpart = "all datasets" if len(ds) == len(DATASET_ORDER) else "+".join(DATASETS[d] for d in ds)
    mpart = "all methods" if len(ms) == len(METHOD_ORDER) else "+".join(METHODS[m] for m in ms)
    all_layers = sorted(set(range(8, 12)))
    lpart = "L8-11" if ls == all_layers else "L" + "-".join(map(str, ls))
    return f"{dpart} · {mpart} · {lpart}  (n={len(sub)})"


# ============================================================================
# Stats / drawing helpers
# ============================================================================
def _finite(*arrs):
    arrs = [np.asarray(a, float) for a in arrs]
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= np.isfinite(a)
    return [a[m] for a in arrs], int(m.sum())


def pearson(a, b):
    (a, b), n = _finite(a, b)
    if n < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return np.nan
    ra, rb = a[m].rank(), b[m].rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(ra.corr(rb))


def _safe_norm(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Normalize(0, 1)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        hi = lo + 1e-9
    return Normalize(lo, hi)


def scatter_fit(ax, x, y, color=POINT_COLOR):
    (xx, yy), n = _finite(x, y)
    ax.scatter(xx, yy, s=POINT_SIZE, alpha=POINT_ALPHA, color=color,
               edgecolor="white", linewidth=0.3)
    if FIT_LINE and n >= 3 and np.ptp(xx) > 0:
        b, a = np.polyfit(xx, yy, 1)
        xs = np.linspace(xx.min(), xx.max(), 50)
        ax.plot(xs, b * xs + a, color=FIT_COLOR, lw=2, alpha=0.9, zorder=5)
    r, rho = pearson(x, y), spearman(x, y)
    _annotate_corr(ax, r, rho, n)
    return r, rho, n


def _annotate_corr(ax, r, rho, n):
    ax.text(0.04, 0.96, f"r={r:.2f}  ρ={rho:.2f}\nn={n}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def binned_mean_grid(x, y, v, nbins=HEATMAP_BINS):
    """Mean of v over a quantile-binned (x, y) grid. Returns (grid, xedges,
    yedges) with grid[yi, xi]; NaN where empty. None if insufficient data."""
    (x, y, v), n = _finite(x, y, v)
    if n < 4:
        return None
    nb = max(2, min(nbins, int(round(math.sqrt(n / 2)))))
    xe = np.unique(np.nanquantile(x, np.linspace(0, 1, nb + 1)))
    ye = np.unique(np.nanquantile(y, np.linspace(0, 1, nb + 1)))
    if len(xe) < 2 or len(ye) < 2:
        return None
    nx, ny = len(xe) - 1, len(ye) - 1
    xi = np.clip(np.digitize(x, xe[1:-1]), 0, nx - 1)
    yi = np.clip(np.digitize(y, ye[1:-1]), 0, ny - 1)
    s = np.zeros((ny, nx))
    c = np.zeros((ny, nx))
    for a, b, val in zip(yi, xi, v):
        s[a, b] += val
        c[a, b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return grid, xe, ye


def draw_heatmap_cell(ax, x, y, v, cmap=CMAP_HEATMAP, norm=None):
    res = binned_mean_grid(x, y, v)
    if res is None:
        ax.text(0.5, 0.5, "insufficient\ndata", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="0.4")
        return None
    grid, xe, ye = res
    if norm is None:
        norm = _safe_norm(grid)
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid), cmap=cmap,
                         norm=norm, shading="auto", edgecolors="white", linewidth=0.4)
    return mesh


def tertile_groups(sub, col, labels=REL_TERTILE_LABELS):
    """Split sub into tertiles of `col`. Falls back to a single group when
    there is too little spread."""
    s = pd.to_numeric(sub[col], errors="coerce")
    valid = s.dropna()
    if valid.nunique() < len(labels):
        return [("all", sub)]
    grp = pd.qcut(s.rank(method="first"), len(labels), labels=labels)
    return [(lab, sub[grp == lab]) for lab in labels]


def savefig(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("   saved", p)


def _grid_dims(n, ncols):
    return math.ceil(n / ncols), ncols


# ============================================================================
# [1] & [2]  predictor -> each of the 7 outputs
# ============================================================================
def plot_predictor_vs_outputs(sub, out_dir, ctx, xcol, xlabel, fname, tag):
    outs = OUTPUTS
    nrows, ncols = _grid_dims(len(outs), 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for k, o in enumerate(outs):
        ax = axes[k]
        scatter_fit(ax, sub[xcol], sub[o["col"]])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(o["label"])
        ax.set_title(o["label"], fontsize=12)
    for k in range(len(outs), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"[{tag}] {xlabel} vs outputs  —  {ctx}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_dir, fname)

    if PER_METRIC_FILES:
        sub_dir = out_dir / f"{fname.split('.')[0]}_individual"
        for o in outs:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            scatter_fit(ax, sub[xcol], sub[o["col"]])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(o["label"])
            ax.set_title(f"{xlabel} vs {o['label']}\n{ctx}", fontsize=12)
            fig.tight_layout()
            savefig(fig, sub_dir, f"{o['col']}.png")


# ============================================================================
# [3]  SPEC & REL_INPUT jointly -> each of the 7 outputs
# ============================================================================
def _joint_scatter_cell(ax, sub, o):
    (x, y, c), _ = _finite(sub[SPEC_COL], sub[REL_INPUT_COL], sub[o["col"]])
    norm = _safe_norm(c)
    sc = ax.scatter(x, y, c=c, cmap=CMAP_SCATTER, norm=norm, s=POINT_SIZE + 12,
                    alpha=0.9, edgecolor="0.25", linewidth=0.4)
    ax.set_xlabel(SPEC_LABEL)
    ax.set_ylabel(REL_INPUT_LABEL)
    return sc


def plot_joint_inputs_vs_outputs(sub, out_dir, ctx):
    outs = OUTPUTS
    nrows, ncols = _grid_dims(len(outs), 4)

    # --- colored scatter grid ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for k, o in enumerate(outs):
        ax = axes[k]
        sc = _joint_scatter_cell(ax, sub, o)
        ax.set_title(o["label"], fontsize=12)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(o["label"], fontsize=9)
    for k in range(len(outs), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"[3] {SPEC_LABEL} & {REL_INPUT_LABEL} (color=output)  —  {ctx}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_dir, "03_joint_inputs_vs_outputs_scatter.png")

    # --- binned-mean heatmap grid ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for k, o in enumerate(outs):
        ax = axes[k]
        mesh = draw_heatmap_cell(ax, sub[SPEC_COL], sub[REL_INPUT_COL], sub[o["col"]])
        ax.set_xlabel(SPEC_LABEL)
        ax.set_ylabel(REL_INPUT_LABEL)
        ax.set_title(o["label"], fontsize=12)
        if mesh is not None:
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02).set_label(
                f"mean {o['label']}", fontsize=9)
    for k in range(len(outs), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"[3] {SPEC_LABEL} x {REL_INPUT_LABEL} -> mean output (binned)  —  {ctx}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_dir, "03_joint_inputs_vs_outputs_heatmap.png")

    if PER_METRIC_FILES:
        sub_dir = out_dir / "03_joint_inputs_vs_outputs_individual"
        for o in outs:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
            sc = _joint_scatter_cell(a1, sub, o)
            a1.set_title(f"{o['label']} (scatter)", fontsize=12)
            fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.02)
            mesh = draw_heatmap_cell(a2, sub[SPEC_COL], sub[REL_INPUT_COL], sub[o["col"]])
            a2.set_xlabel(SPEC_LABEL)
            a2.set_ylabel(REL_INPUT_LABEL)
            a2.set_title(f"{o['label']} (binned mean)", fontsize=12)
            if mesh is not None:
                fig.colorbar(mesh, ax=a2, fraction=0.046, pad=0.02)
            fig.suptitle(ctx, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            savefig(fig, sub_dir, f"{o['col']}.png")


# ============================================================================
# [4]  RELEVANCE output vs each of the 6 change metrics
# ============================================================================
def plot_relevance_vs_changes(sub, out_dir, ctx):
    if REL_OUTPUT is None:
        print("   [4] skipped: no relevance output defined.")
        return
    rcol, rlab = REL_OUTPUT["col"], REL_OUTPUT["label"]
    changes = CHANGE_OUTPUTS
    nrows, ncols = _grid_dims(len(changes), 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    stats = {}
    for k, o in enumerate(changes):
        ax = axes[k]
        r, rho, _ = scatter_fit(ax, sub[rcol], sub[o["col"]], color="#55A868")
        stats[o["label"]] = (r, rho)
        ax.set_xlabel(rlab)
        ax.set_ylabel(o["label"])
        ax.set_title(o["label"], fontsize=12)
    for k in range(len(changes), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"[4] {rlab} vs change metrics  —  {ctx}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_dir, "04_relevance_vs_change_scatter.png")

    # --- correlation bar (Pearson vs Spearman) ---
    labels = [o["label"] for o in changes]
    pear = [stats[l][0] for l in labels]
    spear = [stats[l][1] for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(labels)), 5.2))
    ax.bar(x - 0.2, pear, 0.4, label="Pearson r", color="#4C72B0")
    ax.bar(x + 0.2, spear, 0.4, label="Spearman ρ", color="#DD8452")
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("correlation with relevance")
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.set_title(f"[4] corr({rlab}, change metric)  —  {ctx}", fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "04_relevance_vs_change_corrbar.png")

    if PER_METRIC_FILES:
        sub_dir = out_dir / "04_relevance_vs_change_individual"
        for o in changes:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            scatter_fit(ax, sub[rcol], sub[o["col"]], color="#55A868")
            ax.set_xlabel(rlab)
            ax.set_ylabel(o["label"])
            ax.set_title(f"{rlab} vs {o['label']}\n{ctx}", fontsize=12)
            fig.tight_layout()
            savefig(fig, sub_dir, f"{o['col']}.png")


# ============================================================================
# [5]  SPEC & REL_INPUT jointly -> each change metric, conditioned on RELEVANCE
# ============================================================================
def plot_joint_inputs_x_relevance(sub, out_dir, ctx):
    if REL_OUTPUT is None:
        print("   [5] skipped: no relevance output defined.")
        return
    rlab = REL_OUTPUT["label"]
    changes = CHANGE_OUTPUTS
    groups = tertile_groups(sub, REL_OUTPUT["col"])
    ncols = len(groups)
    nrows = len(changes)

    # --- colored scatter: rows=change metric, cols=relevance tertile ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows),
                             squeeze=False)
    for i, o in enumerate(changes):
        norm = _safe_norm(sub[o["col"]])
        for j, (glab, gsub) in enumerate(groups):
            ax = axes[i, j]
            (x, y, c), _ = _finite(gsub[SPEC_COL], gsub[REL_INPUT_COL], gsub[o["col"]])
            ax.scatter(x, y, c=c, cmap=CMAP_SCATTER, norm=norm, s=POINT_SIZE + 10,
                       alpha=0.9, edgecolor="0.25", linewidth=0.4)
            if i == 0:
                ax.set_title(f"{glab}  (n={len(gsub)})", fontsize=12)
            if i == nrows - 1:
                ax.set_xlabel(SPEC_LABEL)
            if j == 0:
                ax.set_ylabel(REL_INPUT_LABEL, fontsize=10)
                ax.annotate(o["label"], xy=(-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=12, fontweight="bold")
        sm = ScalarMappable(norm=norm, cmap=CMAP_SCATTER)
        sm.set_array([])
        fig.colorbar(sm, ax=list(axes[i, :]), fraction=0.018, pad=0.01).set_label(
            o["label"], fontsize=9)
    fig.suptitle(f"[5] {SPEC_LABEL} & {REL_INPUT_LABEL} -> change metric, "
                 f"split by {rlab}  —  {ctx}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_dir, "05_joint_inputs_x_relevance_scatter.png")

    # --- binned-mean heatmap version ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows),
                             squeeze=False)
    for i, o in enumerate(changes):
        # shared norm across the tertile facets of this metric
        grids = []
        for _, gsub in groups:
            res = binned_mean_grid(gsub[SPEC_COL], gsub[REL_INPUT_COL], gsub[o["col"]])
            grids.append(res)
        allvals = np.concatenate([g[0][np.isfinite(g[0])] for g in grids if g is not None]) \
            if any(g is not None for g in grids) else np.array([])
        norm = _safe_norm(allvals)
        mesh = None
        for j, ((glab, gsub), res) in enumerate(zip(groups, grids)):
            ax = axes[i, j]
            if res is not None:
                grid, xe, ye = res
                mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid),
                                     cmap=CMAP_HEATMAP, norm=norm, shading="auto",
                                     edgecolors="white", linewidth=0.4)
            else:
                ax.text(0.5, 0.5, "insufficient\ndata", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="0.4")
            if i == 0:
                ax.set_title(f"{glab}  (n={len(gsub)})", fontsize=12)
            if i == nrows - 1:
                ax.set_xlabel(SPEC_LABEL)
            if j == 0:
                ax.set_ylabel(REL_INPUT_LABEL, fontsize=10)
                ax.annotate(o["label"], xy=(-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=12, fontweight="bold")
        if mesh is not None:
            fig.colorbar(mesh, ax=list(axes[i, :]), fraction=0.018, pad=0.01).set_label(
                f"mean {o['label']}", fontsize=9)
    fig.suptitle(f"[5] {SPEC_LABEL} x {REL_INPUT_LABEL} -> mean change metric (binned), "
                 f"split by {rlab}  —  {ctx}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out_dir, "05_joint_inputs_x_relevance_heatmap.png")

    if PER_METRIC_FILES:
        sub_dir = out_dir / "05_joint_inputs_x_relevance_individual"
        for o in changes:
            norm = _safe_norm(sub[o["col"]])
            fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.4),
                                     squeeze=False)
            for j, (glab, gsub) in enumerate(groups):
                ax = axes[0, j]
                (x, y, c), _ = _finite(gsub[SPEC_COL], gsub[REL_INPUT_COL], gsub[o["col"]])
                sc = ax.scatter(x, y, c=c, cmap=CMAP_SCATTER, norm=norm,
                                s=POINT_SIZE + 10, alpha=0.9, edgecolor="0.25",
                                linewidth=0.4)
                ax.set_title(f"{glab}  (n={len(gsub)})", fontsize=12)
                ax.set_xlabel(SPEC_LABEL)
                if j == 0:
                    ax.set_ylabel(REL_INPUT_LABEL, fontsize=10)
            sm = ScalarMappable(norm=norm, cmap=CMAP_SCATTER)
            sm.set_array([])
            fig.colorbar(sm, ax=list(axes[0, :]), fraction=0.02, pad=0.01).set_label(
                o["label"], fontsize=9)
            fig.suptitle(f"{o['label']} split by {rlab}  —  {ctx}", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            savefig(fig, sub_dir, f"{o['col']}.png")


# ============================================================================
# correlations.csv
# ============================================================================
def write_corr_csv(sub, out_dir):
    rows = []

    def add(predictor, plabel, target, tlabel):
        rows.append(dict(predictor=plabel, target=tlabel,
                         pearson=pearson(sub[predictor], sub[target]),
                         spearman=spearman(sub[predictor], sub[target]),
                         n=int((sub[predictor].notna() & sub[target].notna()).sum())))

    for o in OUTPUTS:
        add(SPEC_COL, SPEC_LABEL, o["col"], o["label"])
        add(REL_INPUT_COL, REL_INPUT_LABEL, o["col"], o["label"])
    if REL_OUTPUT is not None:
        for o in CHANGE_OUTPUTS:
            add(REL_OUTPUT["col"], REL_OUTPUT["label"], o["col"], o["label"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "correlations.csv", index=False)
    print("   saved", out_dir / "correlations.csv")


# ============================================================================
# Main
# ============================================================================
def main():
    df = load_master()
    group_root = PLOTS_ROOT / group_folder_name(POOL_BY)
    pools = build_pools(df, POOL_BY)
    print(f"\nGroup folder: {group_root}")
    print(f"Pooling by  : {[d for d in CANON_DIMS if d in POOL_BY] or '(none)'}"
          f"  -> {len(pools)} pool(s)\n")

    for combo_label, sub in pools:
        out_dir = group_root / combo_label if combo_label else group_root
        ctx = context_label(sub)
        print(f"== pool '{combo_label or 'ALL'}'  [{ctx}]  -> {out_dir}")

        plot_predictor_vs_outputs(sub, out_dir, ctx, SPEC_COL, SPEC_LABEL,
                                  "01_spec_vs_outputs.png", "1")
        plot_predictor_vs_outputs(sub, out_dir, ctx, REL_INPUT_COL, REL_INPUT_LABEL,
                                  "02_relinput_vs_outputs.png", "2")
        plot_joint_inputs_vs_outputs(sub, out_dir, ctx)
        plot_relevance_vs_changes(sub, out_dir, ctx)
        plot_joint_inputs_x_relevance(sub, out_dir, ctx)
        if SAVE_CORR_CSV:
            write_corr_csv(sub, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
