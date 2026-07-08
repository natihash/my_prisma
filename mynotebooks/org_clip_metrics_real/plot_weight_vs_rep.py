#!/usr/bin/env python3
"""
Compare WEIGHT-based change metrics against REPRESENTATION (activation)-based
change metrics for CLIP ViT attention heads during adaptation, across:
    * 3 target datasets  : ImageNet-1k, SUN-395, Synthetic Text
    * 3 adaptation methods: FFT (full finetune), LoRA-r4, LoRA-r16

Two metric families, both stored per head (Layers 8-11 x Heads 0-11 = 48 heads):
  * WEIGHT  : change_metrics_<method>.json      (qk_* / ov_* circuit split)
  * REP/ACT : act_change_metrics_<method>.json  (single value per head)

Selected metrics (swap freely — see the *_METRICS lists below):
  WEIGHT : relative_frobenius, weighted_subspace_overlap   (x qk & ov)
  REP    : CKA_Linear, Subspace_Alignment

Plots produced:
  1. General weight<->rep correlation scatters for the 4 (weight x rep) pairs,
     QK and OV overlaid.
  2. The same joint relations, faceted per adaptation METHOD and per DATASET.
  3. Composite "weight change" group (from the 2 weight metrics) and composite
     "rep change" group (from the 2 rep metrics); how heads distribute over
     these groups — in total, per method, per dataset — plus the joint
     weight-group x rep-group contingency.
  4. Extra views: full metric correlation matrix, composite agreement scatter,
     weight<->rep coupling bars per method/dataset.

Styling (palette, theme, savefig helper, violin/box idiom) is borrowed from
plot_new_task_relevance.py / plot_weight_change.py for visual consistency.

NOTE: this only *creates* the plots; nothing runs on import beyond main().
"""

import json
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Config
# ============================================================================
BASE = Path("/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real")
OUT = BASE / "plots" / "weight_vs_rep"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# METRIC SELECTION
# ----------------------------------------------------------------------------
# WEIGHT metrics are stored per circuit with a qk_/ov_ prefix; list the BASE
# names (no prefix), each expands to its qk_* and ov_* version.
#
# ---- AVAILABLE WEIGHT BASE METRICS (copy/paste into WEIGHT_METRICS) -------
#   "relative_frobenius", "sv_correlation", "energy_ratio_change",
#   "spectral_distance", "mean_cos_angle", "weighted_subspace_overlap",
#   "effective_rank_change", "new_direction_fraction", "within_subspace_fraction",
# ---- AVAILABLE REP/ACT METRICS (copy/paste into REP_METRICS) --------------
#   "CKA_Linear", "Subspace_Alignment", "Principal_Angle_Deg", "Eff_Dim_Orig",
#   "Eff_Dim_Finetuned", "Eff_Dim_Change", "Total_Var_Orig", "Total_Var_Finetuned",
#   "Total_Var_Ratio", "Spectral_Entropy_Orig_bits", "Spectral_Entropy_Finetuned_bits",
#   "Spectral_Entropy_Change", "Centroid_Shift_Cosine", "Centroid_Shift_RelNorm",
#   "Became_More_Specific", "N_Active_Orig", "N_Active_Finetuned",
#   "Top5_Group_Match_Count", "Top5_Group_Match_Score", "Group_Dist_JS_Similarity",
#   "Group_Dist_Cosine",
# --------------------------------------------------------------------------
WEIGHT_METRICS = ["relative_frobenius", "weighted_subspace_overlap"]
REP_METRICS    = ["CKA_Linear", "Subspace_Alignment"]

CIRCUITS = ["qk", "ov"]
CIRCUIT_LABELS = {"qk": "QK", "ov": "OV"}
CIRCUIT_PALETTE = {"qk": "#4C72B0", "ov": "#DD8452"}

# Pretty labels (fallback = cleaned name).
METRIC_LABELS = {
    # weight
    "relative_frobenius":        "Relative Frobenius",
    "weighted_subspace_overlap": "Wgt. subspace overlap",
    "mean_cos_angle":            "Mean cosine angle",
    "new_direction_fraction":    "New-direction fraction",
    "sv_correlation":            "SV correlation",
    "energy_ratio_change":       "Energy-ratio change",
    "spectral_distance":         "Spectral distance",
    "effective_rank_change":     "Eff.-rank change",
    "within_subspace_fraction":  "Within-subspace frac.",
    # rep
    "CKA_Linear":                "Linear CKA",
    "Subspace_Alignment":        "Subspace alignment",
    "Total_Var_Ratio":           "Total-variance ratio",
    "Centroid_Shift_RelNorm":    "Centroid shift (rel-norm)",
    "Principal_Angle_Deg":       "Principal angle (deg)",
    "Spectral_Entropy_Change":   "Spectral-entropy change",
}

# Orientation for the COMPOSITE change score: +1 => higher value means MORE
# change; -1 => higher value means LESS change (so it is flipped before pooling).
METRIC_DIRECTION = {
    "relative_frobenius":        +1,   # large Frobenius delta = big change
    "weighted_subspace_overlap": -1,   # high overlap = little change
    "CKA_Linear":                -1,   # high CKA = representations preserved
    "Subspace_Alignment":        -1,   # high alignment = little change
    # add directions here if you swap in other metrics
}

# ----------------------------------------------------------------------------
# Grouping config (composite change -> ordinal groups via tertiles)
# ----------------------------------------------------------------------------
GROUP_LABELS = ["Low", "Medium", "High"]          # increasing change
N_GROUPS = len(GROUP_LABELS)
GROUP_PALETTE = {"Low": "#4575B4", "Medium": "#FEE090", "High": "#D73027"}

# ----------------------------------------------------------------------------
# DATASETS / METHODS (palettes borrowed from plot_new_task_relevance.py)
# ----------------------------------------------------------------------------
TASKS = {"imagenet1k": "ImageNet-1k", "sun": "SUN-395", "text": "Synthetic Text"}
TASK_ORDER = ["imagenet1k", "sun", "text"]
TASK_PALETTE = {"imagenet1k": "#4C72B0", "sun": "#DD8452", "text": "#55A868"}

METHODS = {"fft": "FFT (full)", "lora4": "LoRA r=4", "lora16": "LoRA r=16"}
METHOD_ORDER = ["fft", "lora4", "lora16"]
METHOD_PALETTE = {"fft": "#C44E52", "lora4": "#8172B3", "lora16": "#937860"}

sns.set_theme(style="whitegrid", context="talk")
_pat = re.compile(r"Layer\s+(\d+)\s+Head\s+(\d+)")

# Derived helpers
WEIGHT_COLS = [f"{c}_{w}" for c in CIRCUITS for w in WEIGHT_METRICS]
PAIRS = list(product(WEIGHT_METRICS, REP_METRICS))   # (weight_base, rep_metric)


def mlabel(m):
    return METRIC_LABELS.get(m, m.replace("_", " "))


def wlabel(circuit, w):
    return f"{CIRCUIT_LABELS[circuit]} {mlabel(w)}"


def savefig(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", p)


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(a[m], b[m])[0, 1]


# ============================================================================
# Load + merge -> one wide row per (task, method, layer, head)
# ============================================================================
def _load_json(path):
    if not path.exists():
        print(f"  WARNING missing file: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def load_all():
    rows = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            wj = _load_json(BASE / task / f"change_metrics_{method}.json")
            aj = _load_json(BASE / task / f"act_change_metrics_{method}.json")
            names = set(wj) | set(aj)
            for name in names:
                mobj = _pat.match(name)
                if not mobj:
                    continue
                layer, head = map(int, mobj.groups())
                row = dict(task=task, task_label=TASKS[task],
                           method=method, method_label=METHODS[method],
                           layer=layer, head=head)
                wm = wj.get(name, {})
                for c in CIRCUITS:
                    for w in WEIGHT_METRICS:
                        row[f"{c}_{w}"] = wm.get(f"{c}_{w}", np.nan)
                am = aj.get(name, {})
                for r in REP_METRICS:
                    row[r] = am.get(r, np.nan)
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(["task", "method", "layer", "head"])
    print(f"Loaded {len(df)} head-rows "
          f"({df[['task','method']].drop_duplicates().shape[0]} task x method files).")
    return df.reset_index(drop=True)


# ============================================================================
# Composite change scores + ordinal groups
# ============================================================================
def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def add_groups(df):
    """Add composite change scores and Low/Medium/High groups.
    - weight_score_<circuit> : mean oriented-z of the weight metrics (per circuit)
    - rep_score              : mean oriented-z of the rep metrics
    Groups are tertiles (rank-based qcut so edges are always unique)."""
    def tertile(score):
        ranks = score.rank(method="first")
        return pd.qcut(ranks, N_GROUPS, labels=GROUP_LABELS)

    for c in CIRCUITS:
        comp = np.mean([METRIC_DIRECTION[w] * _z(df[f"{c}_{w}"])
                        for w in WEIGHT_METRICS], axis=0)
        df[f"weight_score_{c}"] = comp
        df[f"weight_group_{c}"] = tertile(pd.Series(comp, index=df.index))

    rep = np.mean([METRIC_DIRECTION[r] * _z(df[r]) for r in REP_METRICS], axis=0)
    df["rep_score"] = rep
    df["rep_group"] = tertile(pd.Series(rep, index=df.index))
    return df


# ============================================================================
# 1. GENERAL CORRELATION SCATTERS  (4 weight x rep pairs, QK & OV overlaid)
# ============================================================================
def plot_pair_scatters(df):
    nrows, ncols = len(WEIGHT_METRICS), len(REP_METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 6.5 * nrows),
                             squeeze=False)
    for i, w in enumerate(WEIGHT_METRICS):
        for j, r in enumerate(REP_METRICS):
            ax = axes[i, j]
            txt = []
            for c in CIRCUITS:
                xcol = f"{c}_{w}"
                ax.scatter(df[xcol], df[r], s=40, alpha=0.6,
                           color=CIRCUIT_PALETTE[c], edgecolor="white", lw=0.3,
                           label=CIRCUIT_LABELS[c])
                txt.append(f"{CIRCUIT_LABELS[c]} r={pearson(df[xcol], df[r]):.2f}")
            ax.set_xlabel(f"weight: {mlabel(w)}")
            ax.set_ylabel(f"rep: {mlabel(r)}")
            ax.set_title("  |  ".join(txt), fontsize=13)
            ax.legend(fontsize=10, title="Circuit")
    fig.suptitle("Weight-change vs representation-change — all pairs (QK & OV)",
                 y=1.005, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "01_pair_scatters_qk_ov.png")


# ============================================================================
# 2. JOINT RELATIONS PER METHOD / PER DATASET
# ============================================================================
def _facet_scatter(df, facet_key, facet_order, facet_label_map,
                   facet_palette, fname, suptitle):
    """rows = (weight x rep) pairs, cols = facet level. QK & OV overlaid."""
    nrows, ncols = len(PAIRS), len(facet_order)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows),
                             squeeze=False)
    for i, (w, r) in enumerate(PAIRS):
        for j, lvl in enumerate(facet_order):
            ax = axes[i, j]
            sub = df[df[facet_key] == lvl]
            txt = []
            for c in CIRCUITS:
                xcol = f"{c}_{w}"
                ax.scatter(sub[xcol], sub[r], s=32, alpha=0.6,
                           color=CIRCUIT_PALETTE[c], edgecolor="white", lw=0.3,
                           label=CIRCUIT_LABELS[c])
                txt.append(f"{CIRCUIT_LABELS[c]} r={pearson(sub[xcol], sub[r]):.2f}")
            if i == 0:
                ax.set_title(facet_label_map[lvl], fontsize=14,
                             color=facet_palette[lvl], fontweight="bold")
            ax.set_xlabel(f"w:{mlabel(w)}", fontsize=11)
            ax.set_ylabel(f"r:{mlabel(r)}", fontsize=11)
            ax.text(0.04, 0.96, "\n".join(txt), transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0))
            if i == 0 and j == 0:
                ax.legend(fontsize=9, title="Circuit", loc="lower right")
    fig.suptitle(suptitle, y=1.002, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fname)


def plot_joint_by_method(df):
    _facet_scatter(df, "method", METHOD_ORDER, METHODS, METHOD_PALETTE,
                   "02a_joint_by_method.png",
                   "Weight vs rep change per adaptation METHOD (datasets pooled)")


def plot_joint_by_dataset(df):
    _facet_scatter(df, "task", TASK_ORDER, TASKS, TASK_PALETTE,
                   "02b_joint_by_dataset.png",
                   "Weight vs rep change per DATASET (methods pooled)")


# ============================================================================
# 3. COMPOSITE GROUPS + DISTRIBUTIONS
# ============================================================================
def _stacked_fraction(ax, ct, title):
    """ct: index = facet levels (already pretty), columns = GROUP_LABELS."""
    ct = ct.reindex(columns=GROUP_LABELS).fillna(0)
    x = np.arange(len(ct.index))
    bottom = np.zeros(len(ct.index))
    for g in GROUP_LABELS:
        ax.bar(x, ct[g].values, bottom=bottom, color=GROUP_PALETTE[g],
               edgecolor="white", lw=0.6, label=g)
        bottom += ct[g].values
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index, rotation=15, ha="right")
    ax.set_ylabel("fraction of heads")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13)


def plot_group_distribution_total(df):
    """Total head counts per group: weight (qk), weight (ov), rep."""
    series = [(f"weight_group_{c}", f"Weight change — {CIRCUIT_LABELS[c]}")
              for c in CIRCUITS] + [("rep_group", "Representation change")]
    fig, axes = plt.subplots(1, len(series), figsize=(6 * len(series), 6),
                             squeeze=False)
    for ax, (col, title) in zip(axes[0], series):
        counts = df[col].value_counts().reindex(GROUP_LABELS).fillna(0)
        ax.bar(GROUP_LABELS, counts.values,
               color=[GROUP_PALETTE[g] for g in GROUP_LABELS],
               edgecolor="black", lw=0.5)
        for x, v in enumerate(counts.values):
            ax.text(x, v, f"{int(v)}", ha="center", va="bottom", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("# heads")
    fig.suptitle("Head distribution over change groups (total, all heads)",
                 y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03a_group_distribution_total.png")


def plot_group_distribution_by(df, facet_key, facet_order, facet_map, fname,
                               suptitle):
    """Stacked-fraction bars of each grouping across a facet (method/dataset)."""
    series = [(f"weight_group_{c}", f"Weight — {CIRCUIT_LABELS[c]}")
              for c in CIRCUITS] + [("rep_group", "Representation")]
    fig, axes = plt.subplots(1, len(series), figsize=(6 * len(series), 6),
                             squeeze=False)
    for ax, (col, title) in zip(axes[0], series):
        ct = pd.crosstab(df[facet_key], df[col], normalize="index")
        ct.index = [facet_map[i] for i in ct.index]
        ct = ct.reindex([facet_map[k] for k in facet_order])
        _stacked_fraction(ax, ct, title)
    axes[0][-1].legend(title="Group", fontsize=10, bbox_to_anchor=(1.02, 1),
                       loc="upper left")
    fig.suptitle(suptitle, y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fname)


def plot_contingency(df, fname, suptitle, facet_key=None, facet_order=None,
                     facet_map=None, normalize=False):
    """weight-group x rep-group contingency heatmaps.
    Rows = circuits; cols = facet levels (or a single 'all' col)."""
    levels = facet_order if facet_key else ["__all__"]
    fig, axes = plt.subplots(len(CIRCUITS), len(levels),
                             figsize=(4.6 * len(levels), 4.4 * len(CIRCUITS)),
                             squeeze=False)
    for i, c in enumerate(CIRCUITS):
        for j, lvl in enumerate(levels):
            sub = df if lvl == "__all__" else df[df[facet_key] == lvl]
            ct = pd.crosstab(sub[f"weight_group_{c}"], sub["rep_group"])
            ct = ct.reindex(index=GROUP_LABELS, columns=GROUP_LABELS).fillna(0)
            fmt = ".0f"
            if normalize:
                ct = ct.div(ct.sum().sum())
                fmt = ".2f"
            ax = axes[i, j]
            sns.heatmap(ct, ax=ax, cmap="magma", annot=True, fmt=fmt,
                        annot_kws={"size": 11}, linewidths=0.5,
                        linecolor="white", cbar_kws={"label": "count"})
            ttl = CIRCUIT_LABELS[c] if lvl == "__all__" else \
                f"{CIRCUIT_LABELS[c]} — {facet_map[lvl]}"
            ax.set_title(ttl, fontsize=12)
            ax.set_xlabel("rep group")
            ax.set_ylabel(f"{CIRCUIT_LABELS[c]} weight group")
    fig.suptitle(suptitle, y=1.005, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fname)


def plot_repgroup_within_weightgroup(df):
    """Composition view: within each weight-change group, what fraction of heads
    fall into each rep-change group? (one panel per circuit)."""
    fig, axes = plt.subplots(1, len(CIRCUITS), figsize=(7 * len(CIRCUITS), 6),
                             squeeze=False)
    for ax, c in zip(axes[0], CIRCUITS):
        ct = pd.crosstab(df[f"weight_group_{c}"], df["rep_group"],
                         normalize="index").reindex(GROUP_LABELS)
        _stacked_fraction(ax, ct, f"{CIRCUIT_LABELS[c]} weight group")
        ax.set_xlabel(f"{CIRCUIT_LABELS[c]} weight-change group")
    axes[0][-1].legend(title="rep group", fontsize=10,
                       bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Representation-change composition within each weight-change group",
                 y=1.02, fontsize=17, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03f_repgroup_within_weightgroup.png")


# ============================================================================
# 4. EXTRA VIEWS
# ============================================================================
def plot_metric_correlation_matrix(df):
    cols = WEIGHT_COLS + REP_METRICS
    pretty = ([wlabel(c, w) for c in CIRCUITS for w in WEIGHT_METRICS] +
              [mlabel(r) for r in REP_METRICS])
    corr = df[cols].corr()
    corr.index = pretty
    corr.columns = pretty
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 4, 1.0 * len(cols) + 3))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 9},
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix: weight & representation change metrics",
                 fontsize=15)
    # visually separate weight block from rep block
    ax.axhline(len(WEIGHT_COLS), color="k", lw=2)
    ax.axvline(len(WEIGHT_COLS), color="k", lw=2)
    fig.tight_layout()
    savefig(fig, "04a_metric_correlation_matrix.png")


def plot_composite_agreement(df):
    """Composite weight-change score vs composite rep-change score, colored by
    method. One panel per circuit; quadrant lines at 0 (z-scored means)."""
    fig, axes = plt.subplots(1, len(CIRCUITS), figsize=(8 * len(CIRCUITS), 7),
                             squeeze=False)
    for ax, c in zip(axes[0], CIRCUITS):
        for method in METHOD_ORDER:
            sub = df[df.method == method]
            ax.scatter(sub[f"weight_score_{c}"], sub["rep_score"], s=45,
                       alpha=0.65, color=METHOD_PALETTE[method],
                       edgecolor="white", lw=0.3, label=METHODS[method])
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.axvline(0, color="grey", ls="--", lw=1)
        r = pearson(df[f"weight_score_{c}"], df["rep_score"])
        ax.set_title(f"{CIRCUIT_LABELS[c]} circuit   (overall r = {r:.2f})",
                     fontsize=14)
        ax.set_xlabel("composite WEIGHT change  (higher = more change)")
        ax.set_ylabel("composite REP change  (higher = more change)")
        ax.legend(fontsize=10, title="Method")
    fig.suptitle("Do heads that change in weight also change in representation?",
                 y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "04b_composite_agreement.png")


def plot_coupling_bars(df):
    """Weight<->rep composite coupling (Pearson r) per method and per dataset,
    bars split by circuit."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    specs = [("method", METHOD_ORDER, METHODS, axes[0], "Adaptation method"),
             ("task", TASK_ORDER, TASKS, axes[1], "Dataset")]
    for key, order, lab, ax, title in specs:
        x = np.arange(len(order))
        w = 0.38
        for i, c in enumerate(CIRCUITS):
            vals = [pearson(df[df[key] == lvl][f"weight_score_{c}"],
                            df[df[key] == lvl]["rep_score"]) for lvl in order]
            ax.bar(x + (i - 0.5) * w, vals, w, color=CIRCUIT_PALETTE[c],
                   edgecolor="black", lw=0.4, alpha=0.9, label=CIRCUIT_LABELS[c])
        ax.axhline(0, color="grey", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lab[k] for k in order], rotation=15, ha="right")
        ax.set_ylabel("weight–rep composite r")
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10, title="Circuit")
    fig.suptitle("Coupling between weight change and representation change",
                 y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "04c_coupling_bars.png")


# ============================================================================
# 5. LAYERWISE ANALYSIS  (pooled per layer; and per layer x method)
# ============================================================================
def plot_layerwise_scores(df):
    """Mean composite change score per LAYER, pooling over everything
    (all methods + all datasets). Weight score per circuit + rep score."""
    layers = sorted(df["layer"].unique())
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in CIRCUITS:
        g = df.groupby("layer")[f"weight_score_{c}"].mean().reindex(layers)
        ax.plot(layers, g.values, "-o", color=CIRCUIT_PALETTE[c], lw=2.5, ms=10,
                label=f"weight ({CIRCUIT_LABELS[c]})")
    g = df.groupby("layer")["rep_score"].mean().reindex(layers)
    ax.plot(layers, g.values, "-s", color="#55A868", lw=2.5, ms=10,
            label="representation")
    ax.axhline(0, color="grey", ls="--", lw=1, label="global mean = 0")
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("mean composite change score\n(higher = more change)")
    ax.set_title("Layerwise change magnitude (pooled over methods & datasets)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    savefig(fig, "05a_layerwise_scores_pooled.png")


def plot_layerwise_scores_by_method(df):
    """Mean composite change score per LAYER x METHOD (datasets pooled).
    One panel per score (weight-QK, weight-OV, rep); a line per method."""
    layers = sorted(df["layer"].unique())
    scores = [(f"weight_score_{c}", f"Weight change — {CIRCUIT_LABELS[c]}")
              for c in CIRCUITS] + [("rep_score", "Representation change")]
    fig, axes = plt.subplots(1, len(scores), figsize=(6.2 * len(scores), 6),
                             squeeze=False, sharey=True)
    for ax, (col, title) in zip(axes[0], scores):
        for method in METHOD_ORDER:
            g = (df[df.method == method].groupby("layer")[col].mean()
                 .reindex(layers))
            ax.plot(layers, g.values, "-o", color=METHOD_PALETTE[method],
                    lw=2.4, ms=8, label=METHODS[method])
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10, title="Method")
    axes[0][0].set_ylabel("mean composite change score")
    fig.suptitle("Layerwise change magnitude per method (datasets pooled)",
                 y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "05b_layerwise_scores_by_method.png")


def plot_layerwise_coupling(df):
    """Weight<->rep composite coupling (Pearson r) per LAYER, pooling over
    everything. One line per circuit."""
    layers = sorted(df["layer"].unique())
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in CIRCUITS:
        rs = [pearson(df[df.layer == L][f"weight_score_{c}"],
                      df[df.layer == L]["rep_score"]) for L in layers]
        ax.plot(layers, rs, "-o", color=CIRCUIT_PALETTE[c], lw=2.5, ms=10,
                label=CIRCUIT_LABELS[c])
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(layers)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("weight–rep composite r")
    ax.set_title("Layerwise weight–representation coupling\n"
                 "(pooled over methods & datasets)")
    ax.legend(fontsize=11, title="Circuit")
    fig.tight_layout()
    savefig(fig, "05c_layerwise_coupling_pooled.png")


def plot_layerwise_coupling_by_method(df):
    """Weight<->rep composite coupling (Pearson r) per LAYER x METHOD
    (datasets pooled). One panel per circuit; a line per method."""
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, len(CIRCUITS), figsize=(7.5 * len(CIRCUITS), 6.5),
                             squeeze=False, sharey=True)
    for ax, c in zip(axes[0], CIRCUITS):
        for method in METHOD_ORDER:
            sub = df[df.method == method]
            rs = [pearson(sub[sub.layer == L][f"weight_score_{c}"],
                          sub[sub.layer == L]["rep_score"]) for L in layers]
            ax.plot(layers, rs, "-o", color=METHOD_PALETTE[method],
                    lw=2.4, ms=8, label=METHODS[method])
        ax.axhline(0, color="grey", lw=1)
        ax.set_xticks(layers)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Layer")
        ax.set_title(f"{CIRCUIT_LABELS[c]} circuit", fontsize=13)
        ax.legend(fontsize=10, title="Method")
    axes[0][0].set_ylabel("weight–rep composite r")
    fig.suptitle("Layerwise weight–representation coupling per method "
                 "(datasets pooled)", y=1.02, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "05d_layerwise_coupling_by_method.png")


def _layer_palette(layers):
    cmap = plt.cm.viridis(np.linspace(0.12, 0.78, len(layers)))
    return {L: cmap[i] for i, L in enumerate(layers)}


def plot_joint_by_layer(df):
    """The weight x rep pair scatters, faceted PER LAYER (methods & datasets
    pooled). Shows how weight and rep metrics relate within each layer, with a
    per-circuit Pearson r in every panel."""
    layers = sorted(df["layer"].unique())
    label_map = {L: f"Layer {L}" for L in layers}
    _facet_scatter(df, "layer", layers, label_map, _layer_palette(layers),
                   "05e_joint_by_layer.png",
                   "Weight vs rep change per LAYER (methods & datasets pooled)")


def plot_layerwise_corr_matrix(df):
    """Full metric correlation matrix computed SEPARATELY per layer — how every
    weight & rep metric relates to the others within a given layer."""
    cols = WEIGHT_COLS + REP_METRICS
    pretty = ([wlabel(c, w) for c in CIRCUITS for w in WEIGHT_METRICS] +
              [mlabel(r) for r in REP_METRICS])
    layers = sorted(df["layer"].unique())
    ncols = min(2, len(layers))
    nrows = int(np.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=((1.0 * len(cols) + 3) * ncols,
                                      (0.95 * len(cols) + 2.5) * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[len(layers):]:
        ax.set_visible(False)
    for ax, L in zip(axes_flat, layers):
        corr = df[df.layer == L][cols].corr()
        corr.index = pretty
        corr.columns = pretty
        sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": 8},
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Pearson r"})
        ax.axhline(len(WEIGHT_COLS), color="k", lw=2)
        ax.axvline(len(WEIGHT_COLS), color="k", lw=2)
        ax.set_title(f"Layer {L}", fontsize=14, fontweight="bold")
    fig.suptitle("Per-layer correlation of weight & representation metrics\n"
                 "(methods & datasets pooled)",
                 y=1.005, fontsize=18, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "05f_layerwise_corr_matrix.png")


# ============================================================================
# Summary tables
# ============================================================================
def write_summary(df):
    df.to_csv(OUT / "weight_vs_rep_tidy.csv", index=False)
    print("  saved", OUT / "weight_vs_rep_tidy.csv")
    # group cross-tabs per circuit (overall)
    for c in CIRCUITS:
        ct = pd.crosstab(df[f"weight_group_{c}"], df["rep_group"])
        ct.reindex(index=GROUP_LABELS, columns=GROUP_LABELS).to_csv(
            OUT / f"contingency_{c}_overall.csv")
        print("  saved", OUT / f"contingency_{c}_overall.csv")


# ============================================================================
def main():
    df = load_all()
    if df.empty:
        print("No data loaded — check paths / metric lists.")
        return
    df = add_groups(df)

    print("\n[1] general weight<->rep correlation scatters")
    plot_pair_scatters(df)

    print("[2] joint relations per method / per dataset")
    plot_joint_by_method(df)
    plot_joint_by_dataset(df)

    print("[3] composite groups + distributions")
    plot_group_distribution_total(df)
    plot_group_distribution_by(df, "method", METHOD_ORDER, METHODS,
                               "03b_group_distribution_by_method.png",
                               "Head distribution over change groups, per METHOD")
    plot_group_distribution_by(df, "task", TASK_ORDER, TASKS,
                               "03c_group_distribution_by_dataset.png",
                               "Head distribution over change groups, per DATASET")
    plot_contingency(df, "03d_contingency_overall.png",
                     "Weight-group x rep-group contingency (total)")
    plot_contingency(df, "03e_contingency_by_method.png",
                     "Weight-group x rep-group contingency, per METHOD",
                     facet_key="method", facet_order=METHOD_ORDER,
                     facet_map=METHODS)
    plot_repgroup_within_weightgroup(df)

    print("[4] extra views")
    plot_metric_correlation_matrix(df)
    plot_composite_agreement(df)
    plot_coupling_bars(df)

    print("[5] layerwise analysis")
    plot_layerwise_scores(df)
    plot_layerwise_scores_by_method(df)
    plot_layerwise_coupling(df)
    plot_layerwise_coupling_by_method(df)
    plot_joint_by_layer(df)
    plot_layerwise_corr_matrix(df)

    write_summary(df)
    print("\nAll figures + CSVs written to:", OUT)


if __name__ == "__main__":
    main()
