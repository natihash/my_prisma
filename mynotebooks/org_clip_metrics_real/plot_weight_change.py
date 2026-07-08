#!/usr/bin/env python3
"""
Analyze how the QK and OV circuits of CLIP ViT attention heads CHANGE during
adaptation, across:
    * 3 target datasets  : ImageNet-1k, SUN-395, Synthetic Text
    * 3 adaptation methods: FFT (full finetune), LoRA-r4, LoRA-r16

Each `change_metrics_<method>.json` lives under <dataset>/ and stores, for every
one of the 48 heads (Layers 8-11 x Heads 0-11), a battery of weight-change
metrics for the QK and the OV circuit separately (qk_* and ov_* keys).

The script builds a single tidy dataframe and emits plots covering:
    1. QK<->OV correlation of the change (per metric + a general overview)
    2. change metric vs adaptation METHOD
    3. change metric vs adaptation DATASET
    4. layerwise analysis (line trends + Layer x Head heatmaps)

Plot styling (palette, theme, savefig helper, violin/box/strip idiom) is
borrowed from plot_new_task_relevance.py to keep the figure set visually
consistent.

NOTE: this only *creates* the plots; nothing is run on import beyond main().
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# ============================================================================
# LaTeX integration  -- EDIT THESE to match your document.
# ============================================================================
# A figure included with \includegraphics[width=c\textwidth]{...} is rescaled by
# LaTeX by (c*\textwidth)/(figure width); if the figure was authored wider than
# the slot, every label/tick/legend shrinks by that factor (so text fine at
# \textwidth becomes unreadable at 0.5\textwidth).  Fix: author each figure at a
# physical width of exactly c*\textwidth and set fonts to the point size you want
# ON the page -> LaTeX's scale factor is 1 and text keeps that size for any c.
# Set LATEX_WIDTH_FRAC to the same c you use in \includegraphics, set
# TARGET_FONT_PT to the on-page size, and include with width=<c>\textwidth.
TEXTWIDTH_PT = 345.0                            # \the\textwidth of your doc, in pt
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27             # TeX pt -> inch
LATEX_WIDTH_FRAC = 0.7                           # the c in width=c\textwidth
TARGET_FONT_PT = 9.0                             # on-page text size in pt
DPI = 300


def setup(width_frac=None, aspect=0.62, base=None):
    r"""Configure rcParams so the figure is authored at its on-page size: its
    physical width becomes width_frac*\textwidth, so LaTeX does not rescale it at
    \includegraphics[width=width_frac\textwidth] and fonts keep their point size.
    """
    if width_frac is None:
        width_frac = LATEX_WIDTH_FRAC
    if base is None:
        base = TARGET_FONT_PT
    w = width_frac * TEXTWIDTH_IN
    h = w * aspect
    plt.rcParams.update({
        "figure.figsize": (w, h),
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        # constrained_layout fits labels/titles/colorbars/legends INSIDE the
        # canvas, so nothing is clipped and the saved width stays exactly
        # width_frac*textwidth (so don't save with bbox_inches='tight', and don't
        # call tight_layout()).
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        "font.family": "serif",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "figure.titlesize": base + 1,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })
    return w, h


def wrap(text, width=24):
    """Hard-wrap a string to <= `width` chars per line so long titles/labels/
    legend entries break onto new lines instead of overflowing a panel or the
    plotted data.  Hyphenated words and long tokens are kept intact."""
    if not text:
        return text
    return textwrap.fill(str(text), width=max(4, int(width)),
                         break_long_words=False, break_on_hyphens=False)


def chars_per_line(ncols=1, frac=0.85, width_frac=None, base=None):
    """Estimate how many characters fit on one line of a single panel, so
    `wrap()` adapts to the figure width and number of columns."""
    if width_frac is None:
        width_frac = LATEX_WIDTH_FRAC
    if base is None:
        base = TARGET_FONT_PT
    panel_in = (width_frac * TEXTWIDTH_IN) / max(1, ncols) * frac
    avg_glyph_pt = 0.60 * base            # conservative serif glyph advance
    return max(8, int(panel_in * 72.27 / avg_glyph_pt))


def ann_size(scale=1.0, base=None):
    """Font size (pt) for in-plot annotations, relative to the base size."""
    if base is None:
        base = TARGET_FONT_PT
    return max(3.5, base * scale)


# ============================================================================
# Config
# ============================================================================
BASE = Path("/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real")
OUT = BASE / "plots" / "weighg_change"   # (spelling kept as requested)
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# METRIC SELECTION
# ----------------------------------------------------------------------------
# Metrics are stored per circuit with a `qk_` / `ov_` prefix. Below we list the
# BASE names (no prefix). Each base name automatically expands to its qk_* and
# ov_* version, so picking N base metrics gives 2*N plotted series.
#
# ---- FULL LIST OF AVAILABLE BASE METRICS (copy/paste any of these into
#      SELECTED_METRICS to change what gets plotted) -----------------------
#   "relative_frobenius",
#   "sv_correlation",
#   "energy_ratio_change",
#   "spectral_distance",
#   "mean_cos_angle",
#   "weighted_subspace_overlap",
#   "effective_rank_change",
#   "new_direction_fraction",
#   "within_subspace_fraction",
# --------------------------------------------------------------------------
#
# The four metrics requested (expanded to qk_* + ov_* = 8 series total):
SELECTED_METRICS = [
    "relative_frobenius",
    "mean_cos_angle",
    "weighted_subspace_overlap",
    "new_direction_fraction",
]

# Pretty labels for any base metric you might select. Falls back to a
# title-cased name if a metric is missing here.
METRIC_LABELS = {
    "relative_frobenius":        "Relative Frobenius change",
    "sv_correlation":            "Singular-value correlation",
    "energy_ratio_change":       "Energy-ratio change",
    "spectral_distance":         "Spectral distance",
    "mean_cos_angle":            "Mean cosine angle",
    "weighted_subspace_overlap": "Weighted subspace overlap",
    "effective_rank_change":     "Effective-rank change",
    "new_direction_fraction":    "New-direction fraction",
    "within_subspace_fraction":  "Within-subspace fraction",
}

CIRCUITS = ["qk", "ov"]
CIRCUIT_LABELS = {"qk": "QK", "ov": "OV"}

# ----------------------------------------------------------------------------
# DATASETS  (palette + ordering borrowed from plot_new_task_relevance.py)
# ----------------------------------------------------------------------------
TASKS = {
    "imagenet1k": "ImageNet-1k",
    "sun":        "SUN-395",
    "text":       "Synthetic Text",
}
TASK_ORDER = ["imagenet1k", "sun", "text"]
TASK_PALETTE = {"imagenet1k": "#4C72B0", "sun": "#DD8452", "text": "#55A868"}

# ----------------------------------------------------------------------------
# ADAPTATION METHODS  (file suffix -> pretty label, palette)
# ----------------------------------------------------------------------------
METHODS = {
    "fft":    "FFT (full)",
    "lora4":  "LoRA r=4",
    "lora16": "LoRA r=16",
}
METHOD_ORDER = ["fft", "lora4", "lora16"]
METHOD_PALETTE = {"fft": "#C44E52", "lora4": "#8172B3", "lora16": "#937860"}

# Per-circuit accent colors (used when overlaying QK vs OV).
CIRCUIT_PALETTE = {"qk": "#4C72B0", "ov": "#DD8452"}

sns.set_theme(style="whitegrid")

_pat = re.compile(r"Layer\s+(\d+)\s+Head\s+(\d+)")


def mlabel(base):
    return METRIC_LABELS.get(base, base.replace("_", " ").title())


def savefig(fig, name):
    p = OUT / name
    # No bbox_inches='tight': constrained_layout already fits everything inside
    # the canvas, and keeping the figure size preserves the exact
    # width_frac*textwidth width so LaTeX does not rescale (shrink) the text.
    fig.savefig(p)
    plt.close(fig)
    print("  saved", p)


# ============================================================================
# Load -> tidy (long) dataframe
# rows: one per (task, method, layer, head, circuit, metric)
# ============================================================================
def load_all():
    rows = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            path = BASE / task / f"change_metrics_{method}.json"
            if not path.exists():
                print(f"  WARNING missing file: {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            for name, m in data.items():
                layer, head = map(int, _pat.match(name).groups())
                for circuit in CIRCUITS:
                    for base in SELECTED_METRICS:
                        key = f"{circuit}_{base}"
                        if key not in m:
                            continue
                        rows.append(dict(
                            task=task, task_label=TASKS[task],
                            method=method, method_label=METHODS[method],
                            layer=layer, head=head,
                            circuit=circuit, metric=base,
                            value=m[key],
                        ))
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows "
          f"({df[['task','method']].drop_duplicates().shape[0]} task x method files, "
          f"{df['metric'].nunique()} metrics x {len(CIRCUITS)} circuits).")
    return df


def qk_ov_wide(df, metric):
    """Return a wide frame with `qk` and `ov` columns for one base metric,
    indexed implicitly by (task, method, layer, head)."""
    sub = df[df.metric == metric]
    wide = sub.pivot_table(
        index=["task", "task_label", "method", "method_label", "layer", "head"],
        columns="circuit", values="value").reset_index()
    return wide


# ============================================================================
# 1. QK <-> OV CORRELATION
# ============================================================================
def plot_qk_ov_scatter_per_metric(df):
    """One subplot per selected metric: per-head QK change vs OV change,
    colored by adaptation method. Annotates Pearson r + identity line."""
    n = len(SELECTED_METRICS)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    setup(aspect=(nrows / ncols) * 0.85)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    cw = chars_per_line(ncols)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax, metric in zip(axes_flat, SELECTED_METRICS):
        wide = qk_ov_wide(df, metric)
        for method in METHOD_ORDER:
            s = wide[wide.method == method]
            ax.scatter(s["qk"], s["ov"], s=18, alpha=0.65,
                       color=METHOD_PALETTE[method], edgecolor="white", lw=0.4,
                       label=METHODS[method])
        # identity line over the shared data range
        lo = float(np.nanmin([wide["qk"].min(), wide["ov"].min()]))
        hi = float(np.nanmax([wide["qk"].max(), wide["ov"].max()]))
        ax.plot([lo, hi], [lo, hi], ls="--", c="grey", lw=1.2, zorder=0)
        r = wide[["qk", "ov"]].corr().iloc[0, 1]
        ax.set_title(wrap(f"{mlabel(metric)} (Pearson r = {r:.3f})", cw))
        ax.set_xlabel(wrap(f"QK {mlabel(metric)}", cw))
        ax.set_ylabel(wrap(f"OV {mlabel(metric)}", cw))
        ax.legend(title="Method", loc="best", framealpha=0.7,
                  fontsize=ann_size(0.8))
    fig.suptitle(wrap("QK vs OV weight change per head (all datasets & methods)",
                      chars_per_line(1)), fontweight="bold")
    savefig(fig, "01a_qk_ov_scatter_per_metric.png")


def plot_qk_ov_correlation_overview(df):
    """General overview: Pearson r between QK and OV change, per metric,
    broken down by method and overall (pooled)."""
    records = []
    for metric in SELECTED_METRICS:
        wide = qk_ov_wide(df, metric)
        records.append(dict(metric=metric, group="overall",
                            r=wide[["qk", "ov"]].corr().iloc[0, 1]))
        for method in METHOD_ORDER:
            s = wide[wide.method == method]
            records.append(dict(metric=metric, group=METHODS[method],
                                r=s[["qk", "ov"]].corr().iloc[0, 1]))
    cdf = pd.DataFrame(records)

    groups = ["overall"] + [METHODS[m] for m in METHOD_ORDER]
    group_colors = {"overall": "#333333",
                    **{METHODS[m]: METHOD_PALETTE[m] for m in METHOD_ORDER}}
    x = np.arange(len(SELECTED_METRICS))
    w = 0.8 / len(groups)

    setup(aspect=0.6)
    fig, ax = plt.subplots()
    for i, g in enumerate(groups):
        vals = [cdf[(cdf.metric == mtr) & (cdf.group == g)]["r"].values[0]
                for mtr in SELECTED_METRICS]
        ax.bar(x + i * w, vals, w, color=group_colors[g], label=g,
               edgecolor="black", lw=0.4, alpha=0.9)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([wrap(mlabel(m), 16) for m in SELECTED_METRICS],
                       rotation=20, ha="right")
    ax.axhline(0, color="grey", lw=1)
    ax.set_ylabel(wrap("QK–OV Pearson r", chars_per_line(1)))
    ax.set_title(wrap("How coupled are QK and OV changes? (per metric)",
                      chars_per_line(1)))
    ax.legend(title="Group", ncol=2, loc="best", framealpha=0.7,
              fontsize=ann_size(0.8))
    savefig(fig, "01b_qk_ov_correlation_overview.png")


def plot_metric_cross_correlation(df):
    """General cross-metric correlation heatmap of all 8 (qk/ov x metric)
    change series, pooled over every head/task/method."""
    wide = df.pivot_table(
        index=["task", "method", "layer", "head"],
        columns=["circuit", "metric"], values="value")
    wide.columns = [f"{CIRCUIT_LABELS[c]}·{mlabel(m)}" for c, m in wide.columns]
    # keep a sensible column order: all QK then all OV
    order = ([f"QK·{mlabel(m)}" for m in SELECTED_METRICS] +
             [f"OV·{mlabel(m)}" for m in SELECTED_METRICS])
    order = [c for c in order if c in wide.columns]
    corr = wide[order].corr()

    setup(aspect=0.92)
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": ann_size(0.8)},
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Pearson r"})
    ax.set_xticklabels([wrap(c, 16) for c in corr.columns],
                       rotation=30, ha="right")
    ax.set_yticklabels([wrap(c, 16) for c in corr.columns], rotation=0)
    ax.tick_params(labelsize=ann_size(0.75))
    ax.set_title(wrap("Cross-correlation of all change metrics (QK & OV, pooled)",
                      chars_per_line(1)))
    savefig(fig, "01c_metric_cross_correlation.png")


# ============================================================================
# 2. CHANGE METRIC vs ADAPTATION METHOD
# ============================================================================
def plot_metric_vs_method(df):
    """For each selected metric, distribution of the change across methods,
    split by circuit (QK / OV). Violin + box + per-group mean diamond."""
    for metric in SELECTED_METRICS:
        sub = df[df.metric == metric]
        setup(aspect=0.7)
        fig, ax = plt.subplots()
        sns.violinplot(data=sub, x="method", y="value", hue="circuit",
                       order=METHOD_ORDER, hue_order=CIRCUITS,
                       palette=CIRCUIT_PALETTE, cut=0, inner=None,
                       split=True, alpha=0.55, ax=ax)
        sns.boxplot(data=sub, x="method", y="value", hue="circuit",
                    order=METHOD_ORDER, hue_order=CIRCUITS, width=0.25,
                    showfliers=False, boxprops={"facecolor": "white", "zorder": 3},
                    whiskerprops={"zorder": 3}, ax=ax)
        # de-duplicate legend (violin + box both add entries)
        handles, labels = ax.get_legend_handles_labels()
        keep = dict(zip([CIRCUIT_LABELS[c] for c in CIRCUITS],
                        handles[:len(CIRCUITS)]))
        ax.legend(keep.values(), keep.keys(), title="Circuit", loc="best",
                  framealpha=0.7, fontsize=ann_size(0.85))
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([wrap(METHODS[m], 12) for m in METHOD_ORDER])
        ax.set_xlabel("")
        ax.set_ylabel(wrap(mlabel(metric), chars_per_line(1)))
        ax.set_title(wrap(f"{mlabel(metric)} vs adaptation method "
                          "(all datasets, 48 heads each)", chars_per_line(1)))
        savefig(fig, f"02a_vs_method__{metric}.png")


def plot_method_summary_grid(df):
    """Compact overview: mean change (+/- std) per method for every
    metric x circuit. One grouped-bar panel per metric."""
    n = len(SELECTED_METRICS)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    setup(aspect=(nrows / ncols) * 0.7)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    cw = chars_per_line(ncols)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    x = np.arange(len(METHOD_ORDER))
    w = 0.38
    for ax, metric in zip(axes_flat, SELECTED_METRICS):
        sub = df[df.metric == metric]
        for i, circuit in enumerate(CIRCUITS):
            g = (sub[sub.circuit == circuit].groupby("method")["value"]
                 .agg(["mean", "std"]).reindex(METHOD_ORDER))
            ax.bar(x + (i - 0.5) * w, g["mean"], w, yerr=g["std"], capsize=4,
                   color=CIRCUIT_PALETTE[circuit], edgecolor="black", lw=0.4,
                   alpha=0.9, label=CIRCUIT_LABELS[circuit])
        ax.set_xticks(x)
        ax.set_xticklabels([wrap(METHODS[m], 12) for m in METHOD_ORDER])
        ax.set_ylabel(wrap(mlabel(metric), cw))
        ax.set_title(wrap(mlabel(metric), cw))
        ax.legend(title="Circuit", loc="best", framealpha=0.7,
                  fontsize=ann_size(0.8))
    fig.suptitle(wrap("Mean weight change by adaptation method (±std over heads)",
                      chars_per_line(1)), fontweight="bold")
    savefig(fig, "02b_method_summary_grid.png")


# ============================================================================
# 3. CHANGE METRIC vs ADAPTATION DATASET
# ============================================================================
def plot_metric_vs_dataset(df):
    """For each metric, distribution of the change across datasets, split by
    circuit. Violin + box, dataset palette borrowed from the reference script."""
    for metric in SELECTED_METRICS:
        sub = df[df.metric == metric]
        setup(aspect=0.7)
        fig, ax = plt.subplots()
        sns.violinplot(data=sub, x="task", y="value", hue="circuit",
                       order=TASK_ORDER, hue_order=CIRCUITS,
                       palette=CIRCUIT_PALETTE, cut=0, inner=None,
                       split=True, alpha=0.55, ax=ax)
        sns.boxplot(data=sub, x="task", y="value", hue="circuit",
                    order=TASK_ORDER, hue_order=CIRCUITS, width=0.25,
                    showfliers=False, boxprops={"facecolor": "white", "zorder": 3},
                    whiskerprops={"zorder": 3}, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        keep = dict(zip([CIRCUIT_LABELS[c] for c in CIRCUITS],
                        handles[:len(CIRCUITS)]))
        ax.legend(keep.values(), keep.keys(), title="Circuit", loc="best",
                  framealpha=0.7, fontsize=ann_size(0.85))
        ax.set_xticks(range(len(TASK_ORDER)))
        ax.set_xticklabels([wrap(TASKS[t], 12) for t in TASK_ORDER])
        ax.set_xlabel("")
        ax.set_ylabel(wrap(mlabel(metric), chars_per_line(1)))
        ax.set_title(wrap(f"{mlabel(metric)} vs adaptation dataset "
                          "(all methods pooled)", chars_per_line(1)))
        savefig(fig, f"03a_vs_dataset__{metric}.png")


def plot_dataset_method_heatmap(df):
    """Mean change per (dataset x method) cell, one heatmap per
    metric x circuit. Compact summary of where change concentrates."""
    n = len(SELECTED_METRICS)
    ncols = len(CIRCUITS)
    setup(aspect=(n / ncols) * 0.85)
    fig, axes = plt.subplots(n, ncols, squeeze=False)
    cw = chars_per_line(ncols)
    for i, metric in enumerate(SELECTED_METRICS):
        for j, circuit in enumerate(CIRCUITS):
            sub = df[(df.metric == metric) & (df.circuit == circuit)]
            grid = (sub.pivot_table(index="task", columns="method",
                                    values="value", aggfunc="mean")
                    .reindex(index=TASK_ORDER, columns=METHOD_ORDER))
            grid.index = [TASKS[t] for t in grid.index]
            grid.columns = [METHODS[m] for m in grid.columns]
            ax = axes[i, j]
            sns.heatmap(grid, ax=ax, cmap="magma", annot=True, fmt=".3f",
                        annot_kws={"size": ann_size(0.85)}, linewidths=0.5,
                        linecolor="white", cbar_kws={"label": "mean"})
            ax.set_title(wrap(f"{CIRCUIT_LABELS[circuit]} — {mlabel(metric)}", cw))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=ann_size(0.8))
    fig.suptitle(wrap("Mean change per dataset × method", chars_per_line(1)),
                 fontweight="bold")
    savefig(fig, "03b_dataset_method_heatmap.png")


# ============================================================================
# 4. LAYERWISE ANALYSIS
# ============================================================================
def plot_layerwise_trends(df):
    """Mean change vs layer, one line per method, faceted by metric (rows) and
    circuit (cols). Reveals depth trends of the adaptation."""
    layers = sorted(df["layer"].unique())
    n = len(SELECTED_METRICS)
    ncols = len(CIRCUITS)
    setup(aspect=(n / ncols) * 0.7)
    fig, axes = plt.subplots(n, ncols, squeeze=False, sharex=True)
    cw = chars_per_line(ncols)
    for i, metric in enumerate(SELECTED_METRICS):
        for j, circuit in enumerate(CIRCUITS):
            ax = axes[i, j]
            sub = df[(df.metric == metric) & (df.circuit == circuit)]
            for method in METHOD_ORDER:
                g = sub[sub.method == method].groupby("layer")["value"].mean()
                ax.plot(g.index, g.values, "-o", color=METHOD_PALETTE[method],
                        lw=1.6, ms=4, label=METHODS[method])
            ax.set_xticks(layers)
            ax.set_title(wrap(f"{CIRCUIT_LABELS[circuit]} — {mlabel(metric)}", cw))
            if i == n - 1:
                ax.set_xlabel("Layer")
            if j == 0:
                ax.set_ylabel(wrap("mean change", cw))
            if i == 0 and j == 0:
                ax.legend(title="Method", loc="best", framealpha=0.7,
                          fontsize=ann_size(0.8))
    fig.suptitle(wrap("Layerwise weight-change trends (pooled over datasets)",
                      chars_per_line(1)), fontweight="bold")
    savefig(fig, "04a_layerwise_trends.png")


def plot_layer_head_heatmaps(df, metric, circuit, method):
    """Layer x Head heatmap of one metric/circuit/method, one panel per
    dataset. Saved per (metric, circuit, method) combination."""
    sub = df[(df.metric == metric) & (df.circuit == circuit) &
             (df.method == method)]
    if sub.empty:
        return
    setup(aspect=0.42)
    fig, axes = plt.subplots(1, len(TASK_ORDER))
    cw = chars_per_line(len(TASK_ORDER))
    vmin, vmax = sub["value"].min(), sub["value"].max()
    for ax, task in zip(np.atleast_1d(axes), TASK_ORDER):
        s = sub[sub.task == task]
        grid = s.pivot_table(index="layer", columns="head", values="value")
        sns.heatmap(grid, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"label": mlabel(metric)})
        ax.set_title(wrap(TASKS[task], cw))
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.tick_params(labelsize=ann_size(0.8))
    fig.suptitle(wrap(f"{CIRCUIT_LABELS[circuit]} {mlabel(metric)} — "
                      f"{METHODS[method]} (Layer × Head)", chars_per_line(1)),
                 fontweight="bold")
    savefig(fig, f"04b_layerhead__{metric}__{circuit}__{method}.png")


def plot_layerhead_selected(df):
    """Drive the Layer x Head heatmaps for a manageable default subset.
    Edit the loops below to dump more/all combinations."""
    # Default: first selected metric, both circuits, every method.
    for metric in SELECTED_METRICS[:1]:
        for circuit in CIRCUITS:
            for method in METHOD_ORDER:
                plot_layer_head_heatmaps(df, metric, circuit, method)


# ============================================================================
# Summary table
# ============================================================================
def write_summary(df):
    summary = (df.groupby(["metric", "circuit", "task_label", "method_label"])
               ["value"].agg(["mean", "std", "median"]).round(4))
    summary.to_csv(OUT / "summary_change_metrics.csv")
    df.to_csv(OUT / "change_metrics_tidy.csv", index=False)
    print("  saved", OUT / "summary_change_metrics.csv")
    print("  saved", OUT / "change_metrics_tidy.csv")


# ============================================================================
def main():
    df = load_all()
    if df.empty:
        print("No data loaded — check paths / SELECTED_METRICS.")
        return

    print("\n[1] QK<->OV correlation")
    plot_qk_ov_scatter_per_metric(df)
    plot_qk_ov_correlation_overview(df)
    plot_metric_cross_correlation(df)

    print("[2] change vs adaptation method")
    plot_metric_vs_method(df)
    plot_method_summary_grid(df)

    print("[3] change vs adaptation dataset")
    plot_metric_vs_dataset(df)
    plot_dataset_method_heatmap(df)

    print("[4] layerwise analysis")
    plot_layerwise_trends(df)
    plot_layerhead_selected(df)

    write_summary(df)
    print("\nAll figures + CSVs written to:", OUT)


if __name__ == "__main__":
    main()
