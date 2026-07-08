#!/usr/bin/env python3
"""
Analyze how the ACTIVATION (representation) of CLIP ViT attention heads changes
during adaptation, across:
    * 3 target datasets  : ImageNet-1k, SUN-395, Synthetic Text
    * 3 adaptation methods: FFT (full finetune), LoRA-r4, LoRA-r16

Each `act_change_metrics_<method>.json` lives under <dataset>/ and stores, for
every one of the 48 heads (Layers 8-11 x Heads 0-11), a set of activation-based
change metrics. Unlike the weight-change metrics there is NO QK/OV split here —
each metric is a single value per head.

The script builds a tidy dataframe and emits plots covering:
    * head statistics per adaptation METHOD
    * head statistics per adaptation DATASET
    * layerwise analysis (depth trends + Layer x Head heatmaps)
(no metric<->metric correlations, by request.)

Plot styling (palette, theme, savefig helper, violin/box idiom) is borrowed
from plot_new_task_relevance.py / plot_weight_change.py to keep the figure set
visually consistent.

NOTE: this only *creates* the plots; nothing runs on import beyond main().
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
LATEX_WIDTH_FRAC = 0.6                           # the c in width=c\textwidth
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
OUT = BASE / "plots" / "rep_change"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# METRIC SELECTION
# ----------------------------------------------------------------------------
# Activation-based metrics — a single value per head (no qk/ov prefix).
#
# ---- FULL LIST OF AVAILABLE METRICS (copy/paste any of these into
#      SELECTED_METRICS to change what gets plotted) -----------------------
#   "CKA_Linear",
#   "Subspace_Alignment",
#   "Principal_Angle_Deg",
#   "Eff_Dim_Orig",
#   "Eff_Dim_Finetuned",
#   "Eff_Dim_Change",
#   "Total_Var_Orig",
#   "Total_Var_Finetuned",
#   "Total_Var_Ratio",
#   "Spectral_Entropy_Orig_bits",
#   "Spectral_Entropy_Finetuned_bits",
#   "Spectral_Entropy_Change",
#   "Centroid_Shift_Cosine",
#   "Centroid_Shift_RelNorm",
#   "Became_More_Specific",
#   "N_Active_Orig",
#   "N_Active_Finetuned",
#   "Top5_Group_Match_Count",
#   "Top5_Group_Match_Score",
#   "Group_Dist_JS_Similarity",
#   "Group_Dist_Cosine",
# --------------------------------------------------------------------------
#
# The four metrics requested:
SELECTED_METRICS = [
    "CKA_Linear",
    "Subspace_Alignment",
    "Total_Var_Ratio",
    "Centroid_Shift_RelNorm",
]

# Pretty labels for any metric you might select. Falls back to a cleaned-up
# name if a metric is missing here.
METRIC_LABELS = {
    "CKA_Linear":                     "Linear CKA (orig vs finetuned)",
    "Subspace_Alignment":             "Subspace alignment",
    "Principal_Angle_Deg":            "Principal angle (deg)",
    "Eff_Dim_Orig":                   "Effective dim (orig)",
    "Eff_Dim_Finetuned":              "Effective dim (finetuned)",
    "Eff_Dim_Change":                 "Effective-dim change",
    "Total_Var_Orig":                 "Total variance (orig)",
    "Total_Var_Finetuned":            "Total variance (finetuned)",
    "Total_Var_Ratio":               "Total-variance ratio (ft/orig)",
    "Spectral_Entropy_Orig_bits":     "Spectral entropy, orig (bits)",
    "Spectral_Entropy_Finetuned_bits":"Spectral entropy, ft (bits)",
    "Spectral_Entropy_Change":        "Spectral-entropy change",
    "Centroid_Shift_Cosine":          "Centroid shift (cosine)",
    "Centroid_Shift_RelNorm":         "Centroid shift (relative norm)",
    "Became_More_Specific":           "Became more specific (0/1)",
    "N_Active_Orig":                  "# active groups (orig)",
    "N_Active_Finetuned":             "# active groups (finetuned)",
    "Top5_Group_Match_Count":         "Top-5 group match count",
    "Top5_Group_Match_Score":         "Top-5 group match score",
    "Group_Dist_JS_Similarity":       "Group-dist JS similarity",
    "Group_Dist_Cosine":              "Group-dist cosine",
}

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

sns.set_theme(style="whitegrid")

_pat = re.compile(r"Layer\s+(\d+)\s+Head\s+(\d+)")


def mlabel(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


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
# rows: one per (task, method, layer, head, metric)
# ============================================================================
def load_all():
    rows = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            path = BASE / task / f"act_change_metrics_{method}.json"
            if not path.exists():
                print(f"  WARNING missing file: {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            for name, m in data.items():
                layer, head = map(int, _pat.match(name).groups())
                for metric in SELECTED_METRICS:
                    if metric not in m:
                        continue
                    rows.append(dict(
                        task=task, task_label=TASKS[task],
                        method=method, method_label=METHODS[method],
                        layer=layer, head=head,
                        metric=metric, value=m[metric],
                    ))
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows "
          f"({df[['task','method']].drop_duplicates().shape[0]} task x method files, "
          f"{df['metric'].nunique()} metrics).")
    return df


def _grid_axes(n, ncols=2, panel_aspect=0.8):
    """Create an n-panel grid sized for LaTeX.  `panel_aspect` is the height/width
    of a single panel; the whole figure is authored at LATEX_WIDTH_FRAC*\textwidth
    via setup(), so text appears at TARGET_FONT_PT on the page."""
    nrows = int(np.ceil(n / ncols))
    setup(aspect=(nrows / ncols) * panel_aspect)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat


# ============================================================================
# 1. HEAD STATISTICS per ADAPTATION METHOD
# ============================================================================
def plot_metric_vs_method(df):
    """For each metric, distribution of the head statistic across methods.
    Violin + box + jittered points + per-method mean diamond."""
    for metric in SELECTED_METRICS:
        sub = df[df.metric == metric]
        setup(aspect=0.78)
        fig, ax = plt.subplots()
        sns.violinplot(data=sub, x="method", y="value", order=METHOD_ORDER,
                       palette=METHOD_PALETTE, inner=None, cut=0,
                       alpha=0.55, ax=ax)
        sns.boxplot(data=sub, x="method", y="value", order=METHOD_ORDER,
                    width=0.18, showcaps=True, showfliers=False,
                    boxprops={"facecolor": "white", "zorder": 3},
                    whiskerprops={"zorder": 3}, ax=ax)
        sns.stripplot(data=sub, x="method", y="value", order=METHOD_ORDER,
                      color="k", size=2.5, alpha=0.4, jitter=0.12, ax=ax)
        means = sub.groupby("method")["value"].mean()
        for x, method in enumerate(METHOD_ORDER):
            mv = means[method]
            ax.scatter(x, mv, marker="D", s=45, color="crimson",
                       edgecolor="black", lw=0.8, zorder=5,
                       label="per-method mean" if x == 0 else None)
            ax.annotate(f"{mv:.3f}", xy=(x, mv), xytext=(x + 0.22, mv),
                        va="center", fontsize=ann_size(0.85), color="crimson",
                        fontweight="bold")
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([wrap(METHODS[m], 12) for m in METHOD_ORDER])
        ax.set_xlabel("")
        ax.set_ylabel(wrap(mlabel(metric), chars_per_line(1)))
        ax.set_title(wrap(f"{mlabel(metric)} vs adaptation method "
                          "(all datasets, 48 heads each)", chars_per_line(1)))
        ax.legend(loc="best", framealpha=0.7, fontsize=ann_size(0.85))
        savefig(fig, f"01a_vs_method__{metric}.png")


def plot_method_summary_grid(df):
    """Compact overview: mean head statistic (+/- std) per method, one panel
    per metric."""
    fig, axes = _grid_axes(len(SELECTED_METRICS), ncols=2, panel_aspect=0.7)
    cw = chars_per_line(2)
    x = np.arange(len(METHOD_ORDER))
    for ax, metric in zip(axes, SELECTED_METRICS):
        sub = df[df.metric == metric]
        g = sub.groupby("method")["value"].agg(["mean", "std"]).reindex(METHOD_ORDER)
        ax.bar(x, g["mean"], 0.6, yerr=g["std"], capsize=5,
               color=[METHOD_PALETTE[m] for m in METHOD_ORDER],
               edgecolor="black", lw=0.4, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([wrap(METHODS[m], 12) for m in METHOD_ORDER])
        ax.set_ylabel(wrap(mlabel(metric), cw))
        ax.set_title(wrap(mlabel(metric), cw))
    fig.suptitle(wrap("Mean activation change by adaptation method "
                      "(±std over heads)", chars_per_line(1)), fontweight="bold")
    savefig(fig, "01b_method_summary_grid.png")


# ============================================================================
# 2. HEAD STATISTICS per ADAPTATION DATASET
# ============================================================================
def plot_metric_vs_dataset(df):
    """For each metric, distribution of the head statistic across datasets.
    Violin + box + jittered points + per-dataset mean diamond."""
    for metric in SELECTED_METRICS:
        sub = df[df.metric == metric]
        setup(aspect=0.78)
        fig, ax = plt.subplots()
        sns.violinplot(data=sub, x="task", y="value", order=TASK_ORDER,
                       palette=TASK_PALETTE, inner=None, cut=0,
                       alpha=0.55, ax=ax)
        sns.boxplot(data=sub, x="task", y="value", order=TASK_ORDER,
                    width=0.18, showcaps=True, showfliers=False,
                    boxprops={"facecolor": "white", "zorder": 3},
                    whiskerprops={"zorder": 3}, ax=ax)
        sns.stripplot(data=sub, x="task", y="value", order=TASK_ORDER,
                      color="k", size=2.5, alpha=0.4, jitter=0.12, ax=ax)
        means = sub.groupby("task")["value"].mean()
        for x, task in enumerate(TASK_ORDER):
            mv = means[task]
            ax.scatter(x, mv, marker="D", s=45, color="crimson",
                       edgecolor="black", lw=0.8, zorder=5,
                       label="per-dataset mean" if x == 0 else None)
            ax.annotate(f"{mv:.3f}", xy=(x, mv), xytext=(x + 0.22, mv),
                        va="center", fontsize=ann_size(0.85), color="crimson",
                        fontweight="bold")
        ax.set_xticks(range(len(TASK_ORDER)))
        ax.set_xticklabels([wrap(TASKS[t], 12) for t in TASK_ORDER])
        ax.set_xlabel("")
        ax.set_ylabel(wrap(mlabel(metric), chars_per_line(1)))
        ax.set_title(wrap(f"{mlabel(metric)} vs adaptation dataset "
                          "(all methods pooled)", chars_per_line(1)))
        ax.legend(loc="best", framealpha=0.7, fontsize=ann_size(0.85))
        savefig(fig, f"02a_vs_dataset__{metric}.png")


def plot_dataset_method_grouped(df):
    """Grouped bars: mean head statistic per dataset, one bar group per method.
    One panel per metric — shows method x dataset interaction at a glance."""
    fig, axes = _grid_axes(len(SELECTED_METRICS), ncols=2, panel_aspect=0.7)
    cw = chars_per_line(2)
    x = np.arange(len(TASK_ORDER))
    w = 0.8 / len(METHOD_ORDER)
    for ax, metric in zip(axes, SELECTED_METRICS):
        sub = df[df.metric == metric]
        for i, method in enumerate(METHOD_ORDER):
            g = (sub[sub.method == method].groupby("task")["value"].mean()
                 .reindex(TASK_ORDER))
            ax.bar(x + i * w, g.values, w, color=METHOD_PALETTE[method],
                   edgecolor="black", lw=0.4, alpha=0.9, label=METHODS[method])
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([wrap(TASKS[t], 12) for t in TASK_ORDER])
        ax.set_ylabel(wrap(mlabel(metric), cw))
        ax.set_title(wrap(mlabel(metric), cw))
        ax.legend(title="Method", loc="best", framealpha=0.7,
                  fontsize=ann_size(0.8))
    fig.suptitle(wrap("Mean activation change per dataset × method",
                      chars_per_line(1)), fontweight="bold")
    savefig(fig, "02b_dataset_method_grouped.png")


def plot_dataset_method_heatmap(df):
    """Mean head statistic per (dataset x method) cell, one heatmap per metric."""
    fig, axes = _grid_axes(len(SELECTED_METRICS), ncols=2, panel_aspect=0.75)
    cw = chars_per_line(2)
    for ax, metric in zip(axes, SELECTED_METRICS):
        sub = df[df.metric == metric]
        grid = (sub.pivot_table(index="task", columns="method", values="value",
                                aggfunc="mean")
                .reindex(index=TASK_ORDER, columns=METHOD_ORDER))
        grid.index = [TASKS[t] for t in grid.index]
        grid.columns = [METHODS[m] for m in grid.columns]
        sns.heatmap(grid, ax=ax, cmap="magma", annot=True, fmt=".3f",
                    annot_kws={"size": ann_size(0.85)}, linewidths=0.5,
                    linecolor="white", cbar_kws={"label": "mean"})
        ax.set_title(wrap(mlabel(metric), cw))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=ann_size(0.8))
    fig.suptitle(wrap("Mean activation change per dataset × method",
                      chars_per_line(1)), fontweight="bold")
    savefig(fig, "02c_dataset_method_heatmap.png")


# ============================================================================
# 3. LAYERWISE ANALYSIS
# ============================================================================
def plot_layerwise_by_method(df):
    """Mean head statistic vs layer, one line per method, one panel per metric
    (datasets pooled). Depth trend of the activation change."""
    layers = sorted(df["layer"].unique())
    fig, axes = _grid_axes(len(SELECTED_METRICS), ncols=2, panel_aspect=0.7)
    cw = chars_per_line(2)
    for ax, metric in zip(axes, SELECTED_METRICS):
        sub = df[df.metric == metric]
        for method in METHOD_ORDER:
            g = sub[sub.method == method].groupby("layer")["value"].mean()
            ax.plot(g.index, g.values, "-o", color=METHOD_PALETTE[method],
                    lw=1.6, ms=4, label=METHODS[method])
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel(wrap(mlabel(metric), cw))
        ax.set_title(wrap(mlabel(metric), cw))
        ax.legend(title="Method", loc="best", framealpha=0.7,
                  fontsize=ann_size(0.8))
    fig.suptitle(wrap("Layerwise activation-change trends by method "
                      "(datasets pooled)", chars_per_line(1)), fontweight="bold")
    savefig(fig, "03a_layerwise_by_method.png")


def plot_layerwise_by_dataset(df):
    """Mean head statistic vs layer, one line per dataset, one panel per metric
    (methods pooled)."""
    layers = sorted(df["layer"].unique())
    fig, axes = _grid_axes(len(SELECTED_METRICS), ncols=2, panel_aspect=0.7)
    cw = chars_per_line(2)
    for ax, metric in zip(axes, SELECTED_METRICS):
        sub = df[df.metric == metric]
        for task in TASK_ORDER:
            g = sub[sub.task == task].groupby("layer")["value"].mean()
            ax.plot(g.index, g.values, "-o", color=TASK_PALETTE[task],
                    lw=1.6, ms=4, label=TASKS[task])
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel(wrap(mlabel(metric), cw))
        ax.set_title(wrap(mlabel(metric), cw))
        ax.legend(title="Dataset", loc="best", framealpha=0.7,
                  fontsize=ann_size(0.8))
    fig.suptitle(wrap("Layerwise activation-change trends by dataset "
                      "(methods pooled)", chars_per_line(1)), fontweight="bold")
    savefig(fig, "03b_layerwise_by_dataset.png")


def plot_layer_head_heatmaps(df, metric, method):
    """Layer x Head heatmap of one metric/method, one panel per dataset.
    Saved per (metric, method) combination."""
    sub = df[(df.metric == metric) & (df.method == method)]
    if sub.empty:
        return
    vmin, vmax = sub["value"].min(), sub["value"].max()
    setup(aspect=0.42)
    fig, axes = plt.subplots(1, len(TASK_ORDER))
    cw = chars_per_line(len(TASK_ORDER))
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
    fig.suptitle(wrap(f"{mlabel(metric)} — {METHODS[method]} (Layer × Head)",
                      chars_per_line(1)), fontweight="bold")
    savefig(fig, f"03c_layerhead__{metric}__{method}.png")


def plot_layerhead_selected(df):
    """Drive the Layer x Head heatmaps for a manageable default subset.
    Edit the loops below to dump more/all combinations."""
    # Default: first selected metric, every method.
    for metric in SELECTED_METRICS[:1]:
        for method in METHOD_ORDER:
            plot_layer_head_heatmaps(df, metric, method)


# ============================================================================
# Summary table
# ============================================================================
def write_summary(df):
    summary = (df.groupby(["metric", "task_label", "method_label"])
               ["value"].agg(["mean", "std", "median"]).round(4))
    summary.to_csv(OUT / "summary_act_change_metrics.csv")
    df.to_csv(OUT / "act_change_metrics_tidy.csv", index=False)
    print("  saved", OUT / "summary_act_change_metrics.csv")
    print("  saved", OUT / "act_change_metrics_tidy.csv")


# ============================================================================
def main():
    df = load_all()
    if df.empty:
        print("No data loaded — check paths / SELECTED_METRICS.")
        return

    print("\n[1] head stats per adaptation method")
    plot_metric_vs_method(df)
    plot_method_summary_grid(df)

    print("[2] head stats per adaptation dataset")
    plot_metric_vs_dataset(df)
    plot_dataset_method_grouped(df)
    plot_dataset_method_heatmap(df)

    print("[3] layerwise analysis")
    plot_layerwise_by_method(df)
    plot_layerwise_by_dataset(df)
    plot_layerhead_selected(df)

    write_summary(df)
    print("\nAll figures + CSVs written to:", OUT)


if __name__ == "__main__":
    main()
