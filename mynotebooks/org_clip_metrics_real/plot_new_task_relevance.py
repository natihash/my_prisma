#!/usr/bin/env python3
r"""
Summarize the discriminative power ("new-task relevance") of the original
(pre-finetuning) CLIP ViT attention heads across three target tasks:
ImageNet-1k, SUN-395 and a synthetic text-classification dataset.

Two head-level metrics are read from the per-task JSON files:
  * rank_mean   -> R_rank   : a rank-based relevance score, already in [0, 1].
  * top_k_mean  -> R_top-k  : a top-k discriminability score whose RAW magnitude
                              grows strongly with layer depth (a structural
                              artifact of attention concentrating in deep layers).

Because top_k_mean carries this layerwise magnitude trend, comparing raw values
across layers/heads would mostly reflect *depth*, not *task relevance*. We remove
the layerwise magnitude effect by dividing each head's top_k_mean by the
per-layer mean POOLED OVER THE THREE TASKS. This:
   - normalizes every layer to a common scale (~1.0 = average head in that layer),
   - preserves genuine differences BETWEEN tasks (shared denominator per layer),
   - yields a unitless relative relevance comparable across layers and heads.
This pooled-per-layer-mean ratio is what we call R_top-k below.

------------------------------------------------------------------------------
LaTeX-ready text sizing
------------------------------------------------------------------------------
When you drop a figure into LaTeX with

    \includegraphics[width=c\textwidth]{fig.png}

LaTeX rescales the whole image by  (c * \textwidth) / (figure width).  If the
figure was authored wider than the slot it lands in, every label/tick/legend is
shrunk by that same factor -- which is why text that looks fine at
width=\textwidth becomes unreadable at width=0.5\textwidth.

The fix used here: build each figure at a *physical* width of exactly
`c * \textwidth`, and set the fonts to the point size you want to SEE on the
page.  Then LaTeX's scale factor is exactly 1, and text renders at the chosen
point size regardless of `c`.

So: set `LATEX_WIDTH_FRAC` below to the same `c` you will use in
\includegraphics, set `TARGET_FONT_PT` to the on-page size you want, re-run, and
include the figures with  width=<LATEX_WIDTH_FRAC>\textwidth.
------------------------------------------------------------------------------
"""

import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = Path("/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real")
OUT = BASE / "plots" / "new_task_relevance2"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# LaTeX integration  -- EDIT THESE to match your document.
# ----------------------------------------------------------------------------
# Run \the\textwidth in your LaTeX doc and paste the number of points here.
TEXTWIDTH_PT = 345.0
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27          # TeX pt -> inch

# The `c` you will write in \includegraphics[width=c\textwidth]{...}.
# 1.0 = full text width, 0.5 = half, etc.  Text stays the same on-page size.
LATEX_WIDTH_FRAC = 1.2

# Point size the text should *appear* as on the printed/PDF page.
TARGET_FONT_PT = 9.0

DPI = 300

# task_key -> (json path, pretty label)
TASKS = {
    "imagenet1k": (BASE / "imagenet1k" / "new_task_rel_org_clip.json", "ImageNet-1k"),
    "sun":        (BASE / "sun"        / "new_task_rel_org_clip.json", "SUN-395"),
    "text":       (BASE / "text"       / "new_task_rel_org_clip.json", "Synthetic Text"),
}
TASK_ORDER = ["imagenet1k", "sun", "text"]
PALETTE = {"imagenet1k": "#4C72B0", "sun": "#DD8452", "text": "#55A868"}
LABELS = {k: v[1] for k, (_, v) in zip(TASKS, TASKS.items())}

# LaTeX-style labels requested by the user
R_RANK = r"$R_{rank}$"
R_TOPK = r"$R_{top-k}$"

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# LaTeX-aware sizing + text helpers
# ----------------------------------------------------------------------------
def setup(width_frac=None, aspect=0.62, base=None):
    r"""Configure rcParams so the figure is authored at its on-page size.

    The figure's physical width is set to (width_frac * \textwidth) so LaTeX
    does NOT rescale it at \includegraphics[width=width_frac\textwidth] -> fonts
    keep their chosen point size on the page, for any width_frac.
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
        # constrained_layout shrinks the axes to fit labels/titles/colorbars/
        # legends INSIDE the canvas, so nothing is clipped and the saved width
        # stays exactly width_frac*textwidth (do NOT save with bbox_inches='tight',
        # that re-crops and changes the width).
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
    """Hard-wrap a string to <= `width` characters per line.

    Used on every title / axis label / legend entry so long strings break onto
    new lines instead of overflowing into neighbouring panels or the plot area.
    `break_long_words`/`break_on_hyphens` are off so math tokens like
    "$R_{top-k}$" and hyphenated words are never split.
    """
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


# ----------------------------------------------------------------------------
# Load -> tidy dataframe
# ----------------------------------------------------------------------------
_pat = re.compile(r"Layer\s+(\d+)\s+Head\s+(\d+)")


# "rank_mean_cos": 0.31770187616348267,
# "top_k_mean_cos": 0.03515322133898735,
# "rank_mean_mse": 0.19754540920257568,
# "top_k_mean_mse": 0.0002147665072698146

def load_task(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for name, m in data.items():
        layer, head = map(int, _pat.match(name).groups())
        # rows.append(
        #     dict(layer=layer, head=head, name=name,
        #          rank_mean=m["rank_mean_cos"], top_k_mean=m["top_k_mean_cos"],
        #          rank_consistency=m["rank_consistency"],
        #          median_disc=m["median_disc"], gini=m["gini"])
        # )

        rows.append(
            dict(layer=layer, head=head, name=name,
                 rank_mean=m["rank_mean"], top_k_mean=m["top_k_mean"]))
    return pd.DataFrame(rows)


frames = []
for key in TASK_ORDER:
    path, label = TASKS[key]
    df = load_task(path)
    df["task"] = key
    df["task_label"] = label
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

# ----------------------------------------------------------------------------
# Normalize top_k_mean -> R_top-k (remove layerwise magnitude effect)
# Denominator = mean top_k_mean per layer, POOLED across the three tasks.
# ----------------------------------------------------------------------------
layer_pooled_mean = data.groupby("layer")["top_k_mean"].transform("mean")
data["R_topk"] = data["top_k_mean"] / layer_pooled_mean
data["R_rank"] = data["rank_mean"]  # already a normalized rank in [0, 1]

# Report the per-layer denominators (the magnitude effect being removed)
denoms = data.groupby("layer")["top_k_mean"].mean()
print("Per-layer pooled mean of raw top_k_mean (normalization denominator):")
for L, v in denoms.items():
    print(f"  Layer {L}: {v:.3f}")
print(f"\nLoaded {len(data)} rows "
      f"({data['name'].nunique()} heads x {len(TASK_ORDER)} tasks).")

n_layers = data["layer"].nunique()
n_heads = data["head"].nunique()
N_HEADS_TOTAL = n_layers * n_heads  # 48

labels_ordered = [TASKS[k][1] for k in TASK_ORDER]
colors_ordered = [PALETTE[k] for k in TASK_ORDER]


def savefig(fig, name):
    p = OUT / name
    # No bbox_inches='tight': constrained_layout already fits everything inside
    # the canvas, and keeping the figure size intact preserves the exact
    # width_frac*textwidth width so LaTeX does not rescale (and shrink) text.
    fig.savefig(p)
    plt.close(fig)
    print("  saved", p)


# ============================================================================
# FIG 1 - Normalization diagnostic: WHY we normalize, and the effect.
# ============================================================================
setup(aspect=0.5)
fig, axes = plt.subplots(1, 2)
layers = sorted(data["layer"].unique())
cw = chars_per_line(2)

ax = axes[0]
for key in TASK_ORDER:
    sub = data[data.task == key].groupby("layer")["top_k_mean"].mean()
    ax.plot(sub.index, sub.values, "-o", color=PALETTE[key],
            label=TASKS[key][1], lw=1.6, ms=5)
ax.set_title(wrap("Raw top-k mean grows with depth (layerwise magnitude effect)", cw))
ax.set_xlabel("Layer")
ax.set_ylabel(wrap("mean raw top_k_mean", cw))
ax.set_xticks(layers)
ax.legend(loc="best", framealpha=0.7)

ax = axes[1]
for key in TASK_ORDER:
    sub = data[data.task == key].groupby("layer")["R_topk"].mean()
    ax.plot(sub.index, sub.values, "-o", color=PALETTE[key],
            label=TASKS[key][1], lw=1.6, ms=5)
ax.axhline(1.0, ls="--", c="grey", lw=1.2, label="layer avg = 1")
ax.set_title(wrap(f"After normalization: {R_TOPK} (depth effect removed, task gaps kept)", cw))
ax.set_xlabel("Layer")
ax.set_ylabel(wrap(R_TOPK + " (per-layer mean)", cw))
ax.set_xticks(layers)
ax.legend(loc="best", framealpha=0.7)

fig.suptitle(wrap("Layerwise normalization of the top-k relevance metric",
                  chars_per_line(1)), fontweight="bold")
savefig(fig, "fig1_layerwise_normalization.png")


# ============================================================================
# FIG 2 - MAIN: distribution of head relevance per task.
# SEPARATE figure for each metric (R_rank, R_top-k). Violin (shape) + overlaid
# box + jittered points for all 48 heads, with the per-task MEAN marked.
# ============================================================================
def plot_distribution(metric, mlabel, fname):
    setup(aspect=0.82)
    cw = chars_per_line(1)
    fig, ax = plt.subplots()
    sns.violinplot(data=data, x="task", y=metric, order=TASK_ORDER,
                   palette=PALETTE, inner=None, cut=0, ax=ax, alpha=0.55)
    sns.boxplot(data=data, x="task", y=metric, order=TASK_ORDER, width=0.18,
                showcaps=True, boxprops={"facecolor": "white", "zorder": 3},
                showfliers=False, whiskerprops={"zorder": 3}, ax=ax)
    sns.stripplot(data=data, x="task", y=metric, order=TASK_ORDER,
                  color="k", size=2.5, alpha=0.45, jitter=0.12, ax=ax)

    # Per-task mean: diamond marker + value annotation.
    means = data.groupby("task")[metric].mean()
    for x, key in enumerate(TASK_ORDER):
        mval = means[key]
        ax.scatter(x, mval, marker="D", s=45, color="crimson",
                   edgecolor="black", lw=0.8, zorder=5,
                   label="per-task mean" if x == 0 else None)
        ax.annotate(f"mean={mval:.3f}", xy=(x, mval),
                    xytext=(x + 0.22, mval), va="center",
                    fontsize=ann_size(0.85), color="crimson", fontweight="bold")

    ax.set_xticklabels([wrap(l, 12) for l in labels_ordered])
    ax.set_xlabel("")
    ax.set_ylabel(mlabel)
    ax.set_title(wrap(f"{mlabel} across the {N_HEADS_TOTAL} heads", cw))
    ax.legend(loc="best", framealpha=0.7)
    fig.suptitle(wrap(f"Distribution of head relevance per task — {mlabel} "
                      "(original CLIP, pre-finetune)", chars_per_line(1)),
                 fontweight="bold")
    savefig(fig, fname)


plot_distribution("R_rank", R_RANK, "fig2a_relevance_distribution_rank.png")
plot_distribution("R_topk", R_TOPK, "fig2b_relevance_distribution_topk.png")


# ============================================================================
# FIG 2c / 2d - HISTOGRAMS of head relevance per task.
# SEPARATE figure for each metric. Overlaid per-task histograms with the
# per-task MEAN drawn as a vertical line.
# ============================================================================
def plot_histogram(metric, mlabel, fname):
    setup(aspect=0.72)
    cw = chars_per_line(1)
    fig, ax = plt.subplots()
    means = data.groupby("task")[metric].mean()
    # Shared bin edges so the per-task histograms are directly comparable.
    bins = np.histogram_bin_edges(data[metric].values, bins=20)
    for key in TASK_ORDER:
        sub = data[data.task == key]
        ax.hist(sub[metric], bins=bins, color=PALETTE[key], alpha=0.45,
                label=TASKS[key][1], edgecolor="white", lw=0.5)
        # short label (color already identifies the task) so the legend fits.
        ax.axvline(means[key], color=PALETTE[key], ls="--", lw=1.6,
                   label=f"mean = {means[key]:.3f}")
    ax.set_xlabel(mlabel)
    ax.set_ylabel("count (heads)")
    ax.set_title(wrap(f"Histogram of {mlabel} per task  (dashed = per-task mean)", cw))
    # 6 legend entries -> place OUTSIDE the axes so it never covers the bars.
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="outside lower center", ncol=3,
               fontsize=ann_size(0.85), columnspacing=1.0, handlelength=1.4,
               frameon=False)
    fig.suptitle(wrap(f"Per-head relevance histogram — {mlabel}",
                      chars_per_line(1)), fontweight="bold")
    savefig(fig, fname)


plot_histogram("R_rank", R_RANK, "fig2c_relevance_histogram_rank.png")
plot_histogram("R_topk", R_TOPK, "fig2d_relevance_histogram_topk.png")


# ============================================================================
# FIG 3 - Layer x Head heatmaps per task, for both metrics (2 rows x 3 cols).
# ============================================================================
setup(aspect=0.55)
fig, axes = plt.subplots(2, 3)
cw = chars_per_line(3)
for j, key in enumerate(TASK_ORDER):
    sub = data[data.task == key]
    for i, (metric, mlabel, cmap) in enumerate(
            [("R_rank", R_RANK, "viridis"), ("R_topk", R_TOPK, "magma")]):
        grid = sub.pivot(index="layer", columns="head", values=metric)
        ax = axes[i, j]
        sns.heatmap(grid, ax=ax, cmap=cmap,
                    cbar_kws={"label": mlabel},
                    linewidths=0.4, linecolor="white")
        ax.set_title(wrap(f"{TASKS[key][1]} — {mlabel}", cw))
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.tick_params(labelsize=ann_size(0.8))
        ax.figure.axes[-1].tick_params(labelsize=ann_size(0.8))  # colorbar ticks
fig.suptitle(wrap("Per-head relevance maps (Layer x Head) across tasks",
                  chars_per_line(1)), fontweight="bold")
savefig(fig, "fig3_relevance_heatmaps.png")


# ============================================================================
# FIG 4 - R_rank vs R_top-k scatter, colored by task (relationship + separation).
# ============================================================================
setup(aspect=0.78)
cw = chars_per_line(1)
fig, ax = plt.subplots()
for key in TASK_ORDER:
    sub = data[data.task == key]
    ax.scatter(sub["R_rank"], sub["R_topk"], s=22, alpha=0.75,
               color=PALETTE[key], edgecolor="white", lw=0.4,
               label=TASKS[key][1])
ax.axhline(1.0, ls="--", c="grey", lw=1.2)
ax.set_xlabel(R_RANK)
ax.set_ylabel(R_TOPK)
ax.set_title(wrap("Head relevance: rank-based vs top-k (per task, all 48 heads)", cw))
ax.legend(title="Task", loc="best", framealpha=0.7)
savefig(fig, "fig4_rank_vs_topk_scatter.png")


# ============================================================================
# FIG 5 - ECDF + KDE of R_top-k per task (clean view of distribution shift).
# ============================================================================
setup(aspect=0.5)
cw = chars_per_line(2)
fig, axes = plt.subplots(1, 2)
for key in TASK_ORDER:
    sub = data[data.task == key]
    sns.kdeplot(sub["R_topk"], ax=axes[0], color=PALETTE[key], lw=1.6,
                fill=True, alpha=0.18, label=TASKS[key][1], cut=0)
    sns.ecdfplot(sub["R_topk"], ax=axes[1], color=PALETTE[key], lw=1.6,
                 label=TASKS[key][1])
axes[0].axvline(1.0, ls="--", c="grey", lw=1.2)
axes[0].set_xlabel(R_TOPK)
axes[0].set_title(wrap(f"Density of {R_TOPK}", cw))
axes[0].legend(loc="best", framealpha=0.7)
axes[1].axvline(1.0, ls="--", c="grey", lw=1.2)
axes[1].set_xlabel(R_TOPK)
axes[1].set_title(wrap(f"Cumulative distribution of {R_TOPK}", cw))
axes[1].legend(loc="best", framealpha=0.7)
fig.suptitle(wrap(f"How {R_TOPK} relevance is distributed across heads, per task",
                  chars_per_line(1)), fontweight="bold")
savefig(fig, "fig5_topk_distribution.png")


# ============================================================================
# FIG 6 - Top-10 most relevant heads per task (by R_top-k).
# ============================================================================
# setup(aspect=0.5)
# cw = chars_per_line(3)
# fig, axes = plt.subplots(1, 3, sharex=False)
# for ax, key in zip(axes, TASK_ORDER):
#     sub = data[data.task == key].nlargest(10, "R_topk").iloc[::-1]
#     ylabels = [f"L{r.layer} H{r.head}" for r in sub.itertuples()]
#     bars = ax.barh(ylabels, sub["R_topk"], color=PALETTE[key], alpha=0.85,
#                    edgecolor="black", lw=0.5)
#     ax.set_title(wrap(TASKS[key][1], cw))
#     ax.set_xlabel(R_TOPK)
#     # Annotate each bar with its R_top-k value, placed in the whitespace just
#     # past the bar end (never over a bar). Headroom keeps labels on-canvas.
#     xmax = sub["R_topk"].max()
#     ax.set_xlim(0, xmax * 1.22)
#     for bar, val in zip(bars, sub["R_topk"]):
#         y = bar.get_y() + bar.get_height() / 2
#         ax.text(val + xmax * 0.02, y, f"{val:.2f}", va="center", ha="left",
#                 fontsize=ann_size(0.8), color="black")
#     ax.margins(y=0.02)
# fig.suptitle(wrap(f"Top-10 most task-relevant heads by {R_TOPK}",
#                   chars_per_line(1)), fontweight="bold")
# savefig(fig, "fig6_top_heads.png")

setup(aspect=0.5)
cw = chars_per_line(3)
fig, axes = plt.subplots(1, 3, sharex=False)
for ax, key in zip(axes, TASK_ORDER):
    sub = data[data.task == key].nlargest(5, "R_rank").iloc[::-1]
    ylabels = [f"L{r.layer} H{r.head}" for r in sub.itertuples()]
    bars = ax.barh(ylabels, sub["R_rank"], color=PALETTE[key], alpha=0.85,
                   edgecolor="black", lw=0.5)
    ax.set_title(wrap(TASKS[key][1], cw))
    ax.set_xlabel(R_RANK)
    # Annotate each bar with its R_rank value, placed in the whitespace just
    # past the bar end (never over a bar). Headroom keeps labels on-canvas.
    xmax = sub["R_rank"].max()
    ax.set_xlim(0, xmax * 1.22)
    for bar, val in zip(bars, sub["R_rank"]):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(val + xmax * 0.02, y, f"{val:.2f}", va="center", ha="left",
                fontsize=ann_size(0.8), color="black")
    ax.margins(y=0.02)
fig.suptitle(wrap(f"Top-5 most task-relevant heads by {R_RANK}",
                  chars_per_line(1)), fontweight="bold")
savefig(fig, "fig6_top_heads.png")


# ============================================================================
# FIG 7 - Auxiliary metrics from the JSONs: rank consistency & Gini per task.
# (gini = concentration of relevance; rank_consistency = stability of ranking.)
# ============================================================================
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# for ax, metric, title in [
#         (axes[0], "rank_consistency", "Rank consistency\n(stability of head ranking)"),
#         (axes[1], "gini", "Gini\n(concentration of relevance)")]:
#     sns.violinplot(data=data, x="task", y=metric, order=TASK_ORDER,
#                    palette=PALETTE, inner="quartile", cut=0, ax=ax, alpha=0.6)
#     sns.stripplot(data=data, x="task", y=metric, order=TASK_ORDER,
#                   color="k", size=3.5, alpha=0.45, jitter=0.12, ax=ax)
#     ax.set_xticklabels(labels_ordered)
#     ax.set_xlabel("")
#     ax.set_ylabel(metric)
#     ax.set_title(title)
# fig.suptitle("Auxiliary head statistics from the metric JSONs",
#              y=1.02, fontsize=18, fontweight="bold")
# savefig(fig, "fig7_auxiliary_metrics.png")


# ----------------------------------------------------------------------------
# Small summary table (printed + saved as CSV).
# ----------------------------------------------------------------------------
summary = (data.groupby("task_label")
           .agg(R_rank_mean=("R_rank", "mean"),
                R_rank_median=("R_rank", "median"),
                R_topk_mean=("R_topk", "mean"),
                R_topk_median=("R_topk", "median"),
                R_topk_max=("R_topk", "max"))
           .reindex(labels_ordered))
print("\nPer-task summary:")
print(summary.round(3).to_string())
summary.round(4).to_csv(OUT / "summary_per_task.csv")
data.to_csv(OUT / "head_metrics_normalized.csv", index=False)
print("\nAll figures + CSVs written to:", OUT)
print(f"Include in LaTeX with:  "
      f"\\includegraphics[width={LATEX_WIDTH_FRAC:g}\\textwidth]{{...}}")
