#!/usr/bin/env python3
r"""
Visualize per-head spectral / activation metrics for the 48 CLIP attention heads
(Layers 8-11, Heads 0-11).

Source JSON: head_spec_metrics.json
Each entry "Layer L Head H" carries several metrics; here we focus on three and
relabel them to the names used in the paper/notebook:

    PR_attr  <- Act_Eff_Dim        (participation ratio of attention activations)
    PR_mag   <- Act_Mag_Eff_Dim    (participation ratio of activation magnitudes)
    H_attr   <- Act_Entropy_bits   (entropy of attention activations, in bits)

Outputs a set of figures into ./plots/spec/.

------------------------------------------------------------------------------
LaTeX-ready text sizing
------------------------------------------------------------------------------
When you drop a figure into LaTeX with

    \includegraphics[width=c\textwidth]{fig.png}

LaTeX rescales the whole image by  (c * \textwidth) / (figure width).  If the
figure was authored wider than the slot it lands in, every label/tick/legend is
shrunk by that same factor -- which is why 12 pt text that looks fine at
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
import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# LaTeX integration  -- EDIT THESE to match your document.
# --------------------------------------------------------------------------- #
# Run \the\textwidth in your LaTeX doc and paste the number of points here.
TEXTWIDTH_PT = 345.0
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27          # TeX pt -> inch

# The `c` you will write in \includegraphics[width=c\textwidth]{...}.
# 1.0 = full text width, 0.5 = half, etc.  Text stays the same on-page size.
LATEX_WIDTH_FRAC = 1.5

# Point size the text should *appear* as on the printed/PDF page.
TARGET_FONT_PT = 9.0

DPI = 300
PALETTE = "viridis"          # colormap for heatmaps / scatter coloring
LAYER_COLORS = plt.cm.tab10  # categorical colors for layers
HIST_BINS = 12


def setup(width_frac=None, aspect=0.62, base=None):
    r"""Configure rcParams so the figure is authored at its on-page size.

    width_frac : the `c` in \includegraphics[width=c\textwidth] (defaults to
                 LATEX_WIDTH_FRAC).  The figure's physical width is set to
                 c * \textwidth so LaTeX does NOT rescale it -> fonts keep their
                 point size on the page.
    aspect     : figure height / width.
    base       : on-page font size in pt (defaults to TARGET_FONT_PT).
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
        # stays exactly c*textwidth (do NOT pass bbox_inches='tight' on save,
        # that would re-crop and change the width).
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
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })
    return w, h


# --------------------------------------------------------------------------- #
# Text helpers -- keep text inside the plot, never overflowing onto the data.
# --------------------------------------------------------------------------- #
def wrap(text, width=24):
    """Hard-wrap a string to <= `width` characters per line.

    Used on every title / axis label / legend entry so long strings break onto
    new lines instead of overflowing into neighbouring panels or the plot area.
    """
    if not text:
        return text
    return textwrap.fill(str(text), width=max(4, int(width)))


def chars_per_line(ncols=1, frac=0.85, width_frac=None, base=None):
    """Estimate how many characters fit on one line of a single panel.

    Lets `wrap()` adapt automatically: narrower figures (small LATEX_WIDTH_FRAC)
    or more columns -> fewer characters per line -> earlier wrapping.  The glyph
    estimate is deliberately conservative so centered titles (which can spill
    past the figure edge if wider than their panel) wrap early enough to fit.
    """
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


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "head_spec_metrics.json")
OUT_DIR = os.path.join(HERE, "plots", "spec")

# Map: JSON field  ->  display name used everywhere in the plots.
METRIC_MAP = {
    "Act_Eff_Dim":      "PR_attr",
    "Act_Mag_Eff_Dim":  "PR_mag",
    "Act_Entropy_bits": "H_attr",
    "Group_Entropy_bits": "H_group",
}
# Order in which metrics appear in multi-panel figures.
METRICS = ["PR_attr", "PR_mag", "H_attr"]

# Metric used as a proxy for the head's semantic specificity.
SEMANTIC_METRIC = "H_group"

# Nice axis labels (used in titles / colorbars).
METRIC_LABELS = {
    "PR_attr": "PR_attr  (Act. Effective Dim.)",
    "PR_mag":  "PR_mag  (Act. Magnitude Eff. Dim.)",
    "H_attr":  "H_attr  (Act. Entropy, bits)",
    "H_group": "H_group  (Group Entropy, bits — semantic specificity)",
}


# --------------------------------------------------------------------------- #
# LOAD
# --------------------------------------------------------------------------- #
def load_dataframe(path):
    with open(path) as f:
        raw = json.load(f)

    rows = []
    pat = re.compile(r"Layer\s+(\d+)\s+Head\s+(\d+)")
    for key, vals in raw.items():
        m = pat.match(key)
        if not m:
            continue
        layer, head = int(m.group(1)), int(m.group(2))
        row = {"layer": layer, "head": head, "name": key}
        for json_field, disp in METRIC_MAP.items():
            row[disp] = vals.get(json_field, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["layer", "head"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# PLOTS
# --------------------------------------------------------------------------- #
def plot_histograms(df, out_dir):
    """Distribution of each metric across all 48 heads."""
    setup(aspect=0.42)
    ncols = len(METRICS)
    cw = chars_per_line(ncols)
    fig, axes = plt.subplots(1, ncols)
    for ax, metric in zip(axes, METRICS):
        data = df[metric].dropna()
        ax.hist(data, bins=HIST_BINS, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="crimson", ls="--", lw=1.2,
                   label=f"mean = {data.mean():.2f}")
        ax.axvline(data.median(), color="darkorange", ls=":", lw=1.2,
                   label=f"median = {data.median():.2f}")
        ax.set_title(wrap(METRIC_LABELS[metric], cw))
        ax.set_xlabel(wrap(metric, cw))
        ax.set_ylabel("# heads")
        # 'best' keeps the small legend off the bars; semi-transparent box.
        ax.legend(loc="best", framealpha=0.7, fontsize=ann_size(0.8))
    fig.suptitle(wrap("Distribution of activation metrics across 48 heads",
                      chars_per_line(1)))
    fig.savefig(os.path.join(out_dir, "01_histograms.png"))
    plt.close(fig)


def plot_layerwise_box(df, out_dir):
    """Box + strip plot of each metric grouped by layer."""
    setup(aspect=0.5)
    layers = sorted(df["layer"].unique())
    ncols = len(METRICS)
    cw = chars_per_line(ncols)
    fig, axes = plt.subplots(1, ncols)
    for ax, metric in zip(axes, METRICS):
        data_by_layer = [df.loc[df.layer == L, metric].dropna().values for L in layers]
        bp = ax.boxplot(data_by_layer, tick_labels=[f"L{L}" for L in layers],
                        patch_artist=True, showmeans=True,
                        medianprops=dict(color="black"))
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=LAYER_COLORS(i), alpha=0.45)
        # jittered points
        for i, vals in enumerate(data_by_layer):
            x = np.random.normal(i + 1, 0.05, size=len(vals))
            ax.scatter(x, vals, color=LAYER_COLORS(i), edgecolor="k",
                       linewidth=0.3, s=12, zorder=3, alpha=0.8)
        ax.set_title(wrap(METRIC_LABELS[metric], cw))
        ax.set_xlabel("layer")
        ax.set_ylabel(wrap(metric, cw))
    fig.suptitle(wrap("Layer-wise distribution of activation metrics",
                      chars_per_line(1)))
    fig.savefig(os.path.join(out_dir, "02_layerwise_box.png"))
    plt.close(fig)


def plot_heatmaps(df, out_dir):
    """Layer x Head heatmap for each metric."""
    setup(aspect=0.42)
    layers = sorted(df["layer"].unique())
    heads = sorted(df["head"].unique())
    ncols = len(METRICS)
    cw = chars_per_line(ncols)

    # Size the in-cell value labels so a "x.x" string fits inside one cell.
    panel_in = (LATEX_WIDTH_FRAC * TEXTWIDTH_IN) / ncols
    cell_in = panel_in / max(1, len(heads))
    cell_fs = max(3.0, min(TARGET_FONT_PT - 2.0, cell_in * 72.27 / (3.2 * 0.55)))

    fig, axes = plt.subplots(1, ncols)
    for ax, metric in zip(axes, METRICS):
        grid = np.full((len(layers), len(heads)), np.nan)
        for _, r in df.iterrows():
            grid[layers.index(r["layer"]), heads.index(r["head"])] = r[metric]
        im = ax.imshow(grid, aspect="auto", cmap=PALETTE)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels(heads)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{L}" for L in layers])
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(wrap(METRIC_LABELS[metric], cw))
        ax.tick_params(labelsize=cell_fs + 1)
        ax.grid(False)
        # annotate values
        for i in range(len(layers)):
            for j in range(len(heads)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            color="w", fontsize=cell_fs)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=ann_size(0.8))
    fig.suptitle(wrap("Per-head metrics  (Layer × Head)", chars_per_line(1)))
    fig.savefig(os.path.join(out_dir, "03_heatmaps.png"))
    plt.close(fig)


def plot_per_head_bars(df, out_dir):
    """Bar chart per head (x = head index 0..47), colored by layer."""
    setup(aspect=0.95)
    fig, axes = plt.subplots(len(METRICS), 1, sharex=True)
    x = np.arange(len(df))
    layers = sorted(df["layer"].unique())
    colors = [LAYER_COLORS(layers.index(L)) for L in df["layer"]]
    cw = chars_per_line(1)
    for ax, metric in zip(axes, METRICS):
        ax.bar(x, df[metric].values, color=colors, edgecolor="k", linewidth=0.3)
        ax.set_ylabel(wrap(metric, 14))
        ax.set_title(wrap(METRIC_LABELS[metric], cw), loc="left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        df["name"].str.replace("Layer ", "L").str.replace(" Head ", "H"),
        rotation=90, fontsize=ann_size(0.7))
    axes[-1].set_xlabel("head")
    # Layer legend placed OUTSIDE the axes (constrained_layout reserves room),
    # so it never covers the bars.
    handles = [plt.Rectangle((0, 0), 1, 1, color=LAYER_COLORS(i)) for i in range(len(layers))]
    fig.legend(handles, [f"Layer {L}" for L in layers], ncol=len(layers),
               loc="outside lower center", frameon=False)
    fig.suptitle(wrap("Metric value per head", chars_per_line(1)))
    fig.savefig(os.path.join(out_dir, "04_per_head_bars.png"))
    plt.close(fig)


def plot_correlations(df, out_dir):
    """Pairwise scatter matrix + correlation heatmap."""
    layers = sorted(df["layer"].unique())

    # --- correlation heatmap ---
    setup(aspect=0.85)
    cw = chars_per_line(1)
    corr = df[METRICS].corr()
    fig, ax = plt.subplots()
    # aspect="auto" lets constrained_layout shrink the axes to reserve room for
    # the y-tick labels (aspect="equal" would force a square and clip them).
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([wrap(m, 10) for m in METRICS], rotation=30, ha="right")
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([wrap(m, 10) for m in METRICS])
    ax.grid(False)
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="k", fontsize=ann_size(0.9))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    cb.ax.tick_params(labelsize=ann_size(0.8))
    ax.set_title(wrap("Metric correlation (Pearson)", cw))
    fig.savefig(os.path.join(out_dir, "05_correlation_heatmap.png"))
    plt.close(fig)

    # --- pairwise scatter matrix, colored by layer ---
    setup(aspect=0.92)
    n = len(METRICS)
    cw = chars_per_line(n)
    fig, axes = plt.subplots(n, n)
    for i, my in enumerate(METRICS):
        for j, mx in enumerate(METRICS):
            ax = axes[i, j]
            ax.tick_params(labelsize=ann_size(0.75))
            if i == j:
                ax.hist(df[mx].dropna(), bins=HIST_BINS, color="#777", alpha=0.8)
                if j == 0:
                    ax.set_ylabel("count")
            else:
                for k, L in enumerate(layers):
                    sub = df[df.layer == L]
                    ax.scatter(sub[mx], sub[my], color=LAYER_COLORS(k),
                               edgecolor="k", linewidth=0.3, s=10,
                               alpha=0.85, label=f"L{L}")
                r = df[[mx, my]].corr().iloc[0, 1]
                ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=ann_size(0.8), color="crimson", va="top",
                        bbox=dict(fc="white", ec="none", alpha=0.7))
            if i == n - 1:
                ax.set_xlabel(wrap(mx, cw))
            if j == 0 and i != j:
                ax.set_ylabel(wrap(my, cw))
    handles = [plt.Line2D([], [], marker="o", ls="", color=LAYER_COLORS(k),
                          markeredgecolor="k", label=f"Layer {L}")
               for k, L in enumerate(layers)]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(layers),
               frameon=False)
    fig.suptitle(wrap("Pairwise relationships between metrics (colored by layer)",
                      chars_per_line(1)))
    fig.savefig(os.path.join(out_dir, "06_scatter_matrix.png"))
    plt.close(fig)


def plot_semantic_vs_metrics(df, out_dir):
    """Compare the head's semantic specificity (H_group / Group_Entropy_bits)
    against each of the three activation metrics: one scatter panel per metric,
    points colored by layer, with a fitted trend line and Pearson r."""
    setup(aspect=0.5)
    layers = sorted(df["layer"].unique())
    sem = SEMANTIC_METRIC
    ncols = len(METRICS)
    cw = chars_per_line(ncols)

    fig, axes = plt.subplots(1, ncols)
    for ax, metric in zip(axes, METRICS):
        # per-layer colored scatter
        for k, L in enumerate(layers):
            sub = df[df.layer == L]
            ax.scatter(sub[metric], sub[sem], color=LAYER_COLORS(k),
                       edgecolor="k", linewidth=0.3, s=14, alpha=0.85,
                       label=f"Layer {L}")

        # global least-squares trend line + correlation
        d = df[[metric, sem]].dropna()
        r = d[metric].corr(d[sem])
        if len(d) >= 2:
            b, a = np.polyfit(d[metric], d[sem], 1)
            xs = np.linspace(d[metric].min(), d[metric].max(), 100)
            ax.plot(xs, b * xs + a, color="crimson", lw=1.5, ls="--")

        ax.set_xlabel(wrap(METRIC_LABELS[metric], cw))
        if ax is axes[0]:
            ax.set_ylabel(wrap(METRIC_LABELS[sem], cw))
        ax.set_title(wrap(f"H_group vs {metric}  (r = {r:.2f})", cw))

    # one shared layer legend, outside the panels
    handles = [plt.Line2D([], [], marker="o", ls="", color=LAYER_COLORS(k),
                          markeredgecolor="k", label=f"Layer {L}")
               for k, L in enumerate(layers)]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(layers),
               frameon=False)
    fig.suptitle(wrap("Semantic specificity (Group Entropy) vs activation metrics",
                      chars_per_line(1)))
    fig.savefig(os.path.join(out_dir, "08_semantic_vs_metrics.png"))
    plt.close(fig)


def plot_layer_means(df, out_dir):
    """Line plot of mean per layer for each metric (z-scored to compare)."""
    setup(aspect=0.62)
    layers = sorted(df["layer"].unique())
    cw = chars_per_line(1)
    fig, ax = plt.subplots()
    for metric in METRICS:
        m = df.groupby("layer")[metric].mean()
        # z-score so the three metrics share a y-axis
        z = (m - m.mean()) / (m.std() + 1e-9)
        ax.plot(layers, z.values, marker="o", lw=1.6, label=metric)
    ax.set_xlabel("layer")
    ax.set_ylabel(wrap("layer-mean (z-scored across layers)", cw))
    ax.set_title(wrap("Layer-wise trend of each metric (z-scored)", cw))
    ax.set_xticks(layers)
    ax.legend(loc="best", framealpha=0.7)
    fig.savefig(os.path.join(out_dir, "07_layer_means_trend.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(0)

    df = load_dataframe(JSON_PATH)
    print(f"Loaded {len(df)} heads, layers {sorted(df.layer.unique())}.")

    # summary table to stdout + csv
    summary = df.groupby("layer")[METRICS].agg(["mean", "std", "min", "max"])
    print("\nLayer-wise summary:")
    print(summary.round(3))
    df.to_csv(os.path.join(OUT_DIR, "metrics_table.csv"), index=False)

    plot_histograms(df, OUT_DIR)
    plot_layerwise_box(df, OUT_DIR)
    plot_heatmaps(df, OUT_DIR)
    plot_per_head_bars(df, OUT_DIR)
    plot_correlations(df, OUT_DIR)
    plot_semantic_vs_metrics(df, OUT_DIR)
    plot_layer_means(df, OUT_DIR)

    print(f"\nSaved figures + metrics_table.csv to: {OUT_DIR}")
    print(f"Include in LaTeX with:  "
          f"\\includegraphics[width={LATEX_WIDTH_FRAC:g}\\textwidth]{{...}}")
    for f in sorted(os.listdir(OUT_DIR)):
        print("  -", f)


if __name__ == "__main__":
    main()
