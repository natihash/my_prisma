#!/usr/bin/env python3
"""
Plot statistics about the semantic evolution of the last 48 heads of CLIP ViT-B/16
(layers 8-11, 12 heads each) under different adaptation techniques and datasets.

Data sources (three JSON files, slightly different structures):
  - imagenet1k/changes_merged.json
        keys  : "Layer L Head H"
        techs : lora4 / lora16 / lp / fft
        value : {"category", "rank_mean", "top_k_mean"}
        cats  : Ignored / different_function / specialized / same_function
  - sun/semantic_shift_simple.json
  - text/semantic_shift_simple.json
        keys  : "L{L}.H{H}"
        techs : FFT / LoRA4 / LoRA16 / LP
        value : category string ("repurposed", "ignored", "specialized",
                "same function", sometimes suffixed with "(<number>)")

The four semantic-change groups are unified across files as:
        ignored , same_function , specialized , repurposed
(imagenet's "different_function" == sun/text's "repurposed").

Run with the prisma venv:
    ~/prisma/bin/python plot_semantic_evolution.py

All figures are written next to this script.
"""

import json
import os
import re
import textwrap
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# --------------------------------------------------------------------------- #
# LaTeX integration  -- EDIT THESE to match your document.
# --------------------------------------------------------------------------- #
# When a figure is included with \includegraphics[width=c\textwidth]{...}, LaTeX
# rescales it by (c*\textwidth)/(figure width); if the figure was authored wider
# than the slot, every label/tick/legend is shrunk by that factor (which is why
# text fine at \textwidth becomes unreadable at 0.5\textwidth).
#
# Fix: author each figure at a physical width of exactly c*\textwidth and set
# fonts to the point size you want ON the page.  Then LaTeX's scale factor is 1
# and text renders at that point size regardless of c.  Set LATEX_WIDTH_FRAC to
# the same c you use in \includegraphics, set TARGET_FONT_PT to the on-page size,
# re-run, and include with width=<LATEX_WIDTH_FRAC>\textwidth.
TEXTWIDTH_PT = 345.0                          # \the\textwidth of your doc, in pt
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27           # TeX pt -> inch
LATEX_WIDTH_FRAC = 1.0                         # the c in width=c\textwidth
TARGET_FONT_PT = 9.0                           # on-page text size in pt
DPI = 300


def setup(width_frac=None, aspect=0.62, base=None):
    r"""Configure rcParams so the figure is authored at its on-page size.

    The figure's physical width is set to (width_frac * \textwidth) so LaTeX does
    NOT rescale it at \includegraphics[width=width_frac\textwidth] -> fonts keep
    their chosen point size on the page, for any width_frac.
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
        # that re-crops and changes the width).  Do not mix with tight_layout().
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
    """Hard-wrap a string to <= `width` characters per line, so long titles /
    labels / legend entries break onto new lines instead of overflowing into a
    neighbouring panel or onto the plotted data.  Hyphenated words and long
    tokens are kept intact."""
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


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # .../org_clip_metrics_real
OUT = HERE

FILES = {
    "imagenet1k": os.path.join(ROOT, "imagenet1k", "changes_merged.json"),
    "sun":        os.path.join(ROOT, "sun", "semantic_shift_simple.json"),
    "text":       os.path.join(ROOT, "text", "semantic_shift_simple.json"),
}

# --------------------------------------------------------------------------- #
# Canonical vocabularies
# --------------------------------------------------------------------------- #
# Ordered from "least change" to "most change" w.r.t. the original head function.
CATEGORIES = ["ignored", "same_function", "specialized", "repurposed"]
CAT_LABELS = {
    "ignored": "Ignored",
    "same_function": "Same function",
    "specialized": "Specialized",
    "repurposed": "Repurposed",
}
# Sequential-ish, perceptually ordered colours (grey -> green -> orange -> red).
CAT_COLORS = {
    "ignored":       "#9e9e9e",
    "same_function": "#4caf50",
    "specialized":   "#ff9800",
    "repurposed":    "#e53935",
}

TECHNIQUES = ["LP", "LoRA4", "LoRA16", "FFT"]  # ordered by ~capacity / # trainable params
TECH_NORM = {
    "lp": "LP", "lora4": "LoRA4", "lora16": "LoRA16", "fft": "FFT",
    "LP": "LP", "LoRA4": "LoRA4", "LoRA16": "LoRA16", "FFT": "FFT",
}

DATASETS = ["imagenet1k", "sun", "text"]

LAYERS = [8, 9, 10, 11]
HEADS_PER_LAYER = 12

# raw category string  ->  canonical category
CAT_NORM = {
    "ignored": "ignored",
    "Ignored": "ignored",
    "same_function": "same_function",
    "same function": "same_function",
    "specialized": "specialized",
    "repurposed": "repurposed",
    "different_function": "repurposed",
}


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
_paren_re = re.compile(r"\s*\(.*?\)\s*$")
_key_re_words = re.compile(r"Layer\s+(\d+)\s+Head\s+(\d+)")
_key_re_short = re.compile(r"L(\d+)\.H(\d+)")


def normalize_category(raw):
    """Strip a possible '(0.70)' suffix and map to a canonical category."""
    s = _paren_re.sub("", str(raw)).strip()
    if s not in CAT_NORM:
        raise ValueError(f"Unknown category string: {raw!r} (stripped {s!r})")
    return CAT_NORM[s]


def parse_head_key(key):
    """Return (layer, head) from either 'Layer 8 Head 0' or 'L8.H0'."""
    m = _key_re_words.search(key) or _key_re_short.search(key)
    if not m:
        raise ValueError(f"Cannot parse head key: {key!r}")
    return int(m.group(1)), int(m.group(2))


def load_long_dataframe():
    """Load all three files into one tidy long-format DataFrame.

    Columns: dataset, layer, head, head_id, head_order, technique,
             category, rank_mean, top_k_mean (latter two NaN unless imagenet).
    """
    rows = []
    for dataset, path in FILES.items():
        with open(path) as fh:
            data = json.load(fh)
        for head_key, tech_dict in data.items():
            layer, head = parse_head_key(head_key)
            for tech_raw, val in tech_dict.items():
                tech = TECH_NORM[tech_raw]
                if isinstance(val, dict):  # imagenet style
                    cat = normalize_category(val["category"])
                    rank_mean = val.get("rank_mean", np.nan)
                    top_k_mean = val.get("top_k_mean", np.nan)
                else:  # plain string
                    cat = normalize_category(val)
                    rank_mean = np.nan
                    top_k_mean = np.nan
                rows.append({
                    "dataset": dataset,
                    "layer": layer,
                    "head": head,
                    "head_id": f"L{layer}.H{head}",
                    "head_order": (layer - LAYERS[0]) * HEADS_PER_LAYER + head,
                    "technique": tech,
                    "category": cat,
                    "rank_mean": rank_mean,
                    "top_k_mean": top_k_mean,
                })
    df = pd.DataFrame(rows)
    df["category"] = pd.Categorical(df["category"], categories=CATEGORIES, ordered=True)
    df["technique"] = pd.Categorical(df["technique"], categories=TECHNIQUES, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    return df


# --------------------------------------------------------------------------- #
# Generic plotting helpers
# --------------------------------------------------------------------------- #
def pct_table(sub, index, columns="category"):
    """Cross-tab `index` x `columns` as row-normalised percentages (rows sum to 100)."""
    ct = pd.crosstab(sub[index], sub[columns])
    ct = ct.reindex(columns=CATEGORIES, fill_value=0)
    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return pct.fillna(0)


def stacked_bar(ax, pct, title, xlabel, tw=None):
    """Draw a stacked percentage bar chart from a (index x category) % table."""
    if tw is None:
        tw = chars_per_line(1)
    bottoms = np.zeros(len(pct))
    x = np.arange(len(pct))
    for cat in CATEGORIES:
        vals = pct[cat].values
        ax.bar(x, vals, bottom=bottoms, color=CAT_COLORS[cat],
               label=CAT_LABELS[cat], edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 6:  # annotate only reasonably large segments
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=ann_size(0.85), color="white", fontweight="bold")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(str(i), 14) for i in pct.index], rotation=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of heads")
    ax.set_xlabel(wrap(xlabel, tw))
    ax.set_title(wrap(title, tw))


def cat_legend(fig, ncol=4):
    handles = [Patch(facecolor=CAT_COLORS[c], label=CAT_LABELS[c]) for c in CATEGORIES]
    # 'outside lower center' is reserved by constrained_layout, so the legend
    # sits below the axes without covering data and without being clipped on save.
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False)


def save(fig, name):
    path = os.path.join(OUT, name)
    # No bbox_inches='tight': constrained_layout already fits everything inside
    # the canvas, and keeping the figure size preserves the exact
    # width_frac*textwidth width so LaTeX does not rescale (shrink) the text.
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}")


# --------------------------------------------------------------------------- #
# 1. Per-dataset: % heads per category across techniques
# --------------------------------------------------------------------------- #
def plot_per_dataset_technique(df):
    """One stacked-bar panel per dataset; x-axis = technique."""
    setup(width_frac=1.5, aspect=0.5)
    fig, axes = plt.subplots(1, len(DATASETS), sharey=True)
    cw = chars_per_line(len(DATASETS))
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        pct = pct_table(sub, index="technique")
        pct = pct.reindex(TECHNIQUES)
        stacked_bar(ax, pct, f"{ds}", "Adaptation technique", tw=cw)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle(wrap("Semantic-change category distribution per dataset and "
                      "technique (% of the 48 adapted heads)", chars_per_line(1)))
    cat_legend(fig)
    save(fig, "01_per_dataset_per_technique_stacked.png")


def plot_per_dataset_grouped(df):
    """Grouped bars: for each dataset, % of heads in each category (pooled over techniques)."""
    setup(aspect=0.6)
    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(len(DATASETS))
    for i, cat in enumerate(CATEGORIES):
        vals = []
        for ds in DATASETS:
            sub = df[df["dataset"] == ds]
            vals.append((sub["category"] == cat).mean() * 100)
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      color=CAT_COLORS[cat], label=CAT_LABELS[cat])
        ax.bar_label(bars, fmt="%.0f", fontsize=ann_size(0.75), padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel(wrap("% of head-technique pairs", chars_per_line(1)))
    ax.set_xlabel("Adaptation dataset")
    ax.set_title(wrap("Semantic-change category by dataset "
                      "(pooled over the 4 techniques)", chars_per_line(1)))
    ax.margins(y=0.12)            # headroom so bar-top labels stay on-canvas
    ax.set_ylim(bottom=0)
    cat_legend(fig)
    save(fig, "02_per_dataset_grouped.png")


# --------------------------------------------------------------------------- #
# 2. Across techniques, pooled over all datasets
# --------------------------------------------------------------------------- #
def plot_per_technique_pooled(df):
    """Stacked bar: x = technique, pooled over all datasets."""
    setup(aspect=0.7)
    pct = pct_table(df, index="technique").reindex(TECHNIQUES)
    fig, ax = plt.subplots()
    stacked_bar(ax, pct, "Semantic-change category per technique "
                "(all 3 datasets pooled)", "Adaptation technique")
    cat_legend(fig)
    save(fig, "03_per_technique_pooled_stacked.png")


def plot_per_technique_grouped(df):
    """Grouped bars: per technique, % heads in each category, pooled over datasets."""
    setup(aspect=0.6)
    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(len(TECHNIQUES))
    for i, cat in enumerate(CATEGORIES):
        vals = []
        for tech in TECHNIQUES:
            sub = df[df["technique"] == tech]
            vals.append((sub["category"] == cat).mean() * 100)
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      color=CAT_COLORS[cat], label=CAT_LABELS[cat])
        ax.bar_label(bars, fmt="%.0f", fontsize=ann_size(0.75), padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(TECHNIQUES)
    ax.set_ylabel(wrap("% of head-dataset pairs", chars_per_line(1)))
    ax.set_xlabel("Adaptation technique")
    ax.set_title(wrap("Semantic-change category by technique "
                      "(pooled over the 3 datasets)", chars_per_line(1)))
    ax.margins(y=0.12)            # headroom so bar-top labels stay on-canvas
    ax.set_ylim(bottom=0)
    cat_legend(fig)
    save(fig, "04_per_technique_grouped.png")


# --------------------------------------------------------------------------- #
# 3a. Full dataset x technique grid (small multiples) + a master matrix
# --------------------------------------------------------------------------- #
def plot_dataset_technique_grid(df):
    """3x4 grid of stacked single bars (one per dataset/technique) is overkill;
    instead a heatmap of % per category for every dataset-technique combination."""
    combos = [(ds, tech) for ds in DATASETS for tech in TECHNIQUES]
    mat = np.zeros((len(combos), len(CATEGORIES)))
    labels = []
    for r, (ds, tech) in enumerate(combos):
        sub = df[(df["dataset"] == ds) & (df["technique"] == tech)]
        for c, cat in enumerate(CATEGORIES):
            mat[r, c] = (sub["category"] == cat).mean() * 100
        labels.append(f"{ds} / {tech}")

    setup(aspect=1.25)
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([wrap(CAT_LABELS[c], 12) for c in CATEGORIES],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(labelsize=ann_size(0.8))
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center",
                    color="white" if mat[r, c] < 60 else "black",
                    fontsize=ann_size(0.8))
    # separate datasets with horizontal lines
    for k in range(1, len(DATASETS)):
        ax.axhline(k * len(TECHNIQUES) - 0.5, color="white", linewidth=2)
    ax.set_title(wrap("% of heads per category for every dataset / technique "
                      "combination", chars_per_line(1)))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="% of heads")
    cb.ax.tick_params(labelsize=ann_size(0.8))
    save(fig, "05_dataset_technique_category_heatmap.png")


# --------------------------------------------------------------------------- #
# 3b. Per-head category heatmaps (head x technique), one panel per dataset
# --------------------------------------------------------------------------- #
def plot_head_technique_heatmaps(df):
    cat_to_int = {c: i for i, c in enumerate(CATEGORIES)}
    cmap = ListedColormap([CAT_COLORS[c] for c in CATEGORIES])
    norm = BoundaryNorm(np.arange(-0.5, len(CATEGORIES) + 0.5), cmap.N)

    head_order = (df[["head_order", "head_id"]]
                  .drop_duplicates().sort_values("head_order"))
    ordered_ids = head_order["head_id"].tolist()

    setup(aspect=0.85)
    fig, axes = plt.subplots(1, len(DATASETS), sharey=True)
    cw = chars_per_line(len(DATASETS))
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        piv = (sub.pivot_table(index="head_id", columns="technique",
                               values="category", observed=True,
                               aggfunc=lambda s: cat_to_int[s.iloc[0]])
               .reindex(index=ordered_ids, columns=TECHNIQUES))
        ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(TECHNIQUES)))
        ax.set_xticklabels(TECHNIQUES, rotation=45, ha="right")
        ax.set_title(wrap(ds, cw))
        ax.tick_params(labelsize=ann_size(0.8))
        # layer boundaries
        for k in range(1, len(LAYERS)):
            ax.axhline(k * HEADS_PER_LAYER - 0.5, color="black", linewidth=1.2)
        if ax is axes[0]:
            ax.set_yticks(range(len(ordered_ids)))
            ax.set_yticklabels(ordered_ids, fontsize=ann_size(0.6))
            ax.set_ylabel(wrap("Head (grouped by layer 8->11)", chars_per_line(1)))
    fig.suptitle(wrap("Per-head semantic category by technique and dataset",
                      chars_per_line(1)))
    cat_legend(fig)
    save(fig, "06_per_head_category_heatmaps.png")


# --------------------------------------------------------------------------- #
# 3c. Layer-wise category composition
# --------------------------------------------------------------------------- #
def plot_layerwise(df):
    """For each dataset, stacked bars of category composition per layer (pooled techniques)."""
    setup(aspect=0.5)
    fig, axes = plt.subplots(1, len(DATASETS), sharey=True)
    cw = chars_per_line(len(DATASETS))
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        pct = pct_table(sub, index="layer").reindex(LAYERS)
        pct.index = [f"L{l}" for l in pct.index]
        stacked_bar(ax, pct, f"{ds}", "Layer", tw=cw)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle(wrap("Category composition by layer (pooled over techniques)",
                      chars_per_line(1)))
    cat_legend(fig)
    save(fig, "07_layerwise_composition.png")


def plot_repurposed_by_layer(df):
    """Line plot: fraction of heads that are 'repurposed' (most changed) per layer,
    one line per technique, faceted by dataset."""
    setup(aspect=0.45)
    fig, axes = plt.subplots(1, len(DATASETS), sharey=True)
    cw = chars_per_line(len(DATASETS))
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        for tech in TECHNIQUES:
            ys = []
            for l in LAYERS:
                s = sub[(sub["technique"] == tech) & (sub["layer"] == l)]
                changed = s["category"].isin(["specialized", "repurposed"]).mean() * 100
                ys.append(changed)
            ax.plot(LAYERS, ys, marker="o", label=tech)
        ax.set_title(wrap(ds, cw))
        ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS)
        if ax is axes[0]:
            ax.set_ylabel(wrap("% heads specialized or repurposed", chars_per_line(1)))
    axes[-1].legend(title="Technique", frameon=False, loc="best",
                    framealpha=0.7, fontsize=ann_size(0.85))
    fig.suptitle(wrap("Degree of functional change vs. depth "
                      "(% of heads that are specialized or repurposed)",
                      chars_per_line(1)))
    save(fig, "08_change_vs_layer.png")


# --------------------------------------------------------------------------- #
# 3d. Cross-technique agreement
# --------------------------------------------------------------------------- #
def plot_technique_agreement(df):
    """For each dataset, heatmap of pairwise technique agreement
    (fraction of heads assigned the same category)."""
    setup(aspect=0.42)
    fig, axes = plt.subplots(1, len(DATASETS))
    cw = chars_per_line(len(DATASETS))
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        piv = sub.pivot_table(index="head_id", columns="technique",
                              values="category", aggfunc="first", observed=True)
        n = len(TECHNIQUES)
        agree = np.eye(n)
        for i, j in combinations(range(n), 2):
            a = piv[TECHNIQUES[i]].astype(str)
            b = piv[TECHNIQUES[j]].astype(str)
            frac = (a.values == b.values).mean()
            agree[i, j] = agree[j, i] = frac
        im = ax.imshow(agree, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(TECHNIQUES, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(TECHNIQUES)
        ax.tick_params(labelsize=ann_size(0.8))
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{agree[i, j]:.2f}", ha="center", va="center",
                        color="white" if agree[i, j] < 0.6 else "black",
                        fontsize=ann_size(0.85))
        ax.set_title(wrap(ds, cw))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=ann_size(0.8))
    fig.suptitle(wrap("Cross-technique agreement on head category "
                      "(fraction of the 48 heads given the same label)",
                      chars_per_line(1)))
    save(fig, "09_technique_agreement.png")


def plot_consistency(df):
    """How many distinct categories does each head receive across the 4 techniques?
    Histogram per dataset (1 = fully consistent, 4 = fully inconsistent)."""
    setup(aspect=0.6)
    fig, ax = plt.subplots()
    width = 0.25
    x = np.arange(1, 5)
    for k, ds in enumerate(DATASETS):
        sub = df[df["dataset"] == ds]
        ndistinct = sub.groupby("head_id", observed=True)["category"].nunique()
        counts = np.array([(ndistinct == v).sum() for v in x], dtype=float)
        counts = counts / counts.sum() * 100
        ax.bar(x + (k - 1) * width, counts, width, label=ds)
    ax.set_xticks(x)
    ax.set_xlabel(wrap("# distinct categories a head receives across the 4 "
                       "techniques", chars_per_line(1)))
    ax.set_ylabel("% of heads")
    ax.set_title(wrap("Consistency of a head's behaviour across adaptation "
                      "techniques", chars_per_line(1)))
    ax.legend(title="Dataset", frameon=False, loc="best",
              framealpha=0.7, fontsize=ann_size(0.85))
    save(fig, "10_head_consistency_across_techniques.png")


def plot_dataset_agreement(df):
    """Pairwise dataset agreement on head category, per technique (heatmap grid)."""
    setup(aspect=0.32)
    fig, axes = plt.subplots(1, len(TECHNIQUES))
    cw = chars_per_line(len(TECHNIQUES))
    for ax, tech in zip(axes, TECHNIQUES):
        sub = df[df["technique"] == tech]
        piv = sub.pivot_table(index="head_id", columns="dataset",
                              values="category", aggfunc="first", observed=True)
        n = len(DATASETS)
        agree = np.eye(n)
        for i, j in combinations(range(n), 2):
            a = piv[DATASETS[i]].astype(str)
            b = piv[DATASETS[j]].astype(str)
            frac = (a.values == b.values).mean()
            agree[i, j] = agree[j, i] = frac
        im = ax.imshow(agree, cmap="cividis", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels([wrap(d, 9) for d in DATASETS], rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels([wrap(d, 9) for d in DATASETS])
        ax.tick_params(labelsize=ann_size(0.75))
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{agree[i, j]:.2f}", ha="center", va="center",
                        color="white" if agree[i, j] < 0.6 else "black",
                        fontsize=ann_size(0.8))
        ax.set_title(wrap(tech, cw))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=ann_size(0.75))
    fig.suptitle(wrap("Cross-dataset agreement on head category (per technique)",
                      chars_per_line(1)))
    save(fig, "11_dataset_agreement.png")


# --------------------------------------------------------------------------- #
# 3e. ImageNet-only numeric metrics (rank_mean, top_k_mean)
# --------------------------------------------------------------------------- #
def plot_imagenet_metrics(df):
    sub = df[(df["dataset"] == "imagenet1k") & df["rank_mean"].notna()].copy()
    if sub.empty:
        return

    # (a) Boxplots of each metric by category
    for metric, fname, ylabel in [
        ("rank_mean", "12_imagenet_rankmean_by_category.png", "rank_mean"),
        ("top_k_mean", "13_imagenet_topkmean_by_category.png", "top_k_mean"),
    ]:
        setup(aspect=0.62)
        fig, ax = plt.subplots()
        data = [sub[sub["category"] == c][metric].values for c in CATEGORIES]
        bp = ax.boxplot(data,
                        tick_labels=[wrap(CAT_LABELS[c], 12) for c in CATEGORIES],
                        patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], CATEGORIES):
            patch.set_facecolor(CAT_COLORS[c]); patch.set_alpha(0.7)
        # jittered points
        for i, c in enumerate(CATEGORIES, start=1):
            ys = sub[sub["category"] == c][metric].values
            xs = np.random.normal(i, 0.05, size=len(ys))
            ax.scatter(xs, ys, s=10, color="black", alpha=0.4, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Semantic category")
        ax.set_title(wrap(f"ImageNet: {ylabel} distribution by semantic category "
                          "(all techniques pooled)", chars_per_line(1)))
        save(fig, fname)

    # (b) Scatter rank_mean vs top_k_mean coloured by category, per technique
    setup(aspect=0.4)
    fig, axes = plt.subplots(1, len(TECHNIQUES), sharex=True, sharey=True)
    cw = chars_per_line(len(TECHNIQUES))
    for ax, tech in zip(axes, TECHNIQUES):
        s = sub[sub["technique"] == tech]
        for c in CATEGORIES:
            sc = s[s["category"] == c]
            ax.scatter(sc["rank_mean"], sc["top_k_mean"], s=14,
                       color=CAT_COLORS[c], label=CAT_LABELS[c],
                       edgecolor="k", linewidth=0.3, alpha=0.85)
        ax.set_title(wrap(tech, cw))
        ax.set_xlabel("rank_mean")
        if ax is axes[0]:
            ax.set_ylabel("top_k_mean")
    cat_legend(fig)
    fig.suptitle(wrap("ImageNet: rank_mean vs top_k_mean by technique, "
                      "coloured by category", chars_per_line(1)))
    save(fig, "14_imagenet_rank_vs_topk_scatter.png")

    # (c) top_k_mean by layer & technique (mean +/- std) -> how strongly heads
    #     activate the residual stream as depth increases
    setup(aspect=0.62)
    fig, ax = plt.subplots()
    for tech in TECHNIQUES:
        means, stds = [], []
        for l in LAYERS:
            s = sub[(sub["technique"] == tech) & (sub["layer"] == l)]["top_k_mean"]
            means.append(s.mean()); stds.append(s.std())
        ax.errorbar(LAYERS, means, yerr=stds, marker="o", capsize=3, label=tech)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Layer")
    ax.set_ylabel(wrap("top_k_mean (mean ± std)", chars_per_line(1)))
    ax.set_title(wrap("ImageNet: top_k_mean magnitude vs depth, per technique",
                      chars_per_line(1)))
    ax.legend(title="Technique", frameon=False, loc="best",
              framealpha=0.7, fontsize=ann_size(0.85))
    save(fig, "15_imagenet_topk_vs_layer.png")


# --------------------------------------------------------------------------- #
# 3f. Overall category mix (one bar) and a compact summary
# --------------------------------------------------------------------------- #
def plot_overall_mix(df):
    setup(aspect=0.9)
    fig, ax = plt.subplots()
    pct = pct_table(df.assign(all="all"), index="all")
    stacked_bar(ax, pct, "Overall semantic-change mix "
                "(all heads, datasets, techniques)", "")
    ax.set_xticks([0])
    ax.set_xticklabels(["all 3 datasets\n× 4 techniques"])
    cat_legend(fig, ncol=2)
    save(fig, "16_overall_mix.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    np.random.seed(0)
    df = load_long_dataframe()

    # quick sanity print
    print(f"Loaded {len(df)} head-technique-dataset records "
          f"({df['head_id'].nunique()} heads x {len(TECHNIQUES)} techniques "
          f"x {len(DATASETS)} datasets).")
    print("Category counts:\n", df["category"].value_counts().reindex(CATEGORIES))

    # also dump the tidy table for reference / further analysis
    df.to_csv(os.path.join(OUT, "semantic_evolution_long.csv"), index=False)
    print(f"  saved semantic_evolution_long.csv")

    print("Generating figures...")
    plot_per_dataset_technique(df)        # 01
    plot_per_dataset_grouped(df)          # 02
    plot_per_technique_pooled(df)         # 03
    plot_per_technique_grouped(df)        # 04
    plot_dataset_technique_grid(df)       # 05
    plot_head_technique_heatmaps(df)      # 06
    plot_layerwise(df)                    # 07
    plot_repurposed_by_layer(df)          # 08
    plot_technique_agreement(df)          # 09
    plot_consistency(df)                  # 10
    plot_dataset_agreement(df)            # 11
    plot_imagenet_metrics(df)             # 12-15
    plot_overall_mix(df)                  # 16
    print(f"Done. Figures written to {OUT}")


if __name__ == "__main__":
    main()
