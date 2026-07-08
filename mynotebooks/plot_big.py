"""
Visualize ViT head semantics as a (heads x layers) grid.
Each cell = one horizontal bar, normalized to that head's total score:
  - colored/patterned segments for a chosen budget of important groups
  - one grey "Other" segment for everything else
This shows the dominant group AND how peaked/spread each head is.

Group-selection scheme (95 groups is too many to color):
  1. LAST layer: take the single top group of EACH head -> solid color.
     (These are the deepest, most task-relevant signatures; every last-layer
      head's dominant group is guaranteed a unique style.)
  2. EARLIER layers (the few layers before the last): take the top group of each
     head, then keep the groups that recur most often across those heads. Ties
     (e.g. everything appears once) are broken by the largest score gap between a
     head's #1 and #2 group -> hatched color.
  3. Everything outside the budget collapses into a single grey "Other".

Because >12 distinct colors are hard to tell apart, the earlier-layer groups
reuse the palette but carry a dense hatch, so color + pattern keeps them
separable. The legend separates the solid (last-layer) groups from the hatched
(earlier-layer) ones.
"""
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- config ----------------
INPUT_JSON     = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/all_group_scores.json"
N_LAST_LAYER   = 12            # cap on solid-colored groups from the LAST layer
N_EXTRA        = 6             # hatched groups pulled from the earlier layers
N_EXTRA_LAYERS = 3             # how many layers *before* the last to scan
OUT_BASE       = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/plots3/head_semantics"
DEEP_HATCH     = "xxxx"        # dense (double) crosshatch for earlier-layer groups
# -----------------------------------------

def short(name, n=30):
    name = name.strip()
    return name if len(name) <= n else name[:n - 1].rstrip() + "…"

# ---- load & parse ----
with open(INPUT_JSON) as f:
    data = json.load(f)

heads_data = {}                       # (layer, head) -> {group: score}
layers, heads = set(), set()
for entry in data.values():
    m = re.search(r"Layer\s+(\d+)\s+Head\s+(\d+)", entry["Head_Name"])
    l, h = int(m.group(1)), int(m.group(2))
    layers.add(l); heads.add(h)
    heads_data[(l, h)] = entry["Scores"]
layers, heads = sorted(layers), sorted(heads)

# ---- per-head dominant group (and its margin over #2) within a layer ----
def top_group_per_head(layer):
    """List of (group, gap) for the #1 group of each head in `layer`."""
    out = []
    for h in heads:
        s = heads_data.get((layer, h))
        if not s:
            continue
        order = sorted(s.items(), key=lambda kv: kv[1], reverse=True)
        if not order or order[0][1] <= 0:
            continue
        gap = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        out.append((order[0][0], gap))
    return out

def rank_by_freq_then_gap(per_head_tops):
    """Order groups by how often they are a head's #1, ties broken by max gap."""
    freq, gap = Counter(), {}
    for g, gp in per_head_tops:
        freq[g] += 1
        gap[g] = max(gap.get(g, 0.0), gp)
    return sorted(freq, key=lambda g: (freq[g], gap[g]), reverse=True), freq

# ---- criterion 1: every per-head top group of the LAST layer (solid) ----
last_layer = layers[-1]
primary, last_freq = rank_by_freq_then_gap(top_group_per_head(last_layer))
primary = primary[:N_LAST_LAYER]
primary_set = set(primary)

# ---- criterion 2: most-recurring per-head top groups of the EARLIER layers (hatched) ----
prev_layers = [l for l in layers if l != last_layer][-N_EXTRA_LAYERS:]
prev_tops = []
for l in prev_layers:
    prev_tops += top_group_per_head(l)
prev_ranked, prev_freq = rank_by_freq_then_gap(prev_tops)
extra = [g for g in prev_ranked if g not in primary_set][:N_EXTRA]

budget = primary + extra
budget_set = set(budget)

# ---- styles: solid colors for last layer, same colors + dense hatch for earlier ----
PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#e6ab02",  # gold
    "#1b9e77",  # teal
    "#e7298a",  # magenta
]
GREY = (0.82, 0.82, 0.82)
plt.rcParams["hatch.linewidth"] = 0.9   # make the pattern read clearly (not too faint)

styles = {}                              # group -> (color, hatch)
for i, g in enumerate(primary):
    # if a dataset ever has more last-layer groups than colors, hatch the overflow
    styles[g] = (PALETTE[i % len(PALETTE)], "" if i < len(PALETTE) else "....")
for j, g in enumerate(extra):
    styles[g] = (PALETTE[j % len(PALETTE)], DEEP_HATCH)

print(f"{len(budget_set)} groups styled: {len(primary)} solid (last layer {last_layer}), "
      f"{len(extra)} hatched (layers {prev_layers}); rest -> Other.")
print("Last-layer top groups (solid):")
for g in primary:
    print(f"   {last_freq[g]}x  {g}")
print("Earlier-layer recurring groups (hatched):")
for g in extra:
    print(f"   {prev_freq[g]}x  {g}")

# ---- build the figure ----
nrows, ncols = len(heads), len(layers)
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(2.2 * ncols, 0.46 * nrows + 1.6),
                         squeeze=False)

for ri, h in enumerate(heads):
    for ci, l in enumerate(layers):
        ax = axes[ri][ci]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        # subtle cell frame so empty cells and bar extents are readable
        ax.add_patch(plt.Rectangle((0, 0.12), 1, 0.76, fill=False,
                                   edgecolor=(0.88, 0.88, 0.88), linewidth=0.5))
        scores = heads_data.get((l, h))
        if not scores or sum(scores.values()) <= 0:
            ax.text(0.5, 0.5, "–", ha="center", va="center",
                    fontsize=8, color=(0.7, 0.7, 0.7))
            continue
        total = sum(scores.values())
        # styled segments first (sorted big->small), then grey Other
        seg = sorted(((g, s) for g, s in scores.items() if g in budget_set and s > 0),
                     key=lambda kv: kv[1], reverse=True)
        other = sum(s for g, s in scores.items() if g not in budget_set)
        left = 0.0
        for g, s in seg:
            w = s / total
            color, hatch = styles[g]
            ax.barh(0.5, w, left=left, height=0.7, color=color, hatch=hatch,
                    edgecolor="white", linewidth=0.4)
            left += w
        if other > 0:
            ax.barh(0.5, other / total, left=left, height=0.7,
                    color=GREY, edgecolor="white", linewidth=0.4)

# row labels (heads) on the left, column labels (layers) on top
for ri, h in enumerate(heads):
    axes[ri][0].annotate(f"Head {h}", xy=(0, 0.5), xytext=(-6, 0),
                         textcoords="offset points", xycoords="axes fraction",
                         ha="right", va="center", fontsize=8)
for ci, l in enumerate(layers):
    title = f"Layer {l}" + (" " if l == last_layer else "")
    axes[0][ci].set_title(title, fontsize=10, pad=6,
                          fontweight="bold" if l == last_layer else "normal")

# ---- two separated legends: solid (last layer) vs hatched (earlier layers) ----
solid_handles = [Patch(facecolor=styles[g][0], edgecolor="white", label=short(g))
                 for g in primary]
hatch_handles = [Patch(facecolor=styles[g][0], hatch=DEEP_HATCH, edgecolor="white",
                       label=short(g)) for g in extra]
hatch_handles.append(Patch(facecolor=GREY, edgecolor="white", label="Other (remaining)"))

leg_solid = fig.legend(handles=solid_handles, loc="upper left",
                       bbox_to_anchor=(0.02, 0.04), ncol=2, frameon=False,
                       fontsize=7.5, handlelength=1.6, handleheight=1.4,
                       columnspacing=1.4, labelspacing=0.6,
                       title=f"Last layer (L{last_layer}): top group of each head",
                       title_fontsize=8.5)
leg_solid._legend_box.align = "left"

leg_hatch = fig.legend(handles=hatch_handles, loc="upper right",
                       bbox_to_anchor=(0.98, 0.04), ncol=1, frameon=False,
                       fontsize=7.5, handlelength=1.6, handleheight=1.4,
                       columnspacing=1.4, labelspacing=0.6,
                       title=f"Earlier layers (L{min(prev_layers)}–L{max(prev_layers)}): "
                             f"recurring top groups",
                       title_fontsize=8.5)
leg_hatch._legend_box.align = "left"

fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.16,
                    wspace=0.08, hspace=0.25)

fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_BASE}.png", dpi=200, bbox_inches="tight")
print(f"saved {OUT_BASE}.pdf / .png")
