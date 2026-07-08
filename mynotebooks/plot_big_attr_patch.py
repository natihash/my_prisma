"""
Same head x layer semantics grid as plot_big.py, but for the attribution-patching
scores. The COLOR/HATCH assignment and the legend labels are taken UNCHANGED from
the original CLIP-metrics JSON (the "color reference"); this file only supplies the
bar CONTENT (which head shows how much of each group).

So a given group keeps the exact same color/pattern across both figures, even
though the selection of styled groups was decided purely from the reference file's
last / earlier layers. We do NOT re-derive top groups from this file.
"""
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- config ----------------
COLOR_REF_JSON = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/all_group_scores.json"  # decides colors/labels
# CONTENT_JSON   = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/all_group_scores_attr_patch.json"             # decides bar content
CONTENT_JSON = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_group_scores_sae_fixed3.json"
N_LAST_LAYER   = 12            # cap on solid-colored groups from the LAST layer (of the reference)
N_EXTRA        = 6             # hatched groups pulled from the earlier layers (of the reference)
N_EXTRA_LAYERS = 3             # how many layers *before* the last to scan
OUT_BASE       = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/plots3/head_semantics_sae3"
DEEP_HATCH     = "xxxx"        # dense (double) crosshatch for earlier-layer groups
# -----------------------------------------

def short(name, n=30):
    name = name.strip()
    return name if len(name) <= n else name[:n - 1].rstrip() + "…"

def load(path):
    """Return {(layer, head): {group: score}}, sorted layers, sorted heads."""
    with open(path) as f:
        data = json.load(f)
    hd, layers, heads = {}, set(), set()
    for entry in data.values():
        m = re.search(r"Layer\s+(\d+)\s+Head\s+(\d+)", entry["Head_Name"])
        l, h = int(m.group(1)), int(m.group(2))
        layers.add(l); heads.add(h)
        hd[(l, h)] = entry["Scores"]
    return hd, sorted(layers), sorted(heads)

# ---- per-head dominant group (and its margin over #2) within a layer ----
def top_group_per_head(hd, heads, layer):
    """List of (group, gap) for the #1 group of each head in `layer`."""
    out = []
    for h in heads:
        s = hd.get((layer, h))
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

# ====================================================================
# 1) SELECTION + STYLES  -- derived ONLY from the color-reference file
# ====================================================================
ref_hd, ref_layers, ref_heads = load(COLOR_REF_JSON)
last_layer = ref_layers[-1]

primary, last_freq = rank_by_freq_then_gap(top_group_per_head(ref_hd, ref_heads, last_layer))
primary = primary[:N_LAST_LAYER]
primary_set = set(primary)

prev_layers = [l for l in ref_layers if l != last_layer][-N_EXTRA_LAYERS:]
prev_tops = []
for l in prev_layers:
    prev_tops += top_group_per_head(ref_hd, ref_heads, l)
prev_ranked, prev_freq = rank_by_freq_then_gap(prev_tops)
extra = [g for g in prev_ranked if g not in primary_set][:N_EXTRA]

budget = primary + extra
budget_set = set(budget)

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
plt.rcParams["hatch.linewidth"] = 0.9

styles = {}                              # group -> (color, hatch)  (same mapping as plot_big.py)
for i, g in enumerate(primary):
    styles[g] = (PALETTE[i % len(PALETTE)], "" if i < len(PALETTE) else "....")
for j, g in enumerate(extra):
    styles[g] = (PALETTE[j % len(PALETTE)], DEEP_HATCH)

# ====================================================================
# 2) CONTENT  -- bars drawn from the attribution-patching file
# ====================================================================
heads_data, layers, heads = load(CONTENT_JSON)

missing = [g for g in budget if not any(g in heads_data.get((l, h), {})
                                        for l in layers for h in heads)]
if missing:
    print(f"NOTE: {len(missing)} styled groups are absent from the content file "
          f"(legend only): {missing}")

print(f"{len(budget_set)} groups styled from reference: {len(primary)} solid (last layer "
      f"{last_layer}), {len(extra)} hatched (layers {prev_layers}); rest -> Other.")

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
