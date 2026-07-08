#!/usr/bin/env python3
"""Build semantic_shift.json for SUN397 adaptation of CLIP ViT.

For every head (Layer 8-11, heads 0-11) and every adaptation technique
(fft, lora4, lora16) label the head's semantic-function change into one of:
  - ignored      : low top_k_mean vs the other heads of the same layer (math only)
  - stable       : top groups roughly the same before vs after
  - specialized  : partial group overlap; a former *secondary* original group
                   becomes/rises to primary while the original primary is
                   lost or demoted
  - repurposed   : top groups are different; the new primary group is foreign
                   to the original repertoire

LP is labeled with the same logic as the other techniques once its
post-adaptation group/text/relevance data is available.

Notes added to each entry support manual review; uncertain calls get "flags".
"""
import json, os, statistics

ROOT = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real"
SUN = os.path.join(ROOT, "sun")
TECHS = ["fft", "lora4", "lora16", "lp"]

# ---- tunables -------------------------------------------------------------
MAX_GROUPS = 6          # max top groups to consider
DROP_RATIO = 0.5        # "sudden drop": a group < 0.5x the previous one cuts the list
DROP_AMBIG = (0.40, 0.60)  # if the cut ratio lands here, flag the group set as sensitive
IGNORE_Z = -1.0         # top_k_mean z <= this (within layer) -> ignored
BORDER_Z = -0.75        # between this and IGNORE_Z -> borderline (use rank_mean tiebreak)
LAYER_CV_FLOOR = 0.08   # if a layer's top_k spread is tighter than this, nobody is ignored
JACCARD_LOW = 0.34      # set-overlap considered "low"
JACCARD_HIGH = 0.50     # set-overlap considered "high"

# ---- semantically-related group clusters ----------------------------------
# Conservative near-synonym clusters. Two groups "match" if identical OR in the
# same cluster. This implements the rule: a head is only "repurposed" if the
# original head had nothing like the new primary group ("...or something closer
# to that class") in its repertoire. Groups not listed match by exact name only.
_CLUSTERS = [
    {"Architecture & Buildings", "Structures & Barriers", "Urban & City Scenes",
     "Industrial & Manufacturing"},
    {"Natural Landscapes", "Trees & Forests", "Plants & Botanical",
     "Natural Landforms & Geology", "Outdoor & Camping", "Fungi & Mushrooms"},
    {"Water & Aquatic", "Marine & Aquatic Life"},
    {"Sky & Space", "Weather & Atmosphere", "Sunsets & Sunrises",
     "Fog, Mist & Haze", "Night & Darkness"},
    {"Animals", "Birds", "Cats & Felines", "Dogs & Canines",
     "Insects & Arachnids", "Reptiles & Amphibians", "Primates"},
    {"Food & Cuisine", "Fruits", "Vegetables & Produce", "Kitchen & Cooking Equipment"},
    {"People & Portraits", "Facial Expressions & Emotions", "Body Parts",
     "Children & Youth", "Professions & Occupations", "Social Interactions & Connections"},
    {"Letters, Text & Writing", "Numbers & Numerals"},
    {"Artistic Styles & Movements", "Caricatures & Illustrations",
     "Abstract Art & Patterns", "Graffiti & Street Art"},
    {"Geometric Shapes (General)", "Circles & Round Objects", "Triangles & Cones",
     "Squares, Rectangles & Grids", "Polygons & Multi-sided Shapes",
     "Spirals, Swirls & Vortexes", "Symmetry & Balance", "Fractals & Complex Math"},
    {"Sports & Athletics", "Toys, Games & Recreation", "Music, Dance & Performance"},
]
_GROUP2CL = {}
for _i, _c in enumerate(_CLUSTERS):
    for _g in _c:
        _GROUP2CL[_g] = _i

def related(a, b):
    """True if groups are identical or in the same semantic cluster."""
    if a == b:
        return True
    ca, cb = _GROUP2CL.get(a), _GROUP2CL.get(b)
    return ca is not None and ca == cb

def related_in(group, group_list):
    """Return the first group in group_list related to `group`, else None."""
    for g in group_list:
        if related(group, g):
            return g
    return None

# ---- helpers --------------------------------------------------------------
def load(p):
    with open(os.path.join(SUN, p) if not os.path.isabs(p) else p) as f:
        return json.load(f)

def head_index_map(group_scores):
    """Head_X -> 'Layer L Head H' (canonical name)."""
    return {hk: v["Head_Name"] for hk, v in group_scores.items()}

def select_top_groups(scores):
    """Return (list[(group,score)], cut_ratio_at_break_or_None)."""
    items = sorted(((g, s) for g, s in scores.items() if s > 0), key=lambda x: -x[1])
    if not items:
        return [], None
    cand = items[:MAX_GROUPS]
    selected = [cand[0]]
    cut_ratio = None
    for i in range(1, len(cand)):
        prev = selected[-1][1]
        cur = cand[i][1]
        ratio = cur / prev if prev > 0 else 1.0
        if ratio < DROP_RATIO:
            cut_ratio = ratio
            break
        selected.append(cand[i])
    return selected, cut_ratio

def top_classes(top_texts_dict, n=8):
    """head_top_texts: class -> list; rank by len(list) desc."""
    ranked = sorted(top_texts_dict.items(), key=lambda kv: -len(kv[1]))
    return [f"{c} ({len(v)})" for c, v in ranked[:n]]

def layer_of(name):
    return int(name.split("Head")[0].split("Layer")[1].strip())

# ---- load all sources -----------------------------------------------------
org_scores = load(os.path.join(ROOT, "all_group_scores.json"))
org_text = load(os.path.join(ROOT, "group_text_org.json"))   # keyed by 'Layer L Head H'
idx2name = head_index_map(org_scores)

post_scores = {t: load(f"all_group_scores_{t}.json") for t in TECHS}
post_texts = {t: load(f"head_top_texts_{t}.json") for t in TECHS}
rel = {t: load(f"new_task_rel_{t}.json") for t in TECHS}   # keyed by 'Layer L Head H'

# ---- precompute per-layer ignored stats -----------------------------------
# per technique: layer -> {'mean','std','rmean','rstd'}
layer_stats = {}
for t in TECHS:
    layers = {}
    for name, v in rel[t].items():
        L = layer_of(name)
        layers.setdefault(L, {"tk": [], "rk": []})
        layers[L]["tk"].append(v["top_k_mean"])
        layers[L]["rk"].append(v["rank_mean"])
    stats = {}
    for L, d in layers.items():
        tk, rk = d["tk"], d["rk"]
        stats[L] = {
            "mean": statistics.mean(tk), "std": statistics.pstdev(tk),
            "rmean": statistics.mean(rk), "rstd": statistics.pstdev(rk),
        }
    layer_stats[t] = stats

# ---- precompute original top groups per head ------------------------------
org_top = {}  # Head_X -> selected groups
for hk, v in org_scores.items():
    sel, cut = select_top_groups(v["Scores"])
    org_top[hk] = (sel, cut)

# ---- classification -------------------------------------------------------
def decide_ignored(t, name):
    s = layer_stats[t][layer_of(name)]
    v = rel[t][name]
    tk, rk = v["top_k_mean"], v["rank_mean"]
    z = (tk - s["mean"]) / s["std"] if s["std"] > 0 else 0.0
    rz = (rk - s["rmean"]) / s["rstd"] if s["rstd"] > 0 else 0.0
    cv = s["std"] / s["mean"] if s["mean"] else 0.0
    metrics = {
        "top_k_mean": round(tk, 4), "rank_mean": round(rk, 4),
        "layer_topk_mean": round(s["mean"], 4), "layer_topk_std": round(s["std"], 4),
        "topk_z": round(z, 3), "rank_z": round(rz, 3), "layer_cv": round(cv, 3),
    }
    flags = []
    if cv < LAYER_CV_FLOOR:
        return False, metrics, flags  # everything close -> all relevant
    if z <= IGNORE_Z:
        return True, metrics, flags
    if z <= BORDER_Z:
        # borderline: lean on rank_mean as secondary signal
        if rz <= IGNORE_Z:
            flags.append("ignored_borderline_confirmed_by_rank")
            return True, metrics, flags
        flags.append("ignored_borderline_NOT_ignored_review")
        return False, metrics, flags
    return False, metrics, flags

def classify_shift(hk, name, t):
    o_sel, o_cut = org_top[hk]
    p_sel, p_cut = select_top_groups(post_scores[t][hk]["Scores"])
    O = [g for g, _ in o_sel]
    P = [g for g, _ in p_sel]
    oset, pset = set(O), set(P)
    inter = oset & pset
    union = oset | pset
    jacc = len(inter) / len(union) if union else 0.0
    flags = []
    if not O or not P:
        return "uncertain", {"reason": "empty top-group set"}, ["empty_group_set"], P, p_sel

    o_primary, p_primary = O[0], P[0]

    # related-group overlap (exact OR same semantic cluster)
    rel_overlap = sorted({g for g in P if related_in(g, O)})
    rel_union = len(set(O) | {g for g in P if not related_in(g, O)})
    rel_jacc = len(rel_overlap) / rel_union if rel_union else 0.0

    # group-set sensitivity flags (cut near the drop threshold)
    for tag, cut in (("orig", o_cut), ("post", p_cut)):
        if cut is not None and DROP_AMBIG[0] <= cut <= DROP_AMBIG[1]:
            flags.append(f"{tag}_groupset_sensitive(cut={cut:.2f})")

    if related(p_primary, o_primary):
        label = "stable"
        if p_primary == o_primary:
            reason = f"primary '{o_primary}' preserved"
        else:
            reason = f"primary '{p_primary}' is semantically close to original primary '{o_primary}'"
            flags.append("stable_via_related_groups_review")
        if rel_jacc < JACCARD_LOW:
            flags.append("stable_but_low_secondary_overlap_review")
    else:
        match_o = related_in(p_primary, O)  # post-primary related to an original secondary?
        if match_o is not None:
            # a former secondary (or its close relative) became primary
            o_lost = related_in(o_primary, P) is None
            status = "lost" if o_lost else "demoted"
            label = "specialized"
            via = "" if match_o == p_primary else f" (via related group '{match_o}')"
            reason = (f"original secondary '{match_o}'{via} promoted to primary as "
                      f"'{p_primary}'; original primary '{o_primary}' {status}")
            if match_o != p_primary:
                flags.append("specialized_via_related_groups_review")
            if rel_jacc >= JACCARD_HIGH:
                flags.append("specialized_but_high_overlap_review")
        else:
            label = "repurposed"
            reason = (f"new primary '{p_primary}' (and relatives) absent from original "
                      f"top groups; original primary '{o_primary}'")
            if related_in(o_primary, P) is not None:
                reason += " still present (demoted)"
            if rel_jacc >= JACCARD_HIGH:
                flags.append("repurposed_but_high_overlap_review")
    info = {
        "reason": reason,
        "jaccard_exact": round(jacc, 3),
        "jaccard_related": round(rel_jacc, 3),
        "orig_primary": o_primary,
        "post_primary": p_primary,
        "group_overlap_exact": sorted(inter),
        "group_overlap_related": rel_overlap,
    }
    return label, info, flags, P, p_sel

# ---- build output ---------------------------------------------------------
out = {}
for hk in sorted(org_scores, key=lambda k: int(k.split("_")[1])):
    name = idx2name[hk]
    o_sel, o_cut = org_top[hk]
    entry = {
        "original_top_groups": [[g, round(s, 3)] for g, s in o_sel],
        "original_top_texts": org_text.get(name, {}).get("top_texts", []),
    }
    for t in TECHS:
        ignored, imetrics, iflags = decide_ignored(t, name)
        if ignored:
            entry[t] = {
                "label": "ignored",
                "reason": f"top_k_mean z={imetrics['topk_z']} within layer (<= {IGNORE_Z})",
                "ignored_metrics": imetrics,
                "flags": iflags,
                "post_top_groups": [[g, round(s, 3)] for g, s in select_top_groups(post_scores[t][hk]["Scores"])[0]],
                "post_top_classes": top_classes(post_texts[t][hk]["Top_Texts"]),
            }
        else:
            label, info, sflags, P, p_sel = classify_shift(hk, name, t)
            entry[t] = {
                "label": label,
                "reason": info["reason"],
                "details": {k: v for k, v in info.items() if k != "reason"},
                "ignored_metrics": imetrics,
                "post_top_groups": [[g, round(s, 3)] for g, s in p_sel],
                "post_top_classes": top_classes(post_texts[t][hk]["Top_Texts"]),
                "flags": iflags + sflags,
            }
    out[name] = entry

with open(os.path.join(SUN, "semantic_shift.json"), "w") as f:
    json.dump(out, f, indent=2)

# ---- summary --------------------------------------------------------------
from collections import Counter
print("Wrote semantic_shift.json with", len(out), "heads.")
for t in TECHS:
    c = Counter(out[n][t]["label"] for n in out)
    flagged = sum(1 for n in out if out[n][t]["flags"])
    print(f"  {t:7s}: {dict(c)}  | flagged_for_review={flagged}")
