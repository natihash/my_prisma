"""
Builds two analysis JSONs:
  1. comp_across_task.json     -- FFT only, head semantic evolution across imagenet1k / sun397 / text
  2. comp_across_technique.json -- imagenet1k only, head semantic evolution across fft / lp / lora4 / lora16

Evidence (top groups, top texts, relevance metrics) is pulled programmatically from the
source metric files. Semantic-change labels for imagenet1k come from changes_merged.json
(authoritative). Labels for sun397 and text are decided here following the same rules,
documented in the "rules" block of comp_across_task.json.
"""
import json, os
BASE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real"
def load(p): return json.load(open(os.path.join(BASE, p)))
def dump(p, obj): json.dump(obj, open(os.path.join(BASE, p), "w"), indent=2, ensure_ascii=False)

# ---------- helpers ----------
def top_groups(scores, n=5, rel_floor=0.02):
    """Sorted top groups; cap at n; drop trailing entries < rel_floor * max (noise);
    honor a sudden drop: stop early if next < 8% of current after >=3 kept."""
    items = sorted(((k, float(v)) for k, v in scores.items()), key=lambda x: -x[1])
    items = [it for it in items if it[1] > 0]
    if not items:
        return {}
    mx = items[0][1]
    out = []
    for i, (g, v) in enumerate(items[:n]):
        if v < rel_floor * mx:
            break
        if len(out) >= 3 and out[-1][1] > 0 and v < 0.08 * out[-1][1]:
            break
        out.append((g, v))
    return {g: round(v, 3) for g, v in out}

def by_name(group_file):
    d = load(group_file)
    return {hv["Head_Name"]: hv["Scores"] for hv in d.values()}

def top_texts(head_top_texts_obj, n=6):
    tt = head_top_texts_obj["Top_Texts"]
    ranked = sorted(tt.items(), key=lambda x: -len(x[1]))
    return [{"text": t, "count": len(v)} for t, v in ranked[:n]]

def txt_by_name(file):
    d = load(file)
    return {hv["Head_Name"]: hv for hv in d.values()}

# ---------- load sources ----------
org_groups   = by_name("all_group_scores.json")              # shared "before" (org CLIP)
imn_fft_grp  = by_name("imagenet1k/all_group_scores_fft.json")
sun_fft_grp  = by_name("sun/all_group_scores_fft.json")
txt_fft      = txt_by_name("text/head_top_texts_fft.json")

merged       = load("imagenet1k/changes_merged.json")        # authoritative imagenet labels
imn_just     = load("imagenet1k/changes_fft.json")           # imagenet fft justifications
imn_rel      = load("imagenet1k/new_task_rel_fft.json")
sun_rel      = load("sun/new_task_rel_fft.json")
txt_rel      = load("text/new_task_rel_fft.json")

LABEL_MAP = {  # changes_merged -> shared vocabulary
    "same_function": "same_function",
    "different_function": "different_function",
    "specialized": "specialized",
    "Ignored": "became_useless",
}

def tk(rel, nm): return round(rel[nm]["top_k_mean"], 3)
def rk(rel, nm): return round(rel[nm]["rank_mean"], 3)

# per-layer top_k_mean means (for relative "useless" judgement)
def layer_mean(rel, L):
    vals = [rel[f"Layer {L} Head {H}"]["top_k_mean"] for H in range(12)]
    return round(sum(vals) / len(vals), 3)

# =====================================================================================
# TASK 1 : FFT across the three tasks. Judgments for sun397 / text decided below.
# (imagenet label is taken from changes_merged.json automatically.)
# =====================================================================================
# For each selected head: sun & text semantic_change + justification, text assigned group,
# and a cross-task summary. imagenet label/justification pulled automatically.
T1 = {
 "Layer 8 Head 0": {
   "sun":  ("same_function", "Place/geography head -> Architecture & Buildings / urban scenes: the place-scene thread is retained, though weak in this early layer (tk 0.89 ~ layer mean)."),
   "text": ("different_function", "Becomes a numeral reader (top text '50'); the geographic function is gone.", "Numbers"),
   "summary": "A geography/places head diverges three ways: scattered animals/objects under ImageNet (different), keeps a weak scene role under SUN (same), and turns into a numeral reader under text."},

 "Layer 8 Head 4": {
   "sun":  ("became_useless", "Face/portrait head collapses to a weak scatter (Architecture/Toys/Markets); tk 0.74 is bottom-tier for the layer."),
   "text": ("became_useless", "Only a few weak texts ('guitar'); tk 0.64 is the lowest in the layer.", "Letters, Text & Writing"),
   "summary": "A facial-expression/portrait head that has no purchase on any of the three target datasets: useless under ImageNet, SUN and text alike (consistent death)."},

 "Layer 8 Head 8": {
   "sun":  ("different_function", "Text/signage thread is lost; drifts to Sports/Transportation/Architecture. tk 0.99 ~ layer mean (weak)."),
   "text": ("same_function", "Top texts are words ('question','carpenter'); the org Letters,Text&Writing reading function is preserved and is relevant (tk 1.31, above layer mean 0.80).", "Letters, Text & Writing"),
   "summary": "An org text/signage head: ImageNet specializes it onto objects, SUN scatters it, but the text-classification task re-uses exactly its original text-reading function (same)."},

 "Layer 8 Head 9": {
   "sun":  ("same_function", "Night/urban head -> Architecture & Buildings / Natural Landscapes / landforms: the urban-scene thread is retained and it is the most relevant layer-8 head for SUN (tk 1.58, well above layer mean)."),
   "text": ("became_useless", "Top text '30' (numeral) but tk 0.54 is below the layer mean -> not relevant.", "Numbers"),
   "summary": "An urban/night-scene head: becomes a useful scene detector under SUN (same), scatters under ImageNet (different), and is effectively dead for text."},

 "Layer 9 Head 8": {
   "sun":  ("same_function", "Built/commercial-space head -> Architecture/Markets/Artistic: the built-environment function is preserved and relevant (tk 2.28, above layer mean 1.72)."),
   "text": ("different_function", "Re-used as a word reader ('kernel','marathon','monastery'); relevant (tk 2.06 > layer mean 1.32) but a different function from built spaces.", "Letters, Text & Writing"),
   "summary": "A robust built-environment head: keeps the same function under both ImageNet and SUN, and is repurposed into a useful word reader for text."},

 "Layer 9 Head 9": {
   "sun":  ("became_useless", "Futuristic/tech/sci-fi head scatters to Toys/Architecture/Markets; tk 0.89 is below layer mean 1.72."),
   "text": ("different_function", "Becomes a strong, dedicated numeral reader ('9','58','43','88','34' all numbers) and is highly relevant (tk 2.44, well above layer mean 1.32).", "Numbers"),
   "summary": "Headline divergence: a sci-fi/tech-illustration head that is useless for ImageNet and SUN turns into one of the strongest numeral readers for the text task."},

 "Layer 10 Head 3": {
   "sun":  ("different_function", "Retro/art head -> Letters,Text&Writing + Markets (signage/typography emerges); same destination as ImageNet. tk 2.59 ~ layer mean 3.12."),
   "text": ("different_function", "Top texts are numerals ('11','65','30'); a numeral reader. tk 3.62 ~ layer mean.", "Numbers"),
   "summary": "An art/aesthetic head whose latent typography thread surfaces: it becomes a text/signage head under both ImageNet and SUN, and a numeral reader under text."},

 "Layer 10 Head 6": {
   "sun":  ("same_function", "Retail/cultural-venue head -> Food/Architecture (venue/scene preserved); tk 2.89 ~ layer mean 3.12."),
   "text": ("different_function", "Becomes a numeral reader ('61','84','86','77','85'); relevant (tk 3.85 > layer mean 3.53) but a different function.", "Numbers"),
   "summary": "A commercial/cultural-venue head: keeps its scene function under both ImageNet and SUN, and is repurposed into a useful numeral reader for text."},

 "Layer 11 Head 9": {
   "sun":  ("different_function", "The number-reading function has no scene use; drifts to Sports/Architecture/Landscapes. tk 3.85 ~ layer mean 3.87."),
   "text": ("same_function", "Org was a dedicated Numbers & Numerals head; under text it stays a numeral reader ('96','29','46','27','50') and is highly relevant (tk 4.48, high absolute). Original function preserved.", "Numbers"),
   "summary": "Headline result: the dedicated numeral head is wasted/repurposed under ImageNet (specialized, number-reading lost) and SUN (different), but the text-classification task re-activates its exact original number-reading function (same)."},

 "Layer 11 Head 11": {
   "sun":  ("different_function", "Letter-reading lost; -> Architecture/Religious/Landscapes scene head. tk 4.21, above layer mean 3.87."),
   "text": ("became_useless", "Despite being the org letter/text head, under text it gives a scattered word/number mix and tk 3.05 is the lowest in the layer (clear gap below the rest).", "Letters, Text & Writing"),
   "summary": "Instructive contrast with L11H9: this dedicated letter/text head does NOT survive the text task (lowest relevance in its layer), while the numeral head does -- the text dataset leans numeral-heavy."},

 "Layer 11 Head 5": {
   "sun":  ("specialized", "Animal head -> Plants/Trees/Food with Animals still in the top-5: the organic/natural thread is retained but narrowed toward vegetation/markets. tk 4.47, above layer mean."),
   "text": ("different_function", "A few word texts ('uranium','mongoose','kindness'); mildly relevant (tk 5.53 > layer mean 4.87) but sparse and unrelated to animals.", "Letters, Text & Writing"),
   "summary": "The canonical animal head: preserved under ImageNet (same), narrowed toward natural/vegetation scenes under SUN (specialized), weak word reader under text."},

 "Layer 11 Head 6": {
   "sun":  ("same_function", "Scenery head -> Architecture/Structures/Sports: scene function preserved and relevant (tk 3.96, above layer mean 3.87)."),
   "text": ("different_function", "Sparse word texts dominated by 'baseball'; high tk (5.85) but very few distinct texts -- relevance is concentrated, function changed.", "Letters, Text & Writing"),
   "summary": "A natural-scenery head that is robust across both image tasks (same function under ImageNet and SUN, a scene/landscape detector), and a sparse word reader under text."},
}

def build_task1():
    out = {
      "analysis": "Semantic-functionality evolution of CLIP attention heads under FULL fine-tuning (FFT), compared across three adaptation targets.",
      "method": "FFT",
      "tasks": ["imagenet1k", "sun397", "text_classification"],
      "head_scope": "Last 4 transformer blocks (Layers 8-11), 12 heads each = 48 heads. Only heads with interesting cross-task behaviour are reported.",
      "categories": {
        "same_function": "Top groups close/similar to org CLIP, OR the underlying computation (e.g. texture, colour, number reading) is preserved.",
        "different_function": "Top groups change to a very different theme.",
        "specialized": "Some org functions dropped while others are retained (the head narrows).",
        "became_useless": "Relevance to the new task is low relative to other heads in the same layer (top_k_mean in the bottom tier, judged via new_task_rel_*.json), regardless of which groups appear on top."
      },
      "rules_for_deciding": {
        "source_of_truth_imagenet": "imagenet1k semantic_change is taken verbatim from imagenet1k/changes_merged.json (fft entry).",
        "before": "Top groups of org CLIP (all_group_scores.json) -- shared baseline for all tasks.",
        "after_imagenet_sun": "Top groups from all_group_scores_fft.json (sorted desc, top ~5, noise/elbow trimmed).",
        "after_text": "text dataset covers only two semantic groups (Numbers, and Letters/Text). It has no group scores, so the top texts (head_top_texts_fft.json, ranked by how many prompts map to each text) are used and the head is assigned to one of the two groups.",
        "useless_rule": "Compare a head's top_k_mean to the other heads in the SAME layer (same new_task_rel file); a clear bottom-tier value with a gap below the rest = became_useless.",
        "semantic_rule": "Compare org top groups vs after top groups: heavy overlap/preserved computation = same_function; partial overlap with dropped functions = specialized; disjoint theme = different_function."
      },
      "layer_topk_means": {
        "imagenet1k": {f"L{L}": layer_mean(imn_rel, L) for L in range(8, 12)},
        "sun397":     {f"L{L}": layer_mean(sun_rel, L) for L in range(8, 12)},
        "text":       {f"L{L}": layer_mean(txt_rel, L) for L in range(8, 12)},
      },
      "heads": {}
    }
    for nm, j in T1.items():
        L = int(nm.split()[1])
        imn_cat = LABEL_MAP[merged[nm]["fft"]["category"]]
        sun_cat, sun_just = j["sun"]
        txt_cat, txt_just, txt_grp = j["text"]
        out["heads"][nm] = {
          "before_org_top_groups": top_groups(org_groups[nm]),
          "imagenet1k": {
            "after_top_groups": top_groups(imn_fft_grp[nm]),
            "semantic_change": imn_cat,
            "top_k_mean": tk(imn_rel, nm),
            "rank_mean": rk(imn_rel, nm),
            "layer_topk_mean": layer_mean(imn_rel, L),
            "justification": imn_just[nm]["justification"],
          },
          "sun397": {
            "after_top_groups": top_groups(sun_fft_grp[nm]),
            "semantic_change": sun_cat,
            "top_k_mean": tk(sun_rel, nm),
            "rank_mean": rk(sun_rel, nm),
            "layer_topk_mean": layer_mean(sun_rel, L),
            "justification": sun_just,
          },
          "text_classification": {
            "after_top_texts": top_texts(txt_fft[nm]),
            "assigned_group": txt_grp,
            "semantic_change": txt_cat,
            "top_k_mean": tk(txt_rel, nm),
            "rank_mean": rk(txt_rel, nm),
            "layer_topk_mean": layer_mean(txt_rel, L),
            "justification": txt_just,
          },
          "cross_task_summary": j["summary"],
        }
    dump("comp_across_task.json", out)
    print("wrote comp_across_task.json with", len(out["heads"]), "heads")

# =====================================================================================
# TASK 2 : ImageNet1k across techniques (fft / lp / lora4 / lora16).
# Labels come straight from changes_merged.json; we add evidence + a cross-technique summary.
# =====================================================================================
TECH_GROUPS = {
  "fft":    by_name("imagenet1k/all_group_scores_fft.json"),
  "lp":     by_name("imagenet1k/all_group_scores_lp.json"),
  "lora4":  by_name("imagenet1k/all_group_scores_lora4.json"),
  "lora16": by_name("imagenet1k/all_group_scores_lora16.json"),
}
ORDER = ["lora4", "lora16", "lp", "fft"]

T2 = {
 "Layer 9 Head 2": "Most divergent head in the set: a texture/pattern head read four different ways -- LoRA (4 & 16) -> specialized, LP -> different_function, FFT -> same_function (the texture computation survives, now applied to dog-coat texture). Adapter capacity changes the verdict.",
 "Layer 11 Head 3": "Capacity paradox: the low-capacity LoRA adapters preserve this professions/transport/marine head (same_function), while the high-capacity LP and FFT overwrite it into uselessness (Ignored). More tuning is not always more preservation.",
 "Layer 8 Head 0": "Opposite of L11H3: LoRA leaves this places head essentially untouched (Ignored / no engagement), while LP and FFT actively repurpose it (different_function). Full tuning is needed to recruit it.",
 "Layer 9 Head 4": "Only FFT manages to recruit this texture/subject-count head (different_function); LoRA4, LoRA16 and LP all leave it useless. A head that needs full capacity to be activated.",
 "Layer 9 Head 6": "LoRA4/16 specialize this aerial/landscape head (retain the aquatic thread), whereas LP and FFT push it all the way to a different function (reptiles/marine). Smooth capacity gradient: specialize -> repurpose.",
 "Layer 10 Head 0": "Stable under LoRA4/16 and LP (same_function, round-object/vessel thread), but FFT alone specializes it onto containers/kitchen equipment.",
 "Layer 10 Head 4": "Transport/sport/animal head held as same_function by LoRA4/16 and LP, but FFT changes it (different_function) by adding fashion and dropping the dynamic-motion framing.",
 "Layer 10 Head 9": "LoRA4/16 read it as different_function while LP and FFT read it as specialized (slender/elongated-object thread retained into implements/medical). Adapter vs full-tune disagree on whether a thread survives.",
 "Layer 11 Head 5": "Similarity anchor: the dominant animal head is same_function under ALL four techniques -- a function so strong and so aligned with ImageNet that adaptation method does not matter.",
 "Layer 11 Head 7": "Similarity anchor: graspable-objects head is specialized under all four techniques (-> instruments/tools/implements). Consistent narrowing regardless of method.",
 "Layer 11 Head 1": "Similarity anchor: reflections/mood/composition head is different_function under all four techniques -- abstract aesthetic functions are reliably overwritten on ImageNet.",
 "Layer 8 Head 1": "Similarity anchor (death): a facial-expression/emotion head that is Ignored / useless under every technique -- no method can give it ImageNet purchase.",
}

def build_task2():
    out = {
      "analysis": "Semantic-functionality evolution of CLIP attention heads when adapting to ImageNet-1k, compared across four adaptation techniques.",
      "task": "imagenet1k",
      "techniques": ORDER,
      "head_scope": "Last 4 transformer blocks (Layers 8-11). Only heads with interesting cross-technique behaviour (divergence or notable agreement) are reported.",
      "labels_source": "Per-technique semantic_change, top_k_mean and rank_mean are taken from imagenet1k/changes_merged.json (Ignored is reported as 'became_useless').",
      "categories": {
        "same_function": "Preserved org function / computation.",
        "different_function": "Theme overwritten.",
        "specialized": "Narrowed: some org functions retained, others dropped.",
        "became_useless": "Low top_k_mean relative to the layer (reported as 'Ignored' in the source)."
      },
      "heads": {}
    }
    for nm, summary in T2.items():
        entry = {"before_org_top_groups": top_groups(org_groups[nm]), "techniques": {}}
        for t in ORDER:
            m = merged[nm][t]
            entry["techniques"][t] = {
              "after_top_groups": top_groups(TECH_GROUPS[t][nm]),
              "semantic_change": LABEL_MAP[m["category"]],
              "top_k_mean": round(m["top_k_mean"], 3),
              "rank_mean": round(m["rank_mean"], 3),
            }
        entry["cross_technique_summary"] = summary
        out["heads"][nm] = entry
    dump("comp_across_technique.json", out)
    print("wrote comp_across_technique.json with", len(out["heads"]), "heads")

if __name__ == "__main__":
    build_task1()
    build_task2()
