import json, os
BASE="/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real"

def load(p): return json.load(open(os.path.join(BASE,p)))

# org baseline top groups (shared "before") -- flat Head_0..47 with Head_Name + Scores
org = load("all_group_scores.json")
def topg(scores, n=6):
    return sorted(((k,round(v,3)) for k,v in scores.items()), key=lambda x:-x[1])[:n]

org_by_name={}
for hk,hv in org.items():
    org_by_name[hv["Head_Name"]] = topg(hv["Scores"])

# sun fft
sun = load("sun/all_group_scores_fft.json")
sun_by_name={}
for hk,hv in sun.items():
    sun_by_name[hv["Head_Name"]] = topg(hv["Scores"])
sun_rel = load("sun/new_task_rel_fft.json")  # keyed by "Layer X Head Y"

# imagenet fft after-groups
imn = load("imagenet1k/all_group_scores_fft.json")
imn_by_name={}
for hk,hv in imn.items():
    imn_by_name[hv["Head_Name"]] = topg(hv["Scores"])

# imagenet merged categories (fft) + justification
merged = load("imagenet1k/changes_merged.json")
imnet_fft_just = load("imagenet1k/changes_fft.json")
imnet_rel = load("imagenet1k/new_task_rel_fft.json")

# text fft top texts
txt = load("text/head_top_texts_fft.json")
txt_by_name={}
for hk,hv in txt.items():
    tt=hv["Top_Texts"]
    ranked=sorted(tt.items(), key=lambda x:-len(x[1]))
    txt_by_name[hv["Head_Name"]] = [(t, len(v)) for t,v in ranked[:8]]
txt_rel = load("text/new_task_rel_fft.json")

# build ordered head names layers 8-11
names=[f"Layer {L} Head {H}" for L in range(8,12) for H in range(12)]

out=[]
for L in range(8,12):
    # per-layer top_k for sun, imnet, text to judge "useless"
    sun_tk={H:sun_rel[f"Layer {L} Head {H}"]["top_k_mean"] for H in range(12)}
    im_tk ={H:imnet_rel[f"Layer {L} Head {H}"]["top_k_mean"] for H in range(12)}
    tx_tk ={H:txt_rel[f"Layer {L} Head {H}"]["top_k_mean"] for H in range(12)}
    out.append(f"\n===== LAYER {L}  (sun_tk mean {sum(sun_tk.values())/12:.2f} | imnet_tk mean {sum(im_tk.values())/12:.2f} | text_tk mean {sum(tx_tk.values())/12:.2f}) =====")
    for H in range(12):
        nm=f"Layer {L} Head {H}"
        out.append(f"\n--- {nm} ---")
        out.append("  ORG before : "+", ".join(f"{g}={s}" for g,s in org_by_name[nm][:5]))
        out.append(f"  IMNET fft  : cat={merged[nm]['fft']['category']:18s} tk={im_tk[H]:.2f}  | just: {imnet_fft_just[nm]['justification']}")
        out.append("    imnet after groups: "+", ".join(f"{g}={s}" for g,s in imn_by_name[nm][:5]))
        out.append("  SUN  fft   : tk={:.2f}  groups: ".format(sun_tk[H])+", ".join(f"{g}={s}" for g,s in sun_by_name[nm][:5]))
        out.append("  TEXT fft   : tk={:.2f}  toptexts: ".format(tx_tk[H])+", ".join(f"{t}({c})" for t,c in txt_by_name[nm][:6]))

print("\n".join(out))
