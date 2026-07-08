import sys
sys.path.insert(0, "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/src")

import os
# NOTE: CUDA_LAUNCH_BLOCKING=1 serializes every GPU kernel (debug-only) and
# slows GPU work massively. Leave it off for normal runs; re-enable only when
# you need a precise CUDA error location.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import vit_prisma
from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
from vit_prisma.utils import prisma_utils

import numpy as np
import torch
from fancy_einsum import einsum
from collections import defaultdict

import plotly.graph_objs as go
import plotly.express as px

import matplotlib.colors as mcolors

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class ConvertTo3Channels:
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

transform = transforms.Compose([
    ConvertTo3Channels(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# from vit_prisma.models.base_vit import HookedViT

# # model_name = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_linear_probe_real"
# # model_name = "vit_base_patch16_224"
# # model_name = "vit_base_patch16_clip_224.laion2b_ft_in1k"
# # model_name = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft"
# # model_name = "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

# model_name = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_lora_r16_merged"
# # model_name = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_lora_r4_merged"
# # model_name = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft_latest"

# model = HookedViT.from_pretrained(model_name,
#                                         center_writing_weights=True,
#                                         center_unembed=True,
#                                         fold_ln=True,
#                                         refactor_factored_attn_matrices=True,
#                                         device="cuda"
#                                     )

# model = model.to("cuda:0")
# model.cfg.device = "cuda:0"
# print("Model device config:", model.cfg.device)
# print("Is CUDA available?:", torch.cuda.is_available())

import json
LOCAL_JSON_PATH = "/home/nfm/ViT-Prisma/demos/imagenet_class_index.json"
with open(LOCAL_JSON_PATH, 'r') as f:
    imagenet_class_index = json.load(f)

wnid_to_name = {}
for idx, (wnid, class_name) in imagenet_class_index.items():
    safe_class_name = class_name.replace(" ", "_").replace("/", "_").replace(",", "")
    wnid_to_name[wnid] = safe_class_name

wnid_to_idx = {wnid: int(idx) for idx, (wnid, name) in imagenet_class_index.items()}
idx_to_name = {int(idx): name for idx, (wnid, name) in imagenet_class_index.items()}

idx_to_wnid = {int(idx): wnid for idx, (wnid, name) in imagenet_class_index.items()}
name_to_idx = {name: int(idx) for idx, (wnid, name) in imagenet_class_index.items()}

import os
import json
import torch
from torch import einsum

LOCAL_JSON_PATH = "/home/nfm/ViT-Prisma/demos/imagenet_class_index.json"

with open(LOCAL_JSON_PATH, 'r') as f:
    imagenet_class_index = json.load(f)

wnid_to_idx = {wnid: int(idx) for idx, (wnid, name) in imagenet_class_index.items()}
idx_to_name = {int(idx): name for idx, (wnid, name) in imagenet_class_index.items()}


def residual_stack_to_logit_attn(centered_residual_stack, answer_residual_direction):    
    raw_dla = torch.einsum(
        "h b d, c d -> h b c",  # h=heads, b=batch, d=d_model, c=classes
        centered_residual_stack,
        answer_residual_direction,
    )
    return raw_dla 

def calculate_dla(model, head_idx, TENSORS_DIR, num_classes, text_dict=None, is_clip=True):
    if is_clip:
        proj_head = model.head.W_H
        target_direction = text_dict @ proj_head.T
        target_direction = target_direction.to(model.cfg.device)
    else:
        print("Normal ViT Model, so just use direct logit attribution using the head")
        all_inds = [*range(num_classes)]
        target_direction = model.tokens_to_residual_directions(all_inds)

    if len(head_idx) >= 12 and target_direction.shape[0] > 1000:
        print("Warning: You are slicing more than 12 heads and using more than 1000 text descriptions. This may lead to very large tensors that could exceed your GPU memory. Consider reducing the number of heads or text descriptions for a more manageable tensor size.")
        return None

    pths = sorted(f for f in os.listdir(TENSORS_DIR) if f.endswith('.pt'))

    # Files can have different sample counts (e.g. SUN val classes with 15 vs 30
    # samples), so we can't pre-allocate a fixed [files, heads, samples, classes]
    # tensor. Instead collect each file's per-head attribution [heads, b, classes]
    # and concatenate along the sample axis. find_top_texts_topk / the PART 1 loop
    # only index dim1 (heads) and read the last (class) axis — the leading sample
    # axis is flattened — so a single combined sample axis is equivalent to the old
    # per-file layout. A leading singleton dim keeps the 4D shape they expect.
    chunks = []
    for pth in pths:
        filepath = os.path.join(TENSORS_DIR, pth)
        per_head_residual = torch.load(filepath, weights_only=True).to(model.cfg.device)
        per_head_attribution = residual_stack_to_logit_attn(per_head_residual[head_idx], target_direction)
        chunks.append(per_head_attribution.detach().cpu())   # [heads, b, classes]
        del per_head_residual, per_head_attribution

    # [heads, total_samples, classes] -> [1, heads, total_samples, classes]
    final_tensor = torch.cat(chunks, dim=1).unsqueeze(0)

    return final_tensor

def find_top_texts_topk(final_tensor, head_idx, target_idx, texts, thresh = None, verbose=False):
    all_heads_dict = {}
    for shibu in range(len(target_idx)):
        messi = target_idx[shibu] - head_idx[0]
        if verbose:
            print(f"Analyzing Layer {8+head_idx[messi]//12} Head {head_idx[messi]%12}...")    
        # .float(): the bank is stored as float16; CPU half lacks reliable
        # topk/sort support, so promote this per-head slice (~200 MB) back to float32.
        temp = torch.clone(final_tensor[:,messi,:,:]).float()
        if thresh is None:
            gaga = 500
            values, flat_idx = temp.reshape(-1).topk(gaga, largest=True)
        else:
            # gaga = (temp > thresh[1]).sum().item() + (temp < thresh[0]).sum().item()
            gaga = (temp > thresh[1]).sum().item()
            values, flat_idx = temp.reshape(-1).topk(gaga, largest=True)
            # values = temp[temp > thresh[1]]
            # flat_idx = torch.nonzero(temp > thresh[1]).flatten()
            # values = torch.cat((values, temp[temp < thresh[0]]))
            # flat_idx = torch.cat((flat_idx, torch.nonzero(temp < thresh[0]).flatten()))

        # unravel flat indices to 3D coords
        A, B, C = temp.shape
        i = flat_idx // (B * C)
        j = (flat_idx % (B * C)) // C
        k = flat_idx % C
        indices = torch.stack([i, j, k], dim=1)


        top_names = []
        avg_vals = {}
        for i in range(gaga):
            top_names.append(texts[indices[i][2].item()])
            avg_vals[texts[indices[i][2].item()]] = avg_vals.get(texts[indices[i][2].item()], []) + [values[i].item()]

        #sort avg_vals by mean value
        # avg_vals = {k: sum(v)/len(v) for k, v in avg_vals.items()}
        # avg_vals = dict(sorted(avg_vals.items(), key=lambda item: item[1], reverse=True))

        all_heads_dict[f"Layer {8+head_idx[messi]//12} Head {head_idx[messi]%12}"] = avg_vals


        from collections import Counter
        name_counts = Counter(top_names)
        # print(name_counts.most_common(20))
        # print(list(avg_vals.items())[:20])
        if verbose:
            for name, count in name_counts.most_common(20):
                print(f"{name}: {count} : {sum(avg_vals[name])/len(avg_vals[name]):.3f}", end="||")
            print("\n" + "-"*50)

        return all_heads_dict

def find_elbows(temp, plot_elbows=False, print_elbows=False, sigma=500, elbows_to_find=2):
    temp_sorted = torch.sort(temp).values
    num_plot = 1_000_000
    idx = torch.arange(
        num_plot,
        device=temp_sorted.device
    ) * (temp_sorted.shape[0] - 1) // (num_plot - 1)
    y = temp_sorted[idx].float().cpu().numpy() if temp_sorted.shape[0] > num_plot else temp_sorted.float().cpu().numpy()
    x = np.linspace(0, 1, len(y))
    # Auto-scale sigma proportionally to data length (targets ~0.05% smoothing).
    # The default 500 was designed for 1M-point subsamples; smaller arrays need a smaller sigma.
    effective_sigma = sigma if len(y) >= num_plot else max(5, int(sigma * len(y) / num_plot))
    y_smooth = gaussian_filter1d(y, sigma=effective_sigma)


    dy = np.gradient(y_smooth)
    d2y = np.gradient(dy)
    curvature = np.abs(d2y)
    peaks, _ = find_peaks(
        curvature,
        distance=len(curvature)//5,
        prominence=np.max(curvature) * 0.1
    )

    top2 = peaks[np.argsort(curvature[peaks])[-elbows_to_find:]]
    top2 = np.sort(top2)

    low_thresh = y[top2][0].item()
    high_thresh = y[top2][1].item() if len(top2) > 1 else None

    if plot_elbows:
        plt.figure(figsize=(12, 6))
        plt.plot(y, linewidth=1)
        for i, p in enumerate(top2):
            plt.axvline(p, color='red', linestyle='--')
            plt.scatter(
                p,
                y[p],
                color='red',
                s=100,
                label=f'Elbow {i+1}' if i == 0 else None
            )

        plt.legend()
        plt.title("Sorted Values with Detected Elbows")
        plt.show()
    if print_elbows:
        print("Elbow indices:", top2)
        print("Elbow values:", y[top2])

    return low_thresh, high_thresh


def find_magnitude_threshold(norms, plot=False, print_result=False):
    """
    Finds a single threshold for activation magnitudes via adaptive elbow detection.
    Falls back to the 90th percentile if no clear elbow is found.
    Returns the scalar threshold value.
    """
    norms_np = norms.float().cpu().numpy()
    sorted_norms = np.sort(norms_np)
    n = len(sorted_norms)

    # Scale sigma to ~1% of data length for small arrays
    sigma = max(5, n // 100)
    y_smooth = gaussian_filter1d(sorted_norms, sigma=sigma)

    dy = np.gradient(y_smooth)
    d2y = np.gradient(dy)
    curvature = np.abs(d2y)

    peaks, _ = find_peaks(
        curvature,
        distance=max(10, n // 5),
        prominence=np.max(curvature) * 0.05,
    )

    if len(peaks) == 0:
        threshold = float(np.percentile(sorted_norms, 90))
        if print_result:
            print(f"No elbow found; falling back to 90th-percentile threshold: {threshold:.4f}")
    else:
        best_peak = peaks[np.argmax(curvature[peaks])]
        threshold = float(sorted_norms[best_peak])
        if print_result:
            print(f"Magnitude threshold (elbow at index {best_peak}/{n}): {threshold:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(sorted_norms, linewidth=0.5)
        marker_x = best_peak if len(peaks) > 0 else int(0.9 * n)
        axes[0].axvline(marker_x, color="red", linestyle="--", label=f"thresh={threshold:.3f}")
        axes[0].set_title("Sorted Norms")
        axes[0].legend()
        axes[1].plot(curvature, linewidth=0.5)
        axes[1].set_title("Curvature")
        plt.tight_layout()
        plt.show()

    return threshold
    
def compute_group_scores(top_texts, group_data):
    from collections import defaultdict

    # Build text/class -> groups lookup (includes both texts and imagenet_classes)
    text_to_groups = defaultdict(list)

    for group_name, group_info in group_data["groups"].items():
        # Get texts from the group
        texts_in_group = group_info.get("texts", [])
        for text in texts_in_group:
            text_to_groups[text].append(group_name)
        
        # Get imagenet classes from the group
        imagenet_classes = group_info.get("imagenet_classes", [])
        for cls_name in imagenet_classes:
            text_to_groups[cls_name].append(group_name)

    # Initialize all groups at 0
    group_scores = {
        group_name: 0.0
        for group_name in group_data["groups"]
    }

    # Accumulate weighted scores
    for text, score_list in top_texts.items():

        weight = sum(score_list)

        groups = text_to_groups.get(text, [])

        # Ignore unassigned texts/classes
        if not groups:
            continue

        for group in groups:
            group_scores[group] += weight

    return group_scores

# group_data = json.load(open("groups_output.json"))
group_data = json.load(open("/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/groups_output_combined_new.json"))


import math
from concurrent.futures import ThreadPoolExecutor

from vit_prisma.models.base_vit import HookedViT

# ----------------------------------------------------------------------------
# Models to process in one run. The model tag is derived from each model name
# (the suffix after the final '.', e.g. "text_lora4"). The matching activation
# folders live under ACT_BASE as <dir_tag> (SUN val) and <dir_tag>_imagenet
# (ImageNet val), where dir_tag drops the leading "text_" (e.g. "lora4").
# ----------------------------------------------------------------------------
MODEL_NAMES = [
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lp",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_fft",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lora4",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lora16",
]

MODEL_NAMES = [
    "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_linear_probe_sun"
]

ACT_BASE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397"
# Original (baseline CLIP) activations are shared across all finetuned models.
orig_clip_inagenet_acts = "/home/nfm/ViT-Prisma/demos/head_acts_clipvit16_FullFT"
orig_clip_sun_acts = os.path.join(ACT_BASE, "clip_orig")

# ----------------------------------------------------------------------------
# Dataset configuration (shared across models)
#   These models are CLIP backbones fine-tuned as 400-class classifiers on the
#   text-overlay dataset, so `texts` (the DLA target classes) are the 400 class
#   names, in the same order as the model's logits / the dataset's val folders.
#   For ImageNet-1k models set DATASET="imagenet".
# ----------------------------------------------------------------------------
DATASET = "text_overlay"   # "imagenet" or any custom dataset name

if DATASET == "imagenet":
    texts = [idx_to_name[i] for i in range(1000)]
else:
    CLASSES_TXT = "/home/nfm/Desktop/rhome/nfm/sun_dataset_folder_sampled/classes.txt"
    with open(CLASSES_TXT) as f:
        texts = [line.strip() for line in f if line.strip()]
NUM_CLASSES = len(texts)   # = number of model output classes (400 here)
print(f"Dataset '{DATASET}': {NUM_CLASSES} classes.")

real_heads = list(range(48))

# ----------------------------------------------------------------------------
# Output configuration
# ----------------------------------------------------------------------------
OUT_DIR = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/sun"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Activation-change analysis config + helpers (defined once, reused per model).
# ----------------------------------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_PCS = 10   # subspace dimensionality used for principal-angle alignment


def _subspace_stats(X, k=N_PCS):
    """Independent statistics of one model's magnitude-filtered head activations.
    X: [n, 768] (already filtered). Returns None if too few samples."""
    n = X.shape[0]
    if n < 3:
        return None
    centroid = X.mean(0)
    Xc = X - centroid
    # Right singular vectors give the principal directions in the 768-d write space.
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    ev = (S.float() ** 2)
    if float(ev.sum()) <= 0:
        return None
    p = ev / ev.sum()
    p_nz = p[p > 0]
    kk = int(min(k, Vh.shape[0], n - 1))
    return {
        "n": n,
        "eff_dim": float((p.sum() ** 2) / (p ** 2).sum()),        # participation ratio
        "total_var": float(ev.sum() / (n - 1)),                    # per-sample variance energy
        "spec_entropy": float(-(p_nz * torch.log2(p_nz)).sum()),   # bits
        "basis": Vh[:kk].T.contiguous(),                           # [768, kk]
        "centroid": centroid,
        "k": kk,
    }


def _linear_cka(X, Y):
    """Paired linear CKA between two [n, d] activation matrices (same rows)."""
    if X.shape[0] < 3:
        return float("nan")
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).norm() ** 2
    xx = (X.T @ X).norm()
    yy = (Y.T @ Y).norm()
    denom = (xx * yy)
    return float(xy / denom) if float(denom) > 0 else float("nan")


def _principal_angles(Vo, Vf):
    """Principal angles between two orthonormal column-bases [768, k]."""
    k = min(Vo.shape[1], Vf.shape[1])
    sv = torch.linalg.svdvals(Vo[:, :k].T @ Vf[:, :k]).clamp(-1, 1)
    mean_cos = float(sv.mean())
    mean_angle_deg = float(torch.arccos(sv.clamp(-1, 1)).mean() * 180.0 / math.pi)
    return mean_cos, mean_angle_deg


def _load_half(path):
    return torch.load(path, weights_only=True).half()   # [48, b, 768]


# ============================================================================
# Run PART 1 (top texts) + PART 3 (activation-change metrics) for every model.
# ============================================================================
for model_name in MODEL_NAMES:
    # model_tag = model_name.split(".")[-1].replace("text_", "")   # "lp"/"fft"/"lora4"/"lora16"
    model_tag = "lp"
    is_lp = (model_tag == "lp")

    is_lp = True

    if is_lp:
        # Linear probing leaves the backbone/attention heads unchanged, so lp's
        # activations are identical to the original CLIP model. Reuse the original
        # CLIP activations for the top-texts DLA, and skip PART 3 entirely below
        # (there is no representational change to measure).
        head_new_acts = orig_clip_sun_acts
        head_imagenet_acts = orig_clip_inagenet_acts
    else:
        head_new_acts = os.path.join(ACT_BASE, model_tag)
        head_imagenet_acts = os.path.join(ACT_BASE, model_tag + "_imagenet")

    # PART 1 (top texts) needs head_new_acts; PART 3 (skipped for lp) also needs head_imagenet_acts.
    needed = [head_new_acts] if is_lp else [head_new_acts, head_imagenet_acts]
    if not all(os.path.isdir(d) for d in needed):
        print(f"[skip] {model_tag}: missing activation dir(s) {needed}")
        continue

    print(f"\n========== Processing {model_tag} ({model_name}) ==========")
    model = HookedViT.from_pretrained(model_name,
                                            center_writing_weights=True,
                                            center_unembed=True,
                                            fold_ln=True,
                                            refactor_factored_attn_matrices=True,
                                            device="cuda"
                                        )
    model = model.to("cuda:0")
    model.cfg.device = "cuda:0"
    print("Model device config:", model.cfg.device)


    # ========================================================================
    # PART 1 — New functionality of each head: top texts (via DLA), per model.
    #   Direct logit attribution through the model's own 400-class classifier head
    #   (is_clip=False), probed on this model's in-distribution text-overlay val
    #   activations. Top-texts json only (group scores skipped for this run).
    # ========================================================================
    DTA = calculate_dla(model, head_idx=real_heads, TENSORS_DIR=head_new_acts,
                        num_classes=NUM_CLASSES, is_clip=False)

    all_top_texts_dict = {}
    all_group_scores_dict = {}
    for head in real_heads:
        head_adj = head - real_heads[0]

        # Per-head adaptive thresholds on the direct-logit-attribution tensor.
        temp = torch.clone(DTA[:, head_adj, :, :]).flatten()
        low_thresh, high_thresh = find_elbows(temp, plot_elbows=False, print_elbows=False)

        # find_elbows can return high_thresh=None when <2 elbows are detected;
        # fall back to the top-k default in that case.
        thresh = (low_thresh, high_thresh) if high_thresh is not None else None

        top_texts_dict = find_top_texts_topk(
            DTA, head_idx=real_heads, target_idx=[head], texts=texts, thresh=thresh
        )
        head_name = list(top_texts_dict.keys())[0]

        all_top_texts_dict[f"Head_{head}"] = {
            "Head_Name": head_name,
            "Top_Texts": top_texts_dict[head_name],
        }

        # For each top text of this head, find which group(s) it belongs to in
        # group_data and accumulate the sum of its attribution list into those
        # groups. compute_group_scores starts every group at 0.0, so groups that
        # never get a hit stay 0.0 in the output.
        group_scores = compute_group_scores(top_texts_dict[head_name], group_data)
        all_group_scores_dict[f"Head_{head}"] = {
            "Head_Name": head_name,
            "Scores": group_scores,
        }
        print(f"[top-texts] {model_tag} {head_name} done.")

    with open(os.path.join(OUT_DIR, f"head_top_texts_{model_tag}.json"), "w") as f:
        json.dump(all_top_texts_dict, f, indent=4)
    print(f"Saved top texts for {len(all_top_texts_dict)} heads -> head_top_texts_{model_tag}.json")

    with open(os.path.join(OUT_DIR, f"all_group_scores_{model_tag}.json"), "w") as f:
        json.dump(all_group_scores_dict, f, indent=4)
    print(f"Saved group scores for {len(all_group_scores_dict)} heads -> all_group_scores_{model_tag}.json")

    del DTA

    # lp leaves the heads unchanged -> no representational change to measure; skip PART 3.
    if is_lp:
        print(f"[skip-part3] {model_tag}: linear probe leaves heads unchanged; "
              f"no activation-change metrics computed.")
        del model
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
        continue


    # (PART 2 group-score comparison removed: group scores are not computed in this run.)


    # ========================================================================
    # PART 3 — Representational change of each head (activation-based).
    #   Compare ORIGINAL (baseline CLIP) vs. this FINETUNED model's activation
    #   subspace, magnitude-filtered to the directions each head actually writes.
    #   Metrics: CKA_Linear, Subspace_Alignment, Principal_Angle_Deg,
    #   Eff_Dim_*, Total_Var_*, Spectral_Entropy_*, Centroid_Shift_*,
    #   Became_More_Specific. (Group-score comparison metrics dropped this run.)
    # ========================================================================


    # ---- Load this model's + the baseline's activations into memory. --------
    # Each tuple pairs the finetuned-model dir with the original-CLIP dir for the
    # SAME inputs (text-overlay val + ImageNet val); common files load in identical
    # sorted order so row i is the same sample in both banks -> paired CKA is valid.
    # float16 keeps memory modest (~4.5 GB per [48, total_samples, 768] bank).
    ACT_PAIRS = [
        ("text",     head_new_acts,      orig_clip_sun_acts),
        ("imagenet", head_imagenet_acts, orig_clip_inagenet_acts),
    ]

    # Reads release the GIL, so a thread pool overlaps the (latency-bound) NFS
    # round-trips for a big speedup. Keep per-file chunks as a list rather than
    # torch.cat-ing into one giant bank (that cat would briefly double RAM).
    f_chunks, o_chunks = [], []
    for tag, f_dir, o_dir in ACT_PAIRS:
        common = sorted(fn for fn in (set(os.listdir(f_dir)) & set(os.listdir(o_dir))) if fn.endswith('.pt'))
        print(f"Loading {len(common)} '{tag}' activation files from both models (parallel)...")
        with ThreadPoolExecutor(max_workers=16) as ex:
            f_loaded = list(ex.map(_load_half, [os.path.join(f_dir, fn) for fn in common]))
            o_loaded = list(ex.map(_load_half, [os.path.join(o_dir, fn) for fn in common]))
        for fn, ft, ot in zip(common, f_loaded, o_loaded):
            if ft.shape != ot.shape:
                print(f"  skipping {tag}/{fn}: shape mismatch {tuple(ft.shape)} vs {tuple(ot.shape)}")
                continue
            f_chunks.append(ft)
            o_chunks.append(ot)
        print(f"  loaded {len(f_chunks)} files total so far.")

    n_samples = sum(c.shape[1] for c in f_chunks)
    print(f"Activation banks ready: {len(f_chunks)} files, {n_samples} total samples per head.")

    change_metrics_dict = {}

    for head in real_heads:
        head_name = f"Layer {8 + head // 12} Head {head % 12}"

        # [total_samples, 768] paired activation matrices for this head, gathered from
        # the per-file chunks ([48, b, 768] -> [b, 768] for this head, then stacked).
        Fh = torch.cat([c[head] for c in f_chunks], dim=0).float().to(DEVICE)
        Oh = torch.cat([c[head] for c in o_chunks], dim=0).float().to(DEVICE)

        # Magnitude filtering: keep only the activations the head actually drives.
        f_norm, o_norm = Fh.norm(dim=1), Oh.norm(dim=1)
        f_thr = find_magnitude_threshold(f_norm)
        o_thr = find_magnitude_threshold(o_norm)
        f_mask = f_norm >= f_thr
        o_mask = o_norm >= o_thr
        union = f_mask | o_mask   # samples relevant to at least one model -> paired CKA

        stats_o = _subspace_stats(Oh[o_mask])
        stats_f = _subspace_stats(Fh[f_mask])

        metrics = {}
        if stats_o is not None and stats_f is not None:
            mean_cos, mean_angle = _principal_angles(stats_o["basis"], stats_f["basis"])
            co, cf = stats_o["centroid"], stats_f["centroid"]
            centroid_cos = float((co @ cf) / (co.norm() * cf.norm() + 1e-8))
            centroid_relnorm = float((cf - co).norm() / (0.5 * (co.norm() + cf.norm()) + 1e-8))

            eff_dim_change = stats_f["eff_dim"] - stats_o["eff_dim"]
            entropy_change = stats_f["spec_entropy"] - stats_o["spec_entropy"]

            metrics = {
                "CKA_Linear": round(_linear_cka(Oh[union], Fh[union]), 4),
                "Subspace_Alignment": round(mean_cos, 4),
                "Principal_Angle_Deg": round(mean_angle, 4),
                "Eff_Dim_Orig": round(stats_o["eff_dim"], 4),
                "Eff_Dim_Finetuned": round(stats_f["eff_dim"], 4),
                "Eff_Dim_Change": round(eff_dim_change, 4),
                "Total_Var_Orig": round(stats_o["total_var"], 4),
                "Total_Var_Finetuned": round(stats_f["total_var"], 4),
                "Total_Var_Ratio": round(stats_f["total_var"] / (stats_o["total_var"] + 1e-8), 4),
                "Spectral_Entropy_Orig_bits": round(stats_o["spec_entropy"], 4),
                "Spectral_Entropy_Finetuned_bits": round(stats_f["spec_entropy"], 4),
                "Spectral_Entropy_Change": round(entropy_change, 4),
                "Centroid_Shift_Cosine": round(centroid_cos, 4),
                "Centroid_Shift_RelNorm": round(centroid_relnorm, 4),
                "Became_More_Specific": int(eff_dim_change < 0 and entropy_change < 0),
                "N_Active_Orig": int(stats_o["n"]),
                "N_Active_Finetuned": int(stats_f["n"]),
            }

        change_metrics_dict[head_name] = metrics
        print(f"[change-metrics] {model_tag} {head_name} done.")

        del Fh, Oh
        if DEVICE != "cpu":
            torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, f"act_change_metrics_{model_tag}.json"), "w") as f:
        json.dump(change_metrics_dict, f, indent=4)
    print(f"Saved activation change metrics for {len(change_metrics_dict)} heads -> "
          f"act_change_metrics_{model_tag}.json")

    # ---- free this model before moving to the next one ----------------------
    del model, f_chunks, o_chunks
    if DEVICE != "cpu":
        torch.cuda.empty_cache()
