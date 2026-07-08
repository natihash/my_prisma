import sys
sys.path.insert(0, "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/src")

import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from vit_prisma.utils.constants import BASE_DIR, DATA_DIR, MODEL_DIR, DEVICE, MODEL_CHECKPOINTS_DIR
from vit_prisma.utils.tutorial_utils import calculate_clean_accuracy, load_clip_models, plot_image, get_feature_activations
from vit_prisma.utils.data_utils.loader import load_dataset
from vit_prisma.utils.tutorial_utils import plot_act_distribution
from vit_prisma.utils.tutorial_utils import plot_top_imgs_for_features

DEVICE = 'cuda' # change to cpu if cpu only paradigm
BATCH_SIZE = 16

# Load an SAE
from huggingface_hub import hf_hub_download, list_repo_files
from vit_prisma.sae import SparseAutoencoder
import glob

def load_sae(repo_id, file_name, config_name):
    # Step 1: Download SAE from Hugginface
    sae_path = hf_hub_download(repo_id, file_name) # Download weights
    hf_hub_download(repo_id, config_name) # Download config

    # sae_path = repo_id
    # Step 2: Load the pretrained SAE weights from the downloaded path
    print(f"Loading SAE from {sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(sae_path) # This now automatically gets config.json and converts into the VisionSAERunnerConfig object
    return sae

def my_load_sae(pth):
    config_path = os.path.join(pth, "config.json")
    # weights_path = os.path.join(pth, "n_images_13000704.pt")
    weights = glob.glob(os.path.join(pth, "*.pt"))
    weights.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    weights = [w for w in weights if not "log_" in os.path.basename(w)]
    weights_path = weights[-1]

    print(f"----------------Loading SAE from {weights_path} ----------------")

    sae = SparseAutoencoder.load_from_pretrained(weights_path=weights_path, config_path=config_path) # This now automatically gets config.json and converts into the VisionSAERunnerConfig object
    sae.eval()
    sae.to(DEVICE)
    
    return sae

# repo_id = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_mlp_out-l1-1e-05" # Change this to your chosen SAE. See /docs for a list of SAEs.
# repo_id2 = "/home/nfm/ViT-Prisma/demos/sae_ckpts/ee158df6-tinyclip_sae_16_hyperparam_sweep_lr"
# repo_id3 = "/home/nfm/ViT-Prisma/demos/sae_ckpts/892ec2b9-tinyclip_sae_16_hyperparam_sweep_lr"
# repo_id4 = "/home/nfm/ViT-Prisma/demos/sae_ckpts/f103614d-tinyclip_sae_16_hyperparam_sweep_lr"
# repo_id5 = "/home/nfm/ViT-Prisma/demos/sae_ckpts/59fcae83-sae_training_clip_b16"
# repo_id6 = "/home/nfm/ViT-Prisma/demos/sae_ckpts/82d22c1d-sae_training_clip_b16_cls_only"
# repo_id7 = "/home/nfm/ViT-Prisma/demos/sae_ckpts/2e0029a5-claude_clip2_topk"
repo_id7 = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/demos/sae_ckpts/29e0111c-claude_topk_all"
file_name = "weights.pt"
config_name = "config.json"
sae = my_load_sae(repo_id7)
# sae = load_sae(repo_id, file_name, config_name)


# Load model
from vit_prisma.models.model_loader import load_hooked_model

model_name = sae.cfg.model_name
model = load_hooked_model(model_name)
model.to(DEVICE); # Move to device

import os
import json
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import open_clip
IMAGENET_VAL_DIR = '/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'
LOCAL_JSON_PATH = "/home/nfm/ViT-Prisma/demos/imagenet_class_index.json"
BATCH_SIZE = 50
DEVICE = 'cuda'
_, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
dataset = ImageFolder(IMAGENET_VAL_DIR, transform=preprocess) 
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

OUTPUT_PATH = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/demos/activation_cache/corrected_fcs_topk_all_final.pt"
feature_class_scores = torch.load(OUTPUT_PATH, map_location=DEVICE)
feature_class_scores.shape

texts_path = "/home/nfm/clip_text_span/text_descriptions/image_descriptions_general.txt"
with open(texts_path, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

import matplotlib.pyplot as plt


import os
import json
import torch
from torch import einsum

loud_idx = torch.load("/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/loud_features.pt", map_location=DEVICE)
# loud_idx = loud_idx[:200]
# "Loud" features fire almost everywhere, so they dominate attribution without being
# meaningful. Keep their indices in a set so we can skip them during feature selection.
loud_set = set(int(i) for i in loud_idx.flatten().tolist())
print(f"Ignoring {len(loud_set)} loud features during selection")


TENSORS_DIR = '/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/original_acts_all_patches'
ggg = "Original Clip"

LOCAL_JSON_PATH = "/home/nfm/ViT-Prisma/demos/imagenet_class_index.json"

with open(LOCAL_JSON_PATH, 'r') as f:
    imagenet_class_index = json.load(f)

# Create a reverse lookup: "n01440764" -> 0
wnid_to_idx = {wnid: int(idx) for idx, (wnid, name) in imagenet_class_index.items()}
idx_to_name = {int(idx): name for idx, (wnid, name) in imagenet_class_index.items()}

def residual_stack_to_logit_attn(centered_residual_stack, answer_residual_direction):
    # This subtracts the mean across the d_model dimension (dim=-1)
    # centered_residual_stack = residual_stack - residual_stack.mean(dim=-1, keepdim=True)
    
    raw_dla = torch.einsum(
        "h b d, c d -> h b c",  # h=heads, b=batch, d=d_model, c=classes
        centered_residual_stack,
        answer_residual_direction,
    )
    return raw_dla 

sae_encoder_direction = sae.W_enc.T
# sae_encoder_direction = sae.W_dec

group_data = json.load(open("/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/groups_output_combined_new.json"))
all_top_texts_dict = {}
all_group_scores_dict = {}

def select_top_texts_for_feature(feat_vec, texts):
    """Return the single highest-scoring text for a feature.

    `feat_vec` is one row of `feature_class_scores` (a score for every text).
    We take just the top text; if its score is <= 0 we return nothing.

    Returns a list with at most one (text, score) pair so the caller's loop
    stays unchanged (the score is the text's own feature_class_score and is
    used only for display; it is *not* the weight that propagates to the head).
    """
    val, idx = feat_vec.max(dim=0)
    idx, val = int(idx), float(val)
    if val <= 0 or idx >= len(texts):
        return []
    return [(texts[idx], val)]


def compute_group_scores(top_texts, group_data):
    """Aggregate per-text scores into the 95 semantic groups.

    `top_texts` maps text -> list of scores. Each entry in the list is one
    feature score (count * mean attribution) that the text picked up from a
    head's top feature; a text appearing under several features therefore has
    several entries. A text's weight is the sum of its list, and every group it
    belongs to accumulates that weight (a text can land in multiple groups, and
    multiple texts can land in the same group -> both accumulate).

    Returns EVERY group (in group_data order), with 0.0 for groups that got no
    attribution.
    """
    # Build text/class -> groups lookup (includes both texts and imagenet_classes)
    text_to_groups = defaultdict(list)
    for group_name, group_info in group_data["groups"].items():
        for text in group_info.get("texts", []):
            text_to_groups[text].append(group_name)
        for cls_name in group_info.get("imagenet_classes", []):
            text_to_groups[cls_name].append(group_name)

    # Initialize all groups at 0, then accumulate each text's weight.
    group_scores = {group_name: 0.0 for group_name in group_data["groups"]}
    for text, score_list in top_texts.items():
        weight = float(sum(score_list))
        for group in text_to_groups.get(text, []):  # unassigned texts are skipped
            group_scores[group] += weight

    return group_scores


# The cached activations contain only the LAST 48 heads of the model
# (tensor shape (48, 50, 768) == 4 layers x 12 heads). Local head index 0 maps
# to the first of those, so we offset local -> global numbering for the labels.
HEADS_PER_LAYER = model.cfg.n_heads                       # 12
GLOBAL_HEAD_OFFSET = model.cfg.n_layers * model.cfg.n_heads - 48   # 144 - 48 = 96
GLOBAL_LAYER_OFFSET = GLOBAL_HEAD_OFFSET // HEADS_PER_LAYER         # 8

torch.set_grad_enabled(False)  # pure inference/analysis; avoids building any autograd graph

start_inds = [0, 12, 24, 36]  # local indices into the 48 cached heads (covers all of them)
for xx in start_inds:

    head_idx = list(range(xx, xx+12))  # 12 heads per layer

    # ~27.5 GiB in fp32 (1000 x 12 x 50 x 12288). Kept on the GPU (A40, ~45 GiB free)
    # because it won't fit in available CPU RAM; freed at the end of each chunk.
    final_tensor = torch.zeros(
        1000, len(head_idx), 50, sae.cfg.d_in * sae.cfg.expansion_factor, device=DEVICE
    )

    pths = [f for f in os.listdir(TENSORS_DIR) if f.endswith('.pt')]

    for pth in pths:
        # Extract the WNID from the filename (e.g., 'n01440764_tench.pt' -> 'n01440764')
        wnid = pth.split('_')[0]
        
        # Find its exact 0-999 integer index
        true_class_idx = wnid_to_idx[wnid]
        
        # Load and compute
        filepath = os.path.join(TENSORS_DIR, pth)
        per_head_residual = torch.load(filepath, weights_only=True).to("cuda") 
        
        per_head_attribution = residual_stack_to_logit_attn(per_head_residual[head_idx], sae_encoder_direction)
        
        # Insert it into the EXACT right row based on its official ImageNet index
        final_tensor[true_class_idx] = per_head_attribution  # already on GPU

    final_tensor = final_tensor.detach()


    for z in range(final_tensor.shape[1]):
        # True (global) layer/head identity (only the last 48 heads are cached).
        global_layer = GLOBAL_LAYER_OFFSET + xx // HEADS_PER_LAYER
        head_counter = xx + z                         # 0..47 across the cached heads
        head_key = f"Head_{head_counter}"             # json key, e.g. "Head_0"
        head_name = f"Layer {global_layer} Head {z}"  # e.g. "Layer 8 Head 0"

        temp = final_tensor[:, z, :, :]  # (classes, patches, features) view for this head
        gaga = 20000
        values, flat_idx = temp.reshape(-1).topk(gaga, largest=True)

        # We only need the feature index of each top attribution (class/patch unused).
        C = temp.shape[2]
        feat_of_top = (flat_idx % C).cpu()  # -> CPU so the python loop below is cheap
        values = values.cpu()

        # Group the top-`gaga` attributions by feature. EVERY feature that appears in
        # the top-`gaga` attributions is considered (not just the top 20), except
        # "loud" features, which are skipped entirely.
        avg_vals = defaultdict(list)
        for n in range(gaga):
            feat = int(feat_of_top[n])
            if feat in loud_set:
                continue
            avg_vals[feat].append(float(values[n]))

        # (feature, count) for every feature present, strongest (most frequent) first.
        feat_inds = sorted(
            ((feat, len(vals)) for feat, vals in avg_vals.items()),
            key=lambda fc: fc[1], reverse=True,
        )

        # Short summary print of the strongest features only.
        for name, count in feat_inds[:20]:
            print(f"{name}: {count} : {sum(avg_vals[name])/len(avg_vals[name]):.3f}", end="||")
        print(f"\n[{head_key}] {len(feat_inds)} distinct features in top {gaga}")
        print("-"*50)

        # ---- build text & group labels for this head ----
        # text -> list of feature scores (accumulates when a text recurs across features)
        head_top_texts = defaultdict(list)

        for n_feat, (idx, count) in enumerate(feat_inds):
            mean_attr = sum(avg_vals[idx]) / len(avg_vals[idx])
            feat_score = count * mean_attr  # product of count and mean attribution

            feat_vec = feature_class_scores[idx]
            selected = select_top_texts_for_feature(feat_vec, texts)  # single top text

            if n_feat < 20:  # avoid flooding stdout when there are thousands of features
                print(f"\nFeature {idx}: count={count} mean={mean_attr:.3f} score={feat_score:.3f}")
            for text, val in selected:
                # every selected text under this feature gets the feature's score;
                # duplicates across features accumulate in the list
                head_top_texts[text].append(feat_score)
                if n_feat < 20:
                    print(f"  {text:40s}  {val:.4f}")

        head_top_texts = dict(head_top_texts)

        # Raw texts for this head (before group classification): summed/accumulated
        # score per text, sorted high -> low.
        all_top_texts_dict[head_key] = {
            "Head_Name": head_name,
            "Scores": {
                text: float(sum(scores))
                for text, scores in sorted(
                    head_top_texts.items(), key=lambda kv: sum(kv[1]), reverse=True
                )
            },
        }

        # Group labels for this head: each text contributes its summed score to
        # every group it falls into, accumulated across texts. Every group is
        # listed (0.0 if it received no attribution).
        all_group_scores_dict[head_key] = {
            "Head_Name": head_name,
            "Scores": compute_group_scores(head_top_texts, group_data),
        }

    # Free the ~27.5 GiB chunk before the next iteration allocates a fresh one,
    # otherwise the new torch.zeros would briefly need 2x the memory.
    del final_tensor, temp, values, flat_idx
    torch.cuda.empty_cache()


# ---- save the two outputs: raw texts per head, and group scores per head ----
TEXTS_OUT_PATH = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_top_texts_sae_fixed3.json"
GROUPS_OUT_PATH = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_group_scores_sae_fixed3.json"

with open(TEXTS_OUT_PATH, "w") as f:
    json.dump(all_top_texts_dict, f, indent=2)
with open(GROUPS_OUT_PATH, "w") as f:
    json.dump(all_group_scores_dict, f, indent=2)

print(f"\nSaved raw head texts   -> {TEXTS_OUT_PATH}")
print(f"Saved head group scores -> {GROUPS_OUT_PATH}")

