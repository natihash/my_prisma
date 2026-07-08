import sys
sys.path.insert(0, "/home/nfm/ViT-Prisma/src")

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

from vit_prisma.models.base_vit import HookedViT

# model_name = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_linear_probe_real"
# model_name = "vit_base_patch16_224"
# model_name = "vit_base_patch16_clip_224.laion2b_ft_in1k"
# model_name = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft"
model_name = "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
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
print("Is CUDA available?:", torch.cuda.is_available())

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

txt_pth = "/home/nfm/clip_text_span/text_descriptions/image_descriptions_general.txt"
with open(txt_pth, "r") as f:
    texts = [line.strip() for line in f.readlines()]

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

IMAGENET_VAL_DIR = '/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/'
# OUTPUT_DIR = '/home/nfm/ViT-Prisma/demos/head_acts_clipvit16_FullFT'
LOCAL_JSON_PATH = "/home/nfm/ViT-Prisma/demos/imagenet_class_index.json"

BATCH_SIZE = 50 # 50 images per class. 48GB GPU handles this easily!

dataset = ImageFolder(IMAGENET_VAL_DIR) 
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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

def calculate_dla(model, text_dict, head_idx, TENSORS_DIR):
    proj_head = model.head.W_H
    target_direction = text_dict @ proj_head.T
    target_direction = target_direction.to(model.cfg.device)

    if len(head_idx) >= 12 and text_dict.shape[0]>1000:
        print("Warning: You are slicing more than 12 heads and using more than 1000 text descriptions. This may lead to very large tensors that could exceed your GPU memory. Consider reducing the number of heads or text descriptions for a more manageable tensor size.")
        return None
        
    final_tensor = torch.zeros(1000,len(head_idx), 50, len(texts)) # [ImageClass, Head, LogitClass]

    pths = [f for f in os.listdir(TENSORS_DIR) if f.endswith('.pt')]
    for pth in pths:
        # Extract the WNID from the filename (e.g., 'n01440764_tench.pt' -> 'n01440764')
        wnid = pth.split('_')[0]
        
        # Find its exact 0-999 integer index
        true_class_idx = wnid_to_idx[wnid]
        
        # Load and compute
        filepath = os.path.join(TENSORS_DIR, pth)
        per_head_residual = torch.load(filepath, weights_only=True).to(model.cfg.device) 
        
        per_head_attribution = residual_stack_to_logit_attn(per_head_residual[head_idx], target_direction)
        
        # Insert it into the EXACT right row based on its official ImageNet index
        final_tensor[true_class_idx] = per_head_attribution.cpu()

        final_tensor = final_tensor.detach().cpu()
    
    return final_tensor

def find_top_texts_topk(final_tensor, head_idx, target_idx, texts, thresh = None, verbose=False):
    all_heads_dict = {}
    for shibu in range(len(target_idx)):
        messi = target_idx[shibu] - head_idx[0]
        if verbose:
            print(f"Analyzing Layer {8+head_idx[messi]//12} Head {head_idx[messi]%12}...")    
        temp = torch.clone(final_tensor[:,messi,:,:])
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

txt_pth = "/home/nfm/clip_text_span/text_descriptions/image_descriptions_general.txt"
with open(txt_pth, "r") as f:
    texts = [line.strip() for line in f.readlines()]

indices_dict = torch.load("/home/nfm/ViT-Prisma/demos/indices_forclip_dict.pt", weights_only=False)

TENSORS_DIR = '/home/nfm/ViT-Prisma/demos/head_acts_pure_clipvit16'
ggg = "Original Clip"

text_dict = torch.load("/home/nfm/ViT-Prisma/demos/text_dict.pt")
txt_pth = "/home/nfm/clip_text_span/text_descriptions/image_descriptions_general.txt"
with open(txt_pth, "r") as f:
    texts = [line.strip() for line in f.readlines()]
for_print = "With 3500 general text descriptions, model:" + ggg

# text_dict = torch.load("/home/nfm/ViT-Prisma/demos/text_dict_imagenet.pt")
# text_dict = text_dict.to("cuda")
# texts = [idx_to_name[i] for i in range(1000)]
# for_print = "With 1000 ImageNet class name descriptions, model:" + ggg

print(for_print)


import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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


import math
import torch


def calculate_weighted_effective_dimension(
    activations,
    weights=None,
    weight_transform="log",
    return_basis=False,
):
    N, D = activations.shape

    if weights is None:
        weights = torch.ones(N, 1, device=activations.device)

    if weights.dim() == 1:
        weights = weights.unsqueeze(1)

    # transform weights
    if weight_transform == "log":
        w = torch.log1p(weights)
    elif weight_transform == "sqrt":
        w = torch.sqrt(weights)
    else:
        w = weights.clone()

    # normalize
    w = w / w.sum()

    # weighted mean
    weighted_mean = (activations * w).sum(dim=0, keepdim=True)

    # center
    centered = activations - weighted_mean

    # weighted covariance
    cov = (centered * w).T @ centered

    # eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # ascending -> descending
    idx = torch.argsort(eigenvalues, descending=True)

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = torch.clamp(eigenvalues, min=0)

    # participation ratio
    sum_eig = eigenvalues.sum()
    sum_sq_eig = (eigenvalues ** 2).sum()

    eps = 1e-12
    if sum_sq_eig < eps:
        effective_dim = 0.0
    else:
        effective_dim = ((sum_eig ** 2) / sum_sq_eig).item()

    if not return_basis:
        return effective_dim

    n_basis = math.ceil(effective_dim)

    top_eigenvectors = eigenvectors[:, :n_basis]
    top_eigenvalues = eigenvalues[:n_basis]

    explained_variance_ratio = (
        top_eigenvalues.sum() / eigenvalues.sum()
    ).item()

    return {
        "effective_dim": effective_dim,
        "n_basis": n_basis,
        "basis": top_eigenvectors,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "all_eigenvalues": eigenvalues,
    }


import os
import torch

def get_thresholded_activations(dta_matrix, original_head_idx, target_heads, TENSORS_DIR, wnid_to_idx, low_threshold, high_threshold, consider_neg=False):
    """
    Filters and stacks activations for a specific subset of heads based on DTA thresholds.
    """
    # 1. Isolate the target heads in the DTA matrix
    rel_indices = [original_head_idx.index(h) for h in target_heads]
    dta_subset = dta_matrix[:, rel_indices, :, :]
    
    # 2. Apply the head-specific thresholds to the subset
    high_t = torch.tensor(high_threshold, device=dta_matrix.device).view(1, -1, 1, 1)
    low_t = torch.tensor(low_threshold, device=dta_matrix.device).view(1, -1, 1, 1)
    
    if consider_neg:
        condition = (dta_subset > high_t) | (dta_subset < low_t)
    else:
        condition = (dta_subset > high_t)
    mask = condition.any(dim=-1) # Shape: (1000, len(target_heads), 50)
    
    filtered_activations = []
    metadata = []
    
    pths = [f for f in os.listdir(TENSORS_DIR) if f.endswith('.pt')]
    
    for pth in pths:
        wnid = pth.split('_')[0]
        if wnid not in wnid_to_idx:
            continue
            
        true_class_idx = wnid_to_idx[wnid]
        class_mask = mask[true_class_idx] # Shape: (len(target_heads), 50)
        
        # Skip loading if no images in this class met the threshold for the target heads
        if not class_mask.any():
            continue
            
        # Load residual file
        filepath = os.path.join(TENSORS_DIR, pth)
        per_head_residual = torch.load(filepath, map_location='cpu', weights_only=True)
        
        # EXTRACT ONLY THE TARGET HEADS from the saved tensor
        subset_residual = per_head_residual[target_heads] 
        
        # Apply the mask to extract the thresholded activations
        matching_acts = subset_residual[class_mask] 
        filtered_activations.append(matching_acts)
        
        # Store metadata
        indices = class_mask.nonzero(as_tuple=False) 
        for idx in indices:
            rel_head_in_subset = idx[0].item()
            img_id = idx[1].item()
            
            metadata.append({
                'class_idx': true_class_idx,
                'wnid': wnid,
                'head_idx': target_heads[rel_head_in_subset], # Maps back to the exact model head
                'image_idx': img_id
            })
            
    if not filtered_activations:
        print("No activations met the threshold criteria for the target heads.")
        return None, []
        
    final_stacked_acts = torch.cat(filtered_activations, dim=0)
    
    return final_stacked_acts, metadata

def get_all_activations_for_head(head, TENSORS_DIR, wnid_to_idx):
    """Load every stored activation vector for a single head across all classes."""
    all_acts = []
    for pth in sorted(os.listdir(TENSORS_DIR)):
        if not pth.endswith('.pt'):
            continue
        wnid = pth.split('_')[0]
        if wnid not in wnid_to_idx:
            continue
        filepath = os.path.join(TENSORS_DIR, pth)
        per_head_residual = torch.load(filepath, map_location='cpu', weights_only=True)
        all_acts.append(per_head_residual[head])  # (50, d_model)
    return torch.cat(all_acts, dim=0).float()  # (N_images, d_model)


# a function
import torch
import torch.nn.functional as F

def calculate_weighted_sci(text_embeddings, weights):
    """
    Calculates the Weighted Semantic Coherence Index (wSCI).
    
    Args:
        text_embeddings: Tensor of shape (N, 512). The unique top texts.
        weights: Tensor of shape (N,). E.g., The sum of attribution scores for each text.
    Returns:
        wSCI score (float between -1.0 and 1.0)
    """
    N = text_embeddings.shape[0]
    if N <= 1:
        return 1.0 # Perfect coherence if only 1 text survives filtering
        
    # 1. Normalize embeddings to calculate cosine similarity via dot product
    normed_embeds = F.normalize(text_embeddings, p=2, dim=1)
    
    # 2. Pairwise Cosine Similarity Matrix (N x N)
    sim_matrix = normed_embeds @ normed_embeds.T
    
    # 3. Create Pairwise Weight Matrix (N x N)
    # weights.unsqueeze(1) is (N, 1), weights.unsqueeze(0) is (1, N)
    weight_matrix = weights.unsqueeze(1) @ weights.unsqueeze(0)
    
    # 4. Remove Self-Similarity (The Diagonal)
    # We don't want a text compared to itself to inflate the score.
    mask = torch.ones((N, N), device=text_embeddings.device) - torch.eye(N, device=text_embeddings.device)
    
    sim_matrix = sim_matrix * mask
    weight_matrix = weight_matrix * mask
    
    # 5. Calculate Weighted Average
    weighted_sum = torch.sum(sim_matrix * weight_matrix)
    total_weight = torch.sum(weight_matrix)
    
    if total_weight == 0:
        return 0.0
        
    wsci = weighted_sum / total_weight
    return wsci.item()


import torch

def calculate_entropy(values, is_logits=False, base=2, dim=-1):
    """
    Converts a set of values to probabilities and calculates their Shannon entropy.
    
    Args:
        values: torch.Tensor of shape (..., N)
        is_logits: bool. If True, applies Softmax. If False, applies L1 normalization.
        base: int or str. Base of the logarithm (2 for bits, 'e' for nats).
        dim: int. The dimension to compute the entropy over. Default is the last dim.
        
    Returns:
        torch.Tensor containing the entropy values.
    """
    # 1. Convert to probabilities
    if is_logits:
        # Softmax handles negative numbers and scales exponentially to sum to 1
        probs = torch.softmax(values, dim=dim)
    else:
        # Standard normalization: values must be non-negative.
        # We clamp to 0 just in case there are small floating point negatives.
        clamped_values = torch.clamp(values, min=0.0)
        
        # Add epsilon to the sum to prevent division by zero if all values are 0
        total = clamped_values.sum(dim=dim, keepdim=True) + 1e-9
        probs = clamped_values / total

    # 2. Calculate Entropy
    # Add a tiny epsilon to probabilities to avoid log(0) -> -inf -> NaN
    eps = 1e-9
    
    if base == 2:
        log_probs = torch.log2(probs + eps)
    elif base == 'e':
        log_probs = torch.log(probs + eps) # Natural log
    else:
        raise ValueError("Base must be 2 or 'e'")
        
    # Calculate -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=dim)
    
    return entropy

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


import os
import torch

def get_ablation_attribution_tensor(layer, base_dir="/home/nfm/ViT-Prisma/mynotebooks/sims_tensors"):
    # 1. Load the baseline tensor [50000, 3498]
    baseline_path = os.path.join(base_dir, "sims_baseline.pt")
    print(f"Loading baseline from {baseline_path}...")
    baseline_sims = torch.load(baseline_path).cpu()

    # 2. Collect difference tensors for all 12 heads
    num_heads = 12
    diff_tensors = []
    
    for head in range(num_heads):
        ablated_path = os.path.join(base_dir, f"sims_L{layer}_H{head}.pt")
        # print(f"Loading ablated tensor for L{layer}_H{head}...")
        ablated_sims = torch.load(ablated_path).cpu()
        
        # Calculate difference: Baseline - Ablated
        # Positive value = Ablation lowered similarity (Head was contributing positively)
        # Negative value = Ablation increased similarity (Head was a distractor)
        diff = baseline_sims - ablated_sims
        diff_tensors.append(diff)
        
    # 3. Stack the 12 tensors
    # Resulting shape: [12, 50000, 3498]
    print(f"Stacking tensors for layer {layer}...")
    stacked_diffs = torch.stack(diff_tensors, dim=0)
    
    # 4. Reshape to separate the 1000 classes and 50 images
    # 50000 -> 1000 classes, 50 images per class
    # New shape: [12, 1000, 50, 3498]
    reshaped_diffs = stacked_diffs.view(12, 1000, 50, 3498)
    
    # 5. Permute to get the final target shape: [1000, 12, 50, 3498]
    # Dimension mapping: (heads=0, classes=1, images=2, texts=3) -> (classes=1, heads=0, images=2, texts=3)
    final_tensor = reshaped_diffs.permute(1, 0, 2, 3)
    
    # Ensure contiguous memory after permute (good practice before saving or complex math)
    final_tensor = final_tensor.contiguous()
    
    print(f"Successfully created tensor of shape {final_tensor.shape}")
    return final_tensor


import os
import torch
import json
import gc

# 1. Define paths and mappings for the chunked DTA files
dta_files = [
    ("/home/nfm/ViT-Prisma/mynotebooks/files/CLIP_DTA_0_12.pt", list(range(0, 12))),
    ("/home/nfm/ViT-Prisma/mynotebooks/files/CLIP_DTA_12_24.pt", list(range(12, 24))),
    ("/home/nfm/ViT-Prisma/mynotebooks/files/CLIP_DTA_24_36.pt", list(range(24, 36))),
    ("/home/nfm/ViT-Prisma/mynotebooks/files/CLIP_DTA_36_48.pt", list(range(36, 48)))
]

# Paths for the output files
LOGS_OUTPUT_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/head_spec_metrics.json"
GROUP_SCORES_OUTPUT_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/all_group_scores.json"
HEAD_TEXTS_OUTPUT_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/head_top_texts.json"
ELBOWS_CACHE_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/elbows_cache.json"

# Load the text dictionary once (since it's shared across all heads)
text_dict = torch.load("/home/nfm/ViT-Prisma/demos/text_dict.pt")

# Initialize containers for the accumulated data
all_metrics_dict = {}   # head_name -> metric dict (written to JSON)
all_group_scores_dict = {}
all_top_texts_dict = {}

# Load elbows cache so the expensive sort-and-smooth step can be skipped on reruns
if os.path.exists(ELBOWS_CACHE_FILE):
    with open(ELBOWS_CACHE_FILE, 'r') as _f:
        elbows_cache = json.load(_f)
    print(f"Loaded elbows cache with {len(elbows_cache)} entries from {ELBOWS_CACHE_FILE}")
else:
    elbows_cache = {}

print("Starting automated workflow for all heads...")


# 2. Iterate through each file and its corresponding heads
for file_path, real_heads in dta_files:
    print(f"\nLoading DTA matrix: {file_path}")
    DTA_imagent = torch.load(file_path)

    # layer_to_analyze = real_heads[0] // 12 + 8 # Calculate layer number based on the first head index in the chunk
    # DTA_imagent = get_ablation_attribution_tensor(layer_to_analyze)
    # DTA_imagent.shape
    
    # Process each head in the current file chunk
    for head in real_heads:
        print(f"  Processing Head {head}...")
        head_adj = head - real_heads[0] # Calculate the relative index (0 to 11)
        
        # Calculate thresholds (use cache to avoid re-sorting 170M values)
        head_key = str(head)
        if head_key in elbows_cache:
            low_thresh  = elbows_cache[head_key]["low"]
            high_thresh = elbows_cache[head_key]["high"]
        else:
            temp = torch.clone(DTA_imagent[:, head_adj, :, :]).flatten()
            low_thresh, high_thresh = find_elbows(temp, plot_elbows=False, print_elbows=False)
            elbows_cache[head_key] = {"low": low_thresh, "high": high_thresh}
            with open(ELBOWS_CACHE_FILE, 'w') as _f:
                json.dump(elbows_cache, _f)

        # Get top texts
        top_texts_dict = find_top_texts_topk(
            DTA_imagent, head_idx=real_heads, target_idx=[head], 
            texts=texts, thresh=(low_thresh, high_thresh)
        )
        
        # Skip if no texts met the criteria
        if not top_texts_dict:
            print(f"    Warning: No top texts found for Head {head}. Skipping.")
            continue

        head_name = list(top_texts_dict.keys())[0]

        # Extract text embeddings and weights
        stacked_text_embeds = []
        stacked_texts = []
        weights = []
        
        for text, score_list in top_texts_dict[head_name].items():
            text_idx = texts.index(text)
            text_embed = text_dict[text_idx]
            stacked_text_embeds.append(text_embed)
            stacked_texts.append(text)
            weights.append(len(score_list))

        stacked_text_embeds = torch.stack(stacked_text_embeds)
        weights = torch.tensor(weights, dtype=torch.float32, device=stacked_text_embeds.device).unsqueeze(1)

        # Extract thresholded activations
        stacked_activations, metadata = get_thresholded_activations(
            DTA_imagent, original_head_idx=real_heads, target_heads=[head],
            TENSORS_DIR=TENSORS_DIR, wnid_to_idx=wnid_to_idx,
            low_threshold=[low_thresh], high_threshold=[high_thresh]
        )

        # Calculate metrics
        wSCI_score = calculate_weighted_sci(stacked_text_embeds, weights[:, 0])
        acts_dict = calculate_weighted_effective_dimension(stacked_activations, weights=None, weight_transform="log", return_basis=True)
        act_pr = acts_dict["effective_dim"]
        text_pr = calculate_weighted_effective_dimension(stacked_text_embeds, weights=weights, weight_transform="log")

        # Entropy of eigenvalues
        eigenvalues = acts_dict["all_eigenvalues"]
        act_entropy = calculate_entropy(eigenvalues, is_logits=False, base=2)

        # Group scores and entropy
        group_scores = compute_group_scores(top_texts_dict[head_name], group_data)
        group_values = list(group_scores.values())
        group_values_tensor = torch.tensor(group_values, device=stacked_text_embeds.device)
        group_entropy = calculate_entropy(group_values_tensor, is_logits=False, base=2)

        # Magnitude-based effective dimension (threshold by norm, not DTA attribution)
        all_head_acts = get_all_activations_for_head(head, TENSORS_DIR, wnid_to_idx)
        norms = torch.norm(all_head_acts, dim=-1)
        mag_threshold = find_magnitude_threshold(norms)
        high_mag_acts = all_head_acts[norms > mag_threshold]
        if high_mag_acts.shape[0] > 1:
            act_mag_eff_dim = calculate_weighted_effective_dimension(
                high_mag_acts, weights=None, weight_transform="log"
            )
        else:
            act_mag_eff_dim = 1.0
        del all_head_acts, high_mag_acts, norms

        # --- STORE RESULTS ---
        all_metrics_dict[head_name] = {
            "wSCI_Score":        round(wSCI_score.item() if isinstance(wSCI_score, torch.Tensor) else wSCI_score, 4),
            "Act_Eff_Dim":       round(act_pr, 2),
            "Text_Eff_Dim":      round(text_pr, 2),
            "Act_Entropy_bits":  round(act_entropy.item() if isinstance(act_entropy, torch.Tensor) else act_entropy, 4),
            "Group_Entropy_bits":round(group_entropy.item() if isinstance(group_entropy, torch.Tensor) else group_entropy, 4),
            "Act_Mag_Eff_Dim":   round(act_mag_eff_dim, 2),
        }

        print(f"    Metrics for Head {head}: wSCI={wSCI_score:.4f}, ActEffDim={act_pr:.2f}, TextEffDim={text_pr:.2f}, ActEnt={act_entropy:.4f} bits, GroupEnt={group_entropy:.4f} bits, MagEffDim={act_mag_eff_dim:.2f}")

        # 2. Accumulate the full group scores dictionary
        all_group_scores_dict[f"Head_{head}"] = {
            "Head_Name": head_name,
            "Scores": group_scores
        }
        all_top_texts_dict[f"Head_{head}"] = {
            "Head_Name": head_name,
            "Top_Texts": top_texts_dict[head_name]
        }
    # Memory management: Delete the massive DTA chunk and clear cache before loading the next one
    del DTA_imagent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 3. Save the metrics as JSON: {head_name: {metric: value, ...}}
print(f"\nSaving metrics to {LOGS_OUTPUT_FILE}...")
with open(LOGS_OUTPUT_FILE, mode='w', encoding='utf-8') as json_file:
    json.dump(all_metrics_dict, json_file, indent=2)

# 4. Save the full group scores to a JSON file
print(f"Saving full group scores to {GROUP_SCORES_OUTPUT_FILE}...")
with open(GROUP_SCORES_OUTPUT_FILE, mode='w', encoding='utf-8') as json_file:
    json.dump(all_group_scores_dict, json_file, indent=4)

# 5. Save the top texts for each head to a JSON file
print(f"Saving top texts to {HEAD_TEXTS_OUTPUT_FILE}...")
with open(HEAD_TEXTS_OUTPUT_FILE, mode='w', encoding='utf-8') as json_file:
    json.dump(all_top_texts_dict, json_file, indent=4)

print("Workflow complete!")