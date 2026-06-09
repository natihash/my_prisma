import sys
sys.path.insert(0, "/home/nfm/ViT-Prisma/src")

import os
import json
import torch
from tqdm.auto import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vit_prisma.utils import prisma_utils

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

from vit_prisma.transforms import get_clip_val_transforms

# ==========================================
# 1. Setup & Data Loading
# ==========================================

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

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256 # Adjust based on your GPU VRAM

# Load your 3498 normalized text embeddings
# Shape should be [3498, d_model]
target_text_embeds = torch.load("/home/nfm/ViT-Prisma/demos/text_dict.pt").to(device)
if target_text_embeds.dtype != torch.float32:
    target_text_embeds = target_text_embeds.float() 

# Setup ImageNet Dataloader
# (Ensure your transform matches the exact CLIP preprocessing you used for baseline)
val_dir = "/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/" # UPDATE THIS
val_dataset = datasets.ImageFolder(val_dir, transform=get_clip_val_transforms())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Setup ImageNet Class Text Embeddings for Zero-Shot
# Assuming `imagenet_text_embeds` is pre-calculated [1000, d_model] and normalized
# imagenet_text_embeds = ... 
imagenet_text_embeds = torch.load("/home/nfm/ViT-Prisma/demos/text_dict_imagenet.pt") # UPDATE THIS PATH
if imagenet_text_embeds.dtype != torch.float32:
    imagenet_text_embeds = imagenet_text_embeds.float()
imagenet_text_embeds = imagenet_text_embeds.to(device)

# val_dataset is your torchvision.datasets.ImageFolder
dataloader_wnids = val_dataset.classes # These are the folder names, sorted alphabetically

# Create a new, aligned text embedding tensor
aligned_text_embeds = torch.zeros_like(imagenet_text_embeds)

for dataloader_idx, wnid in enumerate(dataloader_wnids):
    # Find what index this wnid was in your original JSON mapping
    original_idx = name_to_idx[wnid_to_name[wnid]] 
    aligned_text_embeds[dataloader_idx] = imagenet_text_embeds[original_idx]

# Now use `aligned_text_embeds` in your evaluation loop!

# ==========================================
# 2. The Hook Function
# ==========================================

def zero_ablate_head_hook(z, hook, head_index):
    """
    Zeros out the activation for a specific attention head.
    z shape is typically: [batch, seq_len, n_heads, d_head]
    """
    z[:, :, head_index, :] = 0.0
    return z

# ==========================================
# 3. Main Evaluation Function
# ==========================================

@torch.no_grad()
def evaluate_ablated_head(
    model, 
    layer, 
    head_index, 
    dataloader, 
    imagenet_embeds, 
    target_embeds, 
    save_dir
):
    # Hook name for the 'z' activations (post-attention, pre-projection)
    hook_name = prisma_utils.get_act_name('z', layer, "attn")
    hook_fn = partial(zero_ablate_head_hook, head_index=head_index)
    
    all_similarities = []
    correct = 0
    total = 0
    
    # Process the dataset in batches
    for images, labels in tqdm(dataloader, desc=f"Evaluating L{layer} H{head_index}", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Run model with the ablation hook
        # Note: Ensure that run_with_hooks returns the final, projected image embeddings
        image_embeds = model.run_with_hooks(
            images,
            fwd_hooks=[(hook_name, hook_fn)]
        )
        
        # Normalize image embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # --- 1. ImageNet Zero-Shot Accuracy ---
        # logits shape: [batch_size, 1000]
        logits = 100.0 * image_embeds @ imagenet_embeds.T 
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # --- 2. Target Texts Similarities ---
        # sims shape: [batch_size, 3498]
        sims = image_embeds @ target_embeds.T
        
        # Cast to bfloat16 to save space, and move to CPU
        all_similarities.append(sims.cpu().to(torch.bfloat16))
        
    accuracy = correct / total
    
    # Concatenate all batches -> Shape: [50000, 3498]
    final_sims_tensor = torch.cat(all_similarities, dim=0)
    
    # Save the tensor
    os.makedirs(save_dir, exist_ok=True)
    tensor_save_path = os.path.join(save_dir, f"sims_L{layer}_H{head_index}.pt")
    torch.save(final_sims_tensor, tensor_save_path)
    
    return accuracy

# ==========================================
# 4. The Loop (Last 4 Layers)
# ==========================================

def run_ablation_study(model):
    save_dir = "/home/nfm/ViT-Prisma/mynotebooks"
    os.makedirs(save_dir, exist_ok=True)
    
    results_dict = {}
    
    # ViT-B/16 typically has 12 layers (0 to 11). Last 4 layers are 8, 9, 10, 11
    layers_to_ablate = [8, 9, 10, 11]
    n_heads = model.cfg.n_heads # Assuming 12 for ViT-B
    
    for layer in layers_to_ablate:
        for head_index in range(n_heads):
            print(f"--- Ablating Layer {layer}, Head {head_index} ---")
            
            acc = evaluate_ablated_head(
                model=model,
                layer=layer,
                head_index=head_index,
                dataloader=val_loader,
                imagenet_embeds=aligned_text_embeds, # Use the aligned text embeddings for ImageNet zero-shot
                target_embeds=target_text_embeds,
                save_dir="/home/nfm/ViT-Prisma/mynotebooks/sims_tensors" # Save tensors in a subdirectory
            )
            
            # Log results
            key = f"L{layer}_H{head_index}"
            results_dict[key] = acc
            print(f"Accuracy for {key}: {acc:.4f}\n")
            
            # Save JSON incrementally so you don't lose data if the script crashes
            json_path = os.path.join(save_dir, "imagenet_ablation_accuracies.json")
            with open(json_path, "w") as f:
                json.dump(results_dict, f, indent=4)
                
    print("Ablation study complete! All tensors and accuracies saved.")


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

run_ablation_study(model)