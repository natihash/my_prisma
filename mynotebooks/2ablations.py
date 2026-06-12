import sys
sys.path.insert(0, "/home/nfm/ViT-Prisma/src")

import os
import json
import torch
from tqdm.auto import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import vit_prisma
from vit_prisma.utils.data_utils.imagenet.imagenet_dict import IMAGENET_DICT
from vit_prisma.utils import prisma_utils
from vit_prisma.transforms import get_clip_val_transforms
from vit_prisma.models.base_vit import HookedViT

from PIL import Image

# ==========================================
# 1. Setup & Data Loading
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256 # Adjust based on your GPU VRAM

LOCAL_JSON_PATH = "/home/nfm/ViT-Prisma/demos/imagenet_class_index.json"
with open(LOCAL_JSON_PATH, 'r') as f:
    imagenet_class_index = json.load(f)

wnid_to_name = {}
for idx, (wnid, class_name) in imagenet_class_index.items():
    safe_class_name = class_name.replace(" ", "_").replace("/", "_").replace(",", "")
    wnid_to_name[wnid] = safe_class_name

name_to_idx = {name: int(idx) for idx, (wnid, name) in imagenet_class_index.items()}

# Load your 3498 normalized text embeddings
target_text_embeds = torch.load("/home/nfm/ViT-Prisma/demos/text_dict.pt").to(device)
if target_text_embeds.dtype != torch.float32:
    target_text_embeds = target_text_embeds.float() 

# Setup ImageNet Dataloader
val_dir = "/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/" 
val_dataset = datasets.ImageFolder(val_dir, transform=get_clip_val_transforms())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Setup ImageNet Class Text Embeddings
imagenet_text_embeds = torch.load("/home/nfm/ViT-Prisma/demos/text_dict_imagenet.pt")
if imagenet_text_embeds.dtype != torch.float32:
    imagenet_text_embeds = imagenet_text_embeds.float()
imagenet_text_embeds = imagenet_text_embeds.to(device)

dataloader_wnids = val_dataset.classes 
aligned_text_embeds = torch.zeros_like(imagenet_text_embeds)

for dataloader_idx, wnid in enumerate(dataloader_wnids):
    original_idx = name_to_idx[wnid_to_name[wnid]] 
    aligned_text_embeds[dataloader_idx] = imagenet_text_embeds[original_idx]

# ==========================================
# 2. Baseline & Gray Image Processing
# ==========================================

@torch.no_grad()
def compute_baseline(model, dataloader, imagenet_embeds):
    """Computes the normal zero-shot accuracy with no ablations."""
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Calculating Baseline"):
        images, labels = images.to(device), labels.to(device)
        
        image_embeds = model.run_with_hooks(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        logits = 100.0 * image_embeds @ imagenet_embeds.T 
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    acc = correct / total
    print(f"\n---> Baseline Accuracy: {acc:.4f} <---\n")
    return acc

@torch.no_grad()
def compute_baseline_save(model, dataloader, imagenet_embeds, target_embeds, save_dir):
    """
    Computes the normal zero-shot accuracy with no ablations
    and saves the baseline cosine similarities for target texts.
    """
    correct = 0
    total = 0
    all_similarities = []
    
    for images, labels in tqdm(dataloader, desc="Calculating Baseline"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass without any hooks
        image_embeds = model.run_with_hooks(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # --- 1. ImageNet Zero-Shot Accuracy ---
        logits = 100.0 * image_embeds @ imagenet_embeds.T 
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # --- 2. Target Texts Similarities ---
        # sims shape: [batch_size, 3498]
        sims = image_embeds @ target_embeds.T
        all_similarities.append(sims.cpu().to(torch.bfloat16))
        
    acc = correct / total
    print(f"\n---> Baseline Accuracy: {acc:.4f} <---\n")
    
    # --- 3. Save Final Tensor ---
    # Concatenate all batches -> Shape: [50000, 3498]
    final_sims_tensor = torch.cat(all_similarities, dim=0)
    
    os.makedirs(save_dir, exist_ok=True)
    tensor_save_path = os.path.join(save_dir, "sims_baseline.pt")
    torch.save(final_sims_tensor, tensor_save_path)
    
    print(f"Baseline similarities saved to {tensor_save_path}")
    
    return acc

@torch.no_grad()
def get_gray_image_cache(model, layers_to_cache):
    """Passes a neutral gray image through the model and caches 'z' activations."""
    print("Generating neutral gray image cache...")
    
    # Create a solid gray image (128 RGB is 50% gray)
    gray_pil = Image.new("RGB", (224, 224), color=(128, 128, 128))
    # Apply standard CLIP validation transforms
    gray_tensor = get_clip_val_transforms()(gray_pil).unsqueeze(0).to(device)
    
    gray_cache = {}
    
    def cache_hook(z, hook, layer):
        # Save the activation to the cache. Shape: [1, seq_len, n_heads, d_head]
        gray_cache[layer] = z.detach().clone()
        return z

    hooks = []
    for layer in layers_to_cache:
        hook_name = prisma_utils.get_act_name('z', layer, "attn")
        hooks.append((hook_name, partial(cache_hook, layer=layer)))
        
    # Run the model with the caching hooks
    model.run_with_hooks(gray_tensor, fwd_hooks=hooks)
    print("Gray cache generated.\n")
    return gray_cache

# ==========================================
# 3. The Hook Function
# ==========================================

def ablate_head_hook(z, hook, head_index, layer=None, gray_cache=None):
    """
    If gray_cache is provided, patches in the gray image activation for the specific head.
    Otherwise, defaults to zero ablation.
    """
    if gray_cache is not None and layer is not None:
        # Extract the cached gray activation for this head
        # Shape of gray_cache[layer] is [1, seq_len, n_heads, d_head]
        # We slice it to [seq_len, d_head] and PyTorch broadcasts it across the batch dimension
        gray_head_act = gray_cache[layer][0, :, head_index, :] 
        z[:, :, head_index, :] = gray_head_act
    else:
        # Fallback to zero ablation
        z[:, :, head_index, :] = 0.0
        
    return z

# ==========================================
# 4. Main Evaluation Function
# ==========================================

@torch.no_grad()
def evaluate_ablated_head(
    model, 
    layer, 
    head_index, 
    dataloader, 
    imagenet_embeds, 
    target_embeds, 
    save_dir,
    gray_cache=None
):
    hook_name = prisma_utils.get_act_name('z', layer, "attn")
    hook_fn = partial(ablate_head_hook, head_index=head_index, layer=layer, gray_cache=gray_cache)
    
    all_similarities = []
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc=f"Evaluating L{layer} H{head_index}", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        image_embeds = model.run_with_hooks(
            images,
            fwd_hooks=[(hook_name, hook_fn)]
        )
        
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # ImageNet Zero-Shot Accuracy
        logits = 100.0 * image_embeds @ imagenet_embeds.T 
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Target Texts Similarities
        sims = image_embeds @ target_embeds.T
        all_similarities.append(sims.cpu().to(torch.bfloat16))
        
    accuracy = correct / total
    final_sims_tensor = torch.cat(all_similarities, dim=0)
    
    os.makedirs(save_dir, exist_ok=True)
    tensor_save_path = os.path.join(save_dir, f"sims_L{layer}_H{head_index}.pt")
    torch.save(final_sims_tensor, tensor_save_path)
    
    return accuracy

# ==========================================
# 5. The Loop (Last 4 Layers)
# ==========================================

def run_ablation_study(model, use_gray=True):
    save_dir = "/home/nfm/ViT-Prisma/mynotebooks"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Run Baseline
    baseline_acc = compute_baseline(model, val_loader, aligned_text_embeds)
    
    layers_to_ablate = [8, 9, 10, 11]
    n_heads = model.cfg.n_heads 
    
    # 2. Get Gray Cache (if requested)
    gray_cache = None
    if use_gray:
        gray_cache = get_gray_image_cache(model, layers_to_ablate)
    
    # 3. Start Ablation
    results_dict = {"baseline": baseline_acc}
    ablation_type = "Gray Image" if use_gray else "Zero"
    print(f"Starting {ablation_type} Ablation Study...\n")
    
    for layer in layers_to_ablate:
        for head_index in range(n_heads):
            print(f"--- Ablating Layer {layer}, Head {head_index} ---")
            
            acc = evaluate_ablated_head(
                model=model,
                layer=layer,
                head_index=head_index,
                dataloader=val_loader,
                imagenet_embeds=aligned_text_embeds,
                target_embeds=target_text_embeds,
                save_dir="/home/nfm/ViT-Prisma/mynotebooks/sims_tensors",
                gray_cache=gray_cache # Pass the cache dynamically
            )
            
            key = f"L{layer}_H{head_index}"
            results_dict[key] = acc
            print(f"Accuracy for {key}: {acc:.4f}\n")
            
            json_path = os.path.join(save_dir, "imagenet_ablation_accuracies.json")
            with open(json_path, "w") as f:
                json.dump(results_dict, f, indent=4)
                
    print(f"{ablation_type} ablation study complete! All tensors and accuracies saved.")

# ==========================================
# 6. Model Init & Execution
# ==========================================

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

# Run the study (Set use_gray=False to revert to zero ablation)
# run_ablation_study(model, use_gray=True)

# Assuming you have already loaded target_text_embeds and defined your save path
baseline_acc = compute_baseline_save(
    model=model, 
    dataloader=val_loader, 
    imagenet_embeds=aligned_text_embeds,
    target_embeds=target_text_embeds,
    save_dir="/home/nfm/ViT-Prisma/mynotebooks/sims_tensors"
)