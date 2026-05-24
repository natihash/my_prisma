import sys
sys.path.insert(0, "/home/nfm/ViT-Prisma/src")

import argparse
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


# Helper function (ignore)
def plot_image(image):
  plt.figure()
  plt.axis('off')
  plt.imshow(image.permute(1,2,0))

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

def plot_logit_boxplot(average_logits, labels):
  hovertexts = np.array([[IMAGENET_DICT[i] for _ in range(25)] for i in range(1000)])

  fig = go.Figure()
  data = []

  # if tensor, turn to numpy
  if isinstance(average_logits, torch.Tensor):
      average_logits = average_logits.detach().cpu().numpy()

  for i in range(average_logits.shape[1]):  # For each layer
      layer_logits = average_logits[:, i]
      hovertext = hovertexts[:, i]
      box = fig.add_trace(go.Box(
          y=layer_logits,
          name=f'{layer_labels[i]}',
          text=hovertext,
          hoverinfo='y+text',
          boxpoints='suspectedoutliers'
      ))
      data.append(box)


  means = np.mean(average_logits, axis=0)
  fig.add_trace(go.Scatter(
      x = layer_labels,
      y=means,
      mode='markers',
      name='Mean',
      # line=dict(color='gray'),
      marker=dict(size=4, color='red'),
  ))


  fig.update_layout(
      title='Raw Logit Values Per Layer (each dot is 1 ImageNet Class)',
      xaxis=dict(title='Layer'),
      yaxis=dict(title='Logit Values'),
      showlegend=False
  )

  fig.show()


def plot_patched_component(patched_head, title=''):
  """
  Use for plotting Activation Patching.
  """

  fig = go.Figure(data=go.Heatmap(
      z=patched_head.detach().numpy(),
      colorscale='RdBu',  # You can choose any colorscale
      colorbar=dict(title='Value'),  # Customize the color bar
      hoverongaps=False
  ))
  fig.update_layout(
      title=title,
      xaxis_title='Attention Head',
      yaxis_title='Patch Number',
  )

  return fig

def imshow(tensor, **kwargs):
    """
    Use for Activation Patching.
    """
    px.imshow(
          prisma_utils.to_numpy(tensor),
          color_continuous_midpoint=0.0,
          color_continuous_scale="RdBu",
          **kwargs,
      ).show()

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

import torch
import torch.nn.functional as F
from torch.autograd.functional import jvp
from tqdm.auto import tqdm


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def make_gray_corrupted_image(reference_img, device,
                               clip_mean=(0.48145466, 0.4578275, 0.40821073),
                               clip_std =(0.26862954, 0.26130258, 0.27577711)):
    """Uniform gray (pixel=0.5) expressed in CLIP-normalized space."""
    mean = torch.tensor(clip_mean, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(clip_std,  device=device).view(1, 3, 1, 1)
    return (0.5 * torch.ones_like(reference_img) - mean) / std


def _head_idx_to_layer_head(head_idx, n_heads, n_layers, layers_to_keep):
    layer_offset  = n_layers - layers_to_keep
    layer_idx     = layer_offset + (head_idx // n_heads)
    head_in_layer = head_idx % n_heads
    return layer_idx, head_in_layer


def _get_z_activation(model, img, hook_name, head_in_layer):
    """
    Single forward pass → CLS-token hook_z for one head.
    hook_z shape: [batch, seq_len, n_heads, d_head]
    Returns: [d_head]
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(img)
        # position 0 = CLS token
        z = cache[hook_name][0, 0, head_in_layer, :].clone()   # [d_head]
    del cache
    torch.cuda.empty_cache()
    return z


# ──────────────────────────────────────────────────────────────
#  Core: single image × all texts × one head
# ──────────────────────────────────────────────────────────────

def attribution_scores_for_head(
    model,
    clean_img,           # [1, C, H, W] on device, CLIP-normalised
    text_embeddings,     # [N_texts, d_model] on device
    head_idx,            # flat 0..(LAYERS_TO_KEEP*n_heads - 1)
    corrupted_img=None,
    z_corr_cache=None,   # precomputed z_corr to skip the corrupted forward pass
    LAYERS_TO_KEEP=4,
    eps=1e-8,
):
    """
    Attribution patching for one head, all texts, one image.

    Hook used: blocks.{layer}.attn.hook_z  (shape [batch, seq, n_heads, d_head])
    Delta lives in d_head space (64-dim for ViT-B).
    The W_O projection and everything downstream is handled by the JVP.

    Returns
    -------
    attribution_scores : [N_texts]
    cosine_sims        : [N_texts]
    """
    device  = model.cfg.device
    n_heads = model.cfg.n_heads      # 12
    n_layers= model.cfg.n_layers     # 12
    d_head  = model.cfg.d_head       # 64 for ViT-B

    layer_idx, head_in_layer = _head_idx_to_layer_head(
        head_idx, n_heads, n_layers, LAYERS_TO_KEEP
    )
    # This key IS present in the cache (confirmed from your cache.keys() output)
    hook_name = f'blocks.{layer_idx}.attn.hook_z'

    # ── Corrupted z (can be precomputed once for the whole dataset) ──
    if z_corr_cache is not None:
        z_corr = z_corr_cache
    else:
        if corrupted_img is None:
            corrupted_img = make_gray_corrupted_image(clean_img, device)
        z_corr = _get_z_activation(model, corrupted_img, hook_name, head_in_layer)

    # ── Clean z ──
    z_clean = _get_z_activation(model, clean_img, hook_name, head_in_layer)

    # Δz in d_head space — small (64-dim) and efficient
    delta_z = z_corr - z_clean   # [d_head]

    # ── JVP: how does final_output shift when this head's z is nudged by delta_z? ──
    #
    # model_with_z_perturbation(p) adds p to hook_z[0, 0, head_in_layer, :] and
    # lets z @ W_O (and everything after) propagate normally.
    # jvp at p=0 with tangent delta_z gives (output_clean, J·delta_z).

    def model_with_z_perturbation(perturbation):
        def patch_hook(value, hook):
            # value: [batch, seq_len, n_heads, d_head]
            patched = value.clone()
            patched[0, 0, head_in_layer, :] = (
                patched[0, 0, head_in_layer, :] + perturbation
            )
            return patched

        out = model.run_with_hooks(
            clean_img,
            fwd_hooks=[(hook_name, patch_hook)],
        )
        return out[0]   # [d_model]  (CLS token final output)

    zero_z = torch.zeros(d_head, device=device)

    # final_output : [d_model]  clean CLS embedding
    # Jv           : [d_model]  how output shifts due to delta_z through W_O + rest
    final_output, Jv = jvp(
        model_with_z_perturbation,
        (zero_z,),
        (delta_z,),
    )

    # ── Attribution for all texts — one matmul, no backward loops ──
    #
    # attr_t = c_t · Jv
    # c_t    = ∂cosine_sim(final_output, text_t) / ∂final_output
    #        = (text_norm_t - a_norm * sim_t) / ‖a‖

    a      = final_output.detach()
    a_mag  = a.norm().clamp(min=eps)
    a_norm = a / a_mag

    B_norm   = F.normalize(text_embeddings.float(), dim=-1)         # [N, d_model]
    cos_sims = (a_norm * B_norm).sum(dim=1)                         # [N]

    c_t = (B_norm - cos_sims.unsqueeze(1) * a_norm.unsqueeze(0)) / a_mag  # [N, d_model]

    attribution_scores = (c_t * Jv.detach().unsqueeze(0)).sum(dim=1)  # [N]

    return attribution_scores, cos_sims.detach()


# ──────────────────────────────────────────────────────────────
#  Outer loop: full dataset × all texts × one head
# ──────────────────────────────────────────────────────────────

def compute_all_attributions(
    model,
    dataloader,
    text_embeddings,     # [N_texts, d_model]
    head_idx,
    LAYERS_TO_KEEP=4,
    save_path=None,
):
    """
    Runs attribution patching for every image in the dataloader.

    The corrupted (gray) forward pass is run only ONCE — z_corr is reused
    for every image since the corrupted input is always the same.

    Returns
    -------
    all_attrs : [N_images, N_texts]
    all_sims  : [N_images, N_texts]
    """
    device  = model.cfg.device
    n_heads = model.cfg.n_heads
    n_layers= model.cfg.n_layers

    layer_idx, head_in_layer = _head_idx_to_layer_head(
        head_idx, n_heads, n_layers, LAYERS_TO_KEEP
    )
    hook_name = f'blocks.{layer_idx}.attn.hook_z'

    # ── Precompute z_corr once ──
    sample_img    = next(iter(dataloader))[0][:1].to(device)
    corrupted_img = make_gray_corrupted_image(sample_img, device)
    z_corr        = _get_z_activation(model, corrupted_img, hook_name, head_in_layer)

    print(f"Head {head_idx:2d}  →  layer {layer_idx}, head {head_in_layer}  |  hook: {hook_name}")
    print(f"z_corr precomputed: {z_corr.shape}  (d_head={z_corr.shape[0]})")

    all_attrs, all_sims = [], []

    for batch_imgs, _ in tqdm(dataloader, desc=f"Head {head_idx}"):
        batch_attrs, batch_sims = [], []

        for i in range(batch_imgs.shape[0]):
            img = batch_imgs[i : i + 1].to(device)
            attr, sims = attribution_scores_for_head(
                model,
                clean_img      = img,
                text_embeddings= text_embeddings,
                head_idx       = head_idx,
                z_corr_cache   = z_corr,      # no extra forward pass per image
                LAYERS_TO_KEEP = LAYERS_TO_KEEP,
            )
            batch_attrs.append(attr)
            batch_sims.append(sims)

        all_attrs.append(torch.stack(batch_attrs))
        all_sims.append(torch.stack(batch_sims))

    all_attrs = torch.cat(all_attrs, dim=0)   # [N_images, N_texts]
    all_sims  = torch.cat(all_sims,  dim=0)

    if save_path:
        torch.save({
            'attributions': all_attrs.cpu(),
            'cosine_sims' : all_sims.cpu(),
            'head_idx'    : head_idx,
        }, save_path)
        print(f"Saved → {save_path}")

    return all_attrs, all_sims


def attribution_scores_batch(
    model,
    batch_imgs,          # [B, C, H, W] on device
    text_embeddings,     # [N_texts, d_model]
    head_idx,
    z_corr,              # [d_head] precomputed once
    LAYERS_TO_KEEP=4,
    eps=1e-8,
):
    """
    Processes a whole batch of images at once.
    Each image still needs its own JVP (different z_clean → different delta),
    but we vectorize the cosine-sim gradient step over the batch.

    Returns
    -------
    attributions : [B, N_texts]
    cos_sims     : [B, N_texts]
    """
    device   = model.cfg.device
    n_heads  = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    d_head   = model.cfg.d_head
    B        = batch_imgs.shape[0]

    layer_idx, head_in_layer = _head_idx_to_layer_head(
        head_idx, n_heads, n_layers, LAYERS_TO_KEEP
    )
    hook_name = f'blocks.{layer_idx}.attn.hook_z'

    # ── 1. Get ALL clean z activations in one batched forward pass ──
    with torch.no_grad():
        _, cache_clean = model.run_with_cache(batch_imgs)
        # [B, seq, n_heads, d_head] → [B, d_head]
        z_clean_batch = cache_clean[hook_name][:, 0, head_in_layer, :].clone()
    del cache_clean
    torch.cuda.empty_cache()

    # delta per image: [B, d_head]
    delta_batch = z_corr.unsqueeze(0) - z_clean_batch

    # ── 2. JVP per image (unavoidable: each image has a different delta) ──
    #    But we skip the clean cache re-run inside each JVP by reusing batch_imgs[i]
    all_Jv            = []
    all_final_outputs = []

    for i in range(B):
        img_i   = batch_imgs[i : i + 1]   # [1, C, H, W]
        delta_i = delta_batch[i]           # [d_head]

        def model_with_z_perturbation(perturbation, _img=img_i, _head=head_in_layer):
            def patch_hook(value, hook):
                patched = value.clone()
                patched[0, 0, _head, :] = patched[0, 0, _head, :] + perturbation
                return patched
            out = model.run_with_hooks(_img, fwd_hooks=[(hook_name, patch_hook)])
            return out[0]   # [d_model]

        zero_z = torch.zeros(d_head, device=device)
        final_out_i, Jv_i = jvp(model_with_z_perturbation, (zero_z,), (delta_i,))

        all_final_outputs.append(final_out_i.detach())
        all_Jv.append(Jv_i.detach())

        # cleanup per image
        del img_i, delta_i, final_out_i, Jv_i, zero_z
        torch.cuda.empty_cache()

    final_outputs = torch.stack(all_final_outputs)   # [B, d_model]
    Jv_stack      = torch.stack(all_Jv)              # [B, d_model]

    # ── 3. Attribution for all images × all texts — fully batched ──
    a_mag  = final_outputs.norm(dim=-1, keepdim=True).clamp(min=eps)  # [B, 1]
    a_norm = final_outputs / a_mag                                     # [B, d_model]

    B_norm   = F.normalize(text_embeddings.float(), dim=-1)            # [N, d_model]

    # cos_sims[b, t] = dot(a_norm[b], B_norm[t])
    cos_sims = a_norm @ B_norm.T                                       # [B, N]

    # c_t[b, t, :] = (B_norm[t] - a_norm[b] * cos_sims[b,t]) / ||a[b]||
    # Rearranged for matmul:
    # c_t contribution 1: B_norm[t] / ||a[b]||  →  [B, N, d_model]
    B_norm_exp  = B_norm.unsqueeze(0) / a_mag.unsqueeze(1)             # [B, N, d_model]
    # c_t contribution 2: -a_norm[b] * cos_sims[b,t] / ||a[b]||
    correction  = (cos_sims / a_mag) .unsqueeze(-1) * a_norm.unsqueeze(1)  # [B, N, d_model]

    c_t = B_norm_exp - correction                                      # [B, N, d_model]

    # attr[b, t] = sum_d  c_t[b, t, d] * Jv[b, d]
    attributions = (c_t * Jv_stack.unsqueeze(1)).sum(dim=-1)           # [B, N]

    return attributions, cos_sims.detach()


def compute_all_attributions(
    model,
    dataloader,
    text_embeddings,
    head_idx,
    LAYERS_TO_KEEP=4,
    save_path=None,
):
    device  = model.cfg.device
    n_heads = model.cfg.n_heads
    n_layers= model.cfg.n_layers

    layer_idx, head_in_layer = _head_idx_to_layer_head(
        head_idx, n_heads, n_layers, LAYERS_TO_KEEP
    )
    hook_name = f'blocks.{layer_idx}.attn.hook_z'

    # Precompute z_corr once
    sample_img    = next(iter(dataloader))[0][:1].to(device)
    corrupted_img = make_gray_corrupted_image(sample_img, device)
    z_corr        = _get_z_activation(model, corrupted_img, hook_name, head_in_layer)
    print(f"Head {head_idx:2d} → layer {layer_idx}, head {head_in_layer} | {hook_name}")

    all_attrs, all_sims = [], []

    for batch_imgs, _ in tqdm(dataloader, desc=f"Head {head_idx}"):
        batch_imgs = batch_imgs.to(device)
        attrs, sims = attribution_scores_batch(
            model, batch_imgs, text_embeddings, head_idx,
            z_corr=z_corr, LAYERS_TO_KEEP=LAYERS_TO_KEEP,
        )
        all_attrs.append(attrs.cpu())
        all_sims.append(sims.cpu())

        #cleanup
        del batch_imgs, attrs, sims
        torch.cuda.empty_cache()

    all_attrs = torch.cat(all_attrs, dim=0)
    all_sims  = torch.cat(all_sims,  dim=0)

    if save_path:
        torch.save({'attributions': all_attrs, 'cosine_sims': all_sims,
                    'head_idx': head_idx}, save_path)
        print(f"Saved → {save_path}")

    return all_attrs, all_sims


import os
import json
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from vit_prisma.transforms import get_clip_val_transforms
# --- 1. CONFIGURATION ---
IMAGENET_VAL_DIR = '/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/'

mytransform = get_clip_val_transforms()
# Single image, single head, all texts
dataset = ImageFolder(IMAGENET_VAL_DIR, transform=mytransform) 
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)


import os
import torch

# --- Load text embeddings and move to device ---
all_text_embs = torch.load("/home/nfm/ViT-Prisma/demos/text_dict.pt").to(model.cfg.device)

txt_pth = "/home/nfm/clip_text_span/text_descriptions/image_descriptions_general.txt"
with open(txt_pth, "r") as f:
    texts = [line.strip() for line in f.readlines()]

# all_text_embs: [N_texts, d_model]

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Attribution patching for ViT-B heads")
parser.add_argument('--head', type=int, default=45, dest='target_head_idx',
                    help='Target head index (default: 45)')
args = parser.parse_args()
target_head_idx = args.target_head_idx

# --- Quick single-image sanity check ---
sample_img, _ = dataset[0]

attr_scores, cos_sims = attribution_scores_for_head(
    model,
    clean_img       = sample_img.unsqueeze(0).to(model.cfg.device),
    text_embeddings = all_text_embs,
    head_idx        = target_head_idx,
    LAYERS_TO_KEEP  = 4,
)

print("Top promoted texts (attribution):")
topk = attr_scores.topk(10)
for score, idx in zip(topk.values, topk.indices):
    print(f"  [{score:+.4f}] {texts[idx]}")

print("\nTop texts by raw cosine similarity:")
topk_sim = cos_sims.topk(10)
for score, idx in zip(topk_sim.values, topk_sim.indices):
    print(f"  [{score:+.4f}] {texts[idx]}")

# --- Full dataset ---
os.makedirs("attribution_patch", exist_ok=True)

all_attrs, all_sims = compute_all_attributions(
    model,
    dataloader      = dataloader,
    text_embeddings = all_text_embs,
    head_idx        = target_head_idx,
    LAYERS_TO_KEEP  = 4,
    save_path       = f"attribution_patch/attributions_head{target_head_idx}.pt",
)
# all_attrs: [N_images, N_texts]
# all_sims:  [N_images, N_texts]

# --- Summarise: which texts does this head consistently promote? ---
# Mean attribution across all images
mean_attr = all_attrs.mean(dim=0)   # [N_texts]
topk_global = mean_attr.topk(20)

print(f"\nTexts most consistently promoted by head {target_head_idx} across the dataset:")
for score, idx in zip(topk_global.values, topk_global.indices):
    print(f"  [{score:+.5f}] {texts[idx]}")