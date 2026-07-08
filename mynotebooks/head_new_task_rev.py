import sys
sys.path.insert(0, "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/src")

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import json
import torch

torch.set_grad_enabled(False)  # inference only — no autograd graph anywhere below

from vit_prisma.models.base_vit import HookedViT

# ============================================================================
# CONFIG
# ============================================================================
DEVICE = "cuda:0"

# HookedViT (the model whose head we attribute through).
MODEL_NAME = "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

# open_clip text encoder used to (re)compute text embeddings when no saved
# text_dict exists. Must match MODEL_NAME's CLIP variant.
OPEN_CLIP_ARCH = "ViT-B-16"
OPEN_CLIP_PRETRAINED = "laion2b_s34b_b88k"

# Prefix prepended to every class string before encoding (used only when the
# text_dict is computed here rather than loaded from disk).
TEXT_PROMPT_PREFIX = "an image of "

# Force recomputation of the text_dict even if a saved file already exists.
RECOMPUTE_TEXT_DICT = False

# Flat-index range of attention heads stored in the activation tensors.
# Each .pt file has shape (n_heads_saved, n_images, d_model).
HEAD_IDX = list(range(48))

# METRICS_OUT_DIR = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/text"

MODEL_NAMES = [
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lp",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_fft",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lora4",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lora16",
]


DATASETS = {
    "imagenet": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/original_acts",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/demos/imagenet_class_index.json",
        "classes_txt": None,
        # filename -> class-index rule
        "filename_to_class": "imagenet_wnid",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": "/home/nfm/ViT-Prisma/demos/text_dict_imagenet.pt",
        "metrics_name": "new_task_rel_imagenet1k.json",
    },
    "sun397": {
        # filenames look like "abbey.pt", "airplane_cabin.pt"
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397/clip_orig",
        "class_index_json": None,
        # one class string per line; line number == class/text index
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/sun_dataset_folder_sampled/classes.txt",
        "filename_to_class": "stem",
        # computed + saved here on first run (no precomputed file exists)
        "text_dict_path": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397/text_dict_sun397.pt",
        "metrics_name": "new_task_rel_sun397.json",
    },

    "imagenet_ntr": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/ViT-Prisma/demos/head_acts_clipvit16_FullFT",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/demos/imagenet_class_index.json",
        "classes_txt": None,
        # filename -> class-index rule
        "filename_to_class": "imagenet_wnid",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_lp.json",
    },

    "sunfft": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397/fft",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/sun_dataset_folder_sampled/classes.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_fft.json",
        "model_name": "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft_sun",
    },

    "sunlora4": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397/lora4",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/sun_dataset_folder_sampled/classes.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_lora4.json",
        "model_name": "hf_hub:natihash/vit_base_patch16_clip_224.sun_lora_r4_merged",
    },

    "sunlora16": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397/lora16",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/sun_dataset_folder_sampled/classes.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_lora16.json",
        "model_name": "hf_hub:natihash/vit_base_patch16_clip_224.sun_lora_r16_merged",
    },

    "sunlp": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397/clip_orig",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/sun_dataset_folder_sampled/classes.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_lp.json",
        "model_name": "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_linear_probe_sun",
    },
    "textlp": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/text/clip_orig",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/text_overlay_dataset/classnames.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_lp.json",
        "model_name": MODEL_NAMES[0],
    },
    "textfft": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/text/fft",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/text_overlay_dataset/classnames.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_fft.json",
        "model_name": MODEL_NAMES[1],
    },
    "textlora4": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/text/lora4",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/text_overlay_dataset/classnames.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_lora4.json",
        "model_name": MODEL_NAMES[2],
    },
    "textlora16": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/text/lora16",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/text_overlay_dataset/classnames.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_lora16.json",
        "model_name": MODEL_NAMES[3],
    },

    "textorg": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/text/clip_orig",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/text_overlay_dataset/classnames.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/text/text_dict_text.pt",
        "metrics_name": "new_task_rel_org_clip.json",
        "model_name": "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        "is_clip": True,
        "only_acts": True,
    },

    "sunorg": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/sun397/clip_orig",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": None,
        "classes_txt": "/home/nfm/Desktop/rhome/nfm/sun_dataset_folder_sampled/classes.txt",
        # filename -> class-index rule
        "filename_to_class": "stem",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_org_clip.json",
        "model_name": "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        "is_clip": False,
        "only_acts": True,
        "metrics_out_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/sun",
    },

    "imgorg": {
        # per-class head activations: filenames look like "n01440764_tench.pt"
        # "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/imagenet_1k/lora16",
        "activations_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts/original_acts",
        # class ordering / index source (defines the text index for each class)
        "class_index_json": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/demos/imagenet_class_index.json",
        "classes_txt": None,
        # filename -> class-index rule
        "filename_to_class": "imagenet_wnid",
        # precomputed, L2-normalized CLIP text embeddings (n_classes, d_text)
        "text_dict_path": None,
        "metrics_name": "new_task_rel_org_clip.json",
        "model_name": "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        "is_clip": False,
        "only_acts": True,
        "metrics_out_dir": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k"
    },
}

# --- pick the datasets to run here (processed sequentially in a loop) -------
# CONFIG_KEYS = ["sunfft", "sunlora4", "sunlora16", "sunlp"]
# CONFIG_KEYS = ["textlp", "textfft", "textlora4", "textlora16", "textorg"]
CONFIG_KEYS = ["sunorg", "imgorg"]
# ============================================================================
# MODEL
# ============================================================================
def load_model(cfg):
    # MODEL_NAME = "vit_base_patch16_clip_224.laion2b_ft_in1k"
    # MODEL_NAME = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_linear_probe_real"
    # MODEL_NAME = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_lora_r16_merged"
    # MODEL_NAME = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_lora_r4_merged"
    model_name = cfg.get("model_name", MODEL_NAME)
    print(f"Loading model: {model_name}")

    model = HookedViT.from_pretrained(
        model_name,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
        device=DEVICE,
    )
    model = model.to(DEVICE)
    model.cfg.device = DEVICE
    print("Model device config:", model.cfg.device)
    print("Is CUDA available?:", torch.cuda.is_available())
    return model


# ============================================================================
# CLASS MAPPING (generic over datasets)
# ============================================================================
def build_class_mapping(cfg):
    """Return (class_names, file_to_idx).

    class_names[i] is the class at text/class index i (matches text_dict row i).
    file_to_idx(filename) maps an activation .pt filename to its class index.
    """
    if cfg["classes_txt"]:
        with open(cfg["classes_txt"]) as f:
            class_names = [line.strip() for line in f if line.strip()]
        name_to_idx = {name: i for i, name in enumerate(class_names)}
    else:
        with open(cfg["class_index_json"]) as f:
            class_index = json.load(f)
        class_names = [None] * len(class_index)
        wnid_to_idx = {}
        for idx, (wnid, cname) in class_index.items():
            class_names[int(idx)] = cname
            wnid_to_idx[wnid] = int(idx)

    kind = cfg["filename_to_class"]
    if kind == "imagenet_wnid":
        # "n01440764_tench.pt" -> wnid "n01440764" -> official index
        def file_to_idx(filename):
            return wnid_to_idx[filename.split("_")[0]]
    elif kind == "stem":
        # "airplane_cabin.pt" -> "airplane_cabin" -> line in classes.txt
        def file_to_idx(filename):
            return name_to_idx[os.path.splitext(filename)[0]]
    else:
        raise ValueError(f"Unknown filename_to_class rule: {kind}")

    return class_names, file_to_idx


# ============================================================================
# TEXT DICT (load if present, otherwise compute with open_clip)
# ============================================================================
def compute_text_dict(class_names, device):
    """Encode class strings with the open_clip text encoder -> (n_classes, d_text),
    L2-normalized rows (same convention as the saved imagenet text_dict)."""
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms(
        OPEN_CLIP_ARCH, pretrained=OPEN_CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(OPEN_CLIP_ARCH)
    clip_model = clip_model.to(device).eval()

    # Underscores in class names -> spaces for a natural-language prompt.
    # prompts = [TEXT_PROMPT_PREFIX + name.replace("_", " ") for name in class_names]
    # prompts = class_names  # no prefix for text overlay dataset

    temp_pth = "/home/nfm/Desktop/rhome/nfm/text_overlay_dataset/classnames_texts.txt"
    with open(temp_pth) as f:
        prompts = [line.strip() for line in f if line.strip()]

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(prompts), 256):
            tokens = tokenizer(prompts[start:start + 256]).to(device)
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.float().cpu())
    text_dict = torch.cat(embeddings, dim=0)
    print(f"Computed text_dict {tuple(text_dict.shape)} for {len(prompts)} classes")
    return text_dict


def load_or_compute_text_dict(cfg, class_names, device, recompute=False):
    path = cfg["text_dict_path"]
    if path and os.path.exists(path) and not recompute:
        text_dict = torch.load(path, map_location=device).float()
        print(f"Loaded text_dict {tuple(text_dict.shape)} from {path}")
    else:
        text_dict = compute_text_dict(class_names, device)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(text_dict.cpu(), path)
            print(f"Saved computed text_dict to {path}")
    assert text_dict.shape[0] == len(class_names), (
        f"text_dict rows {text_dict.shape[0]} != n_classes {len(class_names)}"
    )
    return text_dict.float().to(device)


# ============================================================================
# DISCRIMINATIVE SCORES (streaming; handles variable images-per-class)
# ============================================================================
def residual_stack_to_logit_attn(centered_residual_stack, answer_residual_direction):
    # h=heads, b=images, d=d_model, c=texts/classes
    return torch.einsum(
        "h b d, c d -> h b c", centered_residual_stack, answer_residual_direction
    )


def compute_act_discriminative_scores(model, head_idx, tensors_dir,
                                      file_to_idx, n_classes, eps=1e-8):
    """Discriminative score from raw head activations (no text attribution).

    Used for CLIP configs where it is hard to write a good text prompt for each
    class (e.g. "image with the word {word} in it", whose text embedding just
    collapses toward the embedding of {word}). Instead of comparing direct text
    attribution (DTA) we compare how differently the head *represents* class-c
    images vs the rest, directly in activation space.

    For each (head h, class c) we compare two d_model-space vectors for head h:
        mean activation vector of class-c images, vs
        mean activation vector of all other images,
    and score the separation two complementary ways:
        cos_score[h, c] = cosine distance (1 - cosine similarity)
                          -> scale-invariant, direction-only separation
        mse_score[h, c] = mean squared error between the two mean vectors
                          -> also captures magnitude differences (e.g. a head
                             that fires strongly for one class and weakly for
                             another), which cosine normalizes away

    Returns (cos_scores, mse_scores), each of shape (n_heads, n_classes).
    """
    dev = model.cfg.device
    n_heads = len(head_idx)

    from concurrent.futures import ThreadPoolExecutor

    def _load_residual(path):
        return torch.load(path, weights_only=True).float()

    pths = [f for f in os.listdir(tensors_dir) if f.endswith(".pt")]
    paths = [os.path.join(tensors_dir, pth) for pth in pths]

    pos_act_sum = None                       # (n_classes, n_heads, d_model)
    total_act_sum = None                     # (n_heads, d_model)
    counts = torch.zeros(n_classes, device=dev)
    total_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        for pth, residual_cpu in zip(pths, executor.map(_load_residual, paths)):
            c = file_to_idx(pth)
            residual = residual_cpu.to(dev)
            del residual_cpu
            acts = residual[head_idx]            # (h, i, d_model)
            del residual
            if pos_act_sum is None:
                d_model = acts.shape[-1]
                pos_act_sum = torch.zeros(n_classes, n_heads, d_model, device=dev)
                total_act_sum = torch.zeros(n_heads, d_model, device=dev)
            n_i = acts.shape[1]
            counts[c] = n_i
            total_count += n_i
            class_sum = acts.sum(dim=1)          # (h, d_model)
            pos_act_sum[c] = class_sum
            total_act_sum += class_sum
            del acts

    # Per-class mean activation vector, and mean activation vector of the rest.
    mean_c = pos_act_sum / counts.clamp_min(1.0).view(n_classes, 1, 1)     # (c, h, d)
    neg_den = (total_count - counts).clamp_min(1.0).view(n_classes, 1, 1)  # (c, 1, 1)
    mean_rest = (total_act_sum.unsqueeze(0) - pos_act_sum) / neg_den       # (c, h, d)

    cos = torch.nn.functional.cosine_similarity(mean_c, mean_rest, dim=-1, eps=eps)  # (c, h)
    cos_scores = 1.0 - cos                                                # (c, h)
    mse_scores = ((mean_c - mean_rest) ** 2).mean(dim=-1)                 # (c, h)

    return cos_scores.T.contiguous(), mse_scores.T.contiguous()          # (n_heads, n_classes)


def compute_discriminative_scores(model, text_dict, head_idx, tensors_dir,
                                  file_to_idx, n_classes, eps=1e-8, is_clip=True):
    """Stream over per-class activation files and return discriminative scores.

    Returns (scores, scores_z), each of shape (n_heads, n_classes), where for
    each (head h, class c):
        score[h, c] = mean attribution of class-c images to text-c
                    - mean attribution of all other images to text-c
        score_z      = score / std of attribution to text-c over all images

    This is mathematically identical to the original dense version but never
    materializes the full (n_classes, n_heads, n_images, n_texts) tensor and
    therefore works when each class has a different number of images.
    """
    if is_clip:
        proj_head = model.head.W_H.float()
        target_direction = (text_dict.float() @ proj_head.T).to(model.cfg.device)  # (n_texts, d_model)
        n_texts = text_dict.shape[0]
        assert n_texts == n_classes, (n_texts, n_classes)
    else:
        print("Normal ViT Model, so just use direct logit attribution using the head")
        all_inds = [*range(n_classes)]
        target_direction = model.tokens_to_residual_directions(all_inds)
        n_texts = target_direction.shape[0]

    dev = model.cfg.device
    n_heads = len(head_idx)

    pos_mean = torch.zeros(n_classes, n_heads, device=dev)
    pos_sum = torch.zeros(n_classes, n_heads, device=dev)
    counts = torch.zeros(n_classes, device=dev)
    total_sum = torch.zeros(n_heads, n_texts, device=dev)    # sum over every image of every class
    total_sumsq = torch.zeros(n_heads, n_texts, device=dev)  # for per-text std
    total_count = 0

    from concurrent.futures import ThreadPoolExecutor

    def _load_residual(path):
        return torch.load(path, weights_only=True).float()

    pths = [f for f in os.listdir(tensors_dir) if f.endswith(".pt")]
    paths = [os.path.join(tensors_dir, pth) for pth in pths]

    # Load files in parallel (torch.load releases the GIL during I/O, so threads
    # genuinely overlap network reads). executor.map preserves order so zip is safe.
    with ThreadPoolExecutor(max_workers=8) as executor:
        for pth, residual_cpu in zip(pths, executor.map(_load_residual, paths)):
            c = file_to_idx(pth)
            residual = residual_cpu.to(dev)
            del residual_cpu
            attr = residual_stack_to_logit_attn(residual[head_idx], target_direction)  # (h, i, t)
            del residual
            n_i = attr.shape[1]

            counts[c] = n_i
            total_count += n_i
            pos_c = attr[:, :, c]                 # (h, i): class-c images -> text c
            pos_sum[c] = pos_c.sum(dim=1)
            pos_mean[c] = pos_c.mean(dim=1)
            total_sum += attr.sum(dim=1)          # (h, t)
            total_sumsq += (attr ** 2).sum(dim=1)
            del attr

    # Negative: every image NOT in class c, attributed to text c.
    neg_sum = total_sum.T - pos_sum                                   # (n_classes, n_heads)
    neg_den = (total_count - counts).clamp_min(1.0).unsqueeze(1)      # (n_classes, 1)
    neg_mean = neg_sum / neg_den
    scores = pos_mean - neg_mean                                      # (n_classes, n_heads)

    # Per-text std over all (class, image) attributions (unbiased, matches torch.std).
    N = total_count
    var = (total_sumsq - total_sum ** 2 / N) / max(N - 1, 1)
    std = var.clamp_min(0).sqrt().T + eps                            # (n_classes, n_heads)
    scores_z = scores / std

    return scores.T.contiguous(), scores_z.T.contiguous()            # (n_heads, n_classes)


# ============================================================================
# HEAD-LEVEL TASK RELEVANCE
# ============================================================================
def compute_head_task_relevance(disc_scores: torch.Tensor, top_k_frac: float = 0.2) -> dict:
    """
    Computes multiple head-level task relevance scalars.

    Args:
        disc_scores:  (n_heads, n_classes) — from compute_discriminative_scores()
                      for the new task classes
        top_k_frac:   fraction of classes to use for top-k mean (default: top 20%)

    Returns dict of (n_heads,) tensors:
        rank_mean       — primary metric: mean rank-percentile across classes
                          (how consistently above-average is this head)
        median_disc     — robust central tendency
        top_k_mean      — mean disc score over the head's best k% of classes
                          (how good at what it's good at)
        rank_consistency — 1 - std of rank percentiles (how reliably above-average)
        gini            — inequality of discriminative power across classes
                          (low = broad, high = specialized)
    """
    n_heads, n_classes = disc_scores.shape

    # ── 1. Rank percentile per class ──────────────────────────────────────────
    ranks = disc_scores.argsort(dim=0).argsort(dim=0).float()  # (n_heads, n_classes)
    ranks_norm = ranks / (n_heads - 1)                          # [0, 1]

    rank_mean = ranks_norm.mean(dim=1)              # (n_heads,)
    rank_consistency = 1.0 - ranks_norm.std(dim=1)  # (n_heads,)

    # ── 2. Median discriminative score ────────────────────────────────────────
    median_disc = disc_scores.median(dim=1).values  # (n_heads,)

    # ── 3. Top-k mean ─────────────────────────────────────────────────────────
    k = max(1, int(top_k_frac * n_classes))
    top_k_mean = disc_scores.topk(k, dim=1).values.mean(dim=1)  # (n_heads,)

    # ── 4. Gini coefficient ───────────────────────────────────────────────────
    s = disc_scores.sort(dim=1).values
    s = s - s.min(dim=1, keepdim=True).values
    idx = torch.arange(1, n_classes + 1, device=disc_scores.device).float()
    gini = (2 * (idx * s).sum(dim=1)) / (n_classes * s.sum(dim=1) + 1e-8) \
           - (n_classes + 1) / n_classes

    return {
        "rank_mean":         rank_mean,
        "rank_consistency":  rank_consistency,
        "median_disc":       median_disc,
        "top_k_mean":        top_k_mean,
        "gini":              gini,
    }


def _rank_mean(disc_scores: torch.Tensor) -> torch.Tensor:
    """Mean rank-percentile (over classes) of each head — see compute_head_task_relevance."""
    n_heads, _ = disc_scores.shape
    ranks = disc_scores.argsort(dim=0).argsort(dim=0).float()  # (n_heads, n_classes)
    ranks_norm = ranks / (n_heads - 1)                          # [0, 1]
    return ranks_norm.mean(dim=1)                               # (n_heads,)


def _top_k_mean(disc_scores: torch.Tensor, top_k_frac: float) -> torch.Tensor:
    """Mean disc score over each head's best k% of classes — see compute_head_task_relevance."""
    _, n_classes = disc_scores.shape
    k = max(1, int(top_k_frac * n_classes))
    return disc_scores.topk(k, dim=1).values.mean(dim=1)        # (n_heads,)


def compute_act_head_task_relevance(cos_scores: torch.Tensor, mse_scores: torch.Tensor,
                                    top_k_frac: float = 0.2) -> dict:
    """Head-level task relevance for the only_acts (raw-activation) path.

    cos_scores / mse_scores: (n_heads, n_classes) from
    compute_act_discriminative_scores(). Returns the rank_mean and top_k_mean
    aggregates for each scoring (the other DTA metrics are not meaningful here).

    Returns dict of (n_heads,) tensors:
        rank_mean_cos, top_k_mean_cos, rank_mean_mse, top_k_mean_mse
    """
    return {
        "rank_mean_cos":  _rank_mean(cos_scores),
        "top_k_mean_cos": _top_k_mean(cos_scores, top_k_frac),
        "rank_mean_mse":  _rank_mean(mse_scores),
        "top_k_mean_mse": _top_k_mean(mse_scores, top_k_frac),
    }


# ============================================================================
# PER-CONFIG DRIVER
# ============================================================================
def process_config(DATASET):
    print(f"\n{'='*78}\n=== Processing config: {DATASET}\n{'='*78}")
    ds_cfg = DATASETS[DATASET]

    # --- model ---------------------------------------------------------------
    model = load_model(ds_cfg)

    # --- class mapping -------------------------------------------------------
    class_names, file_to_idx = build_class_mapping(ds_cfg)
    N_CLASSES = len(class_names)
    print(f"Dataset '{DATASET}': {N_CLASSES} classes")

    # --- text dict (only needed for CLIP models; skipped otherwise) ----------
    # is_clip = ds_cfg.get("is_clip", False)
    is_clip = ds_cfg["is_clip"]
    # only_acts: CLIP-only path that scores heads from raw activations rather
    # than text attribution (useful when good class text prompts are hard to
    # write). Ignored for non-CLIP configs.
    # only_acts = is_clip and ds_cfg.get("only_acts", False)
    only_acts = ds_cfg["only_acts"]
    if is_clip and not only_acts:
        text_dict = load_or_compute_text_dict(
            ds_cfg, class_names, model.cfg.device, recompute=RECOMPUTE_TEXT_DICT
        )
    else:
        # text_dict is unused in the only_acts path (and for non-CLIP models).
        text_dict = None

    # --- discriminative scores + head-level task relevance ------------------
    if only_acts:
        # Raw-activation path: cosine-distance and MSE between the per-head mean
        # activation of class c and the mean activation of the rest.
        cos_scores, mse_scores = compute_act_discriminative_scores(
            model, HEAD_IDX, ds_cfg["activations_dir"], file_to_idx, N_CLASSES,
        )
        print("act disc_scores:", tuple(cos_scores.shape))
        relevance_metrics = compute_act_head_task_relevance(
            cos_scores, mse_scores, top_k_frac=0.2
        )
        n_heads_total = cos_scores.shape[0]
    else:
        disc_scores, disc_scores_z = compute_discriminative_scores(
            model, text_dict, HEAD_IDX, ds_cfg["activations_dir"], file_to_idx, N_CLASSES,
            is_clip=is_clip,
        )
        print("disc_scores:", tuple(disc_scores.shape))
        relevance_metrics = compute_head_task_relevance(disc_scores_z, top_k_frac=0.2)
        n_heads_total = disc_scores.shape[0]

    # ── Save per-head metrics to JSON ─────────────────────────────────────────
    # The activation tensors store the last few layers only, so flat head index 0
    # corresponds to (n_layers - n_saved_layers, head 0).
    N_HEADS_PER_LAYER = model.cfg.n_heads
    n_saved_layers = n_heads_total // N_HEADS_PER_LAYER
    FIRST_LAYER = model.cfg.n_layers - n_saved_layers

    output_metrics = {}
    for flat_idx in range(n_heads_total):
        layer = FIRST_LAYER + flat_idx // N_HEADS_PER_LAYER
        head = flat_idx % N_HEADS_PER_LAYER
        key = f"Layer {layer} Head {head}"
        output_metrics[key] = {
            metric_name: float(metric_tensor[flat_idx])
            for metric_name, metric_tensor in relevance_metrics.items()
        }

    out_dir = ds_cfg["metrics_out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    metrics_name = ds_cfg["metrics_name"]
    if only_acts:
        # Mark the file as activation-only so it doesn't clobber the DTA metrics.
        base, ext = os.path.splitext(metrics_name)
        metrics_name = f"{base}_act{ext}"
    output_path = os.path.join(out_dir, metrics_name)
    with open(output_path, "w") as f:
        json.dump(output_metrics, f, indent=2)
    print(f"Saved metrics for {len(output_metrics)} heads to {output_path}")

    # --- free GPU memory before the next config -----------------------------
    del model, text_dict, relevance_metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    for DATASET in CONFIG_KEYS:
        process_config(DATASET)
