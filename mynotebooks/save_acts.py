import sys
sys.path.insert(0, "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/src")

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # leave off for speed; only enable to debug CUDA errors

import json
import torch
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vit_prisma.models.base_vit import HookedViT  # adjust import if your HookedViT lives elsewhere

# =========================================================================
# 0. TRANSFORM
# =========================================================================
class ConvertTo3Channels:
    def __call__(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

transform = transforms.Compose([
    ConvertTo3Channels(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
SUN_VAL_DIR      = "/home/nfm/Desktop/rhome/nfm/text_overlay_dataset/val/"
IMAGENET_VAL_DIR = "/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/"
LOCAL_JSON_PATH  = "/home/nfm/ViT-Prisma/demos/imagenet_class_index.json"
BASE_OUT         = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/head_acts"

CHUNK_SIZE     = 256   # images per forward pass. 48GB can likely take 256 — bump if VRAM allows.
LAYERS_TO_KEEP = 4
NUM_WORKERS    = 6
DEVICE         = "cuda:0"

# =========================================================================
# 2. IMAGENET NAME MAPS (for the original-style imagenet filenames)
# =========================================================================
with open(LOCAL_JSON_PATH, 'r') as f:
    imagenet_class_index = json.load(f)

wnid_to_name = {}
for idx, (wnid, class_name) in imagenet_class_index.items():
    wnid_to_name[wnid] = class_name.replace(" ", "_").replace("/", "_").replace(",", "")

# =========================================================================
# 3. HELPERS
# =========================================================================
def build_class_batches(dataset, chunk_size):
    """One batch never crosses a class boundary -> still per-class, but big chunks."""
    class_to_indices = defaultdict(list)
    for i, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(i)
    batches = []
    for label in sorted(class_to_indices):
        idxs = class_to_indices[label]
        for s in range(0, len(idxs), chunk_size):
            batches.append(idxs[s:s + chunk_size])
    return batches

def sun_filename(dataset, label):
    # SUN val folders look like "a_abbey", "b_bakery" -> strip leading "<letter>_"
    folder = dataset.classes[label]
    clean = folder.split("_", 1)[1] if "_" in folder else folder
    safe = clean.replace(" ", "_").replace("/", "_").replace(",", "")
    return f"{safe}.pt"

def imagenet_filename(dataset, label):
    folder = dataset.classes[label]                       # wnid
    name = wnid_to_name.get(folder, "unknown_class")
    return f"{folder}_{name}.pt"                           # matches the original code's naming

def load_model(model_name):
    model = HookedViT.from_pretrained(
        model_name,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
        device="cuda",
    )
    model = model.to(DEVICE)
    model.cfg.device = DEVICE
    model.eval()
    return model

def extract_activations(model, dataset, name_fn, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_heads_to_keep = LAYERS_TO_KEEP * model.cfg.n_heads

    # Build batches, then drop any class whose output already exists (resume support)
    batches = build_class_batches(dataset, CHUNK_SIZE)
    batches = [
        b for b in batches
        if not os.path.exists(os.path.join(output_dir, name_fn(dataset, dataset.samples[b[0]][1])))
    ]
    if not batches:
        print(f"  [skip] everything already saved in {output_dir}")
        return

    loader = DataLoader(
        dataset, batch_sampler=batches,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"  Saving {num_heads_to_keep}xNx{model.cfg.d_model} tensors -> {output_dir}")

    current_label = None
    buffer = []

    def flush(label, chunks):
        if not chunks:
            return
        save_path = os.path.join(output_dir, name_fn(dataset, label))
        torch.save(torch.cat(chunks, dim=1), save_path)   # [num_heads_to_keep, N_images, d_model]

    for imgs, labels in tqdm(loader, total=len(batches)):
        label = labels[0].item()                          # batch is single-class by construction
        if current_label is not None and label != current_label:
            flush(current_label, buffer)
            buffer = []
        current_label = label

        imgs = imgs.to(model.cfg.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(imgs)
            all_head_residuals = cache.stack_head_results(layer=-1, pos_slice=0)
            scaled = cache.apply_ln_to_stack(all_head_residuals, layer=-1, pos_slice=0)
            last = scaled[-num_heads_to_keep:, :, :].cpu().clone()

        buffer.append(last)
        del cache, all_head_residuals, scaled, last
        torch.cuda.empty_cache()

    flush(current_label, buffer)                          # final class

# =========================================================================
# 4. LOAD DATASETS ONCE (reused across models)
# =========================================================================
sun_dataset      = ImageFolder(SUN_VAL_DIR, transform=transform)
imagenet_dataset = ImageFolder(IMAGENET_VAL_DIR, transform=transform)

DATASETS = {
    "sun":      (sun_dataset,      sun_filename,      "text"),
    "imagenet": (imagenet_dataset, imagenet_filename, "text"),
}

# =========================================================================
# 5. JOB LIST  (model, dataset, output-tag)
# =========================================================================

# "hf_hub:natihash/vit_base_patch16_clip_224.text_lp",
# "hf_hub:natihash/vit_base_patch16_clip_224.text_fft",
# "hf_hub:natihash/vit_base_patch16_clip_224.text_lora4",
# "hf_hub:natihash/vit_base_patch16_clip_224.text_lora16",

JOBS = [
    # --- three SUN-finetuned models on the SUN dataset ---
    ("hf_hub:natihash/vit_base_patch16_clip_224.text_fft",   "sun", "fft"),
    ("hf_hub:natihash/vit_base_patch16_clip_224.text_lora16",  "sun", "lora16"),
    ("hf_hub:natihash/vit_base_patch16_clip_224.text_lora4",   "sun", "lora4"),

    # --- original CLIP on the SUN dataset ---
    ("open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",                "sun", "clip_orig"),

    # --- original CLIP on ImageNet (the original imagenet act-saving code) ---
    # ("open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",                "imagenet", "clip_orig"),

    # If you also want the SUN-finetuned models on ImageNet, uncomment:
    ("hf_hub:natihash/vit_base_patch16_clip_224.text_fft",  "imagenet", "fft_imagenet"),
    ("hf_hub:natihash/vit_base_patch16_clip_224.text_lora16", "imagenet", "lora16_imagenet"),
    ("hf_hub:natihash/vit_base_patch16_clip_224.text_lora4",  "imagenet", "lora4_imagenet"),
]

# =========================================================================
# 6. RUN
# =========================================================================
print("Is CUDA available?:", torch.cuda.is_available())

for model_name, data_key, tag in JOBS:
    dataset, name_fn, data_dir = DATASETS[data_key]
    output_dir = os.path.join(BASE_OUT, data_dir, tag)

    print(f"\n=== {model_name}  |  {data_key}  ->  {output_dir} ===")
    model = load_model(model_name)
    print("  Model device config:", model.cfg.device)

    extract_activations(model, dataset, name_fn, output_dir)

    del model
    torch.cuda.empty_cache()

print("\nAll jobs complete!")
