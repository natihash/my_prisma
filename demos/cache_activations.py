import sys
import logging
sys.path.insert(0, "/home/nfm/ViT-Prisma/src")
logging.getLogger("PIL").setLevel(logging.WARNING)

import torch
import torchvision
from vit_prisma.sae import VisionModelSAERunnerConfig
from vit_prisma.sae.training.activations_store import VisionActivationsStore
from vit_prisma.transforms import get_clip_val_transforms
from vit_prisma.models.model_loader import load_hooked_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
CACHE_PATH = "/home/nfm/ViT-Prisma/demos/activation_cache/blocks11_resid_post_cls"

imagenet_train_path = '/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
imagenet_validation_path = '/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'

data_transforms = get_clip_val_transforms()
train_dataset = torchvision.datasets.ImageFolder(imagenet_train_path, transform=data_transforms)

model = load_hooked_model(MODEL_NAME)
model.to(DEVICE)

cfg = VisionModelSAERunnerConfig(
    model_name=MODEL_NAME,
    hook_point_layer=11,
    layer_subtype='hook_resid_post',
    cls_token_only=True,
    context_size=1,
    cached_activations_path=CACHE_PATH,
    store_batch_size=512,   # larger batch = faster caching
    num_workers=6,
)

store = VisionActivationsStore(cfg, model, train_dataset, eval_dataset=train_dataset, create_dataloader=False)
print(f"Caching activations to: {CACHE_PATH}")
print(f"Dataset size: {len(train_dataset)} images")
store.generate_cached_activations_from_dataset(tokens_per_file=500_000)
print("Done.")
