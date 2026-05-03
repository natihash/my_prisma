import sys
sys.path.insert(0, "/home/nfm/ViT-Prisma/src")

from vit_prisma.sae import VisionModelSAERunnerConfig
from vit_prisma.sae import VisionSAETrainer
from vit_prisma.transforms import get_clip_val_transforms


import torchvision
import torch

from torch.utils.data import DataLoader, Subset
from pprint import pprint

# Put your ImageNet Paths here
from vit_prisma.transforms import get_clip_val_transforms

imagenet_train_path = '/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
imagenet_validation_path = '/home/nfm/data_prisma/imagenet_val/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'

data_transforms = get_clip_val_transforms()
train_dataset = torchvision.datasets.ImageFolder(imagenet_train_path, transform=data_transforms)
eval_dataset = torchvision.datasets.ImageFolder(imagenet_validation_path, transform=data_transforms)

MODEL_NAME = "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

from vit_prisma.models.model_loader import load_hooked_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_hooked_model(MODEL_NAME)
model.to(DEVICE);

sae_trainer_cfg = VisionModelSAERunnerConfig( 
    model_name=MODEL_NAME,
    hook_point_layer=11,
    layer_subtype='hook_resid_post',
    dataset_name="imagenet",
    feature_sampling_window=1000,
    activation_fn_str='relu',
    wandb_project="sae_training_clip_b16_cls_only",
    expansion_factor=16,
    
    # --- CHANGED FOR ALL PATCHES ---
    cls_token_only=True,  
    context_size=1,      # 196 patches + 1 CLS token
    # -------------------------------
    
    num_workers=6,
    store_batch_size=256,   
    train_batch_size=8192,  
    checkpoint_path='/home/nfm/ViT-Prisma/demos/sae_ckpts',
    num_epochs=10,
    n_checkpoints=5
)

pprint(sae_trainer_cfg)

trainer = VisionSAETrainer(sae_trainer_cfg, model, train_dataset, eval_dataset)
sae = trainer.run()

