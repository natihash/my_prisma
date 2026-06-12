import sys
import logging
sys.path.insert(0, "/home/nfm/ViT-Prisma/src")
logging.getLogger("PIL").setLevel(logging.WARNING)

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

# sae_trainer_cfg = VisionModelSAERunnerConfig( 
#     model_name=MODEL_NAME,
#     hook_point_layer=11,
#     layer_subtype='hook_resid_post',
#     dataset_name="imagenet",
#     feature_sampling_window=1000,
#     activation_fn_str='relu',
#     wandb_project="sae_training_clip_b16_cls_only",
#     expansion_factor=16,
    
#     cls_token_only=True,  
#     context_size=1,
#     # -------------------------------
    
#     num_workers=6,
#     store_batch_size=256,   
#     train_batch_size=8192,  
#     checkpoint_path='/home/nfm/ViT-Prisma/demos/sae_ckpts',
#     num_epochs=10,
#     n_checkpoints=5
# )

sae_trainer_cfg = VisionModelSAERunnerConfig(
    model_name=MODEL_NAME,
    hook_point_layer=11,
    layer_subtype='hook_resid_post',

    dataset_name="imagenet",
    feature_sampling_window=500,

    # TopK selects exactly k features per image, guaranteeing L0=k always.
    # 'identity' postact_fn: topk itself provides sparsity — no additional ReLU
    # needed, and ReLU was causing L0=0 when all pre-activations were negative.
    activation_fn_str='topk',
    activation_fn_kwargs={'k': 32, 'postact_fn': 'identity'},

    # zeros is correct here: normalize_activations='layer_norm' centers the input
    # to ~0 mean, so b_dec should start at 0 (not the geometric median of the raw,
    # unnormalized activations, which is in the wrong space).
    b_dec_init_method='zeros',

    wandb_project="claude_clip2_topk",
    expansion_factor=16,
    use_ghost_grads=False,

    lr=0.0002,
    lr_warm_up_steps=200,

    cls_token_only=True,
    context_size=1,

    num_workers=6,
    store_batch_size=256,
    # n_batches_in_buffer=32 gives data_for_loader = 32*0.75*256 = 6144 tokens,
    # which divides cleanly by train_batch_size=2048 (3 steps per buffer fill).
    # Twice as fast as 64 with no meaningful quality loss.
    n_batches_in_buffer=32,
    train_batch_size=2048,

    cached_activations_path='/home/nfm/ViT-Prisma/demos/activation_cache/blocks11_resid_post_cls',
    use_cached_activations=True,


    checkpoint_path='/home/nfm/ViT-Prisma/demos/sae_ckpts',
    num_epochs=5,
    n_checkpoints=10,
)

pprint(sae_trainer_cfg)

trainer = VisionSAETrainer(sae_trainer_cfg, model, train_dataset, eval_dataset)
sae = trainer.run()

