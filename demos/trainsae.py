import sys
import logging
sys.path.insert(0, "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/src")
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

    wandb_project="claude_topk_cls",
    expansion_factor=16,
    use_ghost_grads=False,

    # Stability tuning to avoid the NaNs from the previous CLS run. The cache is
    # clean (no inf/nan, values in ~[-5, 5]) and inputs are layer-normed to ~unit
    # scale, so the NaNs were training divergence, not bad data. 2e-4 was too hot
    # given the many passes this config makes over the small CLS cache; 5e-5 with
    # a longer warmup keeps it stable. max_grad_norm is the default but set here
    # explicitly (clips global grad norm to 1.0) as an extra NaN guard.
    lr=5e-5,
    lr_warm_up_steps=500,
    max_grad_norm=1.0,

    # CLS-only training: one token (the CLS token) per image, so context_size=1
    # matches the per-image token count the cache reshape uses.
    # (To go back to all-patches: cls_token_only=False, context_size=197, and
    #  point cached_activations_path at the "_all" cache.)
    cls_token_only=True,
    context_size=1,

    num_workers=6,
    store_batch_size=256,
    n_batches_in_buffer=32,
    train_batch_size=2048,

    # CLS-only cache (single file). This is the "_cls" cache, not "_all".
    cached_activations_path='/home/nfm/Desktop/rhome/nfm/ViT-Prisma/demos/activation_cache/blocks11_resid_post_cls',
    use_cached_activations=True,


    checkpoint_path='/home/nfm/Desktop/rhome/nfm/ViT-Prisma/demos/sae_ckpts',
    num_epochs=5,
    n_checkpoints=10,
)

pprint(sae_trainer_cfg)

trainer = VisionSAETrainer(sae_trainer_cfg, model, train_dataset, eval_dataset)
sae = trainer.run()
