import sys
sys.path.insert(0, "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/src")

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import json

from vit_prisma.models.base_vit import HookedViT


def get_head_weights(model, layer: int, head: int):
    attn = model.blocks[layer].attn
    W_Q = attn.W_Q[head].detach()   # [d_model, d_head]
    W_K = attn.W_K[head].detach()   # [d_model, d_head]
    W_V = attn.W_V[head].detach()   # [d_model, d_head]
    W_O = attn.W_O[head].detach()   # [d_head, d_model]
    return W_Q, W_K, W_V, W_O


def compute_circuits(W_Q, W_K, W_V, W_O):
    QK = W_Q @ W_K.T   # [d_model, d_model], rank <= d_head
    OV = W_V @ W_O     # [d_model, d_model], rank <= d_head
    return QK, OV


def relative_frobenius(M_old: torch.Tensor, M_new: torch.Tensor) -> float:
    return (torch.norm(M_new - M_old) / (torch.norm(M_old) + 1e-10)).item()


def svd_spectral_analysis(M_old: torch.Tensor, M_new: torch.Tensor, top_k: int = 10):
    """
    Compare singular value spectra of old and new circuit matrices.

    Returns:
        sv_correlation:       Pearson correlation of full spectra
        energy_ratio_change:  change in fraction of energy in top-k
                              (positive = sharpening/specializing, negative = diffusing)
        spectral_dist:        L2 distance between normalized spectra
    """
    U_old, S_old, _ = torch.linalg.svd(M_old, full_matrices=False)
    U_new, S_new, _ = torch.linalg.svd(M_new, full_matrices=False)

    n = min(len(S_old), len(S_new))
    S_old, S_new = S_old[:n], S_new[:n]

    sv_corr = torch.corrcoef(torch.stack([S_old, S_new]))[0, 1].item()

    k = min(top_k, n)
    energy_old = (S_old[:k] ** 2).sum() / (S_old ** 2).sum()
    energy_new = (S_new[:k] ** 2).sum() / (S_new ** 2).sum()
    energy_ratio_change = (energy_new - energy_old).item()

    S_old_norm = S_old / (S_old.sum() + 1e-10)
    S_new_norm = S_new / (S_new.sum() + 1e-10)
    spectral_dist = torch.norm(S_new_norm - S_old_norm).item()

    return {
        "sv_correlation":       sv_corr,
        "energy_ratio_change":  energy_ratio_change,
        "spectral_distance":    spectral_dist,
    }


def principal_angles(M_old: torch.Tensor, M_new: torch.Tensor, top_k: int = 10):
    """
    Principal angles between the top-k left singular subspaces of M_old and M_new.

    Returns:
        mean_cos_angle:          mean cos(angle) — 1.0 = identical subspaces, 0 = orthogonal
        weighted_subspace_overlap: fraction of old subspace variance explained by new subspace
    """
    U_old, S_old, _ = torch.linalg.svd(M_old, full_matrices=False)
    U_new, S_new, _ = torch.linalg.svd(M_new, full_matrices=False)

    k = min(top_k, U_old.shape[1], U_new.shape[1])
    U_old_k = U_old[:, :k]
    U_new_k = U_new[:, :k]

    cos_angles = torch.linalg.svdvals(U_old_k.T @ U_new_k).clamp(-1.0, 1.0)

    S_weights = S_old[:k] ** 2
    S_weights = S_weights / (S_weights.sum() + 1e-10)
    weighted_overlap = (S_weights * cos_angles ** 2).sum().item()

    return {
        "mean_cos_angle":            cos_angles.mean().item(),
        "weighted_subspace_overlap": weighted_overlap,
    }


def effective_rank(M: torch.Tensor) -> float:
    """exp(entropy of normalized singular values). Higher = more directions used."""
    S = torch.linalg.svdvals(M)
    S = S[S > 1e-10]
    p = S / S.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def effective_rank_change(M_old: torch.Tensor, M_new: torch.Tensor):
    r_old = effective_rank(M_old)
    r_new = effective_rank(M_new)
    return {
        "effective_rank_change": r_new - r_old,
    }


def directional_decomposition_subspace(
    M_old: torch.Tensor,
    M_new: torch.Tensor,
    top_k: int = 10,
):
    """
    Decompose ΔM into within-subspace and out-of-subspace components, where
    "subspace" = col(M_old) ⊗ row(M_old) (the full rank-r SVD subspace of M_old).

    IMPORTANT — what these metrics actually measure:
      - within_subspace_fraction: fraction of ||ΔM|| that is a pure SCALING of
        M_old's singular values (same directions, different magnitudes).
      - new_direction_fraction: everything else — this includes ROTATION of the
        operating subspace as well as genuinely new functionality. Even a tiny
        angular shift of singular vectors generates large Frobenius components
        outside the original col×row subspace, so this metric is dominated by
        subspace rotation, NOT new capabilities. Use mean_cos_angle and
        weighted_subspace_overlap to assess subspace alignment directly.

    Uses the full numerical rank of M_old (auto-detected via singular value
    threshold), not just top_k. top_k is ignored and kept only for API compat.
    """
    U, S, Vt = torch.linalg.svd(M_old, full_matrices=False)

    # Use full rank of M_old, not just top_k (top_k < rank gives artifacts)
    thresh = S[0] * max(M_old.shape) * torch.finfo(S.dtype).eps * 10
    k = int((S > thresh).sum().item())
    k = max(k, 1)

    U_k  = U[:, :k]
    Vt_k = Vt[:k, :]

    delta = M_new - M_old
    delta_within = U_k @ (U_k.T @ delta @ Vt_k.T) @ Vt_k
    delta_outside = delta - delta_within

    norm_within  = torch.norm(delta_within).item()
    norm_outside = torch.norm(delta_outside).item()
    norm_total   = torch.norm(delta).item()

    return {
        "new_direction_fraction":    norm_outside / (norm_total + 1e-10),
        "within_subspace_fraction":  norm_within  / (norm_total + 1e-10),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main comparison entry points
# ═══════════════════════════════════════════════════════════════════════════════

def compare_circuit(M_old: torch.Tensor, M_new: torch.Tensor, top_k: int = 10) -> dict:
    """Run all metrics on a single circuit matrix pair."""
    return {
        "relative_frobenius":       relative_frobenius(M_old, M_new),
        **svd_spectral_analysis(M_old, M_new, top_k=top_k),
        **principal_angles(M_old, M_new, top_k=top_k),
        **effective_rank_change(M_old, M_new),
        **directional_decomposition_subspace(M_old, M_new, top_k=top_k),
    }


def compare_heads(
    model_src,
    model_ft,
    layer: int,
    head: int,
    top_k: int = 10,
) -> dict:
    """
    Full weight-space comparison of one attention head between source and finetuned models.
    Returns dict with keys prefixed by 'qk_' and 'ov_' for each circuit.
    """
    W_Q_src, W_K_src, W_V_src, W_O_src = get_head_weights(model_src, layer, head)
    W_Q_ft,  W_K_ft,  W_V_ft,  W_O_ft  = get_head_weights(model_ft,  layer, head)

    QK_src, OV_src = compute_circuits(W_Q_src, W_K_src, W_V_src, W_O_src)
    QK_ft,  OV_ft  = compute_circuits(W_Q_ft,  W_K_ft,  W_V_ft,  W_O_ft)

    qk_metrics = compare_circuit(QK_src, QK_ft, top_k=top_k)
    ov_metrics = compare_circuit(OV_src, OV_ft, top_k=top_k)

    result = {}
    for k, v in qk_metrics.items():
        result[f"qk_{k}"] = v
    for k, v in ov_metrics.items():
        result[f"ov_{k}"] = v

    result["layer"] = layer
    result["head"]  = head
    return result


_SCALAR_KEYS = [
    "relative_frobenius",
    "sv_correlation",
    "energy_ratio_change",
    "spectral_distance",
    "mean_cos_angle",
    "weighted_subspace_overlap",
    "effective_rank_change",
    "new_direction_fraction",
    "within_subspace_fraction",
]


def compare_all_heads(
    model_src,
    model_ft,
    layers: range = range(8, 12),
    n_heads: int = 12,
    top_k: int = 10,
) -> dict:
    """
    Compare all heads across specified layers.
    Returns dict of (n_total_heads,) tensors for each scalar metric,
    with 'qk_' and 'ov_' prefixes, plus 'layer_ids' and 'head_ids'.
    """
    all_results, layer_ids, head_ids = [], [], []

    for layer in layers:
        for head in range(n_heads):
            result = compare_heads(model_src, model_ft, layer, head, top_k=top_k)
            all_results.append(result)
            layer_ids.append(layer)
            head_ids.append(head)

    out = {
        "layer_ids": torch.tensor(layer_ids),
        "head_ids":  torch.tensor(head_ids),
    }
    for prefix in ["qk_", "ov_"]:
        for key in _SCALAR_KEYS:
            full_key = f"{prefix}{key}"
            out[full_key] = torch.tensor(
                [r[full_key] for r in all_results], dtype=torch.float32
            )
    return out


def print_head_comparison(result: dict):
    """Pretty-print results from compare_heads()."""
    print(f"\n{'='*65}")
    print(f"  Head L{result['layer']}H{result['head']}")
    print(f"{'='*65}")

    for circuit in ["qk", "ov"]:
        label = "QK circuit (where to look)" if circuit == "qk" else "OV circuit (what to extract)"
        print(f"\n  {label}")
        print(f"  {'-'*55}")
        print(f"    Relative Frobenius:        {result[f'{circuit}_relative_frobenius']:.4f}")
        print(f"    SV correlation:            {result[f'{circuit}_sv_correlation']:.4f}")
        print(f"    Energy ratio change:       {result[f'{circuit}_energy_ratio_change']:+.4f}"
              f"  ({'sharpening' if result[f'{circuit}_energy_ratio_change'] > 0 else 'diffusing'})")
        print(f"    Spectral distance:         {result[f'{circuit}_spectral_distance']:.4f}")
        print(f"    Mean cos(principal angle): {result[f'{circuit}_mean_cos_angle']:.4f}")
        print(f"    Weighted subspace overlap: {result[f'{circuit}_weighted_subspace_overlap']:.4f}")
        print(f"    Effective rank change:     {result[f'{circuit}_effective_rank_change']:+.1f}"
              f"  ({'specializing' if result[f'{circuit}_effective_rank_change'] < 0 else 'diversifying'})")
        ndf = result[f'{circuit}_new_direction_fraction']
        wsf = result[f'{circuit}_within_subspace_fraction']
        print(f"    New direction fraction:    {ndf:.4f}  (scaling=0, rotation/new=1)")
        print(f"    Within-subspace fraction:  {wsf:.4f}  (pure scaling only)")


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

load_kwargs = dict(
    center_writing_weights=True,
    center_unembed=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device="cuda",
)

source   = HookedViT.from_pretrained("open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K", **load_kwargs)
# finetuned = HookedViT.from_pretrained("hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft", **load_kwargs)
# finetuned = HookedViT.from_pretrained("hf_hub:natihash/vit_base_patch16_clip_224.laion2b_lora_r16_merged", **load_kwargs)
# finetuned = HookedViT.from_pretrained("hf_hub:natihash/vit_base_patch16_clip_224.laion2b_lora_r4_merged", **load_kwargs)
# finetuned = HookedViT.from_pretrained("hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft_latest", **load_kwargs)
# model_name = "vit_base_patch16_clip_224.laion2b_ft_in1k"



def save_head_metrics_json(
    model_src,
    model_ft,
    layers: range = range(8, 12),
    n_heads: int = 12,
    top_k: int = 10,
    save_path: str = None,
) -> dict:
    """
    Compare all heads in `layers` and save scalar change-metrics to a JSON file.

    JSON structure:
        { "Layer 8 Head 0": { "qk_relative_frobenius": ..., ... }, ... }
    """
    results = {}
    for layer in layers:
        for head in range(n_heads):
            key = f"Layer {layer} Head {head}"
            print(f"  computing {key}...", flush=True)
            r = compare_heads(model_src, model_ft, layer, head, top_k=top_k)

            head_metrics = {}
            for prefix in ["qk_", "ov_"]:
                for metric in _SCALAR_KEYS:
                    full_key = f"{prefix}{metric}"
                    val = r[full_key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    head_metrics[full_key] = round(float(val), 4)

            results[key] = head_metrics

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics for {len(results)} heads → {save_path}")
    else:
        print(f"\nComputed metrics for {len(results)} heads (not saved to file).")
    return results


# model_name0 = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft_sun"
# model_name1 = "hf_hub:natihash/vit_base_patch16_clip_224.sun_lora_r16_merged"
# model_name2 = "hf_hub:natihash/vit_base_patch16_clip_224.sun_lora_r4_merged"

# model_name2 = "hf_hub:natihash/vit_base_patch16_clip_224.laion2b_fullft_last"
# model_name3 = "hf_hub:natihash/vit_base_patch16_clip_224.sun_lora_r16_last"
# model_name4 = "hf_hub:natihash/vit_base_patch16_clip_224.sun_lora_r4_last"

model_names = [
    "hf_hub:natihash/vit_base_patch16_clip_224.text_fft",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lora4",
    "hf_hub:natihash/vit_base_patch16_clip_224.text_lora16",
]

model_tags = ["fft", "lora4", "lora16"]

# model_names = [model_name0, model_name1, model_name2]
# model_names = [model_name2, model_name3, model_name4]


for model_name, model_tag in zip(model_names, model_tags):
    finetuned  = HookedViT.from_pretrained(model_name, **load_kwargs)

    metrics = save_head_metrics_json(
        source,
        finetuned,
        layers=range(8, 12),
        n_heads=12,
        top_k=10,
        save_path=f"/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/text/change_metrics_{model_tag}.json",
    )
