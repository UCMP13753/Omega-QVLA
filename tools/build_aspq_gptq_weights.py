#!/usr/bin/env python
"""Build offline ASPQ-GPTQ quantized weights for GR00T.

This script keeps the existing DuQuant-based ASPQ flow untouched. It builds a
parallel artifact containing dequantized GPTQ weights per layer, optionally
refined inside the ASPQ action subspace.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    DEFAULT_EXCLUDE_REGEX,
    DEFAULT_INCLUDE_REGEX,
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    seed_everything,
)
from tools.aspq_jacobian_sanity import normalized_input_no_inference  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402
from gr00t.quantization.aspq_gptq import (  # noqa: E402
    load_aspq_basis_for_layer,
    solve_aspq_gptq_weight,
    solve_aspq_gptq_weight_with_alpha_select,
)
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402

DEFAULT_LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/ceph/workspace/xinyu/LIBERO")
DEFAULT_LIBERO_DATASET = f"{DEFAULT_LIBERO_ROOT}/datasets/lerobot_libero_10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metric-path", required=True, help="ASPQ metric .pt from collect_aspq_jacobian.py")
    p.add_argument("--output-path", required=True, help="Where to save the ASPQ-GPTQ quantized weights (.pt)")
    p.add_argument("--weight-bits", type=int, default=3)
    p.add_argument("--metric-top-k", type=int, default=0, help="Further truncate the loaded ASPQ basis; 0 keeps all stored vectors.")
    p.add_argument("--min-eig", type=float, default=1e-12)
    p.add_argument("--min-eig-relative", type=float, default=0.0,
                   help="Drop eigvals < min_eig_relative * eigmax per layer (0 disables).")
    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--gptq-err-comp-gamma", type=float, default=1.0,
                   help="GPTQ inter-column error-compensation strength. 1.0 = full GPTQ; "
                        "0.0 = RTN (each block quantized independently). Forwarded to "
                        "gptq_quantize_weight via solve_aspq_gptq_weight.")
    p.add_argument("--missing-metric", choices=["error", "fallback"], default="fallback")
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")

    p.add_argument("--data-source", default="libero", choices=["libero"])
    p.add_argument("--dataset-path", default=DEFAULT_LIBERO_DATASET)
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task", choices=["sequential", "one_per_task", "one_per_trial"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--include-regex", default=DEFAULT_INCLUDE_REGEX)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--cache-layer-inputs", action="store_true",
                   help="Capture sampled layer inputs for all target layers in one pass and reuse them during GPTQ build.")
    p.add_argument("--duquant-rotation", action="store_true",
                   help="Apply DuQuant per-block input rotation as preprocessing BEFORE running ASPQ-GPTQ. "
                        "Builds a hybrid pack: input rotated by R (block-diag of SVD-derived rotations from FP weight), "
                        "then ASPQ-GPTQ on rotated (W,H). Rotation is stored in the record and applied at runtime "
                        "before the standard ASPQ-GPTQ forward.")
    p.add_argument("--duquant-block-size", type=int, default=64)
    p.add_argument("--duquant-rot-mode", default="svd",
                   choices=["svd", "hadamard", "svd_hadamard", "random_hadamard"],
                   help="DuQuant rotation mode. 'svd' (default) = original DuQuant SVD basis; "
                        "'svd_hadamard' = A2-lite (SVD @ Hadamard); 'hadamard' = pure Hadamard; "
                        "'random_hadamard' = D@H (pow-2) or Haar random orthogonal (non-pow-2), "
                        "intended for full-d use.")
    p.add_argument("--duquant-permute", action="store_true",
                   help="Enable zigzag permutation (matches GR00T_DUQUANT_PERMUTE=1 at runtime).")
    p.add_argument("--duquant-output-rotation", action="store_true",
                   help="Also apply block-diag output rotation R_out (matches runtime row_rot=restore). "
                        "Stored as duquant_rotation_out in pack; runtime applies y = y @ R_out after matmul.")
    p.add_argument("--duquant-block-out", type=int, default=0,
                   help="Block size for output rotation; 0 → same as --duquant-block-size.")
    # ---- Per-layer-type overrides (attention vs MLP) ----
    # Each defaults to -1 / "" which means "inherit the duquant_* value".
    # Layer classification: ``attn1`` in module name → attention; ``ff.net`` → MLP.
    p.add_argument("--attn-block-size", type=int, default=-1)
    p.add_argument("--attn-block-out", type=int, default=-1)
    p.add_argument("--attn-rot-mode", default="",
                   choices=["", "svd", "hadamard", "svd_hadamard", "random_hadamard"])
    p.add_argument("--attn-gptq-damp", type=float, default=-1.0)
    p.add_argument("--attn-group-size", type=int, default=0,
                   help="Per-group W4 quantization on attention layers (0 = per-row).")
    p.add_argument("--mlp-block-size", type=int, default=-1)
    p.add_argument("--mlp-block-out", type=int, default=-1)
    p.add_argument("--mlp-rot-mode", default="",
                   choices=["", "svd", "hadamard", "svd_hadamard", "random_hadamard"])
    p.add_argument("--mlp-gptq-damp", type=float, default=-1.0)
    p.add_argument("--mlp-group-size", type=int, default=0)
    return p.parse_args()


def _resolve_per_kind(args, name: str) -> dict:
    """Return effective {block_size, block_out, rot_mode, gptq_damp, group_size} for a layer name."""
    is_attn = "attn1" in name
    prefix = "attn" if is_attn else "mlp"
    def _pick(per, default):
        return per if per not in (-1, -1.0, "") else default
    block_size = _pick(getattr(args, f"{prefix}_block_size"), int(args.duquant_block_size))
    block_out_default = int(args.duquant_block_out) if int(args.duquant_block_out) > 0 else int(args.duquant_block_size)
    block_out = _pick(getattr(args, f"{prefix}_block_out"), block_out_default)
    rot_mode = _pick(getattr(args, f"{prefix}_rot_mode"), args.duquant_rot_mode)
    damp = _pick(getattr(args, f"{prefix}_gptq_damp"), float(args.gptq_damp_percent))
    group_size = int(getattr(args, f"{prefix}_group_size"))
    return {
        "is_attn": is_attn,
        "block_size": int(block_size),
        "block_out": int(block_out),
        "rot_mode": rot_mode,
        "gptq_damp": float(damp),
        "group_size": group_size,
    }


def resolve_target_layers(policy, args) -> list[str]:
    targets = select_targets(
        policy.model,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        scope_prefix=args.scope or None,
        whitelist=None,
        blacklist=None,
    )
    names = [n for n, _ in targets]
    names = names[args.start_layer:]
    if args.max_layers > 0:
        names = names[: args.max_layers]
    return names


def _clear_quant_env() -> None:
    for prefix in ("GR00T_DUQUANT_", "GR00T_ASPQ_GPTQ"):
        for key in list(os.environ.keys()):
            if key == prefix or key.startswith(prefix):
                os.environ.pop(key, None)


def _autocast_context(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE)
    return nullcontext()


def collect_layer_gram(
    policy,
    normalized_samples,
    layer_name: str,
    token_cap: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, int]:
    module = get_named_module(policy.model, layer_name)
    gram: Optional[torch.Tensor] = None
    n_tokens = 0
    gram_device = module.weight.device if hasattr(module, "weight") else torch.device(device)

    def hook(_m, inputs):
        nonlocal gram, n_tokens
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        if token_cap > 0 and x.shape[0] > token_cap:
            idx = torch.from_numpy(rng.choice(x.shape[0], size=token_cap, replace=False)).to(x.device)
            x = x.index_select(0, idx)
        if gram is None:
            gram = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32, device=gram_device)
        x = x.to(device=gram_device, dtype=torch.float32, non_blocking=True)
        gram += x.t() @ x
        n_tokens += int(x.shape[0])

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(), _autocast_context(device):
            for sample in normalized_samples:
                seed_everything(sample["seed"])
                policy.model.get_action(sample["normalized"])
    finally:
        handle.remove()

    if gram is None or n_tokens == 0:
        raise RuntimeError(f"Failed to collect activations for layer '{layer_name}'")
    return gram / float(n_tokens), n_tokens


def collect_cached_layer_inputs(
    policy,
    normalized_samples,
    layer_names: list[str],
    token_cap: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    modules = {name: get_named_module(policy.model, name) for name in layer_names}
    # Cache to CPU by default to avoid eating ~6GB of GPU memory during the build
    # (which competes with concurrent inference servers and causes CUDA OOM).
    # Set GR00T_BUILD_CACHE_ON_GPU=1 to keep caches on GPU (faster but heavier).
    cache_on_gpu = os.environ.get("GR00T_BUILD_CACHE_ON_GPU", "0") not in ("0", "false", "False")
    if cache_on_gpu:
        cache_devices = {
            name: (module.weight.device if hasattr(module, "weight") else torch.device(device))
            for name, module in modules.items()
        }
    else:
        cache_devices = {name: torch.device("cpu") for name in layer_names}
    cached: dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}
    n_tokens: dict[str, int] = {name: 0 for name in layer_names}

    def make_hook(name: str):
        def hook(_m, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            if token_cap > 0 and x.shape[0] > token_cap:
                idx = torch.from_numpy(rng.choice(x.shape[0], size=token_cap, replace=False)).to(x.device)
                x = x.index_select(0, idx)
            x = x.to(device=cache_devices[name], dtype=torch.float16, non_blocking=True)
            cached[name].append(x.contiguous())
            n_tokens[name] += int(x.shape[0])
        return hook

    handles = [module.register_forward_pre_hook(make_hook(name)) for name, module in modules.items()]
    try:
        with torch.no_grad(), _autocast_context(device):
            for sample in normalized_samples:
                seed_everything(sample["seed"])
                policy.model.get_action(sample["normalized"])
    finally:
        for handle in handles:
            handle.remove()

    merged: dict[str, torch.Tensor] = {}
    for name in layer_names:
        if not cached[name] or n_tokens[name] == 0:
            raise RuntimeError(f"Failed to cache activations for layer '{name}'")
        merged[name] = torch.cat(cached[name], dim=0)
    return merged, n_tokens


def main() -> None:
    args = parse_args()
    ensure_libero_runtime()
    _clear_quant_env()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"[ASPQ-GPTQ] loading FP policy ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()

    layer_names = resolve_target_layers(policy, args)
    print(f"[ASPQ-GPTQ] target layers: {len(layer_names)}")

    print(f"[ASPQ-GPTQ] loading {args.num_samples} LIBERO observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(
            {
                "seed": sample["seed"],
                "normalized": normalized_input_no_inference(policy, sample["obs"]),
            }
        )
    print(f"[ASPQ-GPTQ] prepared {len(normalized_samples)} normalized observations")

    # SmoothQuant prerequisite: per-input-channel q99(|x|) is computed only
    # from cached inputs. Fail loudly if the user requested SQ without caching.
    sq_alpha_env = float(os.environ.get("GR00T_ASPQ_SQ_ALPHA", "0"))
    sq_clamp_lo_env = float(os.environ.get("GR00T_ASPQ_SQ_CLAMP_LO", "0.5"))
    sq_clamp_hi_env = float(os.environ.get("GR00T_ASPQ_SQ_CLAMP_HI", "2.0"))
    sq_token_cap = int(os.environ.get("GR00T_ASPQ_SQ_TOKEN_CAP", "8192"))
    sq_alpha_grid_env = os.environ.get("GR00T_ASPQ_SQ_ALPHA_GRID", "").strip()
    sq_select_mode_env = os.environ.get("GR00T_ASPQ_SQ_SELECT_MODE", "fixed").strip().lower()
    sq_score_token_cap = int(os.environ.get("GR00T_ASPQ_SQ_SCORE_TOKEN_CAP", "1024"))
    sq_alpha_grid: Optional[list] = None
    if sq_alpha_grid_env and sq_select_mode_env in ("mse", "action"):
        sq_alpha_grid = [float(a.strip()) for a in sq_alpha_grid_env.split(",") if a.strip()]
        if not sq_alpha_grid:
            raise ValueError("GR00T_ASPQ_SQ_ALPHA_GRID parsed to empty list")
        if not args.cache_layer_inputs:
            raise ValueError(
                "Per-layer alpha selection requires --cache-layer-inputs "
                "to score candidate alphas against calibration X."
            )
        print(
            f"[ASPQ-GPTQ-SQ] alpha-select mode='{sq_select_mode_env}' "
            f"grid={sq_alpha_grid} clamp=[{sq_clamp_lo_env},{sq_clamp_hi_env}] "
            f"score_token_cap={sq_score_token_cap}"
        )
    elif sq_alpha_env > 0.0:
        if not args.cache_layer_inputs:
            raise ValueError(
                "SmoothQuant (GR00T_ASPQ_SQ_ALPHA > 0) requires --cache-layer-inputs "
                "to compute per-channel q99(|x|); the non-cached path only collects H "
                "and would silently disable SQ."
            )
        print(
            f"[ASPQ-GPTQ-SQ] enabled  alpha={sq_alpha_env}  "
            f"clamp=[{sq_clamp_lo_env},{sq_clamp_hi_env}]  "
            f"token_cap={sq_token_cap}"
        )

    cached_inputs = None
    cached_tokens = None
    if args.cache_layer_inputs:
        print("[ASPQ-GPTQ] caching sampled inputs for all target layers in one forward sweep")
        cached_inputs, cached_tokens = collect_cached_layer_inputs(
            policy,
            normalized_samples,
            layer_names,
            args.token_cap,
            rng,
            str(args.device),
        )
        total_mb = sum(t.numel() * t.element_size() for t in cached_inputs.values()) / (1024 ** 2)
        print(f"[ASPQ-GPTQ] cached {len(cached_inputs)} layers of sampled inputs ({total_mb:.1f} MiB on-device)")

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    records: dict[str, dict] = {}

    for idx, layer_name in enumerate(layer_names, start=1):
        module = get_named_module(policy.model, layer_name)
        if not isinstance(module, torch.nn.Linear):
            continue
        print(f"[ASPQ-GPTQ] layer {idx}/{len(layer_names)}: {layer_name}")
        solve_device = module.weight.device
        x_calib_stats: Optional[torch.Tensor] = None
        x_for_score: Optional[torch.Tensor] = None
        if cached_inputs is not None and cached_tokens is not None:
            x_cached = cached_inputs[layer_name].to(device=solve_device, dtype=torch.float32, non_blocking=True)
            n_tokens = int(cached_tokens[layer_name])
            H = (x_cached.t() @ x_cached) / float(n_tokens)
            # SmoothQuant calibration stat: per-input-channel q99(|x|).
            # torch.quantile internally sorts so memory cost ~2× the input matrix
            # plus a per-row sort buffer. Subsample tokens to keep this bounded
            # — q99 only feeds a scale heuristic, full precision is unnecessary.
            sq_active = sq_alpha_env > 0.0 or sq_alpha_grid is not None
            if sq_active:
                if x_cached.shape[0] > sq_token_cap:
                    perm = torch.randperm(x_cached.shape[0], device=x_cached.device)[:sq_token_cap]
                    x_for_sq = x_cached.index_select(0, perm)
                else:
                    x_for_sq = x_cached
                x_calib_stats = torch.quantile(x_for_sq.abs(), 0.99, dim=0).to(dtype=torch.float32)
                del x_for_sq
                # For alpha-select mode we additionally need the raw X matrix
                # to score candidate alphas. Subsample (smaller cap) to bound
                # forward-MSE cost — selection is a heuristic, not precise eval.
                if sq_alpha_grid is not None:
                    if x_cached.shape[0] > sq_score_token_cap:
                        perm2 = torch.randperm(x_cached.shape[0], device=x_cached.device)[:sq_score_token_cap]
                        x_for_score = x_cached.index_select(0, perm2).contiguous()
                    else:
                        x_for_score = x_cached.contiguous()
        else:
            H, n_tokens = collect_layer_gram(
                policy,
                normalized_samples,
                layer_name,
                args.token_cap,
                rng,
                str(args.device),
            )
        basis = load_aspq_basis_for_layer(
            layer_name,
            args.metric_path,
            out_features=module.out_features,
            topk=args.metric_top_k,
            min_eig=args.min_eig,
            min_eig_relative=args.min_eig_relative,
        )
        if basis is None and args.missing_metric == "error":
            raise FileNotFoundError(f"No usable ASPQ metric found for layer '{layer_name}' in {args.metric_path}")
        U = basis[0] if basis is not None else None
        eigvals = basis[1] if basis is not None else None
        W = module.weight.detach().to(dtype=torch.float32, device=solve_device)
        H_dev = H.to(dtype=torch.float32, device=solve_device)
        U_dev = U.to(dtype=torch.float32, device=solve_device) if U is not None else None
        eigvals_dev = eigvals.to(dtype=torch.float32, device=solve_device) if eigvals is not None else None

        # Optional DuQuant rotation as preprocessing for ASPQ-GPTQ.
        # Compose: W' = R_out @ W @ R_combined_in, H' = R_combined_in.T @ H @ R_combined_in.
        # At deploy, runtime applies x' = x @ R_combined_in and y_restored = y @ R_out.
        # ASPQ basis U (output-side) is unaffected by input-side rotation.
        rotation_R = None
        rotation_R_out = None
        per_kind = _resolve_per_kind(args, layer_name)
        if args.duquant_rotation:
            if args.duquant_permute or args.duquant_output_rotation:
                # Full pack: permutation + input rotation + (optional) output rotation
                from gr00t.quantization.duquant_preprocess import pack_weight
                block_size_eff = per_kind["block_size"]
                block_out = per_kind["block_out"]
                rot_mode_eff = per_kind["rot_mode"]
                pack = pack_weight(
                    W,
                    block_size=int(block_size_eff),
                    block_out_size=int(block_out),
                    enable_permute=bool(args.duquant_permute),
                    lambda_smooth=0.15,
                    perm_score="weight",
                    rot_mode=rot_mode_eff,
                )
                in_features = int(W.shape[1])
                out_features = int(W.shape[0])
                # Build R_combined_in [in×in] = P @ block_diag(R_in_blocks)
                R_in_full = torch.zeros((in_features, in_features), dtype=torch.float32, device=solve_device)
                if pack.R_in_blocks:
                    blk = int(pack.meta.get("block_size", int(block_size_eff)))
                    for b, R in pack.R_in_blocks.items():
                        rs, re = b * blk, min((b + 1) * blk, in_features)
                        R_in_full[rs:re, rs:re] = torch.from_numpy(R[: (re - rs), : (re - rs)].astype("float32")).to(solve_device)
                if pack.perm is not None:
                    P = torch.zeros((in_features, in_features), dtype=torch.float32, device=solve_device)
                    perm_t = torch.from_numpy(pack.perm.astype("int64")).to(solve_device)
                    P[perm_t, torch.arange(in_features, device=solve_device)] = 1.0
                    rotation_R = P @ R_in_full
                else:
                    rotation_R = R_in_full
                # Output rotation R_out_full [out×out] (block-diag)
                if args.duquant_output_rotation and pack.R_out_blocks:
                    R_out_full = torch.zeros((out_features, out_features), dtype=torch.float32, device=solve_device)
                    blk_o = int(pack.meta.get("block_out_size", block_out))
                    for b, R in pack.R_out_blocks.items():
                        rs, re = b * blk_o, min((b + 1) * blk_o, out_features)
                        R_out_full[rs:re, rs:re] = torch.from_numpy(R[: (re - rs), : (re - rs)].astype("float32")).to(solve_device)
                    rotation_R_out = R_out_full
                # Apply: W_packed = R_out @ (W @ R_combined_in); H_packed = R_combined_in.T @ H @ R_combined_in
                W = W @ rotation_R
                if rotation_R_out is not None:
                    W = rotation_R_out @ W
                H_dev = rotation_R.t() @ H_dev @ rotation_R
            else:
                # Original code path: input rotation only, no permutation
                from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only
                R_cpu, _perm = compute_duquant_rotation_only(
                    W, block_size=int(per_kind["block_size"]), enable_permute=False,
                    rot_mode=per_kind["rot_mode"],
                )
                rotation_R = R_cpu.to(dtype=torch.float32, device=solve_device)
                W = W @ rotation_R
                H_dev = rotation_R.t() @ H_dev @ rotation_R
            if x_calib_stats is not None:
                # Smooth calib stats also need rotation-equivalent transform.
                # Since x' = x @ R, the per-channel |x'| statistics differ from |x|.
                # Recompute from cached_inputs after rotation if available.
                if cached_inputs is not None:
                    x_cached_dev = cached_inputs[layer_name].to(device=solve_device, dtype=torch.float32)
                    x_rotated = x_cached_dev @ rotation_R
                    n_tokens_local = int(cached_tokens[layer_name])
                    sq_active = sq_alpha_env > 0.0 or sq_alpha_grid is not None
                    if sq_active:
                        if x_rotated.shape[0] > sq_token_cap:
                            perm = torch.randperm(x_rotated.shape[0], device=x_rotated.device)[:sq_token_cap]
                            x_for_sq = x_rotated.index_select(0, perm)
                        else:
                            x_for_sq = x_rotated
                        x_calib_stats = torch.quantile(x_for_sq.abs(), 0.99, dim=0).to(dtype=torch.float32)
                        if sq_alpha_grid is not None:
                            if x_rotated.shape[0] > sq_score_token_cap:
                                perm2 = torch.randperm(x_rotated.shape[0], device=x_rotated.device)[:sq_score_token_cap]
                                x_for_score = x_rotated.index_select(0, perm2).contiguous()
                            else:
                                x_for_score = x_rotated.contiguous()
                        del x_rotated, x_for_sq
                    del x_cached_dev
        gptq_damp_eff = per_kind["gptq_damp"]
        weight_group_size_eff = int(per_kind["group_size"])
        if sq_alpha_grid is not None and x_for_score is not None and x_calib_stats is not None:
            rec = solve_aspq_gptq_weight_with_alpha_select(
                W,
                H_dev,
                bits=args.weight_bits,
                U=U_dev,
                eigvals=eigvals_dev,
                block_size=args.gptq_block_size,
                damp_percent=gptq_damp_eff,
                min_eig=args.min_eig,
                x_calib_stats=x_calib_stats,
                x_calib_inputs=x_for_score,
                sq_alpha_grid=sq_alpha_grid,
                sq_select_mode=sq_select_mode_env,
                sq_clamp=(sq_clamp_lo_env, sq_clamp_hi_env),
                weight_group_size=weight_group_size_eff,
                err_comp_gamma=float(args.gptq_err_comp_gamma),
            )
        else:
            rec = solve_aspq_gptq_weight(
                W,
                H_dev,
                bits=args.weight_bits,
                U=U_dev,
                eigvals=eigvals_dev,
                block_size=args.gptq_block_size,
                damp_percent=gptq_damp_eff,
                min_eig=args.min_eig,
                x_calib_stats=x_calib_stats,
                sq_alpha=sq_alpha_env,
                sq_clamp=(sq_clamp_lo_env, sq_clamp_hi_env),
                weight_group_size=weight_group_size_eff,
                err_comp_gamma=float(args.gptq_err_comp_gamma),
            )
        aspq_rank = rec.rank
        rec_dict = {
            "baseline_q": rec.baseline_q.to(dtype=save_dtype).cpu().contiguous(),
            "U_int8": rec.U_int8.cpu().contiguous(),
            "U_scale": rec.U_scale.to(dtype=save_dtype).cpu().contiguous(),
            "action_q": rec.action_q.to(dtype=save_dtype).cpu().contiguous(),
            "weight_bits": int(args.weight_bits),
            "n_calib_tokens": int(n_tokens),
            "aspq_rank": aspq_rank,
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
            "format_version": 3 if rec.smooth_scale is not None else 2,
        }
        if rec.smooth_scale is not None:
            rec_dict["smooth_scale"] = rec.smooth_scale.to(dtype=save_dtype).cpu().contiguous()
        if rotation_R is not None:
            # Save as FP16 (rotation matrices are orthonormal so values in [-1, 1])
            rec_dict["duquant_rotation"] = rotation_R.to(dtype=torch.float16).cpu().contiguous()
        if rotation_R_out is not None:
            rec_dict["duquant_rotation_out"] = rotation_R_out.to(dtype=torch.float16).cpu().contiguous()
        if rec.sq_stats is not None:
            rec_dict["sq_stats"] = rec.sq_stats
            sel = rec.sq_stats.get("selected_alpha")
            sel_mode = rec.sq_stats.get("select_mode", "fixed")
            sel_str = (
                f" SELECT[{sel_mode}] alpha={sel}"
                if sel is not None
                else ""
            )
            print(
                f"[ASPQ-GPTQ-SQ] {layer_name}:"
                f"{sel_str} "
                f"med_raw={rec.sq_stats.get('median_raw', float('nan')):.4f} "
                f"min={rec.sq_stats.get('min_clamped', float('nan')):.3f} max={rec.sq_stats.get('max_clamped', float('nan')):.3f} "
                f"q1={rec.sq_stats.get('q1_unclamped', float('nan')):.3f} q99={rec.sq_stats.get('q99_unclamped', float('nan')):.3f} "
                f"frac@lo={rec.sq_stats.get('frac_at_lo', float('nan')):.2%} frac@hi={rec.sq_stats.get('frac_at_hi', float('nan')):.2%}"
            )
        records[layer_name] = rec_dict
        print(
            f"[ASPQ-GPTQ] saved {layer_name} "
            f"(shape={tuple(rec.baseline_q.shape)} calib_tokens={n_tokens} aspq_rank={aspq_rank} "
            f"U_int8={tuple(rec.U_int8.shape)} action_q={tuple(rec.action_q.shape)})"
        )

    payload = {
        "__meta__": {
            "checkpoint": args.checkpoint,
            "metric_path": args.metric_path,
            "weight_bits": int(args.weight_bits),
            "num_samples": int(args.num_samples),
            "token_cap": int(args.token_cap),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
            "gptq_err_comp_gamma": float(args.gptq_err_comp_gamma),
            "missing_metric": args.missing_metric,
            "save_dtype": args.save_dtype,
        }
    }
    payload.update(records)
    # Atomic save: write to .tmp then rename, so a concurrent reader (e.g.
    # eval bench launched against the same path) never sees a half-written file.
    tmp_path = str(out_path) + f".tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"[ASPQ-GPTQ] wrote {out_path}")


if __name__ == "__main__":
    main()
