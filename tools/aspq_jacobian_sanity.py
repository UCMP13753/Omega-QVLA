"""ASPQ sanity check.

Goal: verify the Action-Subspace-Projected Quantization motivation on a FP
GR00T policy WITHOUT touching the quant pipeline yet.

For each target Linear layer l we form
    M_l = E_s[ J_l(s)^T J_l(s) ]   # output-side action metric  (d_out x d_out)
    H_l = E_s[ X_l(s)^T X_l(s) ]   # input  Gram                (d_in  x d_in)

where J_l(s) = d a(s) / d h_l(s)  and h_l = X_l W_l^T (post-Linear output).

Cheap stochastic estimator (no need to materialise full J):
    g(s, u) = autograd.grad( <u, a(s)>, h_l(s) )      with u ~ N(0, I_{d_a})
    => E_{s,u}[ g g^T ] = M_l    (unbiased)

We accumulate per-layer:
    tr_M_l, diag(M_l)                       (always; cheap)
    full M_l                                (only for a small set of layers we
                                             want to inspect spectrum on)
    tr_H_l, diag(H_l)                       (always; cheap)

Scoring
-------
Under a layer-uniform quant noise model  E[Δ_l Δ_l^T] = sigma^2 I,
    E[ || J_l Δ_l X_l ||_F^2 ] = sigma^2 * tr(M_l) * tr(H_l) * (1/d_out)
so the per-layer ASPQ score (rank-equivalent to action error contribution
under uniform Δ) is

    score_l := tr(M_l) * tr(H_l) / d_out

We compare the ranking of score_l against the **observed** individual-quant
action RMSE in scenario_summary.csv, and we measure the effective rank of
M_l for a sub-set of layers (= 'how low-dimensional is the action subspace
seen at layer l').

Outputs (under --output-dir):
    aspq_per_layer.csv
    aspq_subset_spectrum.json     (top-K selected layers' eigenvalues)
    plots/score_vs_rmse.png
    plots/top20_score.png
    plots/effective_rank.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.autograd as autograd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from the existing scan script.
from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    DEFAULT_EXCLUDE_REGEX,
    DEFAULT_INCLUDE_REGEX,
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    seed_everything,
)
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE, unsqueeze_dict_values  # noqa: E402
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402

DEFAULT_LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/ceph/workspace/xinyu/LIBERO")
DEFAULT_LIBERO_DATASET = f"{DEFAULT_LIBERO_ROOT}/datasets/lerobot_libero_10"
DEFAULT_QUANTVLA_ROOT = os.environ.get("QUANTVLA_ROOT", "/ceph/workspace/xinyu/custon_asr")
DEFAULT_SCAN_DIR = (
    f"{DEFAULT_QUANTVLA_ROOT}/results/layerwise_quant_2gpu_taskwise_s10_w3a8_gpu67"
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-source", default="libero", choices=["libero"])
    p.add_argument("--dataset-path", default=DEFAULT_LIBERO_DATASET)
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10,
                   help="LIBERO observations to use (one per task is typical).")
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task",
                   choices=["sequential", "one_per_task"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--include-regex", default=DEFAULT_INCLUDE_REGEX)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)

    p.add_argument("--random-dirs", type=int, default=8,
                   help="Random Gaussian probes for stochastic Jacobian estimator.")
    p.add_argument("--token-cap", type=int, default=512,
                   help="Cap tokens kept per layer per sample (random subsample).")
    p.add_argument("--full-spectrum-top-k", type=int, default=20,
                   help="How many layers (top by RMSE) to keep full M_l for spectrum.")
    p.add_argument("--full-spectrum-extra", type=int, default=10,
                   help="Random extra layers for spectrum baseline.")

    p.add_argument("--scan-dir",
                   default=DEFAULT_SCAN_DIR,
                   help="Existing layerwise scan directory (for RMSE join).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--plot-only", action="store_true",
                   help="Only regenerate plots from output-dir/aspq_per_layer.csv.")
    p.add_argument("--metric-output", default="",
                   help="Optional .pt path to save ASPQ eigenspaces for DuQuant.")
    p.add_argument("--metric-top-k", type=int, default=64,
                   help="Top eigenvectors per layer to save in --metric-output.")
    p.add_argument("--metric-all-layers", action="store_true",
                   help="Store full M_l for every selected layer so --metric-output covers all quantized layers.")
    p.add_argument("--sigma-mode", default="none",
                   choices=["none", "min_max", "mean_std"],
                   help="Per-action-dim weighting for the Hutchinson estimator. "
                        "'none' = E[J^T J] (default, current behaviour). "
                        "'min_max' = E[J^T diag(((max-min)/2)^2) J]. "
                        "'mean_std' = E[J^T diag(std^2) J]. "
                        "Reads stats from --checkpoint/experiment_cfg/metadata.json.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hooked policy forward (with grad)
# ---------------------------------------------------------------------------

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


def family_of(name: str) -> str:
    if "language_model" in name:
        return "llm"
    if "transformer_blocks" in name:
        return "dit"
    return "other"


def op_class_of(name: str) -> str:
    if "language_model" in name:
        for op in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            if name.endswith(f".{op}"):
                return f"llm.{op}"
        return "llm.other"
    if "transformer_blocks" in name:
        for op in ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "ff.net.0.proj", "ff.net.2"):
            if op in name:
                return f"dit.{op}"
        return "dit.other"
    return "other"


def short_layer_name(name: str) -> str:
    return (
        name.replace("backbone.eagle_model.language_model.model.layers.", "llm.L")
        .replace("backbone.eagle_model.language_model.layers.", "llm.L")
        .replace("action_head.model.transformer_blocks.", "dit.B")
    )


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    sx = np.asarray(xs)
    sy = np.asarray(ys)
    rx = np.argsort(np.argsort(sx))
    ry = np.argsort(np.argsort(sy))
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass
class LayerAccum:
    d_in: int = 0
    d_out: int = 0
    n_tokens: int = 0
    tr_M: float = 0.0           # sum of column-wise squared grads
    tr_H: float = 0.0           # sum of column-wise squared activations
    diag_M: torch.Tensor | None = None   # CPU float32 [d_out]
    diag_H: torch.Tensor | None = None   # CPU float32 [d_in]
    full_M: torch.Tensor | None = None   # CPU float64 [d_out, d_out]; None if not selected


def unwrap_no_grad_methods(model) -> list[tuple[Any, str, Any]]:
    """Strip @torch.no_grad decorators from inference paths so autograd can
    flow back from the predicted action. Returns a list of (cls, attr, orig)
    tuples for restoration."""
    from gr00t.model.action_head.flow_matching_action_head import (
        FlowmatchingActionHead,
    )
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import (
        Eagle2_5_VLForConditionalGeneration,
    )
    patched = []
    for cls in (FlowmatchingActionHead, Eagle2_5_VLForConditionalGeneration):
        for attr in ("get_action", "forward", "generate"):
            if not hasattr(cls, attr):
                continue
            fn = getattr(cls, attr)
            wrapped = getattr(fn, "__wrapped__", None)
            if wrapped is None:
                continue
            patched.append((cls, attr, fn))
            setattr(cls, attr, wrapped)
            print(f"[ASPQ] unwrapped no_grad on {cls.__name__}.{attr}")
    return patched


def restore_no_grad_methods(patched):
    for cls, attr, fn in patched:
        setattr(cls, attr, fn)


def load_sigma_vec(checkpoint: str, mode: str) -> Optional[torch.Tensor]:
    """Load per-action-dim sigma from checkpoint metadata for sigma-weighted M.

    Returns sigma vector of shape (max_action_dim=32,) or None when mode='none'.
    Action dims used by LIBERO data config are x,y,z,roll,pitch,yaw,gripper (7);
    remaining dims (7..31) are zero-padded — the model output is padded too, so
    a zero sigma there means those padding entries don't contribute to M.
    """
    if mode == "none":
        return None
    import json
    meta_path = Path(checkpoint) / "experiment_cfg" / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    # Walk to action stats: typically meta["new_embodiment"]["statistics"]["action"][dim][stat]
    stats = None
    for emb in meta.values():
        if isinstance(emb, dict) and "statistics" in emb:
            stats = emb["statistics"].get("action")
            break
    if stats is None:
        raise RuntimeError(f"could not find action statistics in {meta_path}")
    dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    sigma = []
    for d in dims:
        s = stats[d]
        if mode == "min_max":
            v = (float(s["max"][0]) - float(s["min"][0])) / 2.0
        elif mode == "mean_std":
            # Goal config: gripper still uses min_max even when others use mean_std.
            # For sigma-weighting that's fine — both give per-dim physical scale.
            if d == "gripper":
                v = (float(s["max"][0]) - float(s["min"][0])) / 2.0
            else:
                v = float(s["std"][0])
        else:
            raise ValueError(f"bad sigma-mode {mode}")
        sigma.append(v)
    # Pad to max_action_dim=32 with zeros (model output is padded too)
    sigma_full = sigma + [0.0] * (32 - len(sigma))
    return torch.tensor(sigma_full, dtype=torch.float32)


def normalized_input_no_inference(policy, sample_obs):
    """Mimic policy.get_action's preprocessing without inference_mode."""
    obs_copy = sample_obs.copy()
    obs_copy = unsqueeze_dict_values(obs_copy)
    for k, v in obs_copy.items():
        if not isinstance(v, np.ndarray):
            obs_copy[k] = np.array(v)
    normalized_input = policy.apply_transforms(obs_copy)
    return normalized_input


def collect_one_sample(
    policy,
    sample,
    layer_names: Sequence[str],
    accums: dict[str, LayerAccum],
    full_M_layers: set[str],
    n_random_dirs: int,
    token_cap: int,
    rng: np.random.Generator,
    sigma_vec: Optional[torch.Tensor] = None,
) -> int:
    """Forward+backward one sample, accumulate per-layer M, H. Returns d_a."""
    modules = {n: get_named_module(policy.model, n) for n in layer_names}
    in_cache: dict[str, torch.Tensor] = {}
    out_cache: dict[str, torch.Tensor] = {}
    # Per-step accumulators: each forward call (one denoising step) appends.
    # When PER_STEP_AGG env is set, M is computed by summing across all steps;
    # otherwise we keep the last-step-only behaviour (back compat).
    in_cache_steps: dict[str, list[torch.Tensor]] = {}
    out_cache_steps: dict[str, list[torch.Tensor]] = {}
    per_step_agg = os.environ.get("GR00T_ASPQ_PER_STEP_AGG", "0") not in ("0", "false", "False")

    def make_pre_hook(name):
        def hook(_m, inputs):
            in_cache[name] = inputs[0]
            if per_step_agg:
                in_cache_steps.setdefault(name, []).append(inputs[0])
        return hook

    def make_post_hook(name):
        def hook(_m, _inputs, output):
            out_cache[name] = output
            if per_step_agg:
                out_cache_steps.setdefault(name, []).append(output)
        return hook

    handles = []
    for n, m in modules.items():
        handles.append(m.register_forward_pre_hook(make_pre_hook(n)))
        handles.append(m.register_forward_hook(make_post_hook(n)))

    try:
        seed_everything(sample["seed"])
        normalized = normalized_input_no_inference(policy, sample["obs"])

        with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = policy.model.get_action(normalized)
        a_pred = model_pred["action_pred"].float()
        a_flat = a_pred.reshape(-1)
        d_a = int(a_flat.numel())

        # ----- accumulate H (no grad needed) ---------------------------------
        for n in layer_names:
            x = in_cache[n].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            # token subsample (deterministic per sample for repeatability)
            if x.shape[0] > token_cap:
                idx = torch.from_numpy(
                    rng.choice(x.shape[0], size=token_cap, replace=False)
                ).to(x.device)
                x = x.index_select(0, idx)
                in_cache[n] = (in_cache[n].reshape(-1, in_cache[n].shape[-1])
                               .index_select(0, idx))
            ac = accums[n]
            if ac.d_in == 0:
                ac.d_in = int(x.shape[1])
                ac.diag_H = torch.zeros(ac.d_in, dtype=torch.float64)
            ac.n_tokens += int(x.shape[0])
            ac.tr_H += float((x * x).sum().item())
            ac.diag_H += (x * x).sum(dim=0).double().cpu()

        # ----- accumulate M via stochastic Jacobian estimator ---------------
        # If sigma_vec is given, sample u ~ N(0, diag(sigma^2)) so that
        # E[g g^T] = J^T diag(sigma^2) J = (diag(sigma) J)^T (diag(sigma) J).
        # That makes M align with physical action MSE (post-denormalization)
        # rather than normalized-action MSE.
        sigma_tile = None
        if sigma_vec is not None:
            # a_pred shape is (B, action_horizon, max_action_dim=32). Tile sigma along (B, H).
            a_shape = a_pred.shape
            sigma_tile = sigma_vec.to(a_flat.device, dtype=a_flat.dtype)
            sigma_tile = sigma_tile.expand(a_shape).reshape(-1).contiguous()
        # Build the list of output tensors to differentiate against.
        # Default: only last step's output (back-compat, M = E[J_last^T J_last]).
        # PER_STEP_AGG: all 8 steps' outputs (M = E[Σ_t J_t^T J_t], captures the
        # full quantization-noise-to-action propagation across denoising rollout).
        if per_step_agg and out_cache_steps:
            num_steps = max(len(out_cache_steps[n]) for n in layer_names if n in out_cache_steps)
            # Flat list: [layer0_step0, layer0_step1, ..., layer1_step0, ...]
            outs_flat: list[torch.Tensor] = []
            outs_index: dict[str, list[int]] = {}
            for n in layer_names:
                steps = out_cache_steps.get(n, [])
                outs_index[n] = list(range(len(outs_flat), len(outs_flat) + len(steps)))
                outs_flat.extend(steps)
        else:
            num_steps = 1
            outs_flat = [out_cache[n] for n in layer_names]
            outs_index = {n: [i] for i, n in enumerate(layer_names)}

        for r in range(n_random_dirs):
            u = torch.randn_like(a_flat)
            if sigma_tile is not None:
                u = u * sigma_tile
            scalar = (u * a_flat).sum()
            grads = autograd.grad(
                scalar, outs_flat,
                retain_graph=(r < n_random_dirs - 1),
                allow_unused=True,
            )
            # Per-direction, per-layer normaliser. With per_step_agg, layer n
            # contributes len(outs_index[n]) outer products per random direction
            # (one per denoising step where it fires). Divide by that count so
            # each layer's M_l is on the same eigenvalue scale as legacy
            # single-step mode regardless of how many times it was called
            # (eigenvectors are scale-invariant either way; this only matters
            # for cross-layer tr_M comparisons used in scoring).
            for n in layer_names:
                ac = accums[n]
                idxs = outs_index.get(n, [])
                if not idxs:
                    continue
                w_per_step = 1.0 / (n_random_dirs * (len(idxs) if per_step_agg else 1))
                for idx in idxs:
                    g = grads[idx]
                    if g is None:
                        continue
                    gflat = g.detach()
                    gflat = gflat.reshape(-1, gflat.shape[-1]).float()
                    if gflat.shape[0] > token_cap:
                        sub_idx = torch.from_numpy(
                            rng.choice(gflat.shape[0], size=token_cap, replace=False)
                        ).to(gflat.device)
                        gflat = gflat.index_select(0, sub_idx)
                    if ac.d_out == 0:
                        ac.d_out = int(gflat.shape[1])
                        ac.diag_M = torch.zeros(ac.d_out, dtype=torch.float64)
                        if n in full_M_layers:
                            ac.full_M = torch.zeros(
                                (ac.d_out, ac.d_out), dtype=torch.float64
                            )
                    ac.tr_M += float((gflat * gflat).sum().item()) * w_per_step
                    ac.diag_M += (gflat * gflat).sum(dim=0).double().cpu() * w_per_step
                    if ac.full_M is not None:
                        contrib = (gflat.T @ gflat).double().cpu()
                        ac.full_M += contrib * w_per_step

    finally:
        for h in handles:
            h.remove()

    return d_a


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_vs_rmse(rows, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys, fams = [], [], []
    for r in rows:
        if r["action_fp_rmse_individual"] is None or r["score"] <= 0 or r["action_fp_rmse_individual"] <= 0:
            continue
        xs.append(r["score"]); ys.append(r["action_fp_rmse_individual"])
        fams.append(r["family"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for f, c in [("llm", "#d95f0e"), ("dit", "#2c7fb8")]:
        sx = [x for x, fm in zip(xs, fams) if fm == f]
        sy = [y for y, fm in zip(ys, fams) if fm == f]
        ax.scatter(sx, sy, s=14, color=c, alpha=0.7, label=f)
    if xs and ys:
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        ax.text(
            0.5, 0.5,
            "No positive RMSE join found.\nPass --scan-dir with scenario_summary.csv.",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_xlabel("ASPQ score: tr(M) * tr(H) / d_out")
    ax.set_ylabel("Observed individual W3A8 action_fp_rmse")
    ax.set_title("ASPQ score vs measured per-layer RMSE")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_top20_score(rows, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: -r["score"])[:20]
    names = [r["layer_name"]
             .replace("backbone.eagle_model.language_model.model.layers.", "llm.L")
             .replace("action_head.model.transformer_blocks.", "dit.B")
             for r in rows]
    vals = [r["score"] for r in rows]
    cols = ["#d95f0e" if r["family"] == "llm" else "#2c7fb8" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 8))
    y = list(range(len(names)))[::-1]
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ASPQ score (log)")
    ax.set_title("Top-20 layers by ASPQ score (FP-only diagnostic)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_eff_rank(spectrum: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = list(spectrum.keys())
    eff_ratios = [spectrum[l]["eff_rank"] / spectrum[l]["d_out"] for l in layers]
    fams = [family_of(l) for l in layers]
    fig, ax = plt.subplots(figsize=(8, 5))
    for f, c in [("llm", "#d95f0e"), ("dit", "#2c7fb8")]:
        v = [e for e, fm in zip(eff_ratios, fams) if fm == f]
        ax.hist(v, bins=20, alpha=0.55, color=c, label=f"{f} (n={len(v)})")
    ax.set_xlabel("effective_rank(M_l) / d_out")
    ax.set_ylabel("count of layers")
    ax.set_title("Action-subspace dimension is small relative to layer width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_takeaway(rows, spectrum: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = [
        r for r in rows
        if r["action_fp_rmse_individual"] is not None
        and r["score"] > 0
        and r["action_fp_rmse_individual"] > 0
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: global diagnostic scatter.
    ax = axes[0]
    for f, c in [("llm", "#d95f0e"), ("dit", "#2c7fb8")]:
        rs = [r for r in paired if r["family"] == f]
        ax.scatter(
            [r["score"] for r in rs],
            [r["action_fp_rmse_individual"] for r in rs],
            s=18,
            color=c,
            alpha=0.72,
            label=f"{f} (n={len(rs)})",
        )
    if paired:
        rho = spearman([r["score"] for r in paired], [r["action_fp_rmse_individual"] for r in paired])
        top = sorted(paired, key=lambda r: -r["action_fp_rmse_individual"])[:3]
        for r in top:
            ax.annotate(
                short_layer_name(r["layer_name"]),
                (r["score"], r["action_fp_rmse_individual"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.text(0.03, 0.97, f"Spearman rho = {rho:.3f}", transform=ax.transAxes, va="top", fontsize=10)
    if paired:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.text(
            0.5, 0.5,
            "No RMSE join.\nASPQ scores were computed,\nbut correlation needs scan-dir.",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_xlabel("ASPQ score")
    ax.set_ylabel("Measured W3A8 action RMSE")
    ax.set_title("Score predicts sensitive layers")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: sorted effective-rank ratios for layers where full M was stored.
    ax = axes[1]
    spec_items = sorted(
        [(name, val) for name, val in spectrum.items() if val.get("eff_rank_ratio") is not None],
        key=lambda kv: kv[1]["eff_rank_ratio"],
    )
    if spec_items:
        vals = [100.0 * val["eff_rank_ratio"] for _, val in spec_items]
        cols = ["#d95f0e" if family_of(name) == "llm" else "#2c7fb8" for name, _ in spec_items]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=cols, alpha=0.85)
        med = float(np.median(vals))
        ax.axhline(med, color="black", linewidth=1.0, linestyle="--")
        ax.text(0.02, 0.94, f"median = {med:.2f}%", transform=ax.transAxes, va="top", fontsize=10)
        ax.set_xlim(-0.7, len(vals) - 0.3)
    else:
        ax.text(
            0.5, 0.5,
            "No full M stored.\nRe-run with spectrum layers selected.",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_xlabel("Full-M layers, sorted")
    ax.set_ylabel("eff_rank(M) / d_out (%)")
    ax.set_title("Action subspace is tiny")
    ax.grid(True, axis="y", alpha=0.25)

    # Panel 3: within-op rank correlation; this removes gross op-family scale effects.
    ax = axes[2]
    op_rows = {}
    for r in paired:
        op_rows.setdefault(op_class_of(r["layer_name"]), []).append(r)
    op_stats = []
    for op, rs in op_rows.items():
        if len(rs) < 3:
            continue
        rho = spearman([r["score"] for r in rs], [r["action_fp_rmse_individual"] for r in rs])
        if np.isfinite(rho):
            op_stats.append((op, rho, len(rs)))
    op_stats = sorted(op_stats, key=lambda item: item[1])
    if op_stats:
        y = np.arange(len(op_stats))
        vals = [rho for _, rho, _ in op_stats]
        labels = [f"{op} ({n})" for op, _, n in op_stats]
        cols = ["#d95f0e" if op.startswith("llm.") else "#2c7fb8" for op, _, _ in op_stats]
        ax.barh(y, vals, color=cols, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-1.0, 1.0)
    else:
        ax.text(
            0.5, 0.5,
            "No within-op correlation\nwithout RMSE join.",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_xlabel("Spearman rho within op class")
    ax.set_title("Correlation is stronger within type")
    ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("ASPQ sanity check: action sensitivity is predictable and low-rank", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _maybe_float(v):
    if v is None or v == "":
        return None
    return float(v)


def _maybe_int(v):
    if v is None or v == "":
        return None
    return int(float(v))


def load_existing_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "layer_name": row["layer_name"],
                "family": row.get("family") or family_of(row["layer_name"]),
                "d_in": _maybe_int(row.get("d_in")),
                "d_out": _maybe_int(row.get("d_out")),
                "n_calib_tokens": _maybe_int(row.get("n_calib_tokens")),
                "tr_M": _maybe_float(row.get("tr_M")),
                "tr_H": _maybe_float(row.get("tr_H")),
                "score": _maybe_float(row.get("score")) or 0.0,
                "eff_rank": _maybe_float(row.get("eff_rank")),
                "eff_rank_ratio": _maybe_float(row.get("eff_rank_ratio")),
                "action_fp_rmse_individual": _maybe_float(row.get("action_fp_rmse_individual")),
            })
    return rows


def regenerate_plots(out: Path):
    rows = load_existing_rows(out / "aspq_per_layer.csv")
    spectrum_path = out / "aspq_subset_spectrum.json"
    spectrum = {}
    if spectrum_path.exists():
        with open(spectrum_path) as f:
            spectrum = json.load(f)

    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_score_vs_rmse(rows, plots_dir / "score_vs_rmse.png")
    plot_top20_score(rows, plots_dir / "top20_score.png")
    if spectrum:
        plot_eff_rank(spectrum, plots_dir / "effective_rank.png")
    plot_takeaway(rows, spectrum, plots_dir / "aspq_takeaway.png")
    print(f"[ASPQ] regenerated plots in {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    if args.plot_only:
        regenerate_plots(out)
        return
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --plot-only is set")
    rng = np.random.default_rng(args.seed)

    print(f"[ASPQ] loading FP policy ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    layer_names = resolve_target_layers(policy, args)
    print(f"[ASPQ] target layers: {len(layer_names)}")

    # Load existing scan RMSE table for join
    rmse_lookup: dict[str, float] = {}
    scan_csv = Path(args.scan_dir) / "scenario_summary.csv"
    if scan_csv.exists():
        for row in csv.DictReader(open(scan_csv)):
            if row["mode"] == "individual" and row["action_fp_rmse"]:
                rmse_lookup[row["focus_layer"]] = float(row["action_fp_rmse"])
        print(f"[ASPQ] joined {len(rmse_lookup)} per-layer individual RMSEs from scan")

    # Pick which layers get full M_l
    full_M_layers: set[str] = set()
    if args.metric_all_layers:
        full_M_layers = set(layer_names)
        print(
            "[ASPQ] metric-all-layers enabled; storing full M for every selected layer. "
            "This can use substantial CPU RAM."
        )
    elif rmse_lookup:
        ranked = sorted(rmse_lookup.items(), key=lambda kv: -kv[1])
        top = [n for n, _ in ranked[: args.full_spectrum_top_k]]
        rest = [n for n in layer_names if n not in set(top)]
        extras = list(rng.choice(rest, size=min(args.full_spectrum_extra, len(rest)),
                                 replace=False)) if rest else []
        full_M_layers = set(top) | set(extras)
    else:
        n_full = min(args.full_spectrum_top_k + args.full_spectrum_extra, len(layer_names))
        if n_full > 0:
            full_M_layers = set(rng.choice(layer_names, size=n_full, replace=False).tolist())
        print(
            "[ASPQ] no RMSE scan joined; storing full M for a random subset "
            f"of {len(full_M_layers)} layers"
        )
    print(f"[ASPQ] storing full M for {len(full_M_layers)} layers")

    # Load samples
    print(f"[ASPQ] loading {args.num_samples} LIBERO observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    print(f"[ASPQ] got {len(samples)} samples")

    # Make sure model parameters require_grad so the autograd graph forms
    for p in policy.model.parameters():
        p.requires_grad_(False)
    # We just need outputs to require grad. Activate grad on the embedding /
    # input only. Easier: enable grad on all params; we'll never call .grad on them.
    for p in policy.model.parameters():
        p.requires_grad_(True)
    policy.model.eval()  # disables dropout etc; eval is fine with grad enabled
    patched = unwrap_no_grad_methods(policy.model)

    accums: dict[str, LayerAccum] = {n: LayerAccum() for n in layer_names}

    sigma_vec = load_sigma_vec(args.checkpoint, args.sigma_mode)
    if sigma_vec is not None:
        non_zero = [(i, float(v)) for i, v in enumerate(sigma_vec) if v > 0]
        print(f"[ASPQ] sigma-mode={args.sigma_mode}: {non_zero}")

    d_a_seen = None
    for i, s in enumerate(samples):
        print(f"[ASPQ] sample {i+1}/{len(samples)}")
        d_a = collect_one_sample(
            policy, s, layer_names, accums, full_M_layers,
            args.random_dirs, args.token_cap, rng,
            sigma_vec=sigma_vec,
        )
        if d_a_seen is None:
            print(f"[ASPQ] action dim d_a = {d_a}")
            d_a_seen = d_a
        torch.cuda.empty_cache()

    # ----- summarize ---------------------------------------------------------
    rows = []
    spectrum: dict[str, dict] = {}
    metric_records: dict[str, dict] = {}
    for n in layer_names:
        ac = accums[n]
        if ac.n_tokens == 0:
            continue
        tr_M_norm = ac.tr_M / max(ac.n_tokens, 1)
        tr_H_norm = ac.tr_H / max(ac.n_tokens, 1)
        score = tr_M_norm * tr_H_norm / max(ac.d_out, 1)
        eff_rank = None; eff_rank_ratio = None
        if ac.full_M is not None:
            M = ac.full_M / max(ac.n_tokens, 1)
            # Eigendecomposition on GPU is dramatically faster for large d_out.
            # We cast to float32 on GPU; precision is fine for top-K eigvals.
            try:
                if torch.cuda.is_available():
                    M_dev = M.to(device="cuda", dtype=torch.float32)
                    eigvals_t, U_t = torch.linalg.eigh(M_dev)
                    eigvals_t = eigvals_t.to(dtype=torch.float64).cpu()
                    U_t = U_t.to(dtype=torch.float64).cpu()
                    del M_dev
                    torch.cuda.empty_cache()
                else:
                    eigvals_t, U_t = torch.linalg.eigh(M)
                eigs = eigvals_t.numpy()
            except Exception:
                eigs_np, U_np = np.linalg.eigh(M.numpy())
                eigvals_t = torch.from_numpy(eigs_np)
                U_t = torch.from_numpy(U_np)
                eigs = eigs_np
            eigs = np.clip(eigs, 0.0, None)
            tr1 = float(eigs.sum())
            tr2 = float((eigs ** 2).sum())
            eff_rank = (tr1 ** 2) / max(tr2, 1e-30)
            eff_rank_ratio = eff_rank / ac.d_out
            order_t = torch.argsort(eigvals_t, descending=True)
            eigvals_sorted = torch.clamp(eigvals_t[order_t], min=0.0)
            U_sorted = U_t[:, order_t]
            if args.metric_output:
                keep = torch.isfinite(eigvals_sorted) & (eigvals_sorted > 0)
                if args.metric_top_k > 0:
                    top = min(int(args.metric_top_k), int(keep.sum().item()))
                    selected = torch.nonzero(keep, as_tuple=False).flatten()[:top]
                else:
                    selected = torch.nonzero(keep, as_tuple=False).flatten()
                metric_records[n] = {
                    "U": U_sorted[:, selected].to(torch.float32).contiguous().cpu(),
                    "eigvals": eigvals_sorted[selected].to(torch.float32).contiguous().cpu(),
                    "d_out": ac.d_out,
                    "n_calib_tokens": ac.n_tokens,
                    "eff_rank": eff_rank,
                    "eff_rank_ratio": eff_rank_ratio,
                }
            spectrum[n] = {
                "d_out": ac.d_out,
                "eff_rank": eff_rank,
                "eff_rank_ratio": eff_rank_ratio,
                "top10_eigs": [float(x) for x in eigs[::-1][:10]],
                "tr_M": tr1,
            }
        rows.append({
            "layer_name": n,
            "family": family_of(n),
            "d_in": ac.d_in,
            "d_out": ac.d_out,
            "n_calib_tokens": ac.n_tokens,
            "tr_M": tr_M_norm,
            "tr_H": tr_H_norm,
            "score": score,
            "eff_rank": eff_rank,
            "eff_rank_ratio": eff_rank_ratio,
            "action_fp_rmse_individual": rmse_lookup.get(n),
        })

    # CSV
    csv_path = out / "aspq_per_layer.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ASPQ] wrote {csv_path}")

    with open(out / "aspq_subset_spectrum.json", "w") as f:
        json.dump(spectrum, f, indent=2)

    if args.metric_output:
        metric_path = Path(args.metric_output)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(metric_records, metric_path)
        print(f"[ASPQ] wrote DuQuant metric eigenspaces for {len(metric_records)} layers to {metric_path}")

    # Spearman correlation
    pairs = [(r["score"], r["action_fp_rmse_individual"])
             for r in rows if r["action_fp_rmse_individual"] is not None]
    if pairs:
        sx = np.array([p[0] for p in pairs])
        sy = np.array([p[1] for p in pairs])
        rx = np.argsort(np.argsort(sx))
        ry = np.argsort(np.argsort(sy))
        rho = np.corrcoef(rx, ry)[0, 1]
        print(f"[ASPQ] Spearman(score, observed RMSE) = {rho:.3f}  over {len(pairs)} layers")

    plot_score_vs_rmse(rows, out / "plots" / "score_vs_rmse.png")
    plot_top20_score(rows, out / "plots" / "top20_score.png")
    if spectrum:
        plot_eff_rank(spectrum, out / "plots" / "effective_rank.png")
    plot_takeaway(rows, spectrum, out / "plots" / "aspq_takeaway.png")
    restore_no_grad_methods(patched)
    print(f"[ASPQ] done. results in {out}")


if __name__ == "__main__":
    main()
