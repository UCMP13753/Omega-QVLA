"""ASPQ-GPTQ runtime wrapper and solver.

This module intentionally lives alongside the existing DuQuant-based ASPQ path.
The old path stays intact; this one follows the ASPQ-GPTQ algorithm family:

1. Quantize the full weight with a standard GPTQ-style solver.
2. Quantize the ASPQ action subspace in the metric eigenbasis.
3. Replace only the action-sensitive subspace of the baseline solution.

When the ASPQ metric provides the full output eigenspace, this reduces to the
paper-style rotate -> GPTQ -> rotate-back recipe. When only a top-k subspace is
available, we keep the baseline GPTQ solution in the orthogonal complement and
apply the ASPQ correction inside span(U).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from .duquant_layers import (
    _aspq_basis_from_record,
    _load_aspq_record,
    _parse_per_layer_wbits,
    select_targets,
)
from .duquant_preprocess import (
    PercentileCalibrator,
    compute_mse_scales,
    fake_quantize_sym,
    qmax,
    sanitize_name,
)


_ASPQ_GPTQ_CACHE: Dict[str, Any] = {}


@dataclass
class AspqGptqConfig:
    enabled: Optional[bool] = None
    path: Optional[str] = None
    act_bits: Optional[int] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    weight_bits: Optional[int] = None
    missing: Optional[str] = None

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = os.environ.get("GR00T_ASPQ_GPTQ", "0") not in ("0", "false", "False")
        if self.path is None:
            self.path = os.environ.get("GR00T_ASPQ_GPTQ_PATH")
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("GR00T_ASPQ_GPTQ_ABITS", 8))
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("GR00T_ASPQ_GPTQ_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("GR00T_ASPQ_GPTQ_CALIB_STEPS", 32))
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("GR00T_ASPQ_GPTQ_WBITS_DEFAULT", 4))
        if self.missing is None:
            self.missing = os.environ.get("GR00T_ASPQ_GPTQ_MISSING", "error").lower()
        if self.enabled and not self.path:
            raise ValueError("GR00T_ASPQ_GPTQ=1 requires GR00T_ASPQ_GPTQ_PATH")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_quantized_container(path: Path) -> Any:
    cache_key = str(path.resolve())
    if cache_key in _ASPQ_GPTQ_CACHE:
        return _ASPQ_GPTQ_CACHE[cache_key]
    if path.suffix not in (".pt", ".pth"):
        raise ValueError(f"Unsupported ASPQ-GPTQ weight file: {path}")
    value = _torch_load(path)
    _ASPQ_GPTQ_CACHE[cache_key] = value
    return value


def _record_get(record: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    if isinstance(record, dict):
        for key in keys:
            if key in record:
                return record[key]
    return None


def _load_quantized_record(layer_name: str, weights_path: str) -> Optional[Any]:
    path = Path(weights_path)
    if path.is_dir():
        safe = sanitize_name(layer_name)
        for suffix in (".pt", ".pth"):
            candidate = path / f"{safe}{suffix}"
            if candidate.exists():
                return _load_quantized_container(candidate)
        return None

    container = _load_quantized_container(path)
    if not isinstance(container, dict):
        return container
    if any(key in container for key in ("baseline_q", "weight_q", "W_q", "quant_weight")):
        return container
    for key in (layer_name, sanitize_name(layer_name)):
        if key in container:
            return container[key]
    return None


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def _quantize_vector_sym(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    max_q = qmax(bits)
    return torch.clamp(torch.round(x / scale), -max_q - 1, max_q) * scale


def _prepare_hessian(
    H: torch.Tensor,
    damp_percent: float,
    reg_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize H, mark dead diag, add damping + absolute regularization.

    Two diagonal additions:
      - damp = damp_percent * mean(diag(H))   (relative; classic GPTQ)
      - reg_lambda                            (absolute; adds λI for L2 reg on
                                               (W-Q), i.e. solves min (W-Q)^T (H + λI)(W-Q))
    """
    H = 0.5 * (H + H.t())
    diag = torch.diag(H).clone()
    dead = diag <= 0
    if dead.any():
        H = H.clone()
        H[dead, dead] = 1.0
    mean_diag = float(diag[~dead].mean().item()) if (~dead).any() else 1.0
    damp = max(float(damp_percent), 0.0) * max(mean_diag, 1e-8)
    total_diag_add = damp + max(float(reg_lambda), 0.0)
    if total_diag_add > 0:
        idx = torch.arange(H.shape[0], device=H.device)
        H = H.clone()
        H[idx, idx] += total_diag_add
    return H, dead


def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    block_size: int = 128,
    damp_percent: float = 0.01,
    mse_alphas: Optional[Sequence[float]] = None,
    reg_lambda: float = 0.0,
    err_comp_gamma: float = 1.0,
    group_size: int = 0,
) -> torch.Tensor:
    """A practical GPTQ-style solver on the input Gram matrix H.

    We quantize each input block independently. This is exact when block_size
    covers the full input dimension and otherwise acts as a standard blockwise
    GPTQ approximation.

    ``mse_alphas`` overrides the per-row MSE candidate grid. Wider grids (e.g.
    up to 5.0) let rows with heavy outliers automatically choose aggressive
    clipping when it strictly reduces row-MSE — this is the principled
    AWQ-style replacement for ad-hoc per-row scale factors. Default falls back
    to ``compute_mse_scales``' built-in narrow grid.

    ``group_size`` (P1.3): if > 0 and < in_features, scales are recomputed every
    ``group_size`` input columns instead of once per block. The output stays a
    dense FP weight (the per-group scale information is "baked in" via the
    rounding), so the runtime needs no change.
    """
    if bits <= 0:
        return W.detach().clone()

    W = W.detach().to(dtype=torch.float32)
    h_dtype = torch.float32 if W.is_cuda else torch.float64
    H = H.detach().to(device=W.device, dtype=h_dtype)
    out_features, in_features = W.shape
    if H.shape != (in_features, in_features):
        raise ValueError(f"H has shape {tuple(H.shape)}, expected {(in_features, in_features)}")

    if block_size <= 0 or block_size >= in_features:
        block_size = in_features

    use_groups = 0 < int(group_size) < in_features
    group_scales: Dict[int, torch.Tensor] = {}
    if use_groups:
        # Precompute per-group per-row scales from the original W. Using the
        # original (not partially-quantized) W matches AWQ/GPTQ-grouped
        # convention and keeps the scale grid stable.
        for g_start in range(0, in_features, int(group_size)):
            g_end = min(g_start + int(group_size), in_features)
            group_scales[g_start] = compute_mse_scales(
                W[:, g_start:g_end], bits, alphas=mse_alphas
            ).to(dtype=W.dtype, device=W.device)

    Q = W.clone()
    gamma = float(err_comp_gamma)
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        W_block = W[:, start:end].clone()
        H_block = H[start:end, start:end].clone()
        H_block, dead = _prepare_hessian(H_block, damp_percent, reg_lambda=reg_lambda)
        if dead.any():
            W_block[:, dead] = 0

        try:
            chol = torch.linalg.cholesky(H_block)
            Hinv = torch.cholesky_inverse(chol).to(dtype=torch.float32)
        except RuntimeError:
            Hinv = torch.linalg.pinv(H_block).to(dtype=torch.float32)

        block_q = torch.zeros_like(W_block)
        if not use_groups:
            row_scales = compute_mse_scales(W_block, bits, alphas=mse_alphas).to(
                dtype=W_block.dtype, device=W_block.device
            )

        for col in range(W_block.shape[1]):
            denom = float(Hinv[col, col].item())
            if not torch.isfinite(Hinv[col, col]) or abs(denom) < 1e-12:
                denom = 1.0
            w_col = W_block[:, col]
            if use_groups:
                global_col = start + col
                g_start = (global_col // int(group_size)) * int(group_size)
                row_scales_g = group_scales[g_start]
            else:
                row_scales_g = row_scales
            q_col = _quantize_vector_sym(w_col, row_scales_g, bits)
            block_q[:, col] = q_col
            err = (w_col - q_col) / denom
            # err_comp_gamma scales the off-diagonal propagation; gamma=0
            # disables error compensation (equivalent to per-block-MSE RTN).
            if gamma > 0 and col + 1 < W_block.shape[1]:
                W_block[:, col + 1:] -= gamma * err[:, None] * Hinv[col, col + 1:][None, :]

        Q[:, start:end] = block_q
    return Q


@dataclass
class AspqGptqRecord:
    """Factored ASPQ-GPTQ outputs for one linear layer.

    Storage pieces (algorithm itself unchanged):
      * ``baseline_q`` ∈ ℝ^[O, I]   — full GPTQ-quantized weight (on the wbit
        integer grid, fp dense storage).
      * ``U_int8`` ∈ ℤ^[O, k]       — int8-quantized action eigenbasis.
      * ``U_scale`` ∈ ℝ^[k]         — per-column dequant scale for U_int8.
      * ``action_q`` ∈ ℝ^[k, I]     — Uᵀ W after the rotated-GPTQ + un-weight
        step (i.e. the quantized projection of W onto span(U)).

    Reconstructing the dense quantized weight:
        U_dq    = U_int8.float() * U_scale
        W_q     = baseline_q + U_dq @ (action_q - U_dq.t() @ baseline_q)

    When ``rank == 0`` (no usable ASPQ subspace) U_int8/U_scale/action_q are
    empty and ``W_q == baseline_q``.

    Coordinate system (important):
      * If ``smooth_scale`` is ``None`` (default), all stored tensors and the
        reconstructed ``W_q`` live in the original input basis. ``W_q ≈ W``.
      * If ``smooth_scale`` is set (SmoothQuant active), all stored tensors
        live in the *smoothed* input basis and ``reconstruct()`` returns
        ``W_q ≈ W · diag(smooth_scale)``, **not** the original ``W``. The
        runtime is responsible for pairing this with ``x' = x / smooth_scale``
        so that ``y = W_q · x' ≈ W · x``. Comparing ``reconstruct()`` against
        the FP ``W`` directly will be misleading; multiply ``W`` by
        ``smooth_scale`` first or divide ``reconstruct()`` by it.
    """

    baseline_q: torch.Tensor
    U_int8: torch.Tensor
    U_scale: torch.Tensor
    action_q: torch.Tensor
    smooth_scale: Optional[torch.Tensor] = None
    sq_stats: Optional[Dict[str, float]] = None

    @property
    def rank(self) -> int:
        return int(self.U_int8.shape[1]) if self.U_int8.numel() > 0 else 0

    def reconstruct(self) -> torch.Tensor:
        if self.U_int8.numel() == 0:
            return self.baseline_q.clone()
        U_dq = self.U_int8.to(dtype=torch.float32) * self.U_scale.to(dtype=torch.float32)
        return (
            self.baseline_q
            + U_dq @ (self.action_q - U_dq.t() @ self.baseline_q)
        )


def quantize_U_int8(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-column symmetric int8 quantization of a [O, k] basis.

    U columns are unit-norm orthogonal so all entries lie in [-1, 1] and a
    per-column max-abs scale is near-lossless.
    """
    if U.numel() == 0:
        return (
            torch.empty(0, 0, dtype=torch.int8, device=U.device),
            torch.empty(0, dtype=torch.float32, device=U.device),
        )
    U_f = U.detach().to(dtype=torch.float32)
    max_abs = U_f.abs().amax(dim=0).clamp_min(1e-8)
    scale = max_abs / 127.0
    codes = torch.clamp(torch.round(U_f / scale), -128.0, 127.0).to(torch.int8)
    return codes, scale


def _empty_aspq_pieces(
    out_features: int, in_features: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(out_features, 0, dtype=torch.int8, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
        torch.empty(0, in_features, dtype=torch.float32, device=device),
    )


def solve_aspq_gptq_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    U: Optional[torch.Tensor],
    eigvals: Optional[torch.Tensor],
    block_size: int = 128,
    damp_percent: float = 0.01,
    min_eig: float = 1e-12,
    x_calib_stats: Optional[torch.Tensor] = None,
    sq_alpha: float = 0.0,
    sq_clamp: Tuple[float, float] = (0.5, 2.0),
    weight_group_size: int = 0,
    err_comp_gamma: float = 1.0,
) -> AspqGptqRecord:
    """Quantize W with GPTQ, then quantize the ASPQ action subspace.

    Algorithm is unchanged from the original; only the return type is
    factored so the build script can persist (baseline_q, U_int8, U_scale,
    action_q) separately. The runtime reconstructs the same dense weight
    from these pieces, so forward behavior matches the previous fused-tensor
    storage.
    """
    # GR00T_ASPQ_WIDE_MSE controls the per-row MSE candidate grid for action_q
    # (and optionally baseline_q). Wider grids let outlier-heavy rows pick
    # AWQ-style aggressive clipping when it strictly reduces MSE.
    #   "0"/unset    → narrow default grid for both (legacy behavior)
    #   "action"/"1" → wide grid for action_q only (recommended; isolates ASPQ
    #                  improvement from generic GPTQ improvement)
    #   "all"        → wide grid for baseline_q AND action_q (generic upgrade)
    wide_mse_env = os.environ.get("GR00T_ASPQ_WIDE_MSE", "0").lower()
    use_wide_action = wide_mse_env in ("1", "true", "action", "all")
    use_wide_baseline = wide_mse_env in ("all",)
    wide_alphas: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)

    # SmoothQuant: per-input-channel rescale s_j = A_j^alpha / B_j^(1-alpha).
    # When sq_alpha > 0 and x_calib_stats is provided, transform W → W·diag(s)
    # and H → diag(1/s) H diag(1/s). Runtime applies x → x/s before A8 quant.
    smooth_scale: Optional[torch.Tensor] = None
    sq_stats: Optional[Dict[str, float]] = None
    if sq_alpha > 0.0 and x_calib_stats is not None:
        eps = 1e-8
        device = W.device
        A = x_calib_stats.detach().to(dtype=torch.float32, device=device).clamp_min(eps)  # (in,)
        B = W.detach().abs().amax(dim=0).to(dtype=torch.float32).clamp_min(eps)  # (in,)
        if A.shape != B.shape:
            raise ValueError(f"x_calib_stats shape {tuple(A.shape)} != W in_features {tuple(B.shape)}")
        raw_s = (A.pow(float(sq_alpha))) / (B.pow(1.0 - float(sq_alpha)))
        med = raw_s.median().clamp_min(eps)
        s_norm = raw_s / med
        lo, hi = float(sq_clamp[0]), float(sq_clamp[1])
        s = torch.clamp(s_norm, min=lo, max=hi)
        # Stats for reporting
        n_lo = int((s_norm < lo).sum().item())
        n_hi = int((s_norm > hi).sum().item())
        n_total = int(s.numel())
        q1 = float(torch.quantile(s_norm, 0.01).item())
        q99 = float(torch.quantile(s_norm, 0.99).item())
        sq_stats = {
            "median_raw": float(med.item()),
            "min_clamped": float(s.min().item()),
            "max_clamped": float(s.max().item()),
            "min_unclamped": float(s_norm.min().item()),
            "max_unclamped": float(s_norm.max().item()),
            "q1_unclamped": q1,
            "q99_unclamped": q99,
            "frac_at_lo": n_lo / max(n_total, 1),
            "frac_at_hi": n_hi / max(n_total, 1),
            "alpha": float(sq_alpha),
        }
        smooth_scale = s
        # Transform W and H to the smoothed input space.
        W = W * s[None, :]
        H = H / s[:, None] / s[None, :]

    baseline_q = gptq_quantize_weight(
        W,
        H,
        bits=bits,
        block_size=block_size,
        damp_percent=damp_percent,
        mse_alphas=wide_alphas if use_wide_baseline else None,
        group_size=int(weight_group_size),
        err_comp_gamma=float(err_comp_gamma),
    )
    out_features, in_features = baseline_q.shape
    device = baseline_q.device

    if U is None or eigvals is None or U.numel() == 0 or eigvals.numel() == 0:
        U_int8, U_scale, action_q = _empty_aspq_pieces(out_features, in_features, device)
        return AspqGptqRecord(baseline_q, U_int8, U_scale, action_q, smooth_scale=smooth_scale, sq_stats=sq_stats)

    U = U.detach().to(dtype=torch.float32, device=device)
    eigvals = eigvals.detach().flatten().to(dtype=torch.float32, device=device)
    keep = torch.isfinite(eigvals) & (eigvals > float(min_eig))
    if int(keep.sum().item()) == 0:
        U_int8, U_scale, action_q = _empty_aspq_pieces(out_features, in_features, device)
        return AspqGptqRecord(baseline_q, U_int8, U_scale, action_q, smooth_scale=smooth_scale, sq_stats=sq_stats)

    U = U[:, keep]
    eigvals = eigvals[keep]
    # NB: row_weights = √λ used to scale W_action multiplicatively, but per-row
    # symmetric quantization is scale-equivariant (Q(αw)/α = Q(w)) so the
    # multiply-then-divide was a mathematical no-op and is removed. λ now only
    # influences the basis selection (top-K eigvecs) and any future
    # mixed-precision bit allocation.

    W_action = U.t() @ W.to(dtype=torch.float32)
    K = W_action.shape[0]

    # GR00T_ASPQ_TOPM_W4: int M ≥ 0. The first M rows of action_q (highest λ
    # eigenvectors, since eigvals are sorted descending) get quantized at W4
    # while the rest stay at the requested bits. Allocates more capacity to the
    # most action-sensitive directions; per-row scale and GPTQ mechanics are
    # untouched.
    topm_w4 = int(os.environ.get("GR00T_ASPQ_TOPM_W4", "0"))
    topm_w4 = max(0, min(topm_w4, K))
    if topm_w4 > 0 and bits >= 4:
        # Already W4 or higher → no benefit from "promote to W4".
        topm_w4 = 0

    # GR00T_ASPQ_ACTION_BLOCK: override block_size for action_q quantization
    # only (baseline_q keeps the requested block_size). Smaller blocks give
    # finer per-block MSE search and tighter Hinv compensation; larger blocks
    # have more data per block to ground the search.
    action_block = int(os.environ.get("GR00T_ASPQ_ACTION_BLOCK", "0"))
    if action_block <= 0:
        action_block = block_size

    action_alphas = wide_alphas if use_wide_action else None
    if topm_w4 > 0:
        action_top = gptq_quantize_weight(
            W_action[:topm_w4],
            H,
            bits=4,
            block_size=action_block,
            damp_percent=damp_percent,
            mse_alphas=action_alphas,
            err_comp_gamma=float(err_comp_gamma),
        )
        action_rest = gptq_quantize_weight(
            W_action[topm_w4:],
            H,
            bits=bits,
            block_size=action_block,
            damp_percent=damp_percent,
            mse_alphas=action_alphas,
            err_comp_gamma=float(err_comp_gamma),
        )
        action_q = torch.cat([action_top, action_rest], dim=0).contiguous()
    else:
        action_q = gptq_quantize_weight(
            W_action,
            H,
            bits=bits,
            block_size=action_block,
            damp_percent=damp_percent,
            mse_alphas=action_alphas,
            err_comp_gamma=float(err_comp_gamma),
        ).contiguous()

    U_int8, U_scale = quantize_U_int8(U)
    return AspqGptqRecord(
        baseline_q=baseline_q,
        U_int8=U_int8,
        U_scale=U_scale,
        action_q=action_q,
        smooth_scale=smooth_scale,
        sq_stats=sq_stats,
    )


def _score_record_for_alpha(
    rec: "AspqGptqRecord",
    W_fp: torch.Tensor,
    X: torch.Tensor,
    U: Optional[torch.Tensor],
    eigvals: Optional[torch.Tensor],
) -> Tuple[float, float]:
    """Score a quantized record on calibration X.

    Returns ``(E_mse, E_action)``. ``E_action`` is ``nan`` when the layer has
    no usable ASPQ basis. Scoring runs in float32 on the device of ``X``.

    Forward in the smoothed coordinate system (the pack stores W·diag(s); at
    runtime ``x' = x/s`` is applied before the linear), so we compute
    ``Y_q = (X / s) @ Wq_s^T`` to mirror runtime behavior.
    """
    Wq_s = rec.reconstruct().to(dtype=torch.float32, device=X.device)
    W_fp_dev = W_fp.detach().to(dtype=torch.float32, device=X.device)
    Y_fp = X @ W_fp_dev.t()
    if rec.smooth_scale is not None and rec.smooth_scale.numel() == X.shape[-1]:
        s = rec.smooth_scale.to(dtype=torch.float32, device=X.device).clamp_min(1e-8)
        X_s = X / s[None, :]
        Y_q = X_s @ Wq_s.t()
    else:
        Y_q = X @ Wq_s.t()
    delta = Y_q - Y_fp
    e_mse = float(delta.pow(2).mean().item())
    if U is None or eigvals is None or U.numel() == 0 or eigvals.numel() == 0:
        return e_mse, float("nan")
    U_dev = U.detach().to(dtype=torch.float32, device=X.device)
    eigvals_dev = eigvals.detach().to(dtype=torch.float32, device=X.device)
    proj = delta @ U_dev  # (N, K)
    lam_norm = eigvals_dev / eigvals_dev.mean().clamp_min(1e-12)
    e_action = float((lam_norm[None, :] * proj.pow(2)).sum(dim=-1).mean().item())
    return e_mse, e_action


def solve_aspq_gptq_weight_with_alpha_select(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    U: Optional[torch.Tensor],
    eigvals: Optional[torch.Tensor],
    block_size: int = 128,
    damp_percent: float = 0.01,
    min_eig: float = 1e-12,
    x_calib_stats: torch.Tensor,
    x_calib_inputs: torch.Tensor,
    sq_alpha_grid: Sequence[float],
    sq_select_mode: str,
    sq_clamp: Tuple[float, float] = (0.5, 2.0),
    weight_group_size: int = 0,
    err_comp_gamma: float = 1.0,
) -> AspqGptqRecord:
    """Per-layer SmoothQuant alpha selection over a small grid.

    For each candidate ``alpha`` in ``sq_alpha_grid``, build a record via the
    existing ``solve_aspq_gptq_weight`` (with that fixed alpha) and score it
    on ``x_calib_inputs``. Pick the alpha whose record minimises:
      * ``E_mse`` if ``sq_select_mode == 'mse'``
      * ``E_action`` if ``sq_select_mode == 'action'``

    Falls back to MSE when the layer lacks a usable ASPQ basis (so ``E_action``
    is undefined). The returned record carries ``selected_alpha`` and the full
    per-alpha score table inside ``sq_stats`` for downstream analysis.

    Important: ``eigvals`` are only consumed by the scoring metric and the
    basis selection inside ``solve_aspq_gptq_weight``; they are NOT used as
    row-wise scaling inside GPTQ (which would be a no-op under per-row
    symmetric quantization scale-equivariance).
    """
    if sq_select_mode not in ("mse", "action"):
        raise ValueError(f"sq_select_mode must be 'mse' or 'action', got '{sq_select_mode}'")
    if not sq_alpha_grid:
        raise ValueError("sq_alpha_grid must be non-empty")

    candidates: list = []  # list of (alpha, record, e_mse, e_action)
    for alpha in sq_alpha_grid:
        rec = solve_aspq_gptq_weight(
            W,
            H,
            bits=bits,
            U=U,
            eigvals=eigvals,
            block_size=block_size,
            damp_percent=damp_percent,
            min_eig=min_eig,
            x_calib_stats=x_calib_stats,
            sq_alpha=float(alpha),
            sq_clamp=sq_clamp,
            weight_group_size=int(weight_group_size),
            err_comp_gamma=float(err_comp_gamma),
        )
        e_mse, e_action = _score_record_for_alpha(rec, W, x_calib_inputs, U, eigvals)
        candidates.append((float(alpha), rec, e_mse, e_action))

    # Choose by selected metric, fall back to MSE if action scores all NaN
    use_action = sq_select_mode == "action" and any(c[3] == c[3] for c in candidates)  # any non-NaN
    keyfn = (lambda c: c[3]) if use_action else (lambda c: c[2])
    valid = [c for c in candidates if c[2] == c[2]]  # filter NaN MSE too (defensive)
    best = min(valid, key=keyfn) if valid else candidates[0]
    best_alpha, best_rec, best_e_mse, best_e_action = best

    # Annotate
    if best_rec.sq_stats is None:
        best_rec.sq_stats = {}
    best_rec.sq_stats["selected_alpha"] = float(best_alpha)
    best_rec.sq_stats["select_mode"] = "action" if use_action else "mse"
    best_rec.sq_stats["select_mode_requested"] = sq_select_mode
    # Persist per-alpha scores as flat keys (json-friendly)
    for alpha_v, _, e_mse, e_action in candidates:
        best_rec.sq_stats[f"e_mse_a{alpha_v}"] = e_mse
        best_rec.sq_stats[f"e_action_a{alpha_v}"] = e_action
    return best_rec


class AspqGptqLinear(nn.Module):
    """Runtime wrapper for offline-built ASPQ-GPTQ quantized weights."""

    def __init__(self, base: nn.Linear, name: str, cfg: AspqGptqConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.cfg = cfg
        self.weight_bits = int(weight_bits if weight_bits is not None else (cfg.weight_bits or 0))
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        # Drop the duplicate FP weight buffer when quantization is available — only
        # allocate it as a fallback if the record is missing. Set
        # GR00T_ASPQ_GPTQ_KEEP_FP=1 to opt back in.
        keep_fp_env = os.environ.get("GR00T_ASPQ_GPTQ_KEEP_FP", "0") not in ("0", "false", "False")
        self._debug_enabled = os.environ.get("GR00T_ASPQ_GPTQ_DEBUG", "0") not in ("0", "false", "False")

        self._act_scale: Optional[torch.Tensor] = None
        self._act_scale_initialized = False
        self.calibrator = PercentileCalibrator(
            percentile=float(cfg.act_percentile or 99.9),
            max_batches=int(cfg.calib_batches or 32),
        )

        record = _load_quantized_record(name, cfg.path or "")
        # SmoothQuant per-input-channel rescale (loaded from pack if present).
        # When set, runtime applies x = x / smooth_scale before A8 fake_quant;
        # the stored W is already W·diag(s).
        self._has_smooth_scale = False
        # Per-DiT-step activation scale table (loaded from pack if present).
        # Shape (num_steps, in_features) — looked up by current_dit_step at
        # forward time. When unset, falls back to single ``_act_scale`` (the
        # legacy step-agnostic A8 calibration).
        self._has_act_scale_table = False
        # SVDQuant low-rank residual branch (loaded from pack if record format
        # == "dit_svdquant_v1"). When set, forward computes
        #   y = linear(x_q, weight_res_q) + (x_s @ B) @ A.T + bias
        # where x_s = x / smooth_scale and x_q = fake_quant(x_s, table[t]).
        self._has_svdquant = False
        # Per-record a_bits override (set by SVDQuant records). Allows mixed
        # precision: LLM records use cfg.act_bits, DiT SVDQuant records use
        # the value baked into the pack at build time.
        self._record_a_bits: Optional[int] = None
        # Init optional-feature flags up-front so fallback path is forward-safe.
        self._has_duquant_rotation = False
        self._has_duquant_rotation_out = False
        # Block-fast path for input/output rotation: when a pack stores the
        # block-diagonal structure explicitly (R_in_blocks: [n_blocks, B, B],
        # perm: [in_features]) alongside the dense R, runtime uses bmm over
        # blocks instead of a full in×in matmul. Critical for pi0.5
        # paligemma down_proj where in=16384 dense matmul = ~1GB/layer.
        self._has_duquant_rotation_blocks = False
        self._has_duquant_rotation_out_blocks = False
        if record is None:
            if cfg.missing == "error":
                raise FileNotFoundError(f"No ASPQ-GPTQ record found for layer '{name}' in {cfg.path}")
            # No quant available — keep FP weight as the live tensor.
            self.register_buffer("_weight_fp", base.weight.detach().clone(), persistent=False)
            self.register_buffer("_weight_q", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self._quant_available = False
            self.register_buffer("_smooth_scale", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self.register_buffer("_act_scale_table", torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self._has_smooth_scale = False
            self._has_act_scale_table = False
            self._has_svdquant = False
        else:
            # Build the quantized dense weight; only keep _weight_q.
            weight_q_t = self._build_weight_from_record(record, name, base.weight)
            self.register_buffer("_weight_q", weight_q_t, persistent=False)
            if keep_fp_env:
                self.register_buffer("_weight_fp", base.weight.detach().clone(), persistent=False)
            else:
                # FP fallback not needed; allocate a zero-element placeholder so
                # attribute access remains valid for the `weight` property.
                self.register_buffer("_weight_fp", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self._quant_available = True
            # Optional SmoothQuant scale: load if present in the record.
            ss_raw = _record_get(record, ("smooth_scale",))
            if ss_raw is not None:
                ss_t = _to_tensor(ss_raw).to(dtype=torch.float32)
                if ss_t.numel() == self.in_features:
                    self.register_buffer("_smooth_scale", ss_t, persistent=False)
                    self._has_smooth_scale = True
                else:
                    self.register_buffer("_smooth_scale", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            else:
                self.register_buffer("_smooth_scale", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            # Optional per-DiT-step activation scale table (only present for
            # DiT layers; LLM layers leave this unset and use _act_scale).
            ast_raw = _record_get(record, ("act_scale_table",))
            if ast_raw is not None:
                ast_t = _to_tensor(ast_raw).to(dtype=torch.float32)
                # Guard against FP16-underflow zeros in act_scale_table:
                # build computes scale = q99.9(|x|) / qmax(a_bits); for channels
                # with very low activation magnitude (q99.9 < ~1e-6) the result
                # is ~8e-9 which underflows FP16's subnormal range (~6e-8) when
                # save_dtype=float16 was used → stored as exact zero → runtime
                # fake_quantize_sym divides by zero → NaN cascade. Clamp on
                # load so existing packs are usable without rebuild.
                ast_t = ast_t.clamp_min(1e-8)
                if ast_t.dim() == 2 and ast_t.shape[1] == self.in_features:
                    self.register_buffer("_act_scale_table", ast_t, persistent=False)
                    # Pre-compute the step-mean fallback buffer at init time so
                    # the forward() fallback path doesn't allocate a fresh
                    # tensor inside a torch.compile + CUDAGraph capture (which
                    # would cause "accessing tensor output of CUDAGraphs that
                    # has been overwritten" RuntimeErrors on pi0.5 inference).
                    self.register_buffer(
                        "_act_scale_table_mean_buf",
                        ast_t.mean(dim=0).contiguous().clone(),
                        persistent=False,
                    )
                    self._has_act_scale_table = True
                else:
                    self.register_buffer(
                        "_act_scale_table",
                        torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                        persistent=False,
                    )
            else:
                self.register_buffer(
                    "_act_scale_table",
                    torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                    persistent=False,
                )
            # SVDQuant lowrank branch — only present for "dit_svdquant_v1"
            # records. lowrank_A is [out, r], lowrank_B is [in, r].
            fmt = _record_get(record, ("format",))
            fmt_str = str(fmt) if fmt is not None else ""
            if fmt_str == "dit_svdquant_v1":
                A_raw = _record_get(record, ("lowrank_A",))
                B_raw = _record_get(record, ("lowrank_B",))
                if A_raw is not None and B_raw is not None:
                    A_t = _to_tensor(A_raw).to(dtype=torch.float32)
                    B_t = _to_tensor(B_raw).to(dtype=torch.float32)
                    if (
                        A_t.dim() == 2
                        and B_t.dim() == 2
                        and A_t.shape[0] == self.out_features
                        and B_t.shape[0] == self.in_features
                        and A_t.shape[1] == B_t.shape[1]
                    ):
                        self.register_buffer("_lowrank_A", A_t, persistent=False)
                        self.register_buffer("_lowrank_B", B_t, persistent=False)
                        self._has_svdquant = True
                # Pull baked-in a_bits so DiT SVDQuant records use their own
                # bit-width regardless of cfg.act_bits (which may be 8 for LLM).
                ab_raw = _record_get(record, ("a_bits",))
                if ab_raw is not None:
                    try:
                        self._record_a_bits = int(ab_raw)
                    except Exception:
                        pass
                    else:
                        print(
                            f"[ASPQ-GPTQ][WARN] {name}: dit_svdquant_v1 record has "
                            f"bad lowrank shapes A={tuple(A_t.shape)} B={tuple(B_t.shape)} "
                            f"expected ({self.out_features}, r) and ({self.in_features}, r)",
                            flush=True,
                        )
            if not self._has_svdquant:
                self.register_buffer(
                    "_lowrank_A",
                    torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                    persistent=False,
                )
                self.register_buffer(
                    "_lowrank_B",
                    torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                    persistent=False,
                )
            # Optional DuQuant input rotation: applied as x' = x @ R BEFORE
            # the standard ASPQ-GPTQ forward path. Used by the "DuQuant + ASPQ"
            # hybrid pack format built with --duquant-rotation.
            #
            # Fast path: when the pack stores `duquant_rotation_blocks` ([n,B,B])
            # + `duquant_rotation_perm` ([in_features] int), runtime uses bmm
            # over blocks: y = x[:, perm].view(T, n, B) @ R_b. This is
            # ~in/B × cheaper than the dense in×in matmul. For pi0.5
            # paligemma down_proj (in=16384, B=64): 256× cheaper.
            blocks_raw = _record_get(record, ("duquant_rotation_blocks",))
            perm_raw = _record_get(record, ("duquant_rotation_perm",))
            if blocks_raw is not None and perm_raw is not None:
                blocks_t = _to_tensor(blocks_raw).to(dtype=torch.float32)
                perm_t = _to_tensor(perm_raw).to(dtype=torch.int64)
                if (
                    blocks_t.dim() == 3
                    and blocks_t.shape[1] == blocks_t.shape[2]
                    and blocks_t.shape[0] * blocks_t.shape[1] == self.in_features
                    and perm_t.dim() == 1
                    and perm_t.numel() == self.in_features
                ):
                    self.register_buffer("_duquant_rotation_blocks", blocks_t, persistent=False)
                    self.register_buffer("_duquant_rotation_perm", perm_t, persistent=False)
                    self._has_duquant_rotation_blocks = True
            # Dense fallback (legacy packs): full in×in R. Slow on large in.
            rot_raw = _record_get(record, ("duquant_rotation",))
            if rot_raw is not None and not self._has_duquant_rotation_blocks:
                rot_t = _to_tensor(rot_raw).to(dtype=torch.float32)
                if rot_t.dim() == 2 and rot_t.shape[0] == self.in_features and rot_t.shape[1] == self.in_features:
                    self.register_buffer("_duquant_rotation", rot_t, persistent=False)
                    self._has_duquant_rotation = True
            # Optional DuQuant output rotation: applied as y_restored = y @ R_out
            # AFTER matmul, BEFORE bias.
            #
            # Fast path: `duquant_rotation_out_blocks` ([n_out, B_out, B_out])
            # — output side has no permutation in our builders, so just bmm.
            blocks_out_raw = _record_get(record, ("duquant_rotation_out_blocks",))
            if blocks_out_raw is not None:
                blocks_out_t = _to_tensor(blocks_out_raw).to(dtype=torch.float32)
                if (
                    blocks_out_t.dim() == 3
                    and blocks_out_t.shape[1] == blocks_out_t.shape[2]
                    and blocks_out_t.shape[0] * blocks_out_t.shape[1] == self.out_features
                ):
                    self.register_buffer("_duquant_rotation_out_blocks", blocks_out_t, persistent=False)
                    self._has_duquant_rotation_out_blocks = True
            # Dense fallback (legacy packs)
            rot_out_raw = _record_get(record, ("duquant_rotation_out",))
            if rot_out_raw is not None and not self._has_duquant_rotation_out_blocks:
                rot_out_t = _to_tensor(rot_out_raw).to(dtype=torch.float32)
                if rot_out_t.dim() == 2 and rot_out_t.shape[0] == self.out_features and rot_out_t.shape[1] == self.out_features:
                    self.register_buffer("_duquant_rotation_out", rot_out_t, persistent=False)
                    self._has_duquant_rotation_out = True
            # Free the now-unneeded base.weight reference; the parent module no
            # longer owns it after Module.replace.
            del weight_q_t

    @staticmethod
    def _build_weight_from_record(
        record: Any, name: str, ref_weight: torch.Tensor
    ) -> torch.Tensor:
        """Materialize the dense quantized weight from an offline record.

        Supports the SVDQuant residual format (weight_res_q + lowrank_A/B),
        the ASPQ factored format (baseline_q + U_int8 + U_scale + action_q),
        and the legacy fused format (weight_q).
        """
        # SVDQuant residual format: dense W_res_q stored at "weight_res_q".
        # The lowrank branch (A, B) is loaded as separate buffers in __init__.
        weight_res_q = _record_get(record, ("weight_res_q",))
        if weight_res_q is not None:
            wrq_t = _to_tensor(weight_res_q).to(dtype=ref_weight.dtype)
            if tuple(wrq_t.shape) != tuple(ref_weight.shape):
                raise ValueError(
                    f"ASPQ-GPTQ weight_res_q for '{name}' has shape {tuple(wrq_t.shape)}, "
                    f"expected {tuple(ref_weight.shape)}"
                )
            return wrq_t
        baseline_q = _record_get(record, ("baseline_q",))
        if baseline_q is not None:
            baseline_t = _to_tensor(baseline_q).to(dtype=torch.float32)
            if tuple(baseline_t.shape) != tuple(ref_weight.shape):
                raise ValueError(
                    f"ASPQ-GPTQ baseline_q for '{name}' has shape {tuple(baseline_t.shape)}, "
                    f"expected {tuple(ref_weight.shape)}"
                )
            U_int8_raw = _record_get(record, ("U_int8",))
            U_scale_raw = _record_get(record, ("U_scale",))
            action_q_raw = _record_get(record, ("action_q",))
            if (
                U_int8_raw is not None
                and U_scale_raw is not None
                and action_q_raw is not None
                and _to_tensor(U_int8_raw).numel() > 0
            ):
                U_int8 = _to_tensor(U_int8_raw).to(dtype=torch.int8)
                U_scale = _to_tensor(U_scale_raw).to(dtype=torch.float32)
                action_q = _to_tensor(action_q_raw).to(dtype=torch.float32)
                if U_int8.dim() != 2 or U_int8.shape[0] != ref_weight.shape[0]:
                    raise ValueError(
                        f"ASPQ-GPTQ U_int8 for '{name}' has shape {tuple(U_int8.shape)}, "
                        f"expected [{ref_weight.shape[0]}, k]"
                    )
                # Runtime K truncation: reuse a K=large pack with a smaller K.
                # Stored columns are sorted descending by eigval, so [:, :K_run]
                # keeps the most action-relevant directions.
                runtime_topk = int(os.environ.get("GR00T_ASPQ_GPTQ_RUNTIME_TOPK", "0"))
                if runtime_topk > 0 and runtime_topk < int(U_int8.shape[1]):
                    U_int8 = U_int8[:, :runtime_topk].contiguous()
                    U_scale = U_scale[:runtime_topk].contiguous()
                    action_q = action_q[:runtime_topk, :].contiguous()
                k = int(U_int8.shape[1])
                if U_scale.shape != (k,):
                    raise ValueError(
                        f"ASPQ-GPTQ U_scale for '{name}' has shape {tuple(U_scale.shape)}, "
                        f"expected ({k},)"
                    )
                if action_q.shape != (k, ref_weight.shape[1]):
                    raise ValueError(
                        f"ASPQ-GPTQ action_q for '{name}' has shape {tuple(action_q.shape)}, "
                        f"expected ({k}, {ref_weight.shape[1]})"
                    )
                U_dq = U_int8.to(dtype=torch.float32) * U_scale
                w = baseline_t + U_dq @ (action_q - U_dq.t() @ baseline_t)
            else:
                w = baseline_t
            return w.to(dtype=ref_weight.dtype)

        # Legacy fused format.
        weight_q = _record_get(record, ("weight_q", "W_q", "quant_weight", "weight"))
        if weight_q is None:
            raise ValueError(f"ASPQ-GPTQ record for '{name}' is missing quantized weight data")
        weight_q_t = _to_tensor(weight_q).to(dtype=ref_weight.dtype)
        if tuple(weight_q_t.shape) != tuple(ref_weight.shape):
            raise ValueError(
                f"ASPQ-GPTQ weight for '{name}' has shape {tuple(weight_q_t.shape)}, "
                f"expected {tuple(ref_weight.shape)}"
            )
        return weight_q_t

    @property
    def weight(self) -> torch.Tensor:
        return self._weight_q if self._quant_available else self._weight_fp

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        if self._act_scale_initialized and self._act_scale is not None:
            return self._act_scale

        with torch.no_grad():
            if self.calibrator is not None and not self.calibrator.is_full():
                self.calibrator.observe(x)
                if self.calibrator.is_full():
                    p_vec = self.calibrator.finalize()
                    max_q = qmax(int(self.cfg.act_bits or 8))
                    scale = torch.clamp(p_vec / max_q, min=1e-6)
                    scale = scale.to(dtype=x.dtype, device=x.device).clone()
                    self._act_scale = scale
                    self._act_scale_initialized = True
                    self._dump_act_stats(p_vec)

            if not self._act_scale_initialized:
                x_abs = torch.abs(x.detach().to(torch.float32))
                x2d = x_abs.reshape(-1, x_abs.shape[-1])
                p_vec = torch.quantile(x2d, float(self.cfg.act_percentile or 99.9) / 100.0, dim=0)
                max_q = qmax(int(self.cfg.act_bits or 8))
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                scale = scale.to(dtype=x.dtype, device=x.device).clone()
                self._act_scale = scale
                self._act_scale_initialized = True
                self._dump_act_stats(p_vec)

        return self._act_scale if self._act_scale is not None else torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

    def _dump_act_stats(self, p_vec: torch.Tensor) -> None:
        """Dump per-channel A8 percentile vector to a JSON file when
        GR00T_ACT_STATS_DIR is set (one file per layer per process).
        """
        out_dir = os.environ.get("GR00T_ACT_STATS_DIR")
        if not out_dir:
            return
        try:
            import json
            from pathlib import Path
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            safe_name = self.name.replace("/", "_")
            (d / f"{safe_name}.json").write_text(json.dumps({
                "layer": self.name,
                "method": "aspq_gptq",
                "act_bits": int(self.cfg.act_bits or 0),
                "percentile": float(self.cfg.act_percentile or 99.9),
                "calib_batches": int(self.cfg.calib_batches or 0),
                "n_channels": int(p_vec.numel()),
                "p_vec": p_vec.detach().cpu().tolist(),
            }))
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DuQuant rotation preprocessing (if present): x' = x @ R
        # Applied BEFORE SmoothQuant scaling because R is on raw input space;
        # the pack's smooth_scale was computed on rotated activations.
        if self._has_duquant_rotation_blocks:
            # Fast path: x[:, perm].view(..., n_blocks, B) @ R_blocks (bmm).
            # Equivalent to x @ (P @ block_diag(R_b)) where P is perm matrix.
            blocks = self._duquant_rotation_blocks.to(dtype=x.dtype, device=x.device)  # [n, B, B]
            perm = self._duquant_rotation_perm.to(device=x.device)
            in_f = blocks.shape[0] * blocks.shape[1]
            B = blocks.shape[1]
            n_blocks = blocks.shape[0]
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1])           # (N, in)
            x2 = x2.index_select(dim=-1, index=perm)     # (N, in)  permuted
            x2 = x2.reshape(-1, n_blocks, B)             # (N, n, B)
            x2 = torch.bmm(x2.transpose(0, 1).contiguous(), blocks)  # (n, N, B)
            x2 = x2.transpose(0, 1).contiguous().reshape(-1, in_f)
            x = x2.reshape(*orig_shape[:-1], in_f)
        elif self._has_duquant_rotation:
            R = self._duquant_rotation.to(dtype=x.dtype, device=x.device)
            x = torch.matmul(x, R)
        # SmoothQuant: divide input by per-channel smooth scale so that the
        # forward sees activations matched to the smoothed-W stored in pack.
        if self._has_smooth_scale and self._smooth_scale.numel() == x.shape[-1]:
            x = x / self._smooth_scale.to(dtype=x.dtype, device=x.device)
        # SVDQuant lowrank branch consumes the smoothed (but un-A-quantized)
        # activation x_s. We compute it before any A4/A8 fake-quant on the
        # residual path so the low-rank correction stays at full precision.
        x_s_for_lowrank: Optional[torch.Tensor] = (
            x if self._has_svdquant else None
        )
        x_q = x
        # Per-record a_bits override: SVDQuant records store their own a_bits
        # (e.g. 4 for W4A4 DiT pack) so that mixed-precision packs can use
        # cfg.act_bits for non-record layers (e.g. LLM A8) and the record's
        # value for SVDQuant DiT layers (A4). Falls back to cfg.act_bits.
        eff_act_bits = (
            int(self._record_a_bits)
            if self._record_a_bits is not None
            else int(self.cfg.act_bits or 0)
        )
        if eff_act_bits > 0:
            # Per-DiT-step scale dispatch: if this layer has an act_scale_table
            # AND a denoising-step context is active AND the step is in range,
            # use that step's row.
            step_scale: Optional[torch.Tensor] = None
            if self._has_act_scale_table:
                from gr00t.quantization.dit_step_context import get_current_dit_step
                cur = get_current_dit_step()
                if cur is not None and 0 <= cur < self._act_scale_table.shape[0]:
                    step_scale = self._act_scale_table[cur].to(dtype=x.dtype, device=x.device)
                elif cur is not None and not getattr(self, "_act_table_oob_warned", False):
                    print(
                        f"[ASPQ-GPTQ][WARN] {self.name}: act_scale_table has "
                        f"{self._act_scale_table.shape[0]} steps but current_dit_step={cur} — "
                        f"falling back to step-mean scale. (warned once per layer)",
                        flush=True,
                    )
                    self._act_table_oob_warned = True
            if step_scale is not None:
                x_q = fake_quantize_sym(x, step_scale, eff_act_bits, label="aspq_gptq_activation_per_step")
            elif self._has_act_scale_table:
                # Deterministic fallback for DiT layers that have a table but
                # the step context is missing/oob: use the step-mean scale.
                # Do NOT fall through to the running PercentileCalibrator,
                # which would otherwise silently corrupt the rollout with a
                # mismatched dynamic estimate.
                if not hasattr(self, "_act_scale_table_mean_buf"):
                    mean_scale = self._act_scale_table.mean(dim=0)
                    self.register_buffer("_act_scale_table_mean_buf", mean_scale, persistent=False)
                fb = self._act_scale_table_mean_buf.to(dtype=x.dtype, device=x.device)
                x_q = fake_quantize_sym(x, fb, eff_act_bits, label="aspq_gptq_activation_table_fallback")
            else:
                s_a = self._get_act_scale(x)
                x_q = fake_quantize_sym(x, s_a, eff_act_bits, label="aspq_gptq_activation")
        weight = self._weight_q if self._quant_available else self._weight_fp
        # Promote main matmul to avoid FP16 overflow: q99.9 × W4_max ×
        # in_features can exceed 65504 (FP16 max), causing catastrophic
        # NaN cascades on SVDQuant W4A8 + LLM-FP16. Real INT4 GEMM (cutlass)
        # uses INT32 accumulator and avoids this; our fake-quant emulation
        # needs explicit promote.
        # Use BF16 (range = FP32, BF16 native on A100/H100, ~zero overhead)
        # over FP32 (~10% slower). Falls back to FP32 on older GPUs.
        out_dtype = x_q.dtype
        promote_dtype = (
            torch.bfloat16
            if (x_q.is_cuda and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        x_q_p = x_q.to(promote_dtype)
        w_p = weight.to(dtype=promote_dtype, device=x_q.device)
        y = torch.nn.functional.linear(x_q_p, w_p, None)
        # SVDQuant: add lowrank correction y += (x_s @ B) @ A.T (also promoted)
        if self._has_svdquant and x_s_for_lowrank is not None:
            B = self._lowrank_B.to(dtype=promote_dtype, device=x_q.device)
            A = self._lowrank_A.to(dtype=promote_dtype, device=x_q.device)
            x_s_p = x_s_for_lowrank.to(promote_dtype)
            z = torch.nn.functional.linear(x_s_p, B.t())   # [..., r]
            y_lr = torch.nn.functional.linear(z, A)         # [..., out]
            y = y + y_lr
        # Optional DuQuant output rotation restore: y_restored = y @ R_out.
        # Applied BEFORE bias (bias lives in the un-rotated output basis).
        if self._has_duquant_rotation_out_blocks:
            # Fast path: bmm over n_out blocks of size B_out.
            blocks_out = self._duquant_rotation_out_blocks.to(dtype=y.dtype, device=y.device)
            out_f = blocks_out.shape[0] * blocks_out.shape[1]
            B_out = blocks_out.shape[1]
            n_out = blocks_out.shape[0]
            orig_shape = y.shape
            y2 = y.reshape(-1, orig_shape[-1])
            y2 = y2.reshape(-1, n_out, B_out)
            y2 = torch.bmm(y2.transpose(0, 1).contiguous(), blocks_out)
            y2 = y2.transpose(0, 1).contiguous().reshape(-1, out_f)
            y = y2.reshape(*orig_shape[:-1], out_f)
        elif self._has_duquant_rotation_out:
            R_out = self._duquant_rotation_out.to(dtype=y.dtype, device=y.device)
            y = torch.matmul(y, R_out)
        # Cast back to original dtype
        y = y.to(dtype=out_dtype)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype, device=y.device)
        if self._debug_enabled and not hasattr(self, "_debug_forward_logged"):
            print(
                f"[GR00T-ASPQ-GPTQ][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y.shape)} "
                f"W{self.weight_bits} A{self.cfg.act_bits} quant={int(self._quant_available)}",
                flush=True,
            )
            self._debug_forward_logged = True
        return y


def wrap_aspq_gptq(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: AspqGptqConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
) -> int:
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    listed = 0
    for name in layer_names:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        if dry_run:
            print(
                f"[GR00T-ASPQ-GPTQ][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits}"
            )
            listed += 1
            continue
        wrapped = AspqGptqLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, wrapped)
        print(
            f"[GR00T-ASPQ-GPTQ][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) "
            f"-> AspqGptqLinear W{wbits} A{cfg.act_bits} quant={int(wrapped._quant_available)}"
        )
        replaced += 1
    if dry_run:
        print(f"[GR00T-ASPQ-GPTQ] Dry-run total layers listed: {listed}")
        return listed
    print(f"[GR00T-ASPQ-GPTQ] Total layers replaced: {replaced}")
    return replaced


def load_aspq_basis_for_layer(
    layer_name: str,
    metric_path: str,
    *,
    out_features: int,
    topk: int = 0,
    min_eig: float = 1e-12,
    min_eig_relative: float = 0.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    record = _load_aspq_record(layer_name, metric_path)
    return _aspq_basis_from_record(
        record,
        out_features=out_features,
        topk=topk,
        min_eig=min_eig,
        min_eig_relative=min_eig_relative,
    )


def enable_aspq_gptq_if_configured(model: nn.Module) -> bool:
    env = os.environ
    activate = env.get("GR00T_ASPQ_GPTQ", "0") not in ("0", "false", "False")
    if not activate:
        return False

    scope = env.get("GR00T_ASPQ_GPTQ_SCOPE", "")
    whitelist = env.get("GR00T_ASPQ_GPTQ_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None
    inc = env.get(
        "GR00T_ASPQ_GPTQ_INCLUDE",
        (
            r".*(?:"
            r"backbone\.eagle_model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"|"
            r"action_head\.model\..*(?:attn1\.to_(?:q|k|v)|attn1\.to_out\.0|ff\.net\.(?:0\.proj|2))"
            r").*"
        ),
    )
    exc = env.get(
        "GR00T_ASPQ_GPTQ_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)"
            r"(?:\.|$)"
        ),
    )
    per_layer_wbits = _parse_per_layer_wbits(env.get("GR00T_ASPQ_GPTQ_WBITS"))
    dry_run = env.get("GR00T_ASPQ_GPTQ_DRYRUN", "0") not in ("0", "false", "False")

    cfg = AspqGptqConfig()
    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[GR00T-ASPQ-GPTQ] SCOPE filter: '{scope}'")
    print(f"[GR00T-ASPQ-GPTQ] Matched Linear layers: {len(layer_names)}")
    wrap_aspq_gptq(model, layer_names, cfg, per_layer_wbits, dry_run=dry_run)
    return True
