"""Experimental packed W4A4 residual linears; no model/runtime dependencies.

Three explicit algorithms, all retaining the denoising-step index:
  fused_float: channel-wise A4, packed W4, dequantize inside a FP16 dot kernel.
  step_weight: channel-wise A4 codes, W4 re-quantized after absorbing each step's
               channel scales. INT8 dot on unpacked INT4 codes, FP32 group sum.
  group_int:   per-step group-shared A4 scales, shared packed W4. Same integer dot.

The integer paths use INT8 tensor-core instructions, NOT native INT4 MMA.
Legacy packs store dense rounded weights without codes/scales. Export therefore
re-quantizes them; the benchmark measures this additional error explicitly.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("fused_float", "step_weight", "group_int")


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Signed [-8,7], adjacent K entries in low/high nibble; pad odd K."""
    q = q.to(torch.int16)
    if ((q < -8) | (q > 7)).any():
        raise ValueError("INT4 codes outside [-8,7]")
    if q.shape[-1] % 2:
        q = F.pad(q, (0, 1))
    return ((q[..., 0::2] & 15) | ((q[..., 1::2] & 15) << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, k: int) -> torch.Tensor:
    v = packed.to(torch.int16)
    q = torch.stack((v & 15, (v >> 4) & 15), dim=-1).flatten(-2)
    return torch.where(q >= 8, q - 16, q)[..., :k].to(torch.int8)


def quantize_weight(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output, per-K-group MSE clipping search; scales always FP32."""
    w = w.float()
    n, k = w.shape
    groups = (k + group_size - 1) // group_size
    codes = torch.zeros((n, k), dtype=torch.int8, device=w.device)
    scales = torch.zeros((n, groups), dtype=torch.float32, device=w.device)
    for g in range(groups):
        lo, hi = g * group_size, min((g + 1) * group_size, k)
        block = w[:, lo:hi]
        base = (block.abs().amax(dim=1) / 7).clamp_min(1e-12)
        best_error = torch.full((n,), float("inf"), device=w.device)
        for alpha in (1.0, .95, .9, .85, .8, .75, .7, .6, .5):
            scale = base * alpha
            q = (block / scale[:, None]).round().clamp(-8, 7)
            error = ((block - q * scale[:, None]) ** 2).mean(dim=1)
            better = error < best_error
            scales[:, g] = torch.where(better, scale, scales[:, g])
            codes[:, lo:hi] = torch.where(better[:, None], q.to(torch.int8), codes[:, lo:hi])
            best_error = torch.minimum(error, best_error)
    return pack_int4(codes), scales


def dequant_weight(packed: torch.Tensor, scales: torch.Tensor, k: int, group_size: int):
    q = unpack_int4(packed, k).float()
    index = torch.arange(k, device=q.device) // group_size
    return q * scales[:, index]


def make_backend_record(w: torch.Tensor, table: torch.Tensor, mode: str,
                        group_size: int = 128) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}")
    if group_size not in (32, 64, 128, 256):
        raise ValueError("group_size must be 32, 64, 128 or 256")
    if w.ndim != 2 or table.ndim != 2 or table.shape[1] != w.shape[1] or table.shape[0] == 0:
        raise ValueError("Expected W[N,K] and activation scale table[T,K]")
    if not torch.isfinite(w).all() or not torch.isfinite(table).all() or (table <= 0).any():
        raise ValueError("Non-finite weights or non-positive activation scales")
    table = table.float().to(w.device)
    k = w.shape[1]
    result = {"version": 1, "mode": mode, "group_size": group_size,
              "in_features": k, "out_features": w.shape[0],
              "act_scale_table": table.cpu(),
              "weight_source": "requantized_legacy_dense", "activation_bits": 4}
    if mode == "step_weight":
        # T+1 banks: the final bank matches the legacy mean-scale fallback.
        banks = [quantize_weight(w.float() * s[None, :], group_size)
                 for s in list(table) + [table.mean(dim=0)]]
        result["packed_weight"] = torch.stack([p.cpu() for p, _ in banks])
        result["weight_scales"] = torch.stack([s.cpu() for _, s in banks])
        result["fallback_bank"] = table.shape[0]
    else:
        p, s = quantize_weight(w, group_size)
        result.update(packed_weight=p.cpu(), weight_scales=s.cpu())
    if mode == "group_int":
        # Conservative envelope of existing channel thresholds. NOT the pooled
        # group percentile. No fabricated calibration claim when X is absent.
        result["group_scale_source"] = "max_of_channel_thresholds"
        result["group_scale_table"] = torch.stack([
            table[:, lo:lo + group_size].amax(dim=1)
            for lo in range(0, k, group_size)], dim=1).cpu()
    return result


class PerStepBackend(nn.Module):
    def __init__(self, record: dict[str, Any]):
        super().__init__()
        if record.get("version") != 1 or record.get("mode") not in MODES:
            raise ValueError("Unsupported perstep backend record")
        self.mode = record["mode"]
        self.group_size = int(record["group_size"])
        self.in_features = int(record["in_features"])
        self.out_features = int(record["out_features"])
        for key in ("packed_weight", "weight_scales", "act_scale_table"):
            self.register_buffer(key, record[key].contiguous())
        self.register_buffer("mean_scale", self.act_scale_table.mean(dim=0))
        if self.mode == "group_int":
            self.register_buffer("group_scale_table", record["group_scale_table"].contiguous())
            self.register_buffer("mean_group_scale", self.group_scale_table.mean(dim=0))
        self._validate_shapes()

    def _validate_shapes(self):
        t = self.act_scale_table.shape[0]
        n, k, g = self.out_features, self.in_features, self.group_size
        if g not in (32, 64, 128, 256) or self.act_scale_table.shape != (t, k) or t == 0:
            raise ValueError("Malformed per-step activation scales or group size")
        prefix = (t + 1,) if self.mode == "step_weight" else ()
        if self.packed_weight.shape != prefix + (n, (k + 1) // 2):
            raise ValueError("Malformed packed weight shape")
        if self.weight_scales.shape != prefix + (n, (k + g - 1) // g):
            raise ValueError("Malformed weight scale shape")
        if self.packed_weight.dtype != torch.uint8:
            raise ValueError("packed_weight must be uint8")
        for value in (self.weight_scales, self.act_scale_table):
            if value.dtype != torch.float32 or not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError("Quantization scales must be finite, positive FP32 tensors")
        if self.mode == "group_int" and self.group_scale_table.shape != (t, (k + g - 1) // g):
            raise ValueError("Malformed group scale shape")

    def selected(self, step: int | None):
        valid = step is not None and 0 <= step < self.act_scale_table.shape[0]
        a = self.act_scale_table[step] if valid else self.mean_scale
        p, s = self.packed_weight, self.weight_scales
        if self.mode == "step_weight":
            bank = step if valid else self.act_scale_table.shape[0]
            p, s = p[bank], s[bank]
        if self.mode == "group_int":
            a = self.group_scale_table[step] if valid else self.mean_group_scale
        return p, s, a

    def forward(self, x: torch.Tensor, step: int | None, act_bits: int = 4,
                engine: str | None = None):
        if act_bits != 4 or x.shape[-1] != self.in_features:
            raise ValueError("Packed per-step backends require A4 and matching input width")
        engine = engine or os.environ.get("GR00T_PERSTEP_ENGINE", "triton")
        p, s, a = self.selected(step)
        if x.device != p.device:
            raise ValueError("Move PerStepBackend to the input device before forwarding")
        if engine == "torch_reference":
            return self.reference(x, p, s, a)
        if engine != "triton" or not x.is_cuda:
            raise RuntimeError("Triton CUDA is required; use engine='torch_reference' only for debugging")
        from .perstep_kernels import residual_linear
        return residual_linear(x, p, s, a, self.group_size, self.mode)

    def reference(self, x, p, s, a):
        """Independent unfused oracle; never reported as GPU kernel speed."""
        k, g = self.in_features, self.group_size
        index = torch.arange(k, device=x.device) // g
        channel_scale = a[index] if self.mode == "group_int" else a
        q = (x.float() / channel_scale).round().clamp(-8, 7)
        w = dequant_weight(p, s, k, g)
        if self.mode == "fused_float":
            # Matches FP16 operand conversion of fused_float, FP32 accumulation.
            xh = (q * channel_scale).half().float()
            return F.linear(xh, w.half().float())
        if self.mode == "step_weight":
            return F.linear(q, w)
        return F.linear(q * channel_scale, w)


def record_weight(record):
    for key in ("weight_res_q", "baseline_q", "weight_q", "W_q", "quant_weight", "weight"):
        if key in record:
            return record[key]
    raise ValueError("Record has no legacy dense quantized weight")


def transform_input(x, record):
    """Same pre-quantization input domain as GptqLinear; for capture replay."""
    if "duquant_rotation_blocks" in record and "duquant_rotation_perm" in record:
        b = record["duquant_rotation_blocks"].to(x.device, x.dtype)
        perm = record["duquant_rotation_perm"].to(x.device, torch.long)
        shape = x.shape
        y = x.reshape(-1, shape[-1])[:, perm].reshape(-1, b.shape[0], b.shape[1])
        x = torch.bmm(y.transpose(0, 1).contiguous(), b).transpose(0, 1).reshape(shape)
    elif "duquant_rotation" in record:
        x = x @ record["duquant_rotation"].to(x.device, x.dtype)
    if "smooth_scale" in record:
        x = x / record["smooth_scale"].to(x.device, x.dtype)
    return x
