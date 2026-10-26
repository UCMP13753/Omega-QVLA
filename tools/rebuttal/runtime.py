"""Strict reference runtime: fake quantization followed by floating-point GEMM.

It intentionally makes no native-INT4 speed claim. Backbone activations are
per-token; head activations use an offline pooled or per-step channel table.
No output-side rotation or high-precision low-rank branch is used.
"""
import torch
from torch import nn
from tools.rebuttal.core import block_transform
from gr00t.quantization.dit_step_context import get_current_dit_step, get_total_dit_steps


class RebuttalLinear(nn.Module):
    def __init__(self, base, rec, config, name):
        super().__init__()
        self.name = name
        self.in_features, self.out_features = base.in_features, base.out_features
        self.bits = config['abits']
        self.steps = config['steps']
        self.scale_mode = config['scaling']
        self.side = rec['side']
        self.register_buffer('weight', rec['weight_q'].to(base.weight.device, dtype=base.weight.dtype))
        self.register_buffer('blocks', rec['blocks'].to(base.weight.device))
        self.register_buffer('perm', rec['perm'].to(base.weight.device))
        self.register_buffer('scales', rec['scales'].to(base.weight.device))
        self.register_buffer('bias', base.bias.detach().clone() if base.bias is not None else None)

    def forward(self, x):
        # FP32 scaling prevents tiny per-channel scales underflowing in FP16.
        with torch.profiler.record_function('rebuttal/input_transform'):
            xr = block_transform(x.float(), self.blocks.float(), self.perm)
        if self.bits:
            with torch.profiler.record_function('rebuttal/activation_fake_quant'):
                if self.side == 'backbone':
                    scale = xr.abs().amax(-1, keepdim=True).clamp_min(1e-8) / (2 ** (self.bits-1)-1)
                elif self.scale_mode == 'perstep':
                    t = get_current_dit_step()
                    if t is None or not 0 <= t < self.steps or get_total_dit_steps() != self.steps:
                        raise RuntimeError(f'{self.name}: invalid denoising context {t}/{get_total_dit_steps()}')
                    scale = self.scales[t]
                else:
                    scale = self.scales[0]
                qmax = 2 ** (self.bits-1)-1
                xr = torch.round(xr / scale).clamp(-qmax, qmax) * scale
        with torch.profiler.record_function('rebuttal/floating_gemm'):
            dtype = torch.bfloat16 if x.is_cuda and torch.cuda.is_bf16_supported() else torch.float32
            y = torch.nn.functional.linear(xr.to(dtype), self.weight.to(dtype), None).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


def install(model, pack, config):
    expected = set(pack['records'])
    found = set()
    for name, base in list(model.named_modules()):
        if name not in expected:
            continue
        if not isinstance(base, nn.Linear):
            raise TypeError(f'{name} is not Linear')
        rec = pack['records'][name]
        if tuple(rec['weight_q'].shape) != tuple(base.weight.shape):
            raise ValueError(f'Weight shape mismatch: {name}')
        parent, _, attr = name.rpartition('.')
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, attr, RebuttalLinear(base, rec, config, name))
        found.add(name)
    if found != expected:
        raise RuntimeError(f'Pack coverage mismatch: missing={sorted(expected-found)}')
    return sorted(found)
