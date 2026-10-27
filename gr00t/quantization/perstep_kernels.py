"""4080-compatible experimental Triton kernels. No native INT4 MMA claim.

Activation quantization is fused into the dot kernel, so the full dequantized
activation/weight matrices are never materialized. Quantization is repeated per
output tile, an explicit prototype tradeoff to measure rather than assume fast.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _linear(X, P, S, A, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            NG: tl.constexpr, PK: tl.constexpr, G: tl.constexpr,
            MODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, G)
    acc = tl.full((BM, BN), 0, tl.float32)
    for group in range(NG):
        kk = group * G + offsets
        x = tl.load(X + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), 0).to(tl.float32)
        if MODE == 2:
            a = tl.load(A + group)
        else:
            a = tl.load(A + kk, kk < K, 1)[None, :]
        aq = tl.minimum(tl.maximum(libdevice.nearbyint(tl.div_rn(x, a)), -8.), 7.)
        byte = tl.load(P + cols[None, :] * PK + kk[:, None] // 2,
                       (cols[None, :] < N) & (kk[:, None] < K), 0).to(tl.int32)
        nibble = (byte >> ((kk[:, None] % 2) * 4)) & 15
        wq = tl.where(nibble >= 8, nibble - 16, nibble)
        scale = tl.load(S + cols * NG + group, cols < N, 0)[None, :]
        if MODE == 0:
            af = (aq * a).to(tl.float16)
            wf = (wq.to(tl.float32) * scale).to(tl.float16)
            acc += tl.dot(af, wf)
        else:
            partial = tl.dot(aq.to(tl.int8), wq.to(tl.int8)).to(tl.float32)
            if MODE == 2:
                partial *= a
            acc += partial * scale
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def residual_linear(x, packed, scales, act_scale, group_size, mode):
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Expected FP16/BF16/FP32 input")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x2.shape
    n = packed.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m:
        _linear[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            x2, packed, scales, act_scale, y, m, n, k,
            triton.cdiv(k, group_size), triton.cdiv(k, 2), group_size,
            {"fused_float": 0, "step_weight": 1, "group_int": 2}[mode],
            16, 32, num_warps=4)
    return y.reshape(*x.shape[:-1], n)
