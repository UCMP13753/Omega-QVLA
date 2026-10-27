"""CPU execution of Triton pointer/mask/packing/accumulation logic.

Run with TRITON_INTERPRET=1 and numpy<2 (Triton 3.1 interpreter requirement).
CUDA libdevice nearbyint is shimmed with NumPy ties-to-even rint because that
external function is unsupported by the interpreter. Actual GPU tests exercise
the real libdevice instruction without this shim.
"""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1",
                                reason="Opt in with TRITON_INTERPRET=1")


@pytest.mark.parametrize("mode", ("fused_float", "step_weight", "group_int"))
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_interpreted_kernel_matches_oracle(mode, group_size, monkeypatch):
    import numpy as np
    import torch
    pytest.importorskip("triton")
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle
    from gr00t.quantization.perstep_backends import make_backend_record, PerStepBackend
    from gr00t.quantization import perstep_kernels as kernels
    monkeypatch.setattr(kernels.libdevice, "nearbyint", lambda x:
                        tl.tensor(TensorHandle(np.rint(x.handle.data), x.handle.dtype), x.type))
    generator = torch.Generator().manual_seed(2)
    w = torch.randn((35, 267), generator=generator) * .1
    table = torch.rand((3, 267), generator=generator) * .1 + .01
    x = torch.randn((19, 267), generator=generator).half()
    table[:, :4] = .25
    x[:, :4] = torch.tensor([.125, .375, -.125, -.375])
    b = PerStepBackend(make_backend_record(w, table, mode, group_size))
    for step in (0, 2, None):
        p, s, a = b.selected(step)
        y = kernels.residual_linear(x, p, s, a, group_size, mode)
        ref = b(x, step, engine="torch_reference")
        error = (y - ref).square().sum() / ref.square().sum().clamp_min(1e-30)
        assert error < 1e-6
