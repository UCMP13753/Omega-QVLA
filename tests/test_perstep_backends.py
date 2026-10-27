"""CPU algebra/integration tests and actual GPU kernel checks when available."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from gr00t.quantization.perstep_backends import (
    MODES, PerStepBackend, dequant_weight, make_backend_record, pack_int4, unpack_int4,
)


def example(k=67, n=35, steps=3):
    generator = torch.Generator().manual_seed(7)
    w = torch.randn((n, k), generator=generator) * .1
    table = torch.rand((steps, k), generator=generator) * .2 + .01
    x = torch.randn((19, k), generator=generator).half()
    return w, table, x


def test_signed_nibbles_odd_width():
    q = torch.arange(-8, 8).repeat(3)[:35].reshape(5, 7).to(torch.int8)
    assert pack_int4(q).dtype == torch.uint8
    torch.testing.assert_close(unpack_int4(pack_int4(q), 7), q)


@pytest.mark.parametrize("mode", MODES)
def test_reference_and_fallback(mode):
    w, table, x = example()
    r = make_backend_record(w, table, mode, 32)
    b = PerStepBackend(r)
    for t in (0, 1, 2, None, 99):
        p, s, a = b.selected(t)
        y = b(x, t, engine="torch_reference")
        w_new = dequant_weight(p, s, w.shape[1], 32)
        scale = a.repeat_interleave(32)[:w.shape[1]] if mode == "group_int" else a
        q = (x.float() / scale).round().clamp(-8, 7)
        if mode == "fused_float":
            ref = (q * scale).half().float() @ w_new.half().float().T
        elif mode == "step_weight":
            ref = q @ w_new.T
        else:
            ref = (q * scale) @ w_new.T
        torch.testing.assert_close(y, ref)
        assert torch.isfinite(y).all()
    torch.testing.assert_close(b(x, None, engine="torch_reference"),
                               b(x, 99, engine="torch_reference"))
    assert not torch.equal(b(x, 0, engine="torch_reference"), b(x, 1, engine="torch_reference"))
    if mode == "step_weight":
        assert b.packed_weight.shape[0] == table.shape[0] + 1


def test_reject_bad_inputs():
    w, table, x = example()
    with pytest.raises(ValueError):
        make_backend_record(w, table * 0, "fused_float")
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([8]))
    b = PerStepBackend(make_backend_record(w, table, "fused_float"))
    with pytest.raises(ValueError):
        b(x, 0, act_bits=8, engine="torch_reference")
    with pytest.raises(RuntimeError):
        b(x, 0, engine="triton")  # never silently use reference for GPU timing


def test_export_cli_preserves_unconverted_records(tmp_path):
    import subprocess
    w, table, _ = example(k=8, n=6)
    original = {"layer": dict(weight_res_q=w, act_scale_table=table, a_bits=4, weight_bits=4),
                "backbone": dict(baseline_q=torch.ones(3, 4)), "__meta__": {"test": True}}
    source = tmp_path / "source.pt"
    output = tmp_path / "converted"
    torch.save(original, source)
    subprocess.run([sys.executable, "-m", "tools.compare_perstep_backends", "--pack", str(source),
                    "--output", str(output), "--export-only", "--device", "cpu"], check=True)
    for mode in MODES:
        pack = torch.load(output / f"{mode}.pt", weights_only=False)
        assert "weight_res_q" not in pack["layer"]
        assert pack["__meta__"] == original["__meta__"]
        torch.testing.assert_close(pack["backbone"]["baseline_q"], original["backbone"]["baseline_q"])
        PerStepBackend(pack["layer"]["perstep_backend"])
    unchanged = torch.load(source, weights_only=False)
    torch.testing.assert_close(unchanged["layer"]["weight_res_q"], w)


@pytest.mark.parametrize("mode", MODES)
def test_gptq_runtime_rotation_lowrank_bias_and_pack_reload(mode, tmp_path, monkeypatch):
    from gr00t.quantization.gptq_layers import GptqConfig, GptqLinear
    from gr00t.quantization.dit_step_context import set_dit_quant_step
    w, table, x = example(k=8, n=6)
    rin, _ = torch.linalg.qr(torch.randn(8, 8))
    rout, _ = torch.linalg.qr(torch.randn(6, 6))
    a, b_lr = torch.randn(6, 2), torch.randn(8, 2)
    smooth = torch.rand(8) + .5
    record = dict(format="dit_svdquant_v1", a_bits=4, lowrank_A=a, lowrank_B=b_lr,
                  duquant_rotation=rin, duquant_rotation_out=rout, smooth_scale=smooth,
                  act_scale_table=table, perstep_backend=make_backend_record(w, table, mode, 32))
    path = tmp_path / "pack.pt"
    torch.save({"layer": record}, path)
    base = torch.nn.Linear(8, 6)
    module = GptqLinear(base, "layer", GptqConfig(enabled=True, path=str(path), act_bits=4))
    monkeypatch.setenv("GR00T_PERSTEP_ENGINE", "torch_reference")
    xx = x.float()
    for step in (0, 2, None, 999):
        with set_dit_quant_step(step):
            y = module(xx)
        transformed = (xx @ rin) / smooth
        residual = module._perstep_backend(transformed, step, engine="torch_reference")
        expected = (residual + transformed @ b_lr @ a.T) @ rout + base.bias
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)
    assert module._weight_q.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs an actual CUDA GPU")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_triton_gpu(mode, group_size):
    pytest.importorskip("triton")
    w, table, x = example(k=267, n=35)
    table[:, :4] = .25
    x[:, :4] = torch.tensor([.125, .375, -.125, -.375])
    b = PerStepBackend(make_backend_record(w, table, mode, group_size)).cuda()
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        xx = x.cuda().to(dtype)
        for step in (0, 2, None):
            y = b(xx, step, engine="triton")
            ref = b(xx, step, engine="torch_reference")
            error = (y - ref).square().sum() / ref.square().sum().clamp_min(1e-30)
            assert error < 1e-6
            assert torch.isfinite(y).all()
