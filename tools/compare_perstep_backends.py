#!/usr/bin/env python
"""Export three packs and benchmark real pack residuals, one layer at a time.

Without --captures, inputs are synthetic draws scaled by the saved table. These
are throughput/numerical smoke tests, NOT model accuracy or LIBERO evidence.
With --captures, replay real raw layer inputs captured by capture_perstep_inputs.
All output errors are relative to the EXISTING quantized model, not an FP model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gr00t.quantization.perstep_backends import (
    MODES, PerStepBackend, dequant_weight, make_backend_record, record_weight, transform_input,
)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    return 0


def nmse(y, ref):
    return float(((y.float() - ref.float()).square().sum() /
                  ref.float().square().sum().clamp_min(1e-30)).item())


def export_all(pack, output, names, group_size, device, captures=None):
    original = load(pack)
    if not isinstance(original, dict):
        raise ValueError("Expected a flat named-layer pack")
    eligible = {n for n, r in original.items()
                if isinstance(r, dict) and "act_scale_table" in r}
    names = names or sorted(eligible)
    if not names or set(names) - eligible:
        raise ValueError("No per-step layers, or requested layer has no scale table")
    outputs = {}
    for mode in MODES:
        converted = dict(original)
        for i, name in enumerate(names):
            r = original[name]
            if int(r.get("weight_bits", 4)) != 4 or int(r.get("a_bits", 4)) != 4:
                raise ValueError(f"{name}: comparison supports W4A4 only")
            w = record_weight(r).to(device)
            table = r["act_scale_table"].float().clamp_min(1e-8).to(device)
            b = make_backend_record(w, table, mode, group_size)
            # Use real inputs for group clipping if provided. Never approximate
            # pooled percentiles by averaging channel percentiles.
            if captures is not None and mode == "group_int":
                values = captures["inputs"][name]
                for t in range(table.shape[0]):
                    raw = values.get(t, values.get(str(t)))
                    if raw is None:
                        raise ValueError(f"Missing real calibration input: {name}, step {t}")
                    x = transform_input(raw.to(device), r).float().reshape(-1, w.shape[1])
                    for g, lo in enumerate(range(0, w.shape[1], group_size)):
                        threshold = torch.quantile(x[:, lo:lo + group_size].abs().flatten(), .999)
                        b["group_scale_table"][t, g] = (threshold / 7).clamp_min(1e-8).cpu()
                b["group_scale_source"] = "pooled_q99.9_real_capture"
            new_record = dict(r)
            for key in ("weight_res_q", "baseline_q", "weight_q", "W_q", "quant_weight", "weight"):
                new_record.pop(key, None)
            new_record["perstep_backend"] = b
            converted[name] = new_record
            if i % 12 == 0:
                print(f"[export] {mode} {i+1}/{len(names)} {name}", flush=True)
            del w
        path = output / f"{mode}.pt"
        torch.save(converted, path)
        outputs[mode] = {"path": str(path.resolve()), "file_bytes": path.stat().st_size,
                         "converted_layers": len(names), "total_perstep_layers": len(eligible)}
        del converted
    return original, names, outputs


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(repeats)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(a.elapsed_time(b) for a, b in events)
    return times[len(times) // 2], max(0, torch.cuda.max_memory_allocated() - resident)


def benchmark(original, names, outputs, args, captures):
    # Read a single backend pack at a time, never all T-bank model weights on GPU.
    rows = []
    for mode in MODES:
        converted = load(outputs[mode]["path"])
        for name in names:
            r = original[name]
            w = record_weight(r).cuda()
            table = r["act_scale_table"].float().clamp_min(1e-8).cuda()
            b = PerStepBackend(converted[name]["perstep_backend"]).cuda()
            for t in range(table.shape[0]):
                for tokens in args.tokens:
                    if captures is None:
                        generator = torch.Generator(device="cuda").manual_seed(args.seed + t)
                        x = torch.randn((tokens, w.shape[1]), device="cuda", generator=generator)
                        x = (x * table[t] * 3).half()
                        source = "synthetic_scale_matched"
                    else:
                        values = captures["inputs"][name]
                        raw = values.get(t, values.get(str(t)))
                        if raw is None:
                            raise ValueError(f"Missing replay input: {name} step={t}")
                        raw = raw.reshape(-1, w.shape[1])[:tokens].cuda()
                        x = transform_input(raw, r)
                        source = "real_layer_input"
                    # Match the original implementation, including its scale
                    # cast to the activation dtype and BF16 operand promotion.
                    a = table[t].to(x.dtype)
                    def legacy():
                        xq = (x / a).round().clamp(-8, 7) * a
                        return torch.nn.functional.linear(xq.bfloat16(), w.bfloat16()).to(x.dtype)
                    ref = legacy().float()
                    p, s, a_new = b.selected(t)
                    weight_target = w.float() * table[t][None, :] if mode == "step_weight" else w.float()
                    weight_nmse = nmse(dequant_weight(p, s, w.shape[1], args.group_size), weight_target)
                    oracle = b.reference(x, p, s, a_new)
                    y = b(x, t)
                    if not torch.isfinite(y).all() or not torch.isfinite(ref).all():
                        raise RuntimeError(f"Nonfinite output {mode}/{name}/{t}")
                    kernel_nmse = nmse(y, oracle)
                    if kernel_nmse > args.kernel_nmse_limit:
                        raise RuntimeError(f"Kernel/oracle NMSE {kernel_nmse} exceeds limit: {mode}/{name}/{t}")
                    legacy_ms, legacy_peak = measure(legacy, args.warmup, args.repeats)
                    ms, peak = measure(lambda: b(x, t), args.warmup, args.repeats)
                    row = dict(mode=mode, layer=name, step=t, tokens=x.shape[0],
                               requested_tokens=tokens, input_dtype=str(x.dtype),
                               input_source=source, repacked_weight_nmse=weight_nmse,
                               residual_nmse_vs_legacy=nmse(y.to(x.dtype), ref),
                               kernel_nmse_vs_oracle=kernel_nmse, legacy_ms=legacy_ms,
                               kernel_ms=ms, speedup=legacy_ms / ms,
                               legacy_peak_extra_bytes=legacy_peak, kernel_peak_extra_bytes=peak,
                               backend_resident_bytes=sum(v.numel()*v.element_size() for v in b.buffers()),
                               scope="residual_only_excludes_rotation_lowrank_bias")
                    rows.append(row)
                    with (Path(args.output) / "benchmark.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
            print(f"[bench] {mode} {name} complete", flush=True)
            del b, w, table, x, y, ref, oracle, p, s, a_new, weight_target
            torch.cuda.empty_cache()
        del converted
    with (Path(args.output) / "benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--captures", help="Held-out raw layer inputs, not calibration inputs")
    p.add_argument("--calibration-captures", help="Independent captures for pooled group calibration")
    p.add_argument("--group-size", type=int, default=128, choices=[32, 64, 128, 256])
    p.add_argument("--layer", action="append", dest="layers")
    p.add_argument("--tokens", type=int, nargs="+", default=[32, 128])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kernel-nmse-limit", type=float, default=1e-6)
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    if min(args.tokens) <= 0 or args.warmup < 0 or args.repeats <= 0:
        p.error("tokens/repeats must be positive, warmup must be nonnegative")
    if not args.export_only and (args.device != "cuda" or not torch.cuda.is_available()):
        p.error("GPU benchmarks require CUDA; CPU export is available with --export-only --device cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        p.error("Use a new empty output directory to keep runs separate")
    if args.captures and args.calibration_captures and Path(args.captures).resolve() == Path(args.calibration_captures).resolve():
        p.error("Use distinct calibration and held-out capture files")
    captures = load(args.captures) if args.captures else None
    calibration = load(args.calibration_captures) if args.calibration_captures else None
    for label, value in (("held-out", captures), ("calibration", calibration)):
        if value is not None and value.get("format") != "perstep_raw_inputs_v1":
            p.error(f"Unsupported {label} capture format")
    if captures and calibration:
        c, h = calibration["args"], captures["args"]
        if c["task_suite_name"] != h["task_suite_name"]:
            p.error("Calibration and held-out captures must use the same suite")
        cal_range = set(range(c["init_offset"], c["init_offset"] + c["libero_num_trials_per_task"]))
        held_range = set(range(h["init_offset"], h["init_offset"] + h["libero_num_trials_per_task"]))
        if cal_range & held_range:
            p.error("Calibration and held-out initial-state windows overlap")
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.time()
    with torch.inference_mode():
        original, names, outputs = export_all(args.pack, output, args.layers,
                                             args.group_size, args.device, calibration)
        rows = [] if args.export_only else benchmark(original, names, outputs, args, captures)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    report = {"args": vars(args), "revision": revision, "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "source_pack_bytes": Path(args.pack).stat().st_size,
              "outputs": outputs, "rows": len(rows), "elapsed_seconds": time.time() - start,
              "accuracy_scope": "local residual drift vs existing quantized pack, not FP accuracy or task success",
              "integer_instruction": "INT8 dot of unpacked INT4 codes"}
    report["summary"] = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        if selected:
            report["summary"][mode] = {
                "geomean_residual_speedup": math.exp(sum(math.log(r["speedup"]) for r in selected) / len(selected)),
                "mean_residual_nmse_vs_legacy": sum(r["residual_nmse_vs_legacy"] for r in selected) / len(selected),
                "max_residual_nmse_vs_legacy": max(r["residual_nmse_vs_legacy"] for r in selected),
                "max_kernel_nmse_vs_oracle": max(r["kernel_nmse_vs_oracle"] for r in selected),
            }
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
