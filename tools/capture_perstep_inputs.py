#!/usr/bin/env python
"""Capture bounded raw inputs of every per-step layer under the original pack.

Use different --init-offset values for calibration and held-out replay files.
Requires the existing GR00T/LIBERO environment. Does not modify the source pack.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-suite-name", default="libero_object")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task")
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--init-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-cap", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.token_cap <= 0 or args.num_samples <= 0:
        p.error("token-cap and num-samples must be positive")
    path = Path(args.output)
    if path.exists():
        p.error("Output already exists")
    pack = torch.load(args.pack, map_location="cpu", weights_only=False)
    targets = {n: r for n, r in pack.items() if isinstance(r, dict) and "act_scale_table" in r}
    if not targets or any("perstep_backend" in r for r in targets.values()):
        p.error("Provide the original dense per-step pack")
    # Match original quantized rollout, not FP activations. Use the pack's layer
    # names as the whitelist, including backbone layers present in merged packs.
    from tools.analyze_layerwise_quant_drift import (
        ensure_libero_runtime, load_libero_samples, seed_everything,
    )
    from gr00t.experiment.data_config import load_data_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.quantization.dit_step_context import get_current_dit_step
    seed_everything(args.seed)
    ensure_libero_runtime()
    os.environ["GR00T_LIBERO_INIT_OFFSET"] = str(args.init_offset)
    os.environ.update(GR00T_GPTQ="1", GR00T_GPTQ_PATH=str(Path(args.pack).resolve()),
                      GR00T_GPTQ_ABITS="4", GR00T_GPTQ_MISSING="error",
                      GR00T_GPTQ_SCOPE="",
                      GR00T_GPTQ_LAYERS=",".join(n for n in pack if n != "__meta__"))
    # These mutually exclusive wrappers should not be inherited from the shell.
    os.environ["GR00T_RTN"] = "0"
    os.environ["GR00T_ATM_ENABLE"] = "0"
    os.environ["GR00T_OHB_ENABLE"] = "0"
    for key in list(os.environ):
        if key.startswith("GR00T_DUQUANT_"):
            del os.environ[key]
    config = load_data_config(args.data_config)
    policy = Gr00tPolicy(model_path=args.checkpoint, modality_config=config.modality_config(),
                         modality_transform=config.transform(),
                         embodiment_tag=EmbodimentTag(args.embodiment_tag),
                         denoising_steps=args.denoising_steps, device=args.device)
    policy.model.eval()
    cache = {n: {} for n in targets}
    def make_hook(name):
        def hook(_module, inputs):
            step = get_current_dit_step()
            if step is None:
                return
            previous = cache[name].get(int(step))
            remaining = args.token_cap - (previous.shape[0] if previous is not None else 0)
            if remaining <= 0:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            chunk = x[:remaining].cpu().clone()
            cache[name][int(step)] = torch.cat((previous, chunk)) if previous is not None else chunk
        return hook
    modules = dict(policy.model.named_modules())
    handles = [modules[n].register_forward_pre_hook(make_hook(n)) for n in targets]
    try:
        samples = load_libero_samples(args, config)
        if isinstance(samples, tuple):
            samples = samples[-1]
        with torch.inference_mode():
            for sample in samples:
                seed_everything(sample["seed"])
                policy.get_action(sample["obs"])
    finally:
        for handle in handles:
            handle.remove()
    missing = [(n, t) for n, r in targets.items()
               for t in range(r["act_scale_table"].shape[0]) if t not in cache[n]]
    if missing:
        raise RuntimeError(f"Incomplete per-step capture; first missing entries: {missing[:10]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "perstep_raw_inputs_v1", "inputs": cache,
                "args": vars(args), "source_pack": str(Path(args.pack).resolve()),
                "note": "observed raw inputs per layer/step, accumulated up to token-cap"}, path)
    print(f"Saved {len(cache)} layers to {path}")


if __name__ == "__main__":
    main()
