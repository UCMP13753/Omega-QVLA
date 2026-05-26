#!/usr/bin/env python
"""Merge a DiT-W4 pack with an LLM-W3 pack to form a mixed-precision pack.

Usage:
    python merge_mixed_precision_pack.py \
        --llm-pack <path/to/llm_w3_pack.pt> \
        --dit-pack <path/to/dit_w4_pack.pt> \
        --output  <path/to/merged.pt>

Logic:
    - Take all entries from llm_pack except those matching the DiT regex.
    - Add all DiT-matched entries from dit_pack (overriding any DiT entries
      that may have been in llm_pack).
    - Carry over __meta__ from llm_pack but mark it as mixed-precision.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

DIT_PATTERN = re.compile(
    r"action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm-pack", required=True)
    p.add_argument("--dit-pack", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    llm = torch.load(args.llm_pack, map_location="cpu", weights_only=False)
    dit = torch.load(args.dit_pack, map_location="cpu", weights_only=False)

    merged: dict = {}
    if "__meta__" in llm:
        meta = dict(llm["__meta__"])
        meta["mixed_precision"] = "DiT W4 + LLM W3"
        meta["llm_pack_source"] = args.llm_pack
        meta["dit_pack_source"] = args.dit_pack
        merged["__meta__"] = meta

    n_llm = 0
    n_dit_in_llm_skipped = 0
    for k, v in llm.items():
        if k == "__meta__":
            continue
        if DIT_PATTERN.search(k):
            n_dit_in_llm_skipped += 1
            continue
        merged[k] = v
        n_llm += 1

    n_dit = 0
    for k, v in dit.items():
        if k == "__meta__":
            continue
        if not DIT_PATTERN.search(k):
            print(f"  warning: dit_pack has non-DiT key '{k}', skipping", file=sys.stderr)
            continue
        merged[k] = v
        n_dit += 1

    print(f"[merge] LLM kept: {n_llm}")
    print(f"[merge] DiT in LLM skipped: {n_dit_in_llm_skipped}")
    print(f"[merge] DiT from DiT pack: {n_dit}")
    print(f"[merge] total layer records: {len(merged) - 1}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(f"[merge] wrote {args.output}")
    # Quick verification: report bits per record sample
    print("[merge] sample weight_bits per layer type:")
    for k, v in merged.items():
        if k == "__meta__" or not isinstance(v, dict):
            continue
        wbits = v.get("weight_bits", "?")
        sq = "+SQ" if "smooth_scale" in v else ""
        print(f"  {k[:80]:80s} bits={wbits}{sq}")
        # Sample a few keys then bail
        if k.endswith("ff.net.2") or k.endswith("self_attn.v_proj"):
            pass


if __name__ == "__main__":
    main()
