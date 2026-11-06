#!/usr/bin/env python3
"""Theoretical, UNTESTED. Plans DiT segmentation and memory for a HunyuanVideo QNN build.

Computes tokens per shape, splits blocks into segments and gives rough weight/activation
sizes per context binary. Pure arithmetic: no model, no SDK needed.
"""
import argparse, json, math, sys

BYTES = {"W4": 0.5, "W8": 1.0, "W16": 2.0}


def tokens(cfg, w, h, frames):
    s, t = cfg["vae"]["spatial"], cfg["vae"]["temporal"]
    pt, ph, pw = cfg["patch"]
    lat_f = (frames - 1) // t + 1
    lw, lh = w // s // pw, h // s // ph  # 16 px per token in both HV 1.5 (16x, patch 1) and 13B (8x, patch 2)
    return lw * lh * (lat_f // pt)


def block_params(hidden, mlp, double):
    attn = 4 * hidden * hidden          # qkv + out
    ffn = 2 * hidden * hidden * mlp      # fc1 + fc2
    mod = 6 * hidden * hidden            # adaLN modulation
    per_stream = attn + ffn + mod
    return per_stream * (2 if double else 1)


def plan(cfg):
    h, mlp = cfg["hidden"], cfg["mlp_ratio"]
    blocks = [("double", i) for i in range(cfg["double_blocks"])] + \
             [("single", i) for i in range(cfg["single_blocks"])]
    per = cfg["segments"]["blocks_per_segment"]
    wb = BYTES[cfg["quant"]["weights"]]
    segs = []
    for k in range(0, len(blocks), per):
        chunk = blocks[k:k + per]
        p = sum(block_params(h, mlp, kind == "double") for kind, _ in chunk)
        segs.append({"index": len(segs), "blocks": [f"{a}_{b}" for a, b in chunk],
                     "params_M": round(p / 1e6, 1), "weights_MB": round(p * wb / 2**20, 1)})
    shapes = []
    for s in cfg["shapes"]:
        n = tokens(cfg, s["w"], s["h"], s["frames"])
        act = n * h * 2 * 8  # fp16, ~8 live tensors of [tokens, hidden] (rough)
        shapes.append({**s, "tokens": n, "activations_MB_est": round(act / 2**20, 1)})
    return {"model": cfg["model"], "segments": segs, "shapes": shapes,
            "note": "Rough estimates only. Measure on device before trusting them."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="cfg/hv15_480p_t2v.json or cfg/hv13b_t2v.json")
    ap.add_argument("-o", "--out", help="write plan JSON here")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = json.dumps(plan(cfg), indent=2)
    if a.out:
        open(a.out, "w").write(out)
    print(out)


if __name__ == "__main__":
    sys.exit(main())
