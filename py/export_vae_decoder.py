#!/usr/bin/env python3
"""Theoretical, UNTESTED. Exports the HunyuanVideo VAE decoder (or TAEHV 1.5) to ONNX
for one fixed latent shape, decoded in temporal chunks.

The planning notes rate this as the most realistic QNN target (no LoRA, fixed shape).
The CausalConv3d with replicate padding must be rewritten as explicit pad + conv3d
(or conv2d over time slices) before export; that rewrite is NOT implemented.
"""
import argparse, sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kind", choices=["full", "taehv15"], default="taehv15")
    ap.add_argument("--latent", default="1,32,3,30,53", help="N,C,T,H,W of one temporal chunk")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("[export_vae_decoder] NOT IMPLEMENTED: decoder adapter (CausalConv3d replicate-pad "
          "rewrite, chunked temporal cache) is pending. See docs/PLAN.md 'VAE'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
