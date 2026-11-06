#!/usr/bin/env python3
"""Theoretical, UNTESTED. Exports one DiT segment (a contiguous run of blocks) to ONNX
with STATIC shapes for one resolution/frame count.

The model-specific adapter (loading HunyuanVideo weights and wiring RoPE, modulation and
text/image streams per block) is NOT implemented. It fails on purpose with a clear message
instead of producing a broken graph. Fill in `load_blocks` and `SegmentWrapper.forward`.
"""
import argparse, json, sys


class NotImplementedAdapter(RuntimeError):
    pass


def load_blocks(checkpoint, block_names):
    raise NotImplementedAdapter(
        "load_blocks: adapter for HunyuanVideo weights not written. Expected: load the "
        "safetensors, build the double/single stream blocks listed in the plan and return "
        "nn.Modules in FP32. See docs/PLAN.md section 'DiT adapter'.")


def build_wrapper(blocks, cfg, n_img, n_txt):
    import torch

    class SegmentWrapper(torch.nn.Module):
        """Inputs: img [1,n_img,H], txt [1,n_txt,H], vec [1,H], rope_cos/sin [n_img+n_txt,head_dim]."""
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(blocks)

        def forward(self, img, txt, vec, rope_cos, rope_sin):
            raise NotImplementedAdapter(
                "SegmentWrapper.forward: per-block call signature for HunyuanVideo not wired. "
                "RoPE must come in as precomputed cos/sin inputs (static shapes, no complex ops).")

    return SegmentWrapper()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan", required=True, help="output of plan_segments.py")
    ap.add_argument("--segment", type=int, required=True)
    ap.add_argument("--shape", type=int, default=0, help="index in config 'shapes'")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="segment_XX.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    cfg, plan = json.load(open(a.config)), json.load(open(a.plan))
    seg, shp = plan["segments"][a.segment], plan["shapes"][a.shape]
    n_img, n_txt, H = shp["tokens"], cfg["text_encoder"]["max_tokens"], cfg["hidden"]
    try:
        blocks = load_blocks(a.checkpoint, seg["blocks"])
        import torch
        m = build_wrapper(blocks, cfg, n_img, n_txt).eval()
        hd = cfg["head_dim"]
        ex = (torch.zeros(1, n_img, H), torch.zeros(1, n_txt, H), torch.zeros(1, H),
              torch.zeros(n_img + n_txt, hd), torch.zeros(n_img + n_txt, hd))
        torch.onnx.export(m, ex, a.out, opset_version=a.opset,
                          input_names=["img", "txt", "vec", "rope_cos", "rope_sin"],
                          output_names=["img_out", "txt_out"], dynamic_axes=None)
    except NotImplementedAdapter as e:
        print(f"[export_dit_segment] NOT IMPLEMENTED: {e}", file=sys.stderr)
        return 2
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
