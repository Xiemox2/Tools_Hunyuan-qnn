#!/usr/bin/env python3
"""Theoretical, UNTESTED. Writes hv.json, the runtime manifest for a HunyuanVideo QNN
package, analogous to zit.json in ZIT-HEXAGON QNN: lists context binaries per segment,
I/O tensor names/shapes and sha256 of every file."""
import argparse, hashlib, json, os, sys


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--pkg", required=True, help="dir with dit_XX.bin, vae_decoder.bin, te/")
    ap.add_argument("--shape", type=int, default=0)
    a = ap.parse_args()
    cfg, plan = json.load(open(a.config)), json.load(open(a.plan))
    shp = plan["shapes"][a.shape]
    files = []
    for s in plan["segments"]:
        f = f"dit_{s['index']}.bin"
        p = os.path.join(a.pkg, f)
        files.append({"name": f, "role": "dit_segment", "blocks": s["blocks"],
                      "sha256": sha256(p) if os.path.exists(p) else None})
    for f, role in (("vae_decoder.bin", "vae_decoder"), ("text_encoder.gguf", "text_encoder")):
        p = os.path.join(a.pkg, f)
        files.append({"name": f, "role": role, "sha256": sha256(p) if os.path.exists(p) else None})
    man = {"format": "hv-qnn-manifest", "version": 0, "status": "theoretical-untested",
           "model": cfg["model"], "width": shp["w"], "height": shp["h"], "frames": shp["frames"],
           "tokens": shp["tokens"], "hidden": cfg["hidden"], "head_dim": cfg["head_dim"],
           "dit_io": {"inputs": ["img", "txt", "vec", "rope_cos", "rope_sin"],
                      "outputs": ["img_out", "txt_out"]},
           "files": files}
    out = os.path.join(a.pkg, "hv.json")
    json.dump(man, open(out, "w"), indent=2)
    missing = [f["name"] for f in files if f["sha256"] is None]
    print(f"wrote {out}" + (f" (missing: {', '.join(missing)})" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
