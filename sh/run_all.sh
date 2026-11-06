#!/usr/bin/env bash
# Theoretical, UNTESTED end-to-end driver. Stops at the first unimplemented adapter.
# Usage: sh/run_all.sh cfg/hv15_480p_t2v.json /path/to/checkpoint out_pkg/
set -euo pipefail
CFG=${1:?config}; CKPT=${2:?checkpoint}; PKG=${3:?output package dir}
HERE=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$PKG/onnx" "$PKG/calib"
python3 "$HERE/py/plan_segments.py" "$CFG" -o "$PKG/plan.json" >/dev/null
N=$(python3 -c "import json;print(len(json.load(open('$PKG/plan.json'))['segments']))")
for ((i=0;i<N;i++)); do
  python3 "$HERE/py/export_dit_segment.py" --config "$CFG" --plan "$PKG/plan.json" \
    --segment $i --checkpoint "$CKPT" --out "$PKG/onnx/segment_$i.onnx"
  # Calibration inputs (raw fp32 tensors from a reference run) must be produced separately.
  "$HERE/sh/convert_segment.sh" "$PKG/onnx/segment_$i.onnx" "$PKG/calib/segment_$i.txt" "$PKG/dit_$i.bin"
done
python3 "$HERE/py/export_vae_decoder.py" --checkpoint "$CKPT" --out "$PKG/onnx/vae_decoder.onnx"
python3 "$HERE/py/make_manifest.py" --config "$CFG" --plan "$PKG/plan.json" --pkg "$PKG"
