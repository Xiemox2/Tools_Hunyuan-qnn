#!/usr/bin/env bash
# Theoretical, UNTESTED. ONNX segment -> QNN DLC -> quantized DLC -> HTP v81 context binary.
# Requires QAIRT (Qualcomm AI Engine Direct SDK) installed locally under your own licence.
# Usage: sh/convert_segment.sh segment_00.onnx calib_00.txt out/dit_0.bin [W8|W4]
set -euo pipefail
ONNX=${1:?onnx}; CALIB=${2:?calibration input list}; OUT=${3:?output .bin}; WQ=${4:-W8}
: "${QAIRT_SDK_ROOT:?set QAIRT_SDK_ROOT to your QAIRT install}"
BIN="$QAIRT_SDK_ROOT/bin/x86_64-linux-clang"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
WBITS=8; [ "$WQ" = "W4" ] && WBITS=4
"$BIN/qairt-converter" --input_network "$ONNX" --output_path "$WORK/seg.dlc"
"$BIN/qairt-quantizer" --input_dlc "$WORK/seg.dlc" --output_dlc "$WORK/seg_q.dlc" \
  --input_list "$CALIB" --weights_bitwidth "$WBITS" --act_bitwidth 16 --bias_bitwidth 32
"$BIN/qnn-context-binary-generator" \
  --backend "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
  --dlc_path "$WORK/seg_q.dlc" \
  --model "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so" \
  --config_file "$(dirname "$0")/../cfg/htp_v81.json" \
  --binary_file "$(basename "${OUT%.bin}")" --output_dir "$(dirname "$OUT")"
echo "context binary: $OUT"
