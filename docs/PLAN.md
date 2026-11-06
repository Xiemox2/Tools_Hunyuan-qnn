# Theoretical QNN plan for HunyuanVideo (UNTESTED)

Status: **theoretical scaffolding only**. No ONNX of a real checkpoint, no QNN context binary
and no on-device run has been produced from these tools. Numbers below are planning estimates
from public model configs and earlier notes; measure everything before trusting them.

## Scope
Primary target: **HunyuanVideo-1.5** T2V at 480p (848×480, ~33 frames), Hexagon **HTP v81**
(SM8850). Optional config sketch for the older **HunyuanVideo 13B** (2024) lives in
`cfg/hv13b_t2v.json` for sizing only.

Out of scope for this repo: Local Dream / Nightmare Mobile sources, ggml-hexagon runtime,
paid SDXL packs, Qualcomm SDK redistributables, and any claim of a working phone app.

## Why segments
HV 1.5 DiT is ~8.3B (54 double-stream blocks, hidden 2048, 16 heads, patch 1×1×1, VAE 16×
spatial / 4× temporal / 32 channels, TE Qwen2.5-VL-7B text states). A single QNN context for
the whole DiT plus long token sequences will not fit VTCM/RAM. Split blocks into **segments**
(default 9 blocks → 6 contexts), export each with **static shapes**, host-side RoPE /
modulation / text packing, then chain them at runtime. Manifest format: `hv.json`
(see `py/make_manifest.py`), analogous to `zit.json` in ZIT-HEXAGON QNN.

## Pipeline (intended, not executed end-to-end here)
1. `py/plan_segments.py cfg/hv15_480p_t2v.json` - pure arithmetic plan (tokens, MB estimates).
2. `py/export_dit_segment.py` - ONNX per segment. **Adapter not implemented** (fails clearly).
3. Provide FP32 calibration inputs from a reference host run (not supplied).
4. `sh/convert_segment.sh` - QAIRT converter → quantizer (W8A16 default) → HTP v81 context
   binary (`cfg/htp_v81.json`). Needs your own QAIRT install under your licence.
5. `py/export_vae_decoder.py` - TAEHV 1.5 or full VAE chunked decode. **Not implemented**
   (CausalConv3d replicate-pad rewrite pending). Planning notes rate a fixed-shape VAE
   decoder as the most realistic first QNN binary if someone continues.
6. Text encoder: keep on CPU/llama.cpp GGUF (intermediate hidden layer). No QNN TE export here.
7. `py/make_manifest.py` - write `hv.json` listing binaries, I/O names and sha256.

Driver: `sh/run_all.sh` stops at the first unimplemented adapter on purpose.

## Honest gaps
- DiT weight load + per-block forward wiring for HunyuanVideo streams.
- VAE CausalConv3d replicate padding → exportable pad+conv.
- Calibration set generation and fold tables.
- Runtime that loads multiple HTP contexts and feeds host tensors between segments.
- I2V / SigLIP / ByT5 / LoRA paths.

## Licences
Toolkit: MIT. Weights and derivatives: Tencent Hunyuan Community License (+ TE/VAE upstream).
Do not ship Qualcomm QAIRT/Hexagon SDK files in this repo.
