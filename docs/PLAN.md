# Plan (theoretical / teórico)

Status: **untested**. Numbers are estimates from planning notes, not measurements.

## Why QNN for Hunyuan is hard
- HV 1.5 DiT is ~8.3B params with 14k–50k tokens. QNN needs static shapes, so one set of binaries per resolution and frame count.
- The DiT must be split into several context binaries (here 6 segments of 9 double blocks) because of VTCM/RAM limits per session.
- W8A16 (1.5) or W4A16 (13B) with calibration is likely needed. Calibration data must come from a reference FP32/BF16 run.
- LoRA: weights are compiled into the binary. Dynamic LoRA would need recompilation or updatable tensors (support on HTP v81 unverified).

## Most realistic first target: VAE decoder
The VAE decoder (or TAEHV 1.5) at a fixed chunk shape has no LoRA and a fixed shape. If decoding is the bottleneck, this is the best first QNN experiment.

## DiT adapter
- Inputs per segment: `img [1,N_img,H]`, `txt [1,N_txt,H]`, `vec [1,H]`, `rope_cos/rope_sin [N_img+N_txt, head_dim]`.
- Keep RoPE outside the graph (precomputed) to avoid complex/unsupported ops.
- HV 1.5: 54 double-stream blocks, 0 single. HV 13B: 20 double + 40 single, plus `vector_in` (CLIP-L pooled) and `guidance_in`.

## VAE
- `CausalConv3d` with replicate padding: rewrite as explicit pad + conv3d (or conv2d over time slices).
- Decode in temporal chunks with a carried cache; one binary per chunk shape.

## Text encoder
- Recommended to stay on ggml-hexagon (HTP0) as in the working Hunyuan NPU app; QNN LLM toolchains may not expose the intermediate hidden layer HV needs.

## Runtime
- `hv.json` lists binaries per segment, I/O names and sha256. A runtime would load segments in order per step (possibly streaming them from disk on 16 GB devices).
