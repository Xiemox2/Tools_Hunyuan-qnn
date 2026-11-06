# Tools_Hunyuan-qnn

> ## ⚠️ Theoretical, untested tools
> None of this has been run end to end or on a phone. No converted models exist and nothing is guaranteed to work.
> The author **won't continue this work for now** due to other priorities. It is published so others can finish it, improve it, or build on it.

Tools to convert **HunyuanVideo 1.5** (and optionally the original **HunyuanVideo 13B**) to **Qualcomm QNN/HTP** context binaries for Hexagon **v81** (SM8850, e.g. Nubia Z80 Ultra), following the same scheme as the ZIT-HEXAGON QNN path for ReaZIT.

[Español](README.md)

## Proposed flow

1. `py/plan_segments.py`: tokens per resolution, DiT split into segments, per-segment memory estimate.
2. `py/export_dit_segment.py`: each segment to ONNX with static shapes (RoPE as precomputed cos/sin inputs).
3. `sh/convert_segment.sh`: ONNX → `qairt-converter` → `qairt-quantizer` (W8A16 or W4A16) → `qnn-context-binary-generator` (v81).
4. `py/export_vae_decoder.py`: VAE / TAEHV 1.5 decoder, fixed shape, temporal chunks.
5. `py/make_manifest.py`: writes `hv.json`, the runtime manifest (like `zit.json`) with sha256s.
6. `sh/run_all.sh`: chains everything and stops at the first unimplemented adapter.

## What is missing (and fails on purpose)

- **DiT adapter:** loading weights and wiring modulation, RoPE and the text/image streams per block (`load_blocks`, `SegmentWrapper.forward`).
- **VAE adapter:** rewriting `CausalConv3d` with replicate padding, plus the chunked temporal cache.
- **Text encoder:** Qwen2.5-VL-7B (1.5) or LLaVA-Llama-3 + CLIP-L (13B). Keeping it on ggml-hexagon rather than QNN is recommended.
- **Calibration:** tensors from a reference run for the quantizer.
- **LoRA:** QNN bakes weights into the binary, so dynamic LoRAs don't fit.

Details in [docs/PLAN.md](docs/PLAN.md).

## Configs

- `cfg/hv15_480p_t2v.json`: hidden 2048, 16 heads, 54 double blocks, 16× VAE, 848×480×33 → 14,310 tokens, 6 segments.
- `cfg/hv13b_t2v.json`: hidden 3072, 24 heads, 20 double + 40 single, 8× VAE + patch 2, 10 segments, W4A16.
- `cfg/htp_v81*.json`: HTP settings taken from the ReaZIT QNN path (8 MB VTCM, O=3). Not validated for Hunyuan.

## Related

- [JiangFeng12/hunyuan-npu-models](https://huggingface.co/JiangFeng12/hunyuan-npu-models): the working Hunyuan NPU app (ggml-hexagon, not QNN).
- [JiangFeng12/ReaZIT-QNN](https://huggingface.co/JiangFeng12/ReaZIT-QNN) and [Xiemox2/ZIT-HEXAGON-QNN](https://github.com/Xiemox2/ZIT-HEXAGON-QNN): the reference QNN path.
- [Xiemox2/Tools_Krea2-npu](https://github.com/Xiemox2/Tools_Krea2-npu): sibling theoretical tools for Krea 2.

## License

Code is MIT (see `LICENSE`). The Qualcomm QAIRT/QNN SDK is **not included**; use it under your own licence. HunyuanVideo weights are under Tencent's licence.
