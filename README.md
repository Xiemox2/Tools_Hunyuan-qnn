# Tools_Hunyuan-qnn

> ## ⚠️ Herramientas teóricas, no probadas
> Nada de esto se ha ejecutado de punta a punta ni en un teléfono. No hay modelos convertidos ni se garantiza que funcione.
> El autor **no va a continuar este trabajo por ahora** por otras prioridades. Se publica para que otros lo terminen, lo mejoren o lo usen de base.

Herramientas para convertir **HunyuanVideo 1.5** (y, como opción, el **HunyuanVideo 13B** original) a *context binaries* de **Qualcomm QNN/HTP** para Hexagon **v81** (SM8850, p. ej. Nubia Z80 Ultra). Siguen el mismo esquema que la ruta QNN de ZIT-HEXAGON con ReaZIT.

[English](README.en.md)

## Flujo propuesto

1. `py/plan_segments.py`: calcula los tokens por resolución, parte el DiT en segmentos y estima la memoria de cada uno.
2. `py/export_dit_segment.py`: exporta cada segmento a ONNX con shapes fijas (RoPE como entradas cos/sin precalculadas).
3. `sh/convert_segment.sh`: ONNX → `qairt-converter` → `qairt-quantizer` (W8A16 o W4A16) → `qnn-context-binary-generator` (v81).
4. `py/export_vae_decoder.py`: decoder VAE / TAEHV 1.5 con forma fija y por trozos de tiempo.
5. `py/make_manifest.py`: escribe `hv.json`, el manifiesto de runtime (como `zit.json`) con los sha256.
6. `sh/run_all.sh`: lo encadena todo y se detiene en el primer adaptador sin implementar.

## Qué falta (y falla a propósito)

- **Adaptador del DiT:** cargar los pesos y conectar modulación, RoPE y los flujos de texto e imagen por bloque (`load_blocks`, `SegmentWrapper.forward`).
- **Adaptador del VAE:** reescribir `CausalConv3d` con padding replicado y la caché temporal por trozos.
- **Encoder de texto:** Qwen2.5-VL-7B (1.5) o LLaVA-Llama-3 + CLIP-L (13B). Se recomienda dejarlo en ggml-hexagon y no pasarlo a QNN.
- **Calibración:** tensores de una corrida de referencia para el cuantizador.
- **LoRA:** en QNN los pesos se hornean dentro del binario, así que las LoRA dinámicas no encajan.

Más detalles en [docs/PLAN.md](docs/PLAN.md).

## Configuraciones

- `cfg/hv15_480p_t2v.json`: hidden 2048, 16 cabezas, 54 bloques double, VAE 16×, 848×480×33 → 14.310 tokens, 6 segmentos.
- `cfg/hv13b_t2v.json`: hidden 3072, 24 cabezas, 20 double + 40 single, VAE 8× + patch 2, 10 segmentos, W4A16.
- `cfg/htp_v81*.json`: configuración HTP tomada de la ruta ReaZIT QNN (VTCM 8 MB, O=3). No validada para Hunyuan.

## Relacionado

- [JiangFeng12/hunyuan-npu-models](https://huggingface.co/JiangFeng12/hunyuan-npu-models): la app Hunyuan NPU funcional (ggml-hexagon, no QNN).
- [JiangFeng12/ReaZIT-QNN](https://huggingface.co/JiangFeng12/ReaZIT-QNN) y [Xiemox2/ZIT-HEXAGON-QNN](https://github.com/Xiemox2/ZIT-HEXAGON-QNN): la ruta QNN de referencia.
- [Xiemox2/Tools_Krea2-npu](https://github.com/Xiemox2/Tools_Krea2-npu): herramientas teóricas hermanas para Krea 2.

## Licencia

El código es MIT (ver `LICENSE` y `NOTICE`). El SDK de Qualcomm QAIRT/QNN **no se incluye**; se usa con tu propia licencia. Los pesos de HunyuanVideo tienen la licencia de Tencent.
