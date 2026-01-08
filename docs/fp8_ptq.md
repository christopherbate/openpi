## JAX FP8 PTQ (amax calibration) — no Transformer Engine

This document describes how to **calibrate FP8 PTQ amax statistics** for OpenPI (JAX) using **JAX only**, without Transformer Engine (TE).

### What this adds (and what it does not)

- **Added**: a calibration path that records **amax = max(abs(tensor))** at key **linear matmul** sites and writes them to JSON.
- **Not added**: FP8 kernels or TE integration. This does **not** make inference faster by itself.

### Instrumented sites

Calibration taps are inserted in:
- `openpi.models.lora.Einsum` (Gemma attention projections: qkv + out)
- `openpi.models.lora.FeedForward` (Gemma MLP matmuls)

These taps are **no-ops unless calibration is enabled**.

### Output schema

The calibration script writes a JSON file with:
- `metadata`: run parameters and notes
- `amax`: a dict mapping site name → observed amax
- `scale`: a dict mapping site name → derived FP8 *multiplier* (assuming FP8 format), defined as \(\\text{fp8\\_max} / \\text{amax}\\)

The default FP8 format assumption is **E4M3FN**.

Important: `mlir_tensorrt_jax`'s `mtrt_quantize` / `mtrt_dequantize` APIs treat `scale` as a **step size** (typical quantization convention: `q = x / scale`, dequant: `x ≈ q * scale`).

So when using the JSON with MLIR-TRT FP8 inference, OpenPI **inverts** the stored multiplier to obtain the quantize step size.
### Run calibration (LIBERO dataset)

If you want calibration to work with `physical-intelligence/libero`, use the `pi05_libero` config:

```bash
uv run third_party/openpi/scripts/calibrate_fp8_amax.py \
  --config-name pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --num-batches 200 \
  --batch-size 4 \
  --num-steps 10 \
  --output-json /tmp/pi05_libero_fp8_amax.json
```

### Run calibration (LeRobot DROID-format dataset)

You need a LeRobot dataset containing DROID-style fields (see `examples/droid/convert_droid_data_to_lerobot.py`).

Run:

```bash
uv run third_party/openpi/scripts/calibrate_fp8_amax.py \
  --config-name pi05_droid \
  --repo-id your_hf_username/my_droid_dataset \
  --checkpoint gs://openpi-assets/checkpoints/pi05_droid \
  --num-batches 200 \
  --batch-size 4 \
  --num-steps 10 \
  --output-json /tmp/pi05_droid_fp8_amax.json
```

Notes:
- `--num_steps` controls pi0.5 sampling steps; fewer steps is faster and usually sufficient for calibration.
- The loader uses **norm stats from the checkpoint** (`<checkpoint>/assets/droid`) so calibration matches the trained model preprocessing.

### Run FP8 inference with MLIR-TensorRT quantize/dequantize

This is **opt-in** and requires:
- running under the **mlir_tensorrt** backend (e.g. `export JAX_PLATFORMS=mlir_tensorrt`)
- providing the calibrated JSON via `--mtrt-fp8-scales-json`

Example (LIBERO server):

```bash
export JAX_PLATFORMS=mlir_tensorrt
uv run third_party/openpi/scripts/serve_policy.py --env LIBERO \
  --mtrt-fp8-scales-json /tmp/pi05_libero_fp8_amax.json
```

Under this mode, OpenPI inserts `mlir_tensorrt_jax.mtrt_ops.mtrt_quantize` / `mtrt_dequantize` around Gemma projection + MLP matmuls (per-tensor FP8).

