"""Export an OpenPI JAX model to StableHLO (MLIR).

This script exports a compiled inference entrypoint (sampling) to a textual MLIR
module using JAX's StableHLO export API.

Notes:
- This exports the *core model* (pi0/pi0.5 style) and expects inputs in the
  `openpi.models.model.Observation` format (already preprocessed / tokenized).
- It captures model parameters as constants in the exported program.

Example (pi05_libero):
uv run third_party/openpi/scripts/export_stablehlo.py \
  --config-name pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --batch-size 1 \
  --num-steps 10 \
  --output-mlir /tmp/pi05_libero.stablehlo.mlir
"""

from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tyro

from openpi.models import model as _model
from openpi.quantization import fp8_ptq as _fp8_ptq
from openpi.shared import download
from openpi.shared import array_typing as at
from openpi.training import config as _config


try:
    # Newer JAX (preferred).
    from jax import export as _jax_export  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    # Older JAX.
    from jax.experimental import export as _jax_export  # type: ignore[no-redef]


@dataclasses.dataclass(frozen=True)
class Args:
    # OpenPI training config name (e.g. pi05_libero).
    config_name: str = "pi05_libero"

    # Checkpoint directory (local path or gs://...). If omitted, uses a default for known configs.
    checkpoint: str | None = None

    # Output path for StableHLO MLIR text.
    output_mlir: str = "model.stablehlo.mlir"

    # Export-time shapes.
    batch_size: int = 1

    # pi0/pi0.5 sampling steps.
    num_steps: int = 10

    # Optional: enable MLIR-TRT FP8 path during export (inserts mtrt_quantize/mtrt_dequantize ops).
    # This expects a JSON produced by calibrate_fp8_amax.py.
    mtrt_fp8_scales_json: str | None = None


def main(args: Args) -> None:
    # Resolve checkpoint.
    if args.checkpoint is None:
        if args.config_name == "pi05_libero":
            checkpoint = "gs://openpi-assets/checkpoints/pi05_libero"
        elif args.config_name == "pi05_droid":
            checkpoint = "gs://openpi-assets/checkpoints/pi05_droid"
        else:
            raise ValueError("--checkpoint must be provided for unknown --config-name")
    else:
        checkpoint = args.checkpoint

    checkpoint_dir = pathlib.Path(download.maybe_download(checkpoint))
    out_path = pathlib.Path(args.output_mlir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load train config + model weights.
    train_cfg = _config.get_config(args.config_name)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_cfg.model.load(params)

    # Freeze module state into a pure function so export is stable.
    graphdef, state = nnx.split(model)

    def sample_actions_fn(rng: jax.Array, obs: _model.Observation) -> jax.Array:
        m = nnx.merge(graphdef, state)
        return m.sample_actions(rng, obs, num_steps=args.num_steps)

    # Compile for a fixed signature.
    jitted = jax.jit(sample_actions_fn)

    # Use abstract input specs.
    obs_spec, _act_spec = train_cfg.model.inputs_spec(batch_size=args.batch_size)
    rng_spec = jax.ShapeDtypeStruct((2,), jnp.uint32)

    # During export, JAX uses placeholder objects (e.g., ArgInfo) for abstract values.
    # OpenPI uses runtime typechecking via jaxtyping/beartype; disable it for tracing/export.
    if args.mtrt_fp8_scales_json is not None:
        scales = _fp8_ptq.load_scales_json(args.mtrt_fp8_scales_json)
        _fp8_ptq.enable_mtrt_fp8(scales)
        print(f"[export_stablehlo] enabled mtrt fp8 using: {args.mtrt_fp8_scales_json}")
    try:
        with at.disable_typechecking():
            exported = _jax_export.export(jitted)(rng_spec, obs_spec)
            mlir_mod = exported.mlir_module()
    finally:
        if args.mtrt_fp8_scales_json is not None:
            _fp8_ptq.disable_mtrt_fp8()

    # Write textual MLIR (StableHLO).
    mlir_text = str(mlir_mod)
    out_path.write_text(mlir_text)
    print(f"[export_stablehlo] wrote: {out_path}")
    print(f"[export_stablehlo] config: {args.config_name}")
    print(f"[export_stablehlo] checkpoint: {checkpoint_dir}")
    print(f"[export_stablehlo] batch_size: {args.batch_size}")
    print(f"[export_stablehlo] num_steps: {args.num_steps}")
    if args.mtrt_fp8_scales_json is not None:
        # Heuristic check: ensure the exported module contains the TensorRT PTQ custom-call modes.
        if "tensorrt.pt_q" not in mlir_text and "tensorrt.pt_dq" not in mlir_text:
            print(
                "[export_stablehlo] WARNING: FP8 was enabled, but the exported MLIR did not contain "
                "expected 'tensorrt.pt_q'/'tensorrt.pt_dq' markers. This usually means the scale keys "
                "did not match the model module paths, so the code fell back to FP32/BF16."
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
