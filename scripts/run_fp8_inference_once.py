"""One-off inference runner for OpenPI policies (optionally with MLIR-TRT FP8).

Example (FP32 / default backend):
uv run third_party/openpi/scripts/run_fp8_inference_once.py \\
  --config-name pi05_libero \\
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero

Example (FP8 on MLIR-TRT):
JAX_PLATFORMS=mlir_tensorrt uv run third_party/openpi/scripts/run_fp8_inference_once.py \\
  --config-name pi05_libero \\
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \\
  --mtrt-fp8-scales-json /tmp/pi05_libero_fp8_scales.json
"""

from __future__ import annotations

import dataclasses
import time

import jax
import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi05_libero"
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    mtrt_fp8_scales_json: str | None = None

    # If provided, will be injected as `prompt` if the input dict doesn't include it.
    prompt: str = "pick up the object"

    num_runs: int = 2


def main(args: Args) -> None:
    print(f"[run_fp8_inference_once] jax.default_backend(): {jax.default_backend()}")
    if args.mtrt_fp8_scales_json is not None and jax.default_backend() != "mlir_tensorrt":
        print(
            "[run_fp8_inference_once] WARNING: FP8 scales JSON was provided but JAX backend is not "
            "'mlir_tensorrt'. You likely want: JAX_PLATFORMS=mlir_tensorrt"
        )

    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        default_prompt=args.prompt,
        mtrt_fp8_scales_json=args.mtrt_fp8_scales_json,
    )

    # Libero-style example input keys (matches openpi/policies/libero_policy.py).
    # Note: you should replace these with real images/state from your environment.
    obs = {
        "observation/state": np.zeros((8,), dtype=np.float32),
        "observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        # `prompt` will be injected from args.prompt if omitted, but we include it explicitly here.
        "prompt": args.prompt,
    }

    # Run a couple times: first run includes JIT compile for JAX models.
    t0 = time.time()
    out = None
    for i in range(args.num_runs):
        out = policy.infer(obs)
        print(
            f"[run_fp8_inference_once] run={i} actions.shape={out['actions'].shape} "
            f"dtype={out['actions'].dtype} infer_ms={out['policy_timing']['infer_ms']:.2f}"
        )
    print(f"[run_fp8_inference_once] total_elapsed_s={time.time() - t0:.2f}")
    assert out is not None
    print(f"[run_fp8_inference_once] first_action: {out['actions'][0]}")


if __name__ == "__main__":
    main(tyro.cli(Args))
