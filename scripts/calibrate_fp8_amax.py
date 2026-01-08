"""Calibrate FP8 PTQ amax statistics for OpenPI JAX models (no TE).

This script runs a small number of batches through a selected OpenPI config and records
per-site amax statistics for linear matmul sites (qkv/out projections + MLP),
emitted by instrumentation in:
- openpi.models.lora.Einsum
- openpi.models.lora.FeedForward

It outputs a standalone JSON file containing:
- metadata (checkpoint, repo_id, fp8 format, etc.)
- per-site amax and derived scale (amax -> scale mapping for the chosen FP8 format)

Example:
uv run third_party/openpi/scripts/calibrate_fp8_amax.py \
  --config_name pi05_droid \
  --repo_id your_hf_username/my_droid_dataset \
  --checkpoint gs://openpi-assets/checkpoints/pi05_droid \
  --num_batches 200 \
  --batch_size 4 \
  --num_steps 10 \
  --output_json /tmp/pi05_droid_fp8_amax.json
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import tyro

from openpi.models import model as _model
from openpi.quantization import fp8_ptq
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class Args:
    # OpenPI config name (determines model + input pipeline), e.g. "pi05_droid" or "pi05_libero".
    config_name: str = "pi05_droid"

    # LeRobot dataset repo id. If omitted, uses the default `repo_id` baked into the config (when available).
    repo_id: str | None = None

    # Checkpoint directory (local path or gs://...). If omitted, uses a default for known configs.
    checkpoint: str | None = None
    # Output JSON file path.
    output_json: str = "fp8_amax.json"

    # Calibration run config.
    num_batches: int = 200
    batch_size: int = 4
    seed: int = 0
    shuffle: bool = True

    # Sampling config (pi0.5 diffusion steps). Lower is faster; calibration does not need many steps.
    num_steps: int = 10

    # FP8 format assumption for converting amax -> scale.
    fp8_format: fp8_ptq.FP8Format = fp8_ptq.FP8Format.E4M3FN


def main(args: Args) -> None:
    if args.checkpoint is None:
        if args.config_name == "pi05_droid":
            checkpoint = "gs://openpi-assets/checkpoints/pi05_droid"
        elif args.config_name == "pi05_libero":
            checkpoint = "gs://openpi-assets/checkpoints/pi05_libero"
        else:
            raise ValueError("--checkpoint must be provided for unknown --config_name")
    else:
        checkpoint = args.checkpoint

    checkpoint_dir = pathlib.Path(download.maybe_download(checkpoint))
    output_path = pathlib.Path(args.output_json)

    # Load the base train config (model architecture).
    base_cfg = _config.get_config(args.config_name)

    # Build a calibration data config.
    #
    # Important: use norm stats from the checkpoint assets so inputs match the trained model's normalization.
    if isinstance(base_cfg.data, _config.LeRobotLiberoDataConfig):
        repo_id = args.repo_id or base_cfg.data.repo_id
        calib_data = dataclasses.replace(
            base_cfg.data,
            repo_id=repo_id,
            assets=_config.AssetsConfig(
                assets_dir=str(checkpoint_dir / "assets"),
                asset_id=base_cfg.data.assets.asset_id,  # default None -> uses repo_id as asset_id
            ),
        )
    else:
        # Default to DROID LeRobot pipeline for pi05_droid-style calibration.
        repo_id = args.repo_id
        if repo_id is None:
            raise ValueError("--repo_id is required for DROID calibration")
        calib_data = _config.LeRobotDROIDDataConfig(
            repo_id=repo_id,
            assets=_config.AssetsConfig(
                assets_dir=str(checkpoint_dir / "assets"),
                asset_id="droid",
            ),
            base_config=_config.DataConfig(prompt_from_task=True),
        )
    calib_cfg = dataclasses.replace(
        base_cfg,
        batch_size=args.batch_size,
        seed=args.seed,
        data=calib_data,
        num_workers=0,  # calibration should be stable/reproducible; avoid extra worker randomness
    )

    # Load model params.
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = calib_cfg.model.load(params)

    # Create a JAX data loader (LeRobot dataset under the hood, with transforms + normalization).
    loader = _data_loader.create_data_loader(
        calib_cfg,
        shuffle=args.shuffle,
        num_batches=args.num_batches,
        skip_norm_stats=False,
        framework="jax",
    )

    # Run calibration: execute sample_actions and block until ready so callbacks fire.
    rng = jax.random.key(args.seed)
    t0 = time.time()
    with fp8_ptq.enable_amax_collection(reset=True):
        for i, (obs, _act) in enumerate(loader):
            rng, step_rng = jax.random.split(rng)
            actions = model.sample_actions(step_rng, obs, num_steps=args.num_steps)
            jax.block_until_ready(actions)
            if (i + 1) % 10 == 0:
                print(f"[calibrate_fp8_amax] processed {i+1}/{args.num_batches} batches")

    elapsed_s = time.time() - t0
    amax = fp8_ptq.get_collected_amax()

    # Convert amax to per-site scales for the chosen FP8 format.
    fp8_max = fp8_ptq.fp8_max_value(args.fp8_format)
    out = {
        "metadata": {
            "model_config": args.config_name,
            "checkpoint": str(checkpoint),
            "checkpoint_dir": str(checkpoint_dir),
            "repo_id": repo_id,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "shuffle": args.shuffle,
            "num_steps": args.num_steps,
            "fp8_format": args.fp8_format.value,
            "fp8_max_value": fp8_max,
            "elapsed_s": elapsed_s,
            "note": "Collected amax stats for linear matmul sites in lora.Einsum / lora.FeedForward. No TE.",
        },
        "amax": amax,
        "scale": {k: fp8_ptq.amax_to_scale(v, fmt=args.fp8_format) for k, v in amax.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[calibrate_fp8_amax] wrote {len(amax)} amax entries to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
