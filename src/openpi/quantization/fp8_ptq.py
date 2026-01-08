"""JAX-only FP8 PTQ helpers (no Transformer Engine).

This module provides:
- A small FP8 format helper (E4M3FN / E5M2) for converting amax -> scale
- A lightweight amax collector that can be enabled during calibration runs

Important:
- This is *calibration only*. It does not change model inference to run in FP8.
- Collection uses `jax.debug.callback`, so it works under JIT/scan, but it is
  intended for offline calibration (it adds overhead).
"""

from __future__ import annotations

import contextlib
import enum
import json
import threading
from typing import Any

import jax
import jax.numpy as jnp

from openpi.shared import download as _download


class FP8Format(str, enum.Enum):
    E4M3FN = "e4m3fn"
    E5M2 = "e5m2"


def fp8_max_value(fmt: FP8Format) -> float:
    """Returns max finite value for the FP8 format.

    We prefer asking JAX for the dtype range, but fall back to known constants if
    the float8 dtype is unavailable in the installed JAX.
    """

    if fmt == FP8Format.E4M3FN:
        # Typical max finite for e4m3fn is 448.
        dtype = getattr(jnp, "float8_e4m3fn", None)
        if dtype is not None:
            return float(jnp.finfo(dtype).max)
        return 448.0
    if fmt == FP8Format.E5M2:
        # Typical max finite for e5m2 is 57344.
        dtype = getattr(jnp, "float8_e5m2", None)
        if dtype is not None:
            return float(jnp.finfo(dtype).max)
        return 57344.0
    raise ValueError(f"Unsupported FP8 format: {fmt}")


def amax_to_scale(amax: float, *, fmt: FP8Format, eps: float = 1e-12) -> float:
    """Convert an observed amax (max(abs(x))) to an FP8 scaling factor.

    We define a scale such that: x_scaled = x * scale fits in FP8 range.
    """

    amax_f = float(amax)
    return fp8_max_value(fmt) / max(amax_f, eps)


_lock = threading.Lock()
_enabled_depth = 0
_amax_by_name: dict[str, float] = {}


@contextlib.contextmanager
def enable_amax_collection(*, reset: bool = True):
    """Enable global amax collection for the duration of the context."""

    global _enabled_depth
    with _lock:
        if _enabled_depth == 0 and reset:
            _amax_by_name.clear()
        _enabled_depth += 1
    try:
        yield
    finally:
        with _lock:
            _enabled_depth = max(0, _enabled_depth - 1)


def is_amax_collection_enabled() -> bool:
    with _lock:
        return _enabled_depth > 0


def get_collected_amax() -> dict[str, float]:
    """Returns a snapshot of the collected amax values (host-side)."""

    with _lock:
        return dict(_amax_by_name)


def _module_path(module: Any) -> str:
    """Best-effort stable module path for naming amax sites."""

    # flax.linen Modules expose `scope` at runtime.
    scope = getattr(module, "scope", None)
    if scope is None:
        return module.__class__.__name__
    path_text = getattr(scope, "path_text", None)
    if isinstance(path_text, str) and path_text:
        return path_text
    # Fallback: join scope.path parts.
    parts = getattr(scope, "path", None)
    if parts:
        return "/".join(str(p) for p in parts)
    return module.__class__.__name__


def record_amax(name: str, x: jax.Array, *, layer_id: jax.Array | int | None = None) -> None:
    """Record amax(x) for the given site name (if collection is enabled).

    If layer_id is provided, we also record a per-layer key:
      f\"{name}/layer_{layer_id}\"
    while still updating the aggregate `name` entry.
    """

    if not is_amax_collection_enabled():
        return

    # Keep the compute cheap/stable: abs -> max -> float32 scalar.
    amax = jnp.max(jnp.abs(x)).astype(jnp.float32)

    def _update(val, lid):
        v = float(val)
        with _lock:
            prev = _amax_by_name.get(name)
            _amax_by_name[name] = v if prev is None else max(prev, v)
            if lid is not None:
                try:
                    li = int(lid)
                except Exception:
                    li = None
                if li is not None:
                    lname = f"{name}/layer_{li:02d}"
                    prev_l = _amax_by_name.get(lname)
                    _amax_by_name[lname] = v if prev_l is None else max(prev_l, v)

    # Use a debug callback to update host-side dict, works under jit/scan.
    jax.debug.callback(_update, amax, layer_id)


def record_module_amax(module: Any, *, tag: str, x: jax.Array, layer_id: jax.Array | int | None = None) -> None:
    """Convenience: record amax with a name derived from module path + tag."""

    record_amax(f"{_module_path(module)}/{tag}", x, layer_id=layer_id)


# -------------------------------------------------------------------------------------------------
# MLIR-TensorRT FP8 runtime mode (inference)
# -------------------------------------------------------------------------------------------------

_mtrt_lock = threading.Lock()
_mtrt_enabled = False
_mtrt_quant_scale_by_name: dict[str, float] = {}


def load_scales_json(path: str) -> dict[str, float]:
    """Load the `scale` mapping from a calibrate_fp8_amax JSON output.

    The calibration script writes `scale` as a *multiplier* (fp8_max / amax). For use with
    mlir_tensorrt_jax `mtrt_quantize`, which expects a *step size* (scale where q = x/scale),
    we convert multiplier -> step_size via inversion when enabling the mode.
    """

    local_path = _download.maybe_download(path)
    data = json.loads(local_path.read_text())
    if "scale" not in data or not isinstance(data["scale"], dict):
        raise ValueError(f"Expected JSON to contain a top-level 'scale' dict: {local_path}")
    out: dict[str, float] = {}
    for k, v in data["scale"].items():
        out[str(k)] = float(v)
    return out


def enable_mtrt_fp8(scales: dict[str, float]) -> None:
    """Enable MLIR-TRT FP8 path using the provided calibration scales.

    `scales` must map site_name -> multiplier (fp8_max/amax). We will invert to get the
    quantize step sizes used by `mtrt_quantize`.
    """

    global _mtrt_enabled, _mtrt_quant_scale_by_name

    from mlir_tensorrt_jax.mtrt_ops import register_all_lowerings  # type: ignore

    register_all_lowerings()

    quant_scale: dict[str, float] = {}
    for name, multiplier in scales.items():
        m = float(multiplier)
        if m <= 0.0:
            raise ValueError(f"Invalid scale: {name} -> {m}")
        quant_scale[name] = 1.0 / m

    with _mtrt_lock:
        _mtrt_quant_scale_by_name = quant_scale
        _mtrt_enabled = True


def disable_mtrt_fp8() -> None:
    global _mtrt_enabled, _mtrt_quant_scale_by_name
    with _mtrt_lock:
        _mtrt_enabled = False
        _mtrt_quant_scale_by_name = {}


def is_mtrt_fp8_enabled() -> bool:
    with _mtrt_lock:
        return _mtrt_enabled


def get_mtrt_quant_scale(module: Any, *, tag: str, layer_id: jax.Array | int | None = None) -> float | None:
    """Lookup quantize step size for a given module+tag.

    The key format matches calibration naming: `<scope.path_text>/<tag>`.
    """

    name = f"{_module_path(module)}/{tag}"
    with _mtrt_lock:
        # Prefer per-layer scales if available, but fall back to the aggregate key.
        if layer_id is not None:
            try:
                li = int(layer_id)
            except Exception:
                li = None
            if li is not None:
                v = _mtrt_quant_scale_by_name.get(f"{name}/layer_{li:02d}")
                if v is not None:
                    return v
        return _mtrt_quant_scale_by_name.get(name)
