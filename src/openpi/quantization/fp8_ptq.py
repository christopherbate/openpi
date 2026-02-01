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

from collections.abc import Sequence
import contextlib
import enum
from functools import lru_cache
import importlib
import json
import threading
from typing import Any

from flax.core import meta
import flax.linen as nn
from flax.linen import initializers
from flax.linen import module
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import _conv_dimension_numbers
from flax.linen.linear import canonicalize_padding
from flax.linen.linear import default_kernel_init
from flax.linen.module import Module
from flax.linen.module import compact
from flax.typing import Array
from flax.typing import ConvGeneralDilatedT
from flax.typing import DotGeneralT
from flax.typing import Dtype
from flax.typing import Initializer
from flax.typing import LaxPadding
from flax.typing import PaddingLike
from flax.typing import PrecisionLike
from flax.typing import PRNGKey as PRNGKey
from flax.typing import Shape as Shape
import jax
from jax.core import ShapedArray
import jax.numpy as jnp
import numpy as np

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

_output_lock = threading.Lock()
_output_scaling_enabled = True


def set_output_scaling_enabled(enabled: bool) -> None:
    """Enable/disable FP8 output scaling/quantization.

    When disabled, we avoid:
    - recording `*/out_amax` sites during calibration runs
    - applying the optional output Q/DQ step in the MLIR-TRT FP8 runtime path
    """

    global _output_scaling_enabled
    with _output_lock:
        _output_scaling_enabled = bool(enabled)


def is_output_scaling_enabled() -> bool:
    with _output_lock:
        return _output_scaling_enabled


@contextlib.contextmanager
def enable_amax_collection(
    *,
    reset: bool = True,
):
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


def record_module_out_amax(module: Any, *, tag: str, x: jax.Array, layer_id: jax.Array | int | None = None) -> None:
    """Record output amax for a site, honoring the output-scaling toggle."""

    if not is_output_scaling_enabled():
        return
    record_module_amax(module, tag=tag, x=x, layer_id=layer_id)


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


def enable_mtrt_fp8(scales: dict[str, float], *, output_scaling: bool = True) -> None:
    """Enable MLIR-TRT FP8 path using the provided calibration scales.

    `scales` must map site_name -> multiplier (fp8_max/amax). We will invert to get the
    quantize step sizes used by `mtrt_quantize`.
    """

    global _mtrt_enabled, _mtrt_quant_scale_by_name
    set_output_scaling_enabled(output_scaling)
    # Optional dependency: only needed when the FP8 runtime mode is enabled.
    importlib.import_module("mlir_tensorrt_jax.mtrt_ops").register_all_lowerings()

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


# -------------------------------------------------------------------------------------------------
# Convenience helpers for model code (reduce boilerplate / centralize "magic strings")
# -------------------------------------------------------------------------------------------------

MTRT_PT_Q_MODE = "tensorrt.pt_q"
MTRT_PT_DQ_MODE = "tensorrt.pt_dq"


def get_mtrt_fp8_dtype():
    """Return the FP8 dtype used for MLIR-TRT quantize ops, if available in this JAX build."""

    try:
        return jnp.float8_e4m3fn
    except AttributeError:
        return None


@lru_cache(maxsize=1)
def _maybe_mtrt_ops():
    """Best-effort import for mlir_tensorrt_jax ops, cached.

    We use `importlib` (not `from ... import`) to avoid hard dependency and to
    keep static analyzers from requiring the module to be present.
    """

    try:
        mod = importlib.import_module("mlir_tensorrt_jax.mtrt_ops")
        return mod.mtrt_quantize, mod.mtrt_dequantize
    except Exception:
        return None


def is_mtrt_fp8_runtime_available() -> bool:
    """True if FP8 runtime mode is enabled and required runtime pieces exist."""

    if not is_mtrt_fp8_enabled():
        return False
    if get_mtrt_fp8_dtype() is None:
        return False
    return _maybe_mtrt_ops() is not None


def maybe_get_mtrt_qdq_scales(
    module: Any,
    *,
    x_tag: str,
    w_tag: str,
    out_tag: str | None = None,
    layer_id: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None] | None:
    """Fetch per-site quantize step sizes as float32 arrays, or None if unavailable."""

    if not is_mtrt_fp8_runtime_available():
        return None
    sx = get_mtrt_quant_scale(module, tag=x_tag, layer_id=layer_id)
    sw = get_mtrt_quant_scale(module, tag=w_tag, layer_id=layer_id)
    if sx is None or sw is None:
        return None

    sy_arr = None
    if out_tag is not None and is_output_scaling_enabled():
        sy = get_mtrt_quant_scale(module, tag=out_tag, layer_id=layer_id)
        if sy is not None:
            sy_arr = jnp.asarray(sy, dtype=jnp.float32)
    return (
        jnp.asarray(sx, dtype=jnp.float32),
        jnp.asarray(sw, dtype=jnp.float32),
        sy_arr,
    )


def fp8_qdq(x: jax.Array, scale: jax.Array):
    """Quantize to FP8 + dequantize back using MLIR-TRT ops (or return None)."""
    if not is_mtrt_fp8_runtime_available():
        raise ValueError("MLIR-TRT FP8 runtime mode is not available")
    fp8_dtype = get_mtrt_fp8_dtype()
    ops = _maybe_mtrt_ops()
    if fp8_dtype is None or ops is None:
        raise ValueError("MLIR-TRT FP8 runtime mode is not available")
    mtrt_quantize, mtrt_dequantize = ops
    x_q = mtrt_quantize(x, scale, mode=MTRT_PT_Q_MODE, output_dtype=fp8_dtype)
    return mtrt_dequantize(x_q, scale, mode=MTRT_PT_DQ_MODE, output_dtype=x.dtype)


def record_linear_sites(
    module: Any,
    *,
    prefix: str,
    x: jax.Array,
    w: jax.Array,
    y: jax.Array | None = None,
    layer_id: jax.Array | int | None = None,
) -> None:
    """Record amax for x/w/(optional) y using `{prefix}/x_amax`, `{prefix}/w_amax`, `{prefix}/out_amax`."""

    record_module_amax(module, tag=f"{prefix}/x_amax", x=x, layer_id=layer_id)
    record_module_amax(module, tag=f"{prefix}/w_amax", x=w, layer_id=layer_id)
    if y is not None:
        record_module_out_amax(module, tag=f"{prefix}/out_amax", x=y, layer_id=layer_id)


class Fp8Dense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    tag: str = "dense"

    @nn.compact
    def __call__(self, inputs: jax.Array, *, layer_id=None) -> jax.Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype) if self.use_bias else None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        record_module_amax(self, tag=f"{self.tag}/x_amax", x=inputs, layer_id=layer_id)
        record_module_amax(self, tag=f"{self.tag}/w_amax", x=kernel, layer_id=layer_id)
        sx = get_mtrt_quant_scale(self, tag=f"{self.tag}/x_amax", layer_id=layer_id)
        sw = get_mtrt_quant_scale(self, tag=f"{self.tag}/w_amax", layer_id=layer_id)
        scales_valid = sx is not None and sw is not None
        y = jax.lax.dot_general(
            inputs if not scales_valid else fp8_qdq(inputs, sx),
            kernel if not scales_valid else fp8_qdq(kernel, sw),
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            record_module_amax(self, tag=f"{self.tag}/bias_add/x_amax", x=y, layer_id=layer_id)
            record_module_amax(self, tag=f"{self.tag}/bias_add/w_amax", x=bias, layer_id=layer_id)
            sbx = get_mtrt_quant_scale(self, tag=f"{self.tag}/bias_add/x_amax", layer_id=layer_id)
            sbw = get_mtrt_quant_scale(self, tag=f"{self.tag}/bias_add/w_amax", layer_id=layer_id)
            scales_valid = scales_valid and sbx is not None and sbw is not None
            y = y if not scales_valid else fp8_qdq(y, sbx)
            bias = bias if not scales_valid else fp8_qdq(bias, sbw)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class Fp8DenseGeneral(nn.Module):
    """DenseGeneral (axis=-1) with optional FP8 PTQ calibration/inference hooks."""

    features: tuple[int, int]
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.normal(stddev=1e-6)
    dtype: str = "float32"
    tag: str = "dense_general"

    @nn.compact
    def __call__(self, x, *, layer_id=None):
        dtype = x.dtype
        if len(self.features) != 2:
            raise ValueError("Fp8DenseGeneral expects features=(num_heads, head_dim)")
        num_heads, head_dim = self.features
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], num_heads, head_dim), self.dtype)
        bias = None
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (num_heads, head_dim), self.dtype)

        record_linear_sites(self, prefix=self.tag, x=x, w=kernel, layer_id=layer_id)

        result = None
        scales = maybe_get_mtrt_qdq_scales(
            self,
            x_tag=f"{self.tag}/x_amax",
            w_tag=f"{self.tag}/w_amax",
            out_tag=f"{self.tag}/out_amax",
            layer_id=layer_id,
        )
        if scales is not None:
            sx_arr, sw_arr, sy_arr = scales
            x_dq = fp8_qdq(x, sx_arr)
            w_dq = fp8_qdq(kernel, sw_arr)
            if x_dq is not None and w_dq is not None:
                y = jnp.einsum("...d,dhm->...hm", x_dq, w_dq)
                if bias is not None:
                    record_module_amax(self, tag=f"{self.tag}/bias_add/x_amax", x=y, layer_id=layer_id)
                    record_module_amax(self, tag=f"{self.tag}/bias_add/w_amax", x=bias, layer_id=layer_id)
                    b_scales = maybe_get_mtrt_qdq_scales(
                        self,
                        x_tag=f"{self.tag}/bias_add/x_amax",
                        w_tag=f"{self.tag}/bias_add/w_amax",
                        out_tag=None,
                        layer_id=layer_id,
                    )
                    if b_scales is not None:
                        sbx_arr, sbw_arr, _ = b_scales
                        y_dq = fp8_qdq(y, sbx_arr)
                        b_dq = fp8_qdq(bias, sbw_arr)
                        y = y_dq + b_dq if y_dq is not None and b_dq is not None else y + bias.astype(y.dtype)
                    else:
                        y = y + bias.astype(y.dtype)
                result = fp8_qdq(y, sy_arr) if sy_arr is not None else y

        if result is None:
            y = jnp.einsum("...d,dhm->...hm", x, kernel.astype(dtype))
            if bias is not None:
                record_module_amax(self, tag=f"{self.tag}/bias_add/x_amax", x=y, layer_id=layer_id)
                record_module_amax(self, tag=f"{self.tag}/bias_add/w_amax", x=bias, layer_id=layer_id)
                b_scales = maybe_get_mtrt_qdq_scales(
                    self,
                    x_tag=f"{self.tag}/bias_add/x_amax",
                    w_tag=f"{self.tag}/bias_add/w_amax",
                    out_tag=None,
                    layer_id=layer_id,
                )
                if b_scales is not None:
                    sbx_arr, sbw_arr, _ = b_scales
                    y_dq = fp8_qdq(y, sbx_arr)
                    b_dq = fp8_qdq(bias, sbw_arr)
                    y = y_dq + b_dq if y_dq is not None and b_dq is not None else y + bias.astype(y.dtype)
                else:
                    y = y + bias.astype(y.dtype)
            result = y

        record_module_out_amax(self, tag=f"{self.tag}/out_amax", x=result, layer_id=layer_id)
        return result


class Fp8DenseGeneralOut(nn.Module):
    """DenseGeneral for attention output (axis=-2,-1) with FP8 PTQ hooks."""

    features: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.normal(stddev=1e-6)
    dtype: str = "float32"
    tag: str = "dense_general_out"

    @nn.compact
    def __call__(self, x, *, layer_id=None):
        dtype = x.dtype
        kernel = self.param("kernel", self.kernel_init, (x.shape[-2], x.shape[-1], self.features), self.dtype)
        bias = None
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)

        record_linear_sites(self, prefix=self.tag, x=x, w=kernel, layer_id=layer_id)

        result = None
        scales = maybe_get_mtrt_qdq_scales(
            self,
            x_tag=f"{self.tag}/x_amax",
            w_tag=f"{self.tag}/w_amax",
            out_tag=f"{self.tag}/out_amax",
            layer_id=layer_id,
        )
        if scales is not None:
            sx_arr, sw_arr, sy_arr = scales
            x_dq = fp8_qdq(x, sx_arr)
            w_dq = fp8_qdq(kernel, sw_arr)
            if x_dq is not None and w_dq is not None:
                y = jnp.einsum("...hm,hmd->...d", x_dq, w_dq)
                if bias is not None:
                    record_module_amax(self, tag=f"{self.tag}/bias_add/x_amax", x=y, layer_id=layer_id)
                    record_module_amax(self, tag=f"{self.tag}/bias_add/w_amax", x=bias, layer_id=layer_id)
                    b_scales = maybe_get_mtrt_qdq_scales(
                        self,
                        x_tag=f"{self.tag}/bias_add/x_amax",
                        w_tag=f"{self.tag}/bias_add/w_amax",
                        out_tag=None,
                        layer_id=layer_id,
                    )
                    if b_scales is not None:
                        sbx_arr, sbw_arr, _ = b_scales
                        y_dq = fp8_qdq(y, sbx_arr)
                        b_dq = fp8_qdq(bias, sbw_arr)
                        y = y_dq + b_dq if y_dq is not None and b_dq is not None else y + bias.astype(y.dtype)
                    else:
                        y = y + bias.astype(y.dtype)
                result = fp8_qdq(y, sy_arr) if sy_arr is not None else y

        if result is None:
            y = jnp.einsum("...hm,hmd->...d", x, kernel.astype(dtype))
            if bias is not None:
                record_module_amax(self, tag=f"{self.tag}/bias_add/x_amax", x=y, layer_id=layer_id)
                record_module_amax(self, tag=f"{self.tag}/bias_add/w_amax", x=bias, layer_id=layer_id)
                b_scales = maybe_get_mtrt_qdq_scales(
                    self,
                    x_tag=f"{self.tag}/bias_add/x_amax",
                    w_tag=f"{self.tag}/bias_add/w_amax",
                    out_tag=None,
                    layer_id=layer_id,
                )
                if b_scales is not None:
                    sbx_arr, sbw_arr, _ = b_scales
                    y_dq = fp8_qdq(y, sbx_arr)
                    b_dq = fp8_qdq(bias, sbw_arr)
                    y = y_dq + b_dq if y_dq is not None and b_dq is not None else y + bias.astype(y.dtype)
                else:
                    y = y + bias.astype(y.dtype)
            result = y

        record_module_out_amax(self, tag=f"{self.tag}/out_amax", x=result, layer_id=layer_id)
        return result


class Fp8Conv(nn.Module):
    """2D Conv (NHWC, HWIO) with FP8 PTQ hooks on operands only.

    Records `x_amax` and `w_amax` and applies optional operand Q/DQ in MLIR-TRT FP8 runtime mode.
    Does NOT record `out_amax` and does NOT apply output Q/DQ.
    """

    features: int
    kernel_size: int | Sequence[int]
    strides: None | int | Sequence[int] = 1
    padding: PaddingLike = 'SAME'
    input_dilation: None | int | Sequence[int] = 1
    kernel_dilation: None | int | Sequence[int] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    mask: Array | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    tag: str = "conv"
    shared_weights: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array, layer_id: int | None = None) -> jax.Array:
        """Applies a (potentially unshared) convolution to the inputs.

        Args:
          inputs: input data with dimensions ``(*batch_dims, spatial_dims..., features)``.
            This is the channels-last convention, i.e. NHWC for a 2d convolution and
            NDHWC for a 3D convolution. Note: this is different from the input convention
            used by ``lax.conv_general_dilated``, which puts the spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap'ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """

        kernel_size: Sequence[int]
        kernel_size = (self.kernel_size,) if isinstance(self.kernel_size, int) else tuple(self.kernel_size)

        def maybe_broadcast(
            x: int | Sequence[int] | None,
        ) -> tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[num_batch_dimensions:]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == "CIRCULAR":
            kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
            zero_pad: list[tuple[int, int]] = [(0, 0)]
            pads = zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
            inputs = jnp.pad(inputs, pads, mode="wrap")
            padding_lax = "VALID"
        elif padding_lax == "CAUSAL":
            if len(kernel_size) != 1:
                raise ValueError("Causal padding is only implemented for 1D convolutions.")
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)
            padding_lax = "VALID"

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        in_features = jnp.shape(inputs)[-1]

        # One shared convolutional kernel for all pixels in the output.
        assert in_features % self.feature_group_count == 0
        kernel_shape = kernel_size + (
            in_features // self.feature_group_count,
            self.features,
        )

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError(
                f"Mask needs to have the same shape as weights. Shapes are: {self.mask.shape}, {kernel_shape}"
            )

        kernel = self.param("kernel", self.kernel_init, kernel_shape, self.param_dtype)

        if self.mask is not None:
            kernel *= self.mask

        if self.use_bias:
            # One bias weight per output channel, shared between pixels.
            bias_shape = (self.features,)
            bias = self.param("bias", self.bias_init, bias_shape, self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        record_module_amax(self, tag=f"{self.tag}/x_amax", x=inputs, layer_id=layer_id)
        record_module_amax(self, tag=f"{self.tag}/w_amax", x=kernel, layer_id=layer_id)
        sx = get_mtrt_quant_scale(self, tag=f"{self.tag}/x_amax", layer_id=layer_id)
        sw = get_mtrt_quant_scale(self, tag=f"{self.tag}/w_amax", layer_id=layer_id)
        scales_valid = sx is not None and sw is not None

        y = jax.lax.conv_general_dilated(
            inputs if not scales_valid else fp8_qdq(inputs, sx),
            kernel if not scales_valid else fp8_qdq(kernel, sw),
            strides,
            padding_lax,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if self.use_bias:
            record_module_amax(self, tag=f"{self.tag}/bias_add/x_amax", x=y, layer_id=layer_id)
            record_module_amax(self, tag=f"{self.tag}/bias_add/w_amax", x=bias, layer_id=layer_id)
            sbx = get_mtrt_quant_scale(self, tag=f"{self.tag}/bias_add/x_amax", layer_id=layer_id)
            sbw = get_mtrt_quant_scale(self, tag=f"{self.tag}/bias_add/w_amax", layer_id=layer_id)
            scales_valid = scales_valid and sbx is not None and sbw is not None
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
            y = y if not scales_valid else fp8_qdq(y, sbx)
            bias = bias if not scales_valid else fp8_qdq(bias, sbw)
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y


class MultiHeadDotProductAttentionFp8(nn.Module):
    """Multi-head attention with FP8 PTQ instrumentation on projections."""

    num_heads: int
    dtype: str = "float32"

    @nn.compact
    def __call__(self, query, key, value=None, *, deterministic=True):
        if value is None:
            value = key
        if query.shape[-1] % self.num_heads != 0:
            raise ValueError("query width must be divisible by num_heads")
        head_dim = query.shape[-1] // self.num_heads

        q = Fp8DenseGeneral(
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            tag="attn/q",
            name="query",
        )(query)
        k = Fp8DenseGeneral(
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            tag="attn/k",
            name="key",
        )(key)
        v = Fp8DenseGeneral(
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            tag="attn/v",
            name="value",
        )(value)

        scale = 1.0 / jnp.sqrt(head_dim)
        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k) * scale
        attn = jax.nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("bhts,bshd->bthd", attn, v)
        out = Fp8DenseGeneralOut(
            features=query.shape[-1],
            dtype=self.dtype,
            tag="attn/out",
            name="out",
        )(out)
        return out
