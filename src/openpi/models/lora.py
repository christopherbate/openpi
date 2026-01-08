import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at
from openpi.quantization import fp8_ptq as _fp8_ptq


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x, *, layer_id=None):
        dtype = x.dtype  # original dtype, could be half-precision
        # FP8 PTQ calibration taps (no-op unless enabled).
        _fp8_ptq.record_module_amax(self, tag="einsum/x_amax", x=x, layer_id=layer_id)
        _fp8_ptq.record_module_amax(self, tag="einsum/w_amax", x=self.w, layer_id=layer_id)
        result = None

        # Optional MLIR-TensorRT FP8 inference path (opt-in).
        if _fp8_ptq.is_mtrt_fp8_enabled():
            sx = _fp8_ptq.get_mtrt_quant_scale(self, tag="einsum/x_amax", layer_id=layer_id)
            sw = _fp8_ptq.get_mtrt_quant_scale(self, tag="einsum/w_amax", layer_id=layer_id)
            sy = _fp8_ptq.get_mtrt_quant_scale(self, tag="einsum/out_amax", layer_id=layer_id)
            fp8_dtype = getattr(jnp, "float8_e4m3fn", None)
            if sx is not None and sw is not None and sy is not None and fp8_dtype is not None:
                from mlir_tensorrt_jax.mtrt_ops import mtrt_dequantize, mtrt_quantize  # type: ignore

                sx_arr = jnp.asarray(sx, dtype=jnp.float32)
                sw_arr = jnp.asarray(sw, dtype=jnp.float32)
                sy_arr = jnp.asarray(sy, dtype=jnp.float32)

                # Quantize x and w to FP8 (per-tensor).
                x_q = mtrt_quantize(x.astype(jnp.bfloat16), sx_arr, mode="tensorrt.pt_q", output_dtype=fp8_dtype)
                x_dq = mtrt_dequantize(x_q, sx_arr, mode="tensorrt.pt_dq", output_dtype=jnp.bfloat16)

                w_q = mtrt_quantize(self.w.astype(jnp.bfloat16), sw_arr, mode="tensorrt.pt_q", output_dtype=fp8_dtype)
                w_dq = mtrt_dequantize(w_q, sw_arr, mode="tensorrt.pt_dq", output_dtype=jnp.bfloat16)

                y = jnp.einsum(eqn, x_dq, w_dq)

                # Force output back to bf16 via explicit quantize + dequantize (per-tensor).
                result_q = mtrt_quantize(y, sy_arr, mode="tensorrt.pt_q", output_dtype=fp8_dtype)
                result_dq = mtrt_dequantize(result_q, sy_arr, mode="tensorrt.pt_dq", output_dtype=jnp.dtype(dtype))
                result = result_dq

        if result is None:
            result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            # Record LoRA weights as well (useful if LoRA is enabled for calibration).
            _fp8_ptq.record_module_amax(self, tag="einsum/lora_a_amax", x=self.w_a, layer_id=layer_id)
            _fp8_ptq.record_module_amax(self, tag="einsum/lora_b_amax", x=self.w_b, layer_id=layer_id)
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        _fp8_ptq.record_module_amax(self, tag="einsum/out_amax", x=result, layer_id=layer_id)
        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x, *, layer_id=None):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
            tag="gating_0",
            layer_id=layer_id,
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
            tag="gating_1",
            layer_id=layer_id,
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora, tag="linear", layer_id=layer_id)
        assert outputs.dtype == dtype
        return outputs

    def _dot(
        self,
        x: at.Array,
        w: at.Array,
        lora_weights: tuple[at.Array, at.Array] | None,
        *,
        tag: str,
        layer_id=None,
    ) -> at.Array:
        # FP8 PTQ calibration taps (no-op unless enabled).
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/x_amax", x=x, layer_id=layer_id)
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/w_amax", x=w, layer_id=layer_id)
        base = None

        if _fp8_ptq.is_mtrt_fp8_enabled():
            sx = _fp8_ptq.get_mtrt_quant_scale(self, tag=f"ffn/{tag}/x_amax", layer_id=layer_id)
            sw = _fp8_ptq.get_mtrt_quant_scale(self, tag=f"ffn/{tag}/w_amax", layer_id=layer_id)
            sy = _fp8_ptq.get_mtrt_quant_scale(self, tag=f"ffn/{tag}/out_amax", layer_id=layer_id)
            fp8_dtype = getattr(jnp, "float8_e4m3fn", None)
            if sx is not None and sw is not None and sy is not None and fp8_dtype is not None:
                try:
                    from mlir_tensorrt_jax.mtrt_ops import mtrt_dequantize, mtrt_quantize  # type: ignore

                    sx_arr = jnp.asarray(sx, dtype=jnp.float32)
                    sw_arr = jnp.asarray(sw, dtype=jnp.float32)
                    sy_arr = jnp.asarray(sy, dtype=jnp.float32)

                    x_q = mtrt_quantize(x.astype(jnp.bfloat16), sx_arr, mode="tensorrt.pt_q", output_dtype=fp8_dtype)
                    x_dq = mtrt_dequantize(x_q, sx_arr, mode="tensorrt.pt_dq", output_dtype=jnp.dtype(x.dtype))
                    w_q = mtrt_quantize(w.astype(jnp.bfloat16), sw_arr, mode="tensorrt.pt_q", output_dtype=fp8_dtype)
                    w_dq = mtrt_dequantize(w_q, sw_arr, mode="tensorrt.pt_dq", output_dtype=jnp.dtype(w.dtype))
                    y = jnp.dot(x_dq, w_dq)
                    base_q = mtrt_quantize(y, sy_arr, mode="tensorrt.pt_q", output_dtype=fp8_dtype)
                    base_dq = mtrt_dequantize(base_q, sy_arr, mode="tensorrt.pt_dq", output_dtype=jnp.dtype(y.dtype))
                    base = base_dq
                except Exception:
                    base = None

        if base is None:
            base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/out_amax", x=base, layer_id=layer_id)
            return base
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/lora_a_amax", x=lora_weights[0], layer_id=layer_id)
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/lora_b_amax", x=lora_weights[1], layer_id=layer_id)
        out = base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/out_amax", x=out, layer_id=layer_id)
        return out
