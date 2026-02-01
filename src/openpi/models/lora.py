import math
import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp

from openpi.quantization import fp8_ptq as _fp8_ptq
import openpi.shared.array_typing as at


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
    def __call__(self, eqn: str, x: jax.Array, *, layer_id=None):
        dtype = x.dtype  # original dtype, could be half-precision
        # FP8 PTQ calibration taps (no-op unless enabled).
        _fp8_ptq.record_module_amax(self, tag="einsum/x_amax", x=x, layer_id=layer_id)
        _fp8_ptq.record_module_amax(self, tag="einsum/w_amax", x=self.w, layer_id=layer_id)
        result = None

        # Optional MLIR-TensorRT FP8 inference path (opt-in).
        scales = _fp8_ptq.maybe_get_mtrt_qdq_scales(
            self,
            x_tag="einsum/x_amax",
            w_tag="einsum/w_amax",
            out_tag="einsum/out_amax",
            layer_id=layer_id,
        )
        if scales is not None:
            sx_arr, sw_arr, sy_arr = scales
            x_dq = _fp8_ptq.fp8_qdq(x, sx_arr)
            w_dq = _fp8_ptq.fp8_qdq(self.w, sw_arr)
            if x_dq is not None and w_dq is not None:
                y = jnp.einsum(eqn, x_dq, w_dq)
                if sy_arr is not None:
                    result = _fp8_ptq.fp8_qdq(y, sy_arr)
                else:
                    result = y.astype(dtype)

        if result is None:
            result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            # Record LoRA weights as well (useful if LoRA is enabled for calibration).
            _fp8_ptq.record_module_amax(self, tag="einsum/lora_a_amax", x=self.w_a, layer_id=layer_id)
            _fp8_ptq.record_module_amax(self, tag="einsum/lora_b_amax", x=self.w_b, layer_id=layer_id)
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            lora_scaled = lora * config.scaling_value
            # Treat the LoRA add as a "bias add"-style op: record + optional operand Q/DQ on both add inputs,
            # independent of whether output Q/DQ is enabled.
            _fp8_ptq.record_module_amax(self, tag="einsum/lora_add/x_amax", x=result, layer_id=layer_id)
            _fp8_ptq.record_module_amax(self, tag="einsum/lora_add/w_amax", x=lora_scaled, layer_id=layer_id)
            add_scales = _fp8_ptq.maybe_get_mtrt_qdq_scales(
                self,
                x_tag="einsum/lora_add/x_amax",
                w_tag="einsum/lora_add/w_amax",
                out_tag=None,
                layer_id=layer_id,
            )
            if add_scales is not None:
                sax_arr, saw_arr, _ = add_scales
                base_dq = _fp8_ptq.fp8_qdq(result, sax_arr)
                lora_dq = _fp8_ptq.fp8_qdq(lora_scaled, saw_arr)
                if base_dq is not None and lora_dq is not None:
                    result = base_dq + lora_dq
                else:
                    result = result + lora_scaled
            else:
                result = result + lora_scaled

        _fp8_ptq.record_module_out_amax(self, tag="einsum/out_amax", x=result, layer_id=layer_id)
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

        scales = _fp8_ptq.maybe_get_mtrt_qdq_scales(
            self,
            x_tag=f"ffn/{tag}/x_amax",
            w_tag=f"ffn/{tag}/w_amax",
            out_tag=f"ffn/{tag}/out_amax",
            layer_id=layer_id,
        )
        if scales is not None:
            sx_arr, sw_arr, sy_arr = scales
            x_dq = _fp8_ptq.fp8_qdq(x, sx_arr)
            w_dq = _fp8_ptq.fp8_qdq(w, sw_arr)
            if x_dq is not None and w_dq is not None:
                y = jnp.dot(x_dq, w_dq)
                if sy_arr is not None:
                    base = _fp8_ptq.fp8_qdq(y, sy_arr)
                else:
                    base = y

        if base is None:
            base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            _fp8_ptq.record_module_out_amax(self, tag=f"ffn/{tag}/out_amax", x=base, layer_id=layer_id)
            return base
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/lora_a_amax", x=lora_weights[0], layer_id=layer_id)
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/lora_b_amax", x=lora_weights[1], layer_id=layer_id)
        lora_update = jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
        # Treat the LoRA add as a "bias add"-style op: record + optional operand Q/DQ on both add inputs,
        # independent of whether output Q/DQ is enabled.
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/lora_add/x_amax", x=base, layer_id=layer_id)
        _fp8_ptq.record_module_amax(self, tag=f"ffn/{tag}/lora_add/w_amax", x=lora_update, layer_id=layer_id)
        add_scales = _fp8_ptq.maybe_get_mtrt_qdq_scales(
            self,
            x_tag=f"ffn/{tag}/lora_add/x_amax",
            w_tag=f"ffn/{tag}/lora_add/w_amax",
            out_tag=None,
            layer_id=layer_id,
        )
        if add_scales is not None:
            sax_arr, saw_arr, _ = add_scales
            base_dq = _fp8_ptq.fp8_qdq(base, sax_arr)
            upd_dq = _fp8_ptq.fp8_qdq(lora_update, saw_arr)
            out = base_dq + upd_dq if base_dq is not None and upd_dq is not None else base + lora_update
        else:
            out = base + lora_update
        _fp8_ptq.record_module_out_amax(self, tag=f"ffn/{tag}/out_amax", x=out, layer_id=layer_id)
        return out
