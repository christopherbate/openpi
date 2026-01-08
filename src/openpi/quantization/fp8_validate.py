"""Validation utilities for FP8 PTQ scale JSON files produced by calibrate_fp8_amax.py."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Any, Iterable


class FP8ScaleJsonError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class FP8ScaleValidationReport:
    model_config: str | None
    num_scales: int
    expected_num_scales: int | None
    missing: list[str]
    extra: list[str]
    invalid_values: list[str]

    def ok(self) -> bool:
        return not self.missing and not self.invalid_values


def load_fp8_scale_json(path: str) -> dict[str, Any]:
    return json.loads(open(path, "r", encoding="utf-8").read())


def _is_finite_positive(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _required_suffixes_pi05_two_expert_qkv_split() -> list[str]:
    """Expected *aggregate* suffixes (no per-layer) for pi05 gemma attention+mlp matmuls.

    This corresponds to 2 experts × (3 attention proj + 3 mlp) × (x/w/out) = 36.
    """

    einsum_sites = [
        "attn/q_einsum",
        "attn/kv_einsum",
        "attn/attn_vec_einsum",
    ]
    ffn_sites = [
        "mlp/ffn/gating_0",
        "mlp/ffn/gating_1",
        "mlp/ffn/linear",
    ]

    suffixes: list[str] = []
    for expert_suffix in ("", "_1"):
        for site in einsum_sites:
            for t in ("x_amax", "w_amax", "out_amax"):
                suffixes.append(f"/{site}{expert_suffix}/einsum/{t}")
        mlp_prefix = "/mlp" if expert_suffix == "" else "/mlp_1"
        for site in ffn_sites:
            sub = site.split("/", 1)[1]
            for t in ("x_amax", "w_amax", "out_amax"):
                suffixes.append(f"{mlp_prefix}/{sub}/{t}")
    return suffixes


_LAYER_SUFFIX_RE = re.compile(r"/layer_\\d+$")


def validate_fp8_scale_dict(
    scale: dict[str, Any],
    *,
    expected_suffixes: Iterable[str] | None = None,
    allow_layered: bool = True,
) -> FP8ScaleValidationReport:
    expected_suffixes = list(expected_suffixes or [])

    invalid_values: list[str] = []
    parsed: dict[str, float] = {}
    for k, v in scale.items():
        try:
            fv = float(v)
        except Exception:
            invalid_values.append(str(k))
            continue
        if not _is_finite_positive(fv):
            invalid_values.append(str(k))
            continue
        parsed[str(k)] = fv

    # Required keys by suffix (aggregate keys).
    missing: list[str] = []
    for suf in expected_suffixes:
        if not any(k.endswith(suf) for k in parsed.keys()):
            missing.append(suf)

    # Extra keys are those that don't match any expected suffix.
    extra: list[str] = []
    if expected_suffixes:
        for k in parsed.keys():
            k_norm = _LAYER_SUFFIX_RE.sub("", k) if allow_layered else k
            if not any(k_norm.endswith(suf) for suf in expected_suffixes):
                extra.append(k)

    expected_num_scales = len(expected_suffixes) if expected_suffixes else None

    return FP8ScaleValidationReport(
        model_config=None,
        num_scales=len(scale),
        expected_num_scales=expected_num_scales,
        missing=missing,
        extra=extra,
        invalid_values=invalid_values,
    )


def validate_fp8_ptq_json(data: dict[str, Any], *, mode: str = "auto") -> FP8ScaleValidationReport:
    if not isinstance(data, dict):
        raise FP8ScaleJsonError("Expected top-level JSON object")

    metadata = data.get("metadata", {})
    model_config = metadata.get("model_config") if isinstance(metadata, dict) else None

    scale = data.get("scale")
    if not isinstance(scale, dict):
        raise FP8ScaleJsonError("Expected JSON to contain top-level 'scale' dict")

    if mode == "auto":
        if model_config in {"pi05_libero", "pi05_droid"}:
            expected = _required_suffixes_pi05_two_expert_qkv_split()
        else:
            expected = []
    elif mode == "pi05_two_expert_qkv_split":
        expected = _required_suffixes_pi05_two_expert_qkv_split()
    elif mode == "none":
        expected = []
    else:
        raise FP8ScaleJsonError(f"Unknown mode: {mode}")

    rep = validate_fp8_scale_dict(scale, expected_suffixes=expected, allow_layered=True)
    return dataclasses.replace(rep, model_config=str(model_config) if model_config is not None else None)


"""Validation utilities for FP8 PTQ scale JSON files produced by calibrate_fp8_amax.py."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Iterable


class FP8ScaleJsonError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class FP8ScaleValidationReport:
    model_config: str | None
    num_scales: int
    expected_num_scales: int | None
    missing: list[str]
    extra: list[str]
    invalid_values: list[str]

    def ok(self) -> bool:
        return not self.missing and not self.invalid_values


def _is_finite_positive(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def load_fp8_scale_json(path: str) -> dict[str, Any]:
    return json.loads(open(path, "r", encoding="utf-8").read())


def _required_suffixes_for_two_expert_qkv_split() -> list[str]:
    # These suffixes intentionally ignore the leading module prefix (e.g. /remat(scan(layers))/...).
    # We validate by checking that each suffix appears at least once as a *full key ending*.
    einsum_sites = [
        "attn/q_einsum",
        "attn/kv_einsum",
        "attn/attn_vec_einsum",
    ]
    ffn_sites = [
        "mlp/ffn/gating_0",
        "mlp/ffn/gating_1",
        "mlp/ffn/linear",
    ]

    suffixes: list[str] = []
    for expert_suffix in ("", "_1"):
        for site in einsum_sites:
            for t in ("x_amax", "w_amax", "out_amax"):
                suffixes.append(f"/{site}{expert_suffix}/einsum/{t}")
        # MLP site naming uses mlp vs mlp_1 (not an _1 suffix on leaf module name).
        mlp_prefix = "/mlp" if expert_suffix == "" else "/mlp_1"
        for site in ffn_sites:
            # site already includes leading "mlp/..." so rewrite with the correct prefix.
            sub = site.split("/", 1)[1]
            for t in ("x_amax", "w_amax", "out_amax"):
                suffixes.append(f"{mlp_prefix}/{sub}/{t}")
    return suffixes


def validate_fp8_scale_dict(
    scale: dict[str, Any],
    *,
    expected_suffixes: Iterable[str] | None = None,
) -> FP8ScaleValidationReport:
    expected_suffixes = list(expected_suffixes or [])

    # Validate values.
    invalid_values: list[str] = []
    parsed: dict[str, float] = {}
    for k, v in scale.items():
        try:
            fv = float(v)
        except Exception:
            invalid_values.append(k)
            continue
        if not _is_finite_positive(fv):
            invalid_values.append(k)
            continue
        parsed[str(k)] = fv

    # Validate required keys by suffix match.
    missing: list[str] = []
    for suf in expected_suffixes:
        if not any(k.endswith(suf) for k in parsed.keys()):
            missing.append(suf)

    # Extra keys are those that don't match any expected suffix (if an expectation is provided).
    extra: list[str] = []
    if expected_suffixes:
        for k in parsed.keys():
            if not any(k.endswith(suf) for suf in expected_suffixes):
                extra.append(k)

    expected_num_scales = len(expected_suffixes) if expected_suffixes else None

    return FP8ScaleValidationReport(
        model_config=None,
        num_scales=len(scale),
        expected_num_scales=expected_num_scales,
        missing=missing,
        extra=extra,
        invalid_values=invalid_values,
    )


def validate_fp8_ptq_json(data: dict[str, Any], *, mode: str = "auto") -> FP8ScaleValidationReport:
    """Validate a JSON file produced by calibrate_fp8_amax.py.

    Args:
      data: Parsed JSON dict.
      mode:
        - "auto": infer expectations from metadata.model_config if present (pi05_libero/pi05_droid), else no suffix checks
        - "pi05_two_expert_qkv_split": enforce the expected 36 key suffixes
        - "none": only validate schema + values
    """

    if not isinstance(data, dict):
        raise FP8ScaleJsonError("Expected top-level JSON object")

    metadata = data.get("metadata", {})
    model_config = metadata.get("model_config") if isinstance(metadata, dict) else None

    scale = data.get("scale")
    if not isinstance(scale, dict):
        raise FP8ScaleJsonError("Expected JSON to contain top-level 'scale' dict")

    if mode == "auto":
        if model_config in {"pi05_libero", "pi05_droid"}:
            expected = _required_suffixes_for_two_expert_qkv_split()
        else:
            expected = []
    elif mode == "pi05_two_expert_qkv_split":
        expected = _required_suffixes_for_two_expert_qkv_split()
    elif mode == "none":
        expected = []
    else:
        raise FP8ScaleJsonError(f"Unknown mode: {mode}")

    rep = validate_fp8_scale_dict(scale, expected_suffixes=expected)
    return dataclasses.replace(rep, model_config=str(model_config) if model_config is not None else None)
