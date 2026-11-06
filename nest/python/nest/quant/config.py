"""PTQ 量化配置。

所有量化语义都必须显式配置：dtype 决定 bit width，累加位宽、粒度、
对称性、舍入与饱和都必须给出；缺省值只是便于调用，不会被后端“猜测”。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from ..errors import QuantizationError
from ..ir.types import QUANTIZED_DTYPES
from .qmath import dtype_bits

GRANULARITIES = ("per_tensor", "per_channel", "per_block")
SCHEMES = ("symmetric", "asymmetric")
CALIBRATIONS = ("minmax", "percentile", "kl", "mse")
ROUNDINGS = ("nearest_even", "nearest_away", "truncate")

_OVERRIDE_FIELDS = frozenset(
    {
        "activation_dtype",
        "weight_dtype",
        "accum_bits",
        "activation_granularity",
        "weight_granularity",
        "activation_scheme",
        "weight_scheme",
        "rounding",
        "saturation",
        "calibration",
        "percentile",
        "kl_bins",
        "block_size",
        "weight_axis",
        "activation_channel_axis",
        "bias_correction",
        "calibration_samples",
        "calibration_strategy",
        "seed",
    }
)


@dataclass(frozen=True)
class QuantConfig:
    """PTQ 量化配置。"""

    activation_dtype: str = "uint8"
    weight_dtype: str = "int8"
    accum_bits: int = 32
    activation_granularity: str = "per_tensor"
    weight_granularity: str = "per_channel"
    activation_scheme: str = "asymmetric"
    weight_scheme: str = "symmetric"
    rounding: str = "nearest_even"
    saturation: bool = True
    calibration: str = "minmax"
    percentile: float = 99.99
    kl_bins: int = 2048
    block_size: int = 32
    # 权重量化轴；None 表示按算子语义推导（Conv=O 轴，Gemm 由 transB 决定）。
    weight_axis: int | None = None
    # 激活 per-channel 量化轴，NCHW 语义下默认通道轴 1。
    activation_channel_axis: int = 1
    # 按 canonical 算子名覆盖量化配置；值中 ``{"float": true}`` 表示回退浮点。
    per_op: dict = field(default_factory=dict)
    # 显式混合精度回退算子集合。
    float_ops: frozenset[str] = frozenset()
    bias_correction: bool = True
    # 校准集由 make_calibration_set 生成时的默认参数。
    calibration_samples: int = 16
    calibration_strategy: str = "normal"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.activation_dtype not in QUANTIZED_DTYPES:
            raise QuantizationError(
                f"非法激活量化 dtype: {self.activation_dtype}"
            )
        if self.weight_dtype not in ("int8", "int16"):
            raise QuantizationError(
                f"权重必须使用有符号量化 dtype，得到 {self.weight_dtype}"
            )
        if self.accum_bits < max(
            dtype_bits(self.activation_dtype), dtype_bits(self.weight_dtype)
        ):
            raise QuantizationError(
                f"accum_bits={self.accum_bits} 小于量化位宽"
            )
        for name, value, allowed in (
            ("activation_granularity", self.activation_granularity, GRANULARITIES),
            ("weight_granularity", self.weight_granularity, GRANULARITIES),
            ("activation_scheme", self.activation_scheme, SCHEMES),
            ("weight_scheme", self.weight_scheme, SCHEMES),
            ("rounding", self.rounding, ROUNDINGS),
            ("calibration", self.calibration, CALIBRATIONS),
        ):
            if value not in allowed:
                raise QuantizationError(f"非法 {name}: {value}")
        if not (0.0 < self.percentile <= 100.0):
            raise QuantizationError(f"percentile 必须在 (0, 100]: {self.percentile}")
        if self.kl_bins <= 1:
            raise QuantizationError(f"kl_bins 必须 > 1: {self.kl_bins}")
        if self.block_size <= 0:
            raise QuantizationError(f"block_size 必须为正数: {self.block_size}")
        if self.activation_channel_axis < 0:
            raise QuantizationError("activation_channel_axis 不能为负")
        if self.calibration_samples <= 0:
            raise QuantizationError("calibration_samples 必须为正数")
        if self.calibration_strategy not in ("normal", "uniform"):
            raise QuantizationError(
                f"非法校准策略: {self.calibration_strategy}"
            )
        for op, overrides in self.per_op.items():
            if not isinstance(overrides, dict):
                raise QuantizationError(f"per_op[{op}] 必须是 dict")
            unknown = set(overrides) - _OVERRIDE_FIELDS - {"float"}
            if unknown:
                raise QuantizationError(
                    f"per_op[{op}] 包含未知字段: {sorted(unknown)}"
                )

    @property
    def activation_bits(self) -> int:
        return dtype_bits(self.activation_dtype)

    @property
    def weight_bits(self) -> int:
        return dtype_bits(self.weight_dtype)

    def for_op(self, canonical_name: str) -> "QuantConfig":
        """返回应用 per-op 覆盖后的配置。"""
        overrides = self.per_op.get(canonical_name)
        if not overrides:
            return self
        data = {key: value for key, value in overrides.items() if key != "float"}
        if not data:
            return self
        return replace(self, **data)

    def op_is_float(self, canonical_name: str) -> bool:
        if canonical_name in self.float_ops:
            return True
        overrides = self.per_op.get(canonical_name, {})
        return bool(overrides.get("float", False))

    def to_dict(self) -> dict:
        return {
            "activation_dtype": self.activation_dtype,
            "weight_dtype": self.weight_dtype,
            "accum_bits": self.accum_bits,
            "activation_granularity": self.activation_granularity,
            "weight_granularity": self.weight_granularity,
            "activation_scheme": self.activation_scheme,
            "weight_scheme": self.weight_scheme,
            "rounding": self.rounding,
            "saturation": self.saturation,
            "calibration": self.calibration,
            "percentile": self.percentile,
            "kl_bins": self.kl_bins,
            "block_size": self.block_size,
            "weight_axis": self.weight_axis,
            "activation_channel_axis": self.activation_channel_axis,
            "per_op": {k: dict(v) for k, v in sorted(self.per_op.items())},
            "float_ops": sorted(self.float_ops),
            "bias_correction": self.bias_correction,
            "calibration_samples": self.calibration_samples,
            "calibration_strategy": self.calibration_strategy,
            "seed": self.seed,
        }

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def from_dict(data: dict) -> "QuantConfig":
        payload = dict(data)
        payload["per_op"] = {
            key: dict(value) for key, value in payload.get("per_op", {}).items()
        }
        payload["float_ops"] = frozenset(payload.get("float_ops", []))
        return QuantConfig(**payload)


__all__ = ["QuantConfig", "GRANULARITIES", "SCHEMES", "CALIBRATIONS", "ROUNDINGS"]
