"""量化数值语义（PTQ 与仿真共用的一份定义）。

规则：
- per-tensor：``scale`` / ``zero_point`` 长度为 1；
- per-channel：长度等于量化轴维度；
- per-block：沿量化轴按 ``block_size`` 分块，长度等于块数。

所有函数都显式要求传入 dtype（QuantInfo 本身不携带 dtype），
避免由后端“猜测”有符号/无符号范围。
"""

from __future__ import annotations

import numpy as np

from ..errors import QuantizationError
from ..ir.types import QuantInfo, QUANTIZED_DTYPES

_DTYPE_BITS = {
    "int8": 8,
    "uint8": 8,
    "int16": 16,
    "uint16": 16,
}

_ROUNDINGS = ("nearest_even", "nearest_away", "truncate")


def dtype_bits(dtype: str) -> int:
    if dtype not in _DTYPE_BITS:
        raise QuantizationError(f"不支持的量化 dtype: {dtype}")
    return _DTYPE_BITS[dtype]


def dtype_limits(dtype: str) -> tuple[int, int]:
    if dtype not in QUANTIZED_DTYPES:
        raise QuantizationError(f"不支持的量化 dtype: {dtype}")
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def compute_scale_zero_point(
    lo: float, hi: float, dtype: str, scheme: str
) -> tuple[float, int]:
    """由浮点范围计算显式 scale / zero_point。"""
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise QuantizationError(f"量化范围包含 NaN/Inf: [{lo}, {hi}]")
    if lo > hi:
        raise QuantizationError(f"量化范围非法: lo={lo} > hi={hi}")
    qmin, qmax = dtype_limits(dtype)
    if scheme == "symmetric":
        if qmin >= 0:
            raise QuantizationError(
                f"对称量化要求有符号量化 dtype，得到 {dtype}"
            )
        max_abs = max(abs(lo), abs(hi))
        if max_abs == 0.0:
            raise QuantizationError("量化范围全为 0，scale 为 0")
        scale = max_abs / abs(qmin)
        return float(scale), 0
    if scheme != "asymmetric":
        raise QuantizationError(f"非法量化方案: {scheme}")
    if hi - lo == 0.0:
        raise QuantizationError(
            f"量化范围退化（lo=hi={lo}），scale 为 0"
        )
    scale = (hi - lo) / float(qmax - qmin)
    zero_point = int(round(qmin - lo / scale))
    zero_point = int(min(max(zero_point, qmin), qmax))
    return float(scale), zero_point


def _param_array(
    param: tuple[float, ...] | tuple[int, ...],
    shape: tuple[int, ...],
    axis: int,
    block_size: int | None,
) -> np.ndarray | float:
    if not param:
        raise QuantizationError("量化参数为空")
    values = np.asarray(param, dtype=np.float64)
    if len(shape) == 0:
        if values.size != 1:
            raise QuantizationError("标量张量只允许 per-tensor 量化参数")
        return float(values[0])
    if axis < 0 or axis >= len(shape):
        raise QuantizationError(f"量化轴 {axis} 超出张量秩 {len(shape)}")
    if values.size == 1:
        return float(values[0])
    dim = int(shape[axis])
    if block_size is not None:
        if block_size <= 0 or dim % block_size != 0:
            raise QuantizationError(
                f"per-block 块长 {block_size} 不能整除量化轴 {dim}"
            )
        blocks = dim // block_size
        if values.size != blocks:
            raise QuantizationError(
                f"per-block 参数长度 {values.size} 与块数 {blocks} 不一致"
            )
        p = values.reshape((blocks,) + (1,) * len(shape))
        moved = np.broadcast_to(
            p, (blocks, block_size) + tuple(shape[:axis]) + tuple(shape[axis + 1:])
        )
        moved = moved.reshape((dim,) + tuple(shape[:axis]) + tuple(shape[axis + 1:]))
        return np.moveaxis(moved, 0, axis)
    if values.size != dim:
        raise QuantizationError(
            f"per-channel 参数长度 {values.size} 与量化轴 {dim} 不一致"
        )
    return values.reshape(
        tuple(1 if i != axis else dim for i in range(len(shape)))
    )


def _round(values: np.ndarray, rounding: str) -> np.ndarray:
    if rounding == "nearest_even":
        return np.rint(values)
    if rounding == "nearest_away":
        return np.where(values >= 0, np.floor(values + 0.5),
                        np.ceil(values - 0.5))
    if rounding == "truncate":
        return np.trunc(values)
    raise QuantizationError(f"非法舍入模式: {rounding}")


def quantize(
    x: np.ndarray,
    quant: QuantInfo,
    dtype: str,
    *,
    rounding: str | None = None,
    saturation: bool | None = None,
) -> np.ndarray:
    """把浮点张量量化为整数张量（直接读取 QuantInfo，不重复推导）。"""
    if quant.mode != "int":
        raise QuantizationError(f"张量量化模式不是 int: {quant.mode}")
    rounding = rounding or quant.rounding
    saturation = quant.saturation if saturation is None else saturation
    if rounding not in _ROUNDINGS:
        raise QuantizationError(f"非法舍入模式: {rounding}")
    shape = tuple(x.shape)
    scale = _param_array(quant.scale or (), shape, quant.axis or 0, quant.block_size)
    if isinstance(scale, float) and scale <= 0.0:
        raise QuantizationError("量化 scale 必须为正数")
    if isinstance(scale, np.ndarray) and np.any(scale <= 0.0):
        raise QuantizationError("量化 scale 必须为正数")
    zero_point = _param_array(
        quant.zero_point or (), shape, quant.axis or 0, quant.block_size
    )
    q = _round(x.astype(np.float64) / scale, rounding) + zero_point
    qmin, qmax = dtype_limits(dtype)
    if saturation:
        q = np.clip(q, qmin, qmax)
    else:
        span = qmax - qmin + 1
        q = (np.mod(q - qmin, span) + qmin)
    return q.astype(dtype, copy=False)


def dequantize(
    q: np.ndarray, quant: QuantInfo, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """把整数张量反量化到 float64。"""
    if quant.mode != "int":
        raise QuantizationError(f"张量量化模式不是 int: {quant.mode}")
    shape = tuple(q.shape) if shape is None else tuple(shape)
    scale = _param_array(quant.scale or (), shape, quant.axis or 0, quant.block_size)
    if isinstance(scale, float) and scale <= 0.0:
        raise QuantizationError("量化 scale 必须为正数")
    if isinstance(scale, np.ndarray) and np.any(scale <= 0.0):
        raise QuantizationError("量化 scale 必须为正数")
    zero_point = _param_array(
        quant.zero_point or (), shape, quant.axis or 0, quant.block_size
    )
    return (q.astype(np.float64) - zero_point) * scale


def quantize_dequantize(
    x: np.ndarray, quant: QuantInfo, dtype: str, out_dtype: str = "float32"
) -> np.ndarray:
    q = quantize(x, quant, dtype)
    return dequantize(q, quant, tuple(x.shape)).astype(out_dtype, copy=False)


__all__ = [
    "compute_scale_zero_point",
    "dequantize",
    "dtype_bits",
    "dtype_limits",
    "quantize",
    "quantize_dequantize",
]
