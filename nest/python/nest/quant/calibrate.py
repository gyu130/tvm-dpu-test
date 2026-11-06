"""PTQ 校准：min-max / percentile / KL / MSE。

校准过程必须可复现：记录校准集哈希、随机种子与采样策略。
出现空样本、NaN/Inf 或退化范围（scale=0）时显式失败。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from ..errors import CalibrationError
from ..ir import ConstantNode, Module
from ..registry import GLOBAL_REGISTRY, OpRegistry
from ..runtime import run_module
from .config import QuantConfig
from .qmath import compute_scale_zero_point, dtype_limits

_EPS = 1e-12


@dataclass(frozen=True)
class CalibrationSample:
    """一组校准输入。"""

    inputs: dict

    def __post_init__(self) -> None:
        if not self.inputs:
            raise CalibrationError("校准样本不能为空")
        payload = {}
        for name, value in self.inputs.items():
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.floating):
                raise CalibrationError(
                    f"校准输入 {name} 必须是浮点张量，得到 {array.dtype}"
                )
            if not np.all(np.isfinite(array)):
                raise CalibrationError(f"校准输入 {name} 包含 NaN/Inf")
            payload[name] = array
        object.__setattr__(self, "inputs", payload)

    def digest(self) -> str:
        digest = hashlib.sha256()
        for name in sorted(self.inputs):
            array = np.ascontiguousarray(self.inputs[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationSet:
    """校准样本集合。"""

    samples: tuple[CalibrationSample, ...]
    seed: int
    strategy: str

    def __post_init__(self) -> None:
        if not self.samples:
            raise CalibrationError("校准集为空，至少需要一个样本")
        expected = set(self.samples[0].inputs)
        for index, sample in enumerate(self.samples):
            if set(sample.inputs) != expected:
                raise CalibrationError(
                    f"校准样本 #{index} 的输入集合与首个样本不一致"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.seed).encode("utf-8"))
        digest.update(self.strategy.encode("utf-8"))
        for sample in self.samples:
            digest.update(sample.digest().encode("utf-8"))
        return digest.hexdigest()

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "strategy": self.strategy,
            "sample_count": len(self.samples),
            "hash": self.digest(),
        }


@dataclass
class CalibrationResult:
    """每个张量的校准范围。"""

    ranges: dict[str, tuple[float, float]]
    channel_ranges: dict[str, tuple[np.ndarray, np.ndarray]]
    tensor_order: tuple[str, ...]
    calibration_hash: str
    seed: int
    strategy: str
    algorithm: str
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "calibration_hash": self.calibration_hash,
            "seed": self.seed,
            "strategy": self.strategy,
            "sample_count": self.sample_count,
            "ranges": {
                name: list(self.ranges[name]) for name in self.tensor_order
            },
            "channel_ranges": {
                name: {
                    "min": lo.tolist(),
                    "max": hi.tolist(),
                }
                for name, (lo, hi) in self.channel_ranges.items()
            },
        }


def make_calibration_set(
    module: Module,
    config: QuantConfig,
    *,
    seed: int | None = None,
    count: int | None = None,
    strategy: str | None = None,
) -> CalibrationSet:
    """根据图输入类型生成确定性校准样本。"""
    seed = config.seed if seed is None else seed
    count = config.calibration_samples if count is None else count
    strategy = config.calibration_strategy if strategy is None else strategy
    if count <= 0:
        raise CalibrationError("校准样本数必须为正数")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(count):
        inputs = {}
        for value in module.graph.inputs:
            if value.type.dtype not in ("float16", "float32", "float64"):
                raise CalibrationError(
                    f"校准输入 {value.name} 必须是浮点类型，得到 "
                    f"{value.type.dtype}"
                )
            if strategy == "normal":
                data = rng.standard_normal(value.type.shape)
            elif strategy == "uniform":
                data = rng.uniform(-1.0, 1.0, size=value.type.shape)
            else:
                raise CalibrationError(f"未知校准策略: {strategy}")
            inputs[value.name] = data.astype(value.type.dtype)
        samples.append(CalibrationSample(inputs))
    return CalibrationSet(tuple(samples), seed=seed, strategy=strategy)


def _reduce_axis(values: np.ndarray, axis: int) -> np.ndarray:
    axes = tuple(index for index in range(values.ndim) if index != axis)
    if not axes:
        return values.astype(np.float64)
    return np.min(values, axis=axes)


def _reduce_axis_max(values: np.ndarray, axis: int) -> np.ndarray:
    axes = tuple(index for index in range(values.ndim) if index != axis)
    if not axes:
        return values.astype(np.float64)
    return np.max(values, axis=axes)


def _range_from_threshold(values: np.ndarray, threshold: float) -> tuple[float, float]:
    if threshold <= 0.0:
        raise CalibrationError("校准范围退化为 0，无法计算 scale")
    if float(values.min()) >= 0.0:
        return 0.0, float(threshold)
    return -float(threshold), float(threshold)


def _quant_dequant_abs(
    values: np.ndarray, lo: float, hi: float, dtype: str, scheme: str
) -> np.ndarray:
    scale, zero_point = compute_scale_zero_point(lo, hi, dtype, scheme)
    qmin, qmax = dtype_limits(dtype)
    q = np.clip(np.rint(values / scale) + zero_point, qmin, qmax)
    return np.abs((q - zero_point) * scale)


def _percentile_range(
    values: np.ndarray, percentile: float, dtype: str, scheme: str
) -> tuple[float, float]:
    threshold = float(np.percentile(np.abs(values), percentile))
    if threshold == 0.0:
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            raise CalibrationError("percentile 校准得到退化范围")
        return lo, hi
    return _range_from_threshold(values, threshold)


def _kl_range(
    values: np.ndarray, bins: int, dtype: str, scheme: str
) -> tuple[float, float]:
    abs_values = np.abs(values.astype(np.float64))
    max_value = float(abs_values.max())
    if max_value == 0.0:
        raise CalibrationError("KL 校准得到全零张量")
    candidates = np.linspace(max_value * 0.5, max_value, 64)
    best_threshold = max_value
    best_divergence = np.inf
    for threshold in candidates:
        lo, hi = _range_from_threshold(values, float(threshold))
        q_abs = _quant_dequant_abs(values, lo, hi, dtype, scheme)
        p_hist, _ = np.histogram(np.minimum(abs_values, threshold),
                                 bins=bins, range=(0.0, float(threshold)))
        q_hist, _ = np.histogram(np.minimum(q_abs, threshold),
                                 bins=bins, range=(0.0, float(threshold)))
        p = p_hist.astype(np.float64) + _EPS
        q = q_hist.astype(np.float64) + _EPS
        p /= p.sum()
        q /= q.sum()
        divergence = float(np.sum(p * np.log(p / q)))
        if divergence < best_divergence:
            best_divergence = divergence
            best_threshold = float(threshold)
    return _range_from_threshold(values, best_threshold)


def _mse_range(
    values: np.ndarray, dtype: str, scheme: str
) -> tuple[float, float]:
    abs_values = np.abs(values.astype(np.float64))
    max_value = float(abs_values.max())
    if max_value == 0.0:
        raise CalibrationError("MSE 校准得到全零张量")
    candidates = {
        float(np.percentile(abs_values, p))
        for p in range(50, 101, 2)
    }
    candidates.add(max_value)
    best_threshold = max_value
    best_mse = np.inf
    for threshold in sorted(candidates):
        if threshold <= 0.0:
            continue
        lo, hi = _range_from_threshold(values, threshold)
        q_abs = _quant_dequant_abs(values, lo, hi, dtype, scheme)
        mse = float(np.mean((np.abs(values) - q_abs) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_threshold = threshold
    return _range_from_threshold(values, best_threshold)


def _compute_range(
    values: np.ndarray, config: QuantConfig, dtype: str, scheme: str
) -> tuple[float, float]:
    flat = values.astype(np.float64).reshape(-1)
    if flat.size == 0:
        raise CalibrationError("校准张量为空")
    if not np.all(np.isfinite(flat)):
        raise CalibrationError("校准张量包含 NaN/Inf")
    if config.calibration == "minmax":
        lo, hi = float(flat.min()), float(flat.max())
        if lo == hi:
            raise CalibrationError(
                f"min-max 校准得到退化范围 [{lo}, {hi}]，scale 为 0"
            )
        return lo, hi
    if config.calibration == "percentile":
        return _percentile_range(flat, config.percentile, dtype, scheme)
    if config.calibration == "kl":
        return _kl_range(flat, config.kl_bins, dtype, scheme)
    if config.calibration == "mse":
        return _mse_range(flat, dtype, scheme)
    raise CalibrationError(f"未知校准算法: {config.calibration}")


def calibrate(
    module: Module,
    calibration_set: CalibrationSet,
    config: QuantConfig,
    *,
    registry: OpRegistry = GLOBAL_REGISTRY,
) -> CalibrationResult:
    """逐样本执行参考实现，收集每个张量的激活范围。"""
    if not calibration_set.samples:
        raise CalibrationError("校准集为空")
    expected_inputs = {value.name for value in module.graph.inputs}
    for sample in calibration_set.samples:
        if set(sample.inputs) != expected_inputs:
            raise CalibrationError(
                "校准样本输入与图输入不一致："
                f"{sorted(sample.inputs)} vs {sorted(expected_inputs)}"
            )

    collected: dict[str, list[np.ndarray]] = {}
    order: list[str] = []
    for value in module.graph.inputs:
        order.append(value.name)
        collected[value.name] = []
    for node in module.graph.nodes:
        if isinstance(node, ConstantNode):
            continue
        for value in node.outputs:
            order.append(value.name)
            collected.setdefault(value.name, [])

    for sample in calibration_set.samples:
        sink: dict[str, np.ndarray] = {}
        run_module(module, sample.inputs, registry=registry, tensor_sink=sink)
        for value in module.graph.inputs:
            collected[value.name].append(np.asarray(sample.inputs[value.name]))
        for name, tensor in sink.items():
            if name not in collected:
                raise CalibrationError(f"校准收集到未知张量: {name}")
            collected[name].append(np.asarray(tensor))

    ranges: dict[str, tuple[float, float]] = {}
    channel_ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    axis = config.activation_channel_axis
    for name in order:
        values = collected[name]
        if not values:
            raise CalibrationError(
                f"张量 {name} 没有校准数据（可能未被参考执行器产出）"
            )
        for tensor in values:
            if not np.all(np.isfinite(tensor)):
                raise CalibrationError(f"张量 {name} 的校准数据包含 NaN/Inf")
        stacked = np.stack([np.asarray(t).astype(np.float64) for t in values])
        ranges[name] = _compute_range(stacked, config, config.activation_dtype,
                                      config.activation_scheme)
        shape = stacked.shape[1:]
        if len(shape) > axis:
            channel_lo = np.min(
                np.stack([_reduce_axis(t, axis) for t in stacked]), axis=0
            )
            channel_hi = np.max(
                np.stack([_reduce_axis_max(t, axis) for t in stacked]), axis=0
            )
            channel_ranges[name] = (channel_lo, channel_hi)

    return CalibrationResult(
        ranges=ranges,
        channel_ranges=channel_ranges,
        tensor_order=tuple(order),
        calibration_hash=calibration_set.digest(),
        seed=calibration_set.seed,
        strategy=calibration_set.strategy,
        algorithm=config.calibration,
        sample_count=len(calibration_set.samples),
    )


def calibration_result_to_json(result: CalibrationResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


__all__ = [
    "CalibrationSample",
    "CalibrationSet",
    "CalibrationResult",
    "make_calibration_set",
    "calibrate",
    "calibration_result_to_json",
]
