"""逐层精度报告：cosine / SQNR / 误差，并自动定位首个发散层。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass

import numpy as np

from ..errors import QuantizationError
from ..ir import ConstantNode, Module
from ..ir.types import SourceSpan


@dataclass(frozen=True)
class AccuracyThresholds:
    min_cosine: float = 0.99
    min_sqnr_db: float = 20.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_cosine <= 1.0):
            raise QuantizationError(f"min_cosine 非法: {self.min_cosine}")
        if self.min_sqnr_db < 0.0:
            raise QuantizationError(f"min_sqnr_db 非法: {self.min_sqnr_db}")

    def to_dict(self) -> dict:
        return {
            "min_cosine": self.min_cosine,
            "min_sqnr_db": self.min_sqnr_db,
        }


@dataclass
class LayerMetrics:
    name: str
    op_type: str
    index: int
    span: SourceSpan | None
    shape: tuple[int, ...]
    dtype: str
    quant_mode: str
    cosine: float
    sqnr_db: float
    max_abs_error: float
    mean_abs_error: float
    min_cosine: float
    min_sqnr_db: float
    is_divergent: bool

    @property
    def sqnr_exact(self) -> bool:
        return math.isinf(self.sqnr_db)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "index": self.index,
            "span": self.span.to_dict() if self.span is not None else None,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "quant_mode": self.quant_mode,
            "cosine": self.cosine,
            "sqnr_db": None if self.sqnr_exact else self.sqnr_db,
            "sqnr_exact": self.sqnr_exact,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "is_divergent": self.is_divergent,
        }


@dataclass
class AccuracyReport:
    layers: tuple[LayerMetrics, ...]
    first_divergent: str | None
    thresholds: AccuracyThresholds
    config_hash: str
    calibration_hash: str
    seed: int
    reproduction: str
    source: str | None
    aggregates: dict

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds.to_dict(),
            "config_hash": self.config_hash,
            "calibration_hash": self.calibration_hash,
            "seed": self.seed,
            "reproduction": self.reproduction,
            "source": self.source,
            "first_divergent": self.first_divergent,
            "layers": [layer.to_dict() for layer in self.layers],
            "aggregates": self.aggregates,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    def to_csv(self) -> str:
        lines = []
        writer = csv.writer(_StringSink(lines))
        writer.writerow(
            [
                "index",
                "name",
                "op_type",
                "shape",
                "dtype",
                "quant_mode",
                "cosine",
                "sqnr_db",
                "max_abs_error",
                "mean_abs_error",
                "is_divergent",
            ]
        )
        for layer in self.layers:
            writer.writerow(
                [
                    layer.index,
                    layer.name,
                    layer.op_type,
                    "x".join(str(dim) for dim in layer.shape),
                    layer.dtype,
                    layer.quant_mode,
                    f"{layer.cosine:.8f}",
                    "inf" if layer.sqnr_exact else f"{layer.sqnr_db:.4f}",
                    f"{layer.max_abs_error:.8f}",
                    f"{layer.mean_abs_error:.8f}",
                    int(layer.is_divergent),
                ]
            )
        return "".join(lines)

    def save_csv(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(self.to_csv())

    def summary_text(self) -> str:
        header = (
            f"{'idx':>4} {'name':<24} {'op':<14} {'cosine':>9} "
            f"{'sqnr(dB)':>9} {'max_abs':>10} {'div':>4}"
        )
        lines = [header, "-" * len(header)]
        for layer in self.layers:
            sqnr = "inf" if layer.sqnr_exact else f"{layer.sqnr_db:.2f}"
            lines.append(
                f"{layer.index:>4} {layer.name:<24} {layer.op_type:<14} "
                f"{layer.cosine:>9.5f} {sqnr:>9} "
                f"{layer.max_abs_error:>10.5f} {int(layer.is_divergent):>4}"
            )
        lines.append("")
        if self.first_divergent is None:
            lines.append(
                "首个发散层: 无（所有层满足 "
                f"cosine>={self.thresholds.min_cosine}, "
                f"SQNR>={self.thresholds.min_sqnr_db} dB）"
            )
        else:
            lines.append(f"首个发散层: {self.first_divergent}")
        lines.append(
            f"config_hash={self.config_hash} "
            f"calibration_hash={self.calibration_hash} seed={self.seed}"
        )
        if self.reproduction:
            lines.append(f"复现: {self.reproduction}")
        return "\n".join(lines)


class _StringSink:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def write(self, text: str) -> None:
        self.lines.append(text)


def _metrics(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float, float, float]:
    ref = reference.astype(np.float64).reshape(-1)
    act = actual.astype(np.float64).reshape(-1)
    if ref.shape != act.shape:
        raise QuantizationError(
            f"张量 shape 不一致: golden={ref.shape} simulated={act.shape}"
        )
    if ref.size == 0:
        return 1.0, math.inf, 0.0, 0.0
    diff = ref - act
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    ref_norm = float(np.linalg.norm(ref))
    act_norm = float(np.linalg.norm(act))
    if ref_norm == 0.0 and act_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or act_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(ref, act) / (ref_norm * act_norm))
        cosine = max(-1.0, min(1.0, cosine))
    noise = float(np.sum(diff ** 2))
    if noise == 0.0:
        sqnr = math.inf
    else:
        signal = float(np.sum(ref ** 2))
        sqnr = math.inf if signal == 0.0 else 10.0 * math.log10(signal / noise)
    return cosine, sqnr, max_abs, mean_abs


def _aggregate(layers: list[LayerMetrics]) -> dict:
    def summarize(items: list[LayerMetrics]) -> dict:
        return {
            "count": len(items),
            "divergent": sum(1 for item in items if item.is_divergent),
            "min_cosine": min((item.cosine for item in items), default=1.0),
            "min_sqnr_db": min(
                (item.sqnr_db for item in items), default=math.inf
            ),
            "max_mean_abs_error": max(
                (item.mean_abs_error for item in items), default=0.0
            ),
        }

    by_op: dict[str, list[LayerMetrics]] = {}
    by_mode: dict[str, list[LayerMetrics]] = {}
    for layer in layers:
        by_op.setdefault(layer.op_type, []).append(layer)
        by_mode.setdefault(layer.quant_mode, []).append(layer)
    return {
        "by_op": {key: summarize(value) for key, value in sorted(by_op.items())},
        "by_quant_mode": {
            key: summarize(value) for key, value in sorted(by_mode.items())
        },
    }


def build_accuracy_report(
    module: Module,
    reference: dict[str, np.ndarray],
    simulated: dict[str, np.ndarray],
    *,
    config_hash: str,
    calibration_hash: str,
    seed: int,
    thresholds: AccuracyThresholds | None = None,
    reproduction: str = "",
    source: str | None = None,
) -> AccuracyReport:
    """按拓扑顺序对比 golden 与 QDQ 仿真张量。"""
    thresholds = thresholds or AccuracyThresholds()
    types = module.graph.tensors
    order: list[tuple[str, str, SourceSpan | None]] = []
    for value in module.graph.inputs:
        order.append((value.name, "input", None))
    for node in module.graph.nodes:
        if isinstance(node, ConstantNode):
            for value in node.outputs:
                order.append((value.name, "constant", node.span))
        else:
            for value in node.outputs:
                order.append((value.name, node.op_type, node.span))

    layers: list[LayerMetrics] = []
    first_divergent: str | None = None
    for index, (name, op_type, span) in enumerate(order):
        if name not in reference:
            raise QuantizationError(f"缺少 golden 张量: {name}")
        if name not in simulated:
            raise QuantizationError(f"缺少仿真张量: {name}")
        cosine, sqnr, max_abs, mean_abs = _metrics(
            reference[name], simulated[name]
        )
        is_divergent = cosine < thresholds.min_cosine or (
            not math.isinf(sqnr) and sqnr < thresholds.min_sqnr_db
        )
        if is_divergent and first_divergent is None:
            first_divergent = name
        tensor_type = types[name].type
        quant_mode = (
            tensor_type.quant.mode if tensor_type.quant is not None else "none"
        )
        layers.append(
            LayerMetrics(
                name=name,
                op_type=op_type,
                index=index,
                span=span,
                shape=tuple(tensor_type.shape),
                dtype=tensor_type.dtype,
                quant_mode=quant_mode,
                cosine=cosine,
                sqnr_db=sqnr,
                max_abs_error=max_abs,
                mean_abs_error=mean_abs,
                min_cosine=thresholds.min_cosine,
                min_sqnr_db=thresholds.min_sqnr_db,
                is_divergent=is_divergent,
            )
        )
    return AccuracyReport(
        layers=tuple(layers),
        first_divergent=first_divergent,
        thresholds=thresholds,
        config_hash=config_hash,
        calibration_hash=calibration_hash,
        seed=seed,
        reproduction=reproduction,
        source=source,
        aggregates=_aggregate(layers),
    )


__all__ = ["AccuracyThresholds", "LayerMetrics", "AccuracyReport",
           "build_accuracy_report"]
