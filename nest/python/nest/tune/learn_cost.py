"""Ansor 的特征抽取与学习型代价模型（纯 numpy GBDT）。

模型只用于**排序剪枝**：最终结论必须来自 Runner 的实测值。
训练标签取 ``log10(cost)``，以改善数量级跨度大的回归条件。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..errors import TuneError
from .build import BuildResult
from .runner import program_stats

FEATURE_NAMES = (
    "log2_conv_tile",
    "log2_gemm_tile",
    "cores",
    "db_depth",
    "split_explicit",
    "stages",
    "log_instructions",
    "log_mul_instructions",
    "log_mem_ops",
    "log_dma_bytes",
    "log_sram_peak",
    "arithmetic_intensity",
    "instructions_per_core",
)


def extract_features(build: BuildResult) -> list[float]:
    """从一次构建产物抽取固定长度特征向量。"""
    stats = [program_stats(program) for program in build.lowered]
    cores = max(build.cores, 1)
    stages = max((item["stages"] for item in stats), default=0)
    instructions = sum(item["instructions"] for item in stats)
    mul = sum(item["mul_instructions"] for item in stats)
    mem_ops = sum(item["mem_ops"] for item in stats)
    dma = sum(item["dma_bytes"] for item in stats)
    conv_tile = float(build.config.get("conv_oc_tile", 1))
    gemm_tile = float(build.config.get("gemm_n_tile", 1))
    split = build.config.get("split_dim", "auto")
    return [
        math.log2(max(conv_tile, 1.0)),
        math.log2(max(gemm_tile, 1.0)),
        float(cores),
        float(build.config.get("db_depth", 1)),
        0.0 if split == "auto" else 1.0,
        float(stages),
        math.log1p(instructions),
        math.log1p(mul),
        math.log1p(mem_ops),
        math.log1p(dma),
        math.log1p(build.sram_peak),
        float(mul) / float(max(dma, 1)),
        float(instructions) / float(cores),
    ]


class _Tree:
    """一层回归树节点（递归结构，便于 JSON 序列化）。"""

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(
        self,
        value: float = 0.0,
        feature: int = -1,
        threshold: float = 0.0,
        left: "_Tree | None" = None,
        right: "_Tree | None" = None,
    ) -> None:
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0

    def predict(self, row: np.ndarray) -> float:
        node = self
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def to_dict(self) -> dict:
        if self.is_leaf:
            return {"value": self.value}
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_Tree":
        if "value" in data:
            return cls(value=float(data["value"]))
        return cls(
            feature=int(data["feature"]),
            threshold=float(data["threshold"]),
            left=cls.from_dict(data["left"]),
            right=cls.from_dict(data["right"]),
        )


def _thresholds(values: np.ndarray, max_bins: int) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= 1:
        return np.empty(0, dtype=float)
    if len(unique) <= max_bins + 1:
        return (unique[:-1] + unique[1:]) / 2.0
    positions = np.linspace(0, len(unique) - 1, max_bins + 1, dtype=int)
    picked = unique[positions]
    return (picked[:-1] + picked[1:]) / 2.0


def _fit_tree(
    X: np.ndarray,
    residual: np.ndarray,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_bins: int,
) -> _Tree:
    n = len(residual)
    mean = float(np.mean(residual))
    if depth >= max_depth or n < 2 * min_samples_leaf:
        return _Tree(value=mean)

    parent_sse = float(np.sum((residual - mean) ** 2))
    best_gain = 0.0
    best: tuple[int, float] | None = None
    for feature in range(X.shape[1]):
        for threshold in _thresholds(X[:, feature], max_bins):
            mask = X[:, feature] <= threshold
            left_count = int(np.sum(mask))
            right_count = n - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            left_mean = float(np.mean(residual[mask]))
            right_mean = float(np.mean(residual[~mask]))
            sse = float(np.sum((residual[mask] - left_mean) ** 2))
            sse += float(np.sum((residual[~mask] - right_mean) ** 2))
            gain = parent_sse - sse
            if gain > best_gain:
                best_gain = gain
                best = (feature, float(threshold))

    if best is None or best_gain <= 1e-12:
        return _Tree(value=mean)

    feature, threshold = best
    mask = X[:, feature] <= threshold
    return _Tree(
        feature=feature,
        threshold=threshold,
        left=_fit_tree(
            X[mask], residual[mask], depth + 1, max_depth, min_samples_leaf, max_bins
        ),
        right=_fit_tree(
            X[~mask], residual[~mask], depth + 1, max_depth, min_samples_leaf, max_bins
        ),
    )


class GBDTRegressor:
    """最小可用的梯度提升回归树（平方损失）。"""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_leaf: int = 3,
        max_bins: int = 16,
    ) -> None:
        if n_estimators <= 0:
            raise TuneError("n_estimators 必须为正数")
        if not 0.0 < learning_rate <= 1.0:
            raise TuneError("learning_rate 必须在 (0, 1]")
        if max_depth <= 0:
            raise TuneError("max_depth 必须为正数")
        if min_samples_leaf <= 0:
            raise TuneError("min_samples_leaf 必须为正数")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.base_ = 0.0
        self.trees_: list[_Tree] = []

    def fit(self, X, y) -> "GBDTRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise TuneError(f"训练特征必须是二维矩阵: {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise TuneError(
                f"特征行数 {X.shape[0]} 与标签数 {y.shape[0]} 不一致"
            )
        if X.shape[0] == 0:
            raise TuneError("训练集为空")

        self.base_ = float(np.mean(y))
        prediction = np.full(X.shape[0], self.base_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            residual = y - prediction
            tree = _fit_tree(
                X, residual, 0, self.max_depth, self.min_samples_leaf, self.max_bins
            )
            prediction += self.learning_rate * np.array(
                [tree.predict(row) for row in X]
            )
            self.trees_.append(tree)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise TuneError(f"预测特征必须是二维矩阵: {X.shape}")
        if not self.trees_:
            raise TuneError("模型尚未训练")
        prediction = np.full(X.shape[0], self.base_)
        for tree in self.trees_:
            prediction += self.learning_rate * np.array(
                [tree.predict(row) for row in X]
            )
        return prediction

    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_bins": self.max_bins,
            "base": self.base_,
            "trees": [tree.to_dict() for tree in self.trees_],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GBDTRegressor":
        model = cls(
            n_estimators=int(data["n_estimators"]),
            learning_rate=float(data["learning_rate"]),
            max_depth=int(data["max_depth"]),
            min_samples_leaf=int(data["min_samples_leaf"]),
            max_bins=int(data.get("max_bins", 16)),
        )
        model.base_ = float(data.get("base", 0.0))
        model.trees_ = [_Tree.from_dict(item) for item in data.get("trees", [])]
        return model


class LearnedCostModel:
    """封装特征规范化与 log10 代价标签的学习型代价模型。"""

    def __init__(self, regressor: GBDTRegressor | None = None) -> None:
        self.regressor = regressor or GBDTRegressor()
        self.samples = 0

    @property
    def ready(self) -> bool:
        return self.samples > 0 and bool(self.regressor.trees_)

    def fit(self, features, costs) -> "LearnedCostModel":
        if len(features) != len(costs):
            raise TuneError("特征数量与代价数量不一致")
        if not features:
            raise TuneError("训练集为空")
        targets = [
            math.log10(max(float(cost), 1e-9)) for cost in costs
        ]
        self.regressor.fit(features, targets)
        self.samples = len(features)
        return self

    def predict(self, feature_vector) -> float:
        return float(self.regressor.predict(np.asarray([feature_vector], dtype=float))[0])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.regressor.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedCostModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(GBDTRegressor.from_dict(data))
        model.samples = 1 if model.regressor.trees_ else 0
        return model

    def summary_text(self) -> str:
        return (
            f"LearnedCostModel: samples={self.samples} "
            f"trees={len(self.regressor.trees_)} "
            f"features={len(FEATURE_NAMES)}"
        )


__all__ = [
    "FEATURE_NAMES",
    "GBDTRegressor",
    "LearnedCostModel",
    "extract_features",
]
