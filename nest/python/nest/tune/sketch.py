"""Ansor 的 sketch 生成。

在这里 sketch 表示**结构选择**：算子模板序列、是否融合、多核切分、双缓冲
深度；tile 因子属于 annotation（由 :mod:`nest.tune.ansor` 的进化搜索填充）。

与 TVM Ansor 的差异：
- 没有 TE/TIR 的 ``create_schedule`` 原语，sketch 直接映射到本仓库的
  ``Stage / Loop / Buffer`` 模板；
- 调度模板目前只支持 conv 按输出通道、gemm 按输出 N 维 tile，
  因此 annotation 轴只暴露这两个维度；空间维度的多级 tiling 需要先扩展
  ``schedule.py`` 的模板。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..errors import TuneError
from ..ir import CallNode, Module
from ..passes.fusion import DEFAULT_PATTERNS, fuse
from ..tile.autoschedule import AutoScheduleOptions, _build_axes
from ..tile.schedule import ScheduleOptions
from .build import BuildResult, build_module
from .space import ConfigSpace, Knob

_TEMPLATE_RULES = (
    (frozenset({"conv2d", "conv2d_relu"}), "conv"),
    (frozenset({"gemm", "gemm_relu"}), "gemm"),
    (frozenset({"max_pool2d", "avg_pool2d", "global_avg_pool2d", "softmax"}), "reduce"),
)

_LEVELS = ("global", "sram", "register")


@dataclass(frozen=True)
class AxisSpec:
    """一个可调 tile 轴。"""

    name: str
    level: str
    candidates: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise TuneError("AxisSpec.name 不能为空")
        if self.level not in _LEVELS:
            raise TuneError(f"非法 AxisSpec.level: {self.level!r}，可选 {_LEVELS}")
        if not self.candidates:
            raise TuneError(f"AxisSpec {self.name} 没有候选因子")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "candidates": list(self.candidates),
        }


def _template_of(canonical: str) -> str:
    for ops, template in _TEMPLATE_RULES:
        if canonical in ops:
            return template
    return "generic"


def _module_structure(module: Module) -> tuple[tuple[str, ...], tuple[str, ...]]:
    templates: list[str] = []
    ops: list[str] = []
    for node in module.graph.nodes:
        if not isinstance(node, CallNode):
            continue
        templates.append(_template_of(node.canonical_name))
        ops.append(node.canonical_name)
    return tuple(templates), tuple(ops)


@dataclass(frozen=True)
class _FuseOptions:
    fusion_patterns: tuple[str, ...] = DEFAULT_PATTERNS


@dataclass(frozen=True)
class Sketch:
    """一个结构 + 一组 annotation 轴。"""

    name: str
    templates: tuple[str, ...]
    ops: tuple[str, ...]
    axes: tuple[AxisSpec, ...]
    cores: int = 1
    split_dim: str = "auto"
    db_depth: int = 1
    fusion_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cores <= 0:
            raise TuneError(f"Sketch.cores 必须为正数: {self.cores}")
        if self.db_depth <= 0:
            raise TuneError(f"Sketch.db_depth 必须为正数: {self.db_depth}")

    @property
    def structural(self) -> dict:
        config = {"cores": self.cores, "db_depth": self.db_depth}
        if self.cores > 1:
            config["split_dim"] = self.split_dim
        return config

    @property
    def space(self) -> ConfigSpace:
        return ConfigSpace(tuple(Knob(axis.name, axis.candidates) for axis in self.axes))

    def config(self, entity: tuple[int, ...]) -> dict:
        config = self.structural
        config.update(self.space.decode(entity))
        return config

    def prepare(self, module: Module) -> Module:
        """按 sketch 的融合策略预处理 Module。"""
        if not self.fusion_patterns:
            return module
        return fuse(module, _FuseOptions(self.fusion_patterns)).module

    def build(
        self,
        module: Module,
        entity: tuple[int, ...],
        base: ScheduleOptions,
    ) -> BuildResult:
        return build_module(
            module, self.config(entity), base, entity=tuple(entity)
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "templates": list(self.templates),
            "ops": list(self.ops),
            "axes": [axis.to_dict() for axis in self.axes],
            "cores": self.cores,
            "split_dim": self.split_dim,
            "db_depth": self.db_depth,
            "fusion_patterns": list(self.fusion_patterns),
        }

    def summary_text(self) -> str:
        axes = ", ".join(
            f"{axis.name}={list(axis.candidates)}" for axis in self.axes
        )
        return (
            f"{self.name}: templates={list(self.templates)} "
            f"ops={list(self.ops)} cores={self.cores} db={self.db_depth} "
            f"axes=[{axes}]"
        )


def _axes_for(module: Module, base: ScheduleOptions) -> tuple[AxisSpec, ...]:
    axes: list[AxisSpec] = []
    candidates = dict(_build_axes(module, AutoScheduleOptions(base=base)))
    if "conv" in candidates:
        axes.append(
            AxisSpec(
                "conv_oc_tile",
                "sram",
                tuple(sorted(set(candidates["conv"]) | {base.conv_oc_tile})),
            )
        )
    if "gemm" in candidates:
        axes.append(
            AxisSpec(
                "gemm_n_tile",
                "sram",
                tuple(sorted(set(candidates["gemm"]) | {base.gemm_n_tile})),
            )
        )
    return tuple(axes)


def generate_sketches(
    module: Module,
    base: ScheduleOptions | None = None,
    *,
    cores: tuple[int, ...] = (1,),
    split_dims: tuple[str, ...] = ("auto",),
    db_depths: tuple[int, ...] = (1, 2),
    fusion_patterns: tuple[tuple[str, ...], ...] = ((), DEFAULT_PATTERNS),
) -> list[Sketch]:
    """枚举结构组合，返回去重后的 sketch 列表。"""
    base = base or ScheduleOptions()
    if not cores or any(int(item) <= 0 for item in cores):
        raise TuneError(f"cores 候选非法: {cores}")
    if not db_depths or any(int(item) <= 0 for item in db_depths):
        raise TuneError(f"db_depths 候选非法: {db_depths}")

    effective_split_dims = split_dims if max(cores) > 1 else ("auto",)
    sketches: list[Sketch] = []
    seen: set[tuple] = set()
    for core_count, split_dim, db_depth, patterns in product(
        cores, effective_split_dims, db_depths, fusion_patterns
    ):
        prepared = module
        if patterns:
            prepared = fuse(module, _FuseOptions(tuple(patterns))).module
        templates, ops = _module_structure(prepared)
        axes = _axes_for(prepared, base)
        key = (templates, ops, int(core_count), str(split_dim), int(db_depth), tuple(patterns))
        if key in seen:
            continue
        seen.add(key)
        name = (
            f"sketch_c{int(core_count)}_db{int(db_depth)}_"
            f"{'-'.join(templates) if templates else 'empty'}"
            + ("_fused" if patterns else "")
        )
        sketches.append(
            Sketch(
                name=name,
                templates=templates,
                ops=ops,
                axes=axes,
                cores=int(core_count),
                split_dim=str(split_dim),
                db_depth=int(db_depth),
                fusion_patterns=tuple(patterns),
            )
        )
    return sketches


__all__ = ["AxisSpec", "Sketch", "generate_sketches"]
