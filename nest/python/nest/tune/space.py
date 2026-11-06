"""AutoTune 的可插拔配置空间。

设计要点：

- 一个 :class:`ConfigSpace` 由若干 :class:`Knob` 组成，空间被编码成整数向量
  （混合进制），这样网格、随机、进化三种搜索共用同一套 ``entity``；
- ``entity`` 同时是日志去重与断点续跑的键；
- 默认空间复用 AutoSchedule 的候选生成（公共因子 + 均匀抽样），
  只把调度模板真正支持的维度暴露为 knob。

配置空间不包含 ``sram_bytes``：它是预算约束，不是可搜索维度。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from ..errors import TuneError
from ..ir import Module
from ..tile.autoschedule import AutoScheduleOptions, _build_axes
from ..tile.schedule import ScheduleOptions


@dataclass(frozen=True)
class Knob:
    """一个可搜索维度及其候选值。"""

    name: str
    values: tuple

    def __post_init__(self) -> None:
        if not self.name:
            raise TuneError("Knob.name 不能为空")
        if not self.values:
            raise TuneError(f"Knob {self.name} 没有候选值")
        if len(set(self.values)) != len(self.values):
            raise TuneError(f"Knob {self.name} 的候选值重复: {self.values}")

    def index_of(self, value) -> int:
        for index, candidate in enumerate(self.values):
            if candidate == value:
                return index
        raise TuneError(f"Knob {self.name} 不包含值 {value!r}，候选 {self.values}")

    def summary_text(self) -> str:
        return f"{self.name}: {list(self.values)}"


@dataclass(frozen=True)
class ConfigSpace:
    """整数向量编码的配置空间；允许零个 knob（退化为单点空间）。"""

    knobs: tuple[Knob, ...] = ()

    def __post_init__(self) -> None:
        names = [knob.name for knob in self.knobs]
        if len(set(names)) != len(names):
            raise TuneError(f"ConfigSpace 存在重复 knob: {names}")

    @property
    def size(self) -> int:
        total = 1
        for knob in self.knobs:
            total *= len(knob.values)
        return total

    @property
    def dimensions(self) -> int:
        return len(self.knobs)

    def decode(self, entity: tuple[int, ...]) -> dict:
        if len(entity) != len(self.knobs):
            raise TuneError(
                f"entity 维度 {len(entity)} 与空间维度 {len(self.knobs)} 不一致"
            )
        config: dict = {}
        for knob, index in zip(self.knobs, entity):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TuneError(f"entity 元素必须是整数下标: {index!r}")
            if index < 0 or index >= len(knob.values):
                raise TuneError(
                    f"entity 下标越界: {knob.name}[{index}]，候选长度 {len(knob.values)}"
                )
            config[knob.name] = knob.values[index]
        return config

    def encode(self, config: dict) -> tuple[int, ...]:
        entity: list[int] = []
        for knob in self.knobs:
            if knob.name not in config:
                raise TuneError(f"config 缺少维度 {knob.name}")
            entity.append(knob.index_of(config[knob.name]))
        return tuple(entity)

    def entity_at(self, index: int) -> tuple[int, ...]:
        """把线性下标解码成整数向量（最后一个 knob 变化最快）。"""
        if index < 0 or index >= self.size:
            raise TuneError(f"线性下标越界: {index}，空间大小 {self.size}")
        entity = [0] * len(self.knobs)
        for position in range(len(self.knobs) - 1, -1, -1):
            width = len(self.knobs[position].values)
            entity[position] = index % width
            index //= width
        return tuple(entity)

    def all_entities(self) -> list[tuple[int, ...]]:
        return [self.entity_at(index) for index in range(self.size)]

    def enumerate(self, limit: int | None = None) -> Iterator[tuple[int, ...]]:
        for index in range(self.size):
            if limit is not None and index >= limit:
                return
            yield self.entity_at(index)

    def sample(self, rng: random.Random) -> tuple[int, ...]:
        return tuple(rng.randrange(len(knob.values)) for knob in self.knobs)

    def sample_distinct(
        self, rng: random.Random, count: int
    ) -> list[tuple[int, ...]]:
        """确定性随机采样 ``count`` 个互不相同的 entity。"""
        if count <= 0:
            return []
        if self.size <= count:
            entities = self.all_entities()
            rng.shuffle(entities)
            return entities
        picked: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        while len(picked) < count:
            entity = self.sample(rng)
            if entity in seen:
                continue
            seen.add(entity)
            picked.append(entity)
        return picked

    def default_entity(self, defaults: dict) -> tuple[int, ...]:
        """按 ``defaults`` 取默认下标；缺失维度取第一个候选值。"""
        entity: list[int] = []
        for knob in self.knobs:
            value = defaults.get(knob.name, knob.values[0])
            entity.append(knob.index_of(value))
        return tuple(entity)

    def summary_text(self) -> str:
        lines = [f"ConfigSpace: size={self.size} dimensions={self.dimensions}"]
        lines.extend(f"  {knob.summary_text()}" for knob in self.knobs)
        return "\n".join(lines)


def default_config(
    base: ScheduleOptions,
    *,
    cores: int = 1,
    split_dim: str = "auto",
    db_depth: int = 1,
) -> dict:
    """由 ScheduleOptions 生成一份完整默认配置。"""
    config = {
        "conv_oc_tile": base.conv_oc_tile,
        "gemm_n_tile": base.gemm_n_tile,
        "cores": cores,
        "db_depth": db_depth,
    }
    if cores > 1:
        config["split_dim"] = split_dim
    return config


def _with_base(candidates: list[int], base_value: int) -> tuple[int, ...]:
    return tuple(sorted(set(candidates) | {base_value}))


def autotune_space(
    module: Module,
    base: ScheduleOptions | None = None,
    *,
    cores: tuple[int, ...] = (1,),
    split_dims: tuple[str, ...] = ("auto",),
    db_depths: tuple[int, ...] = (1, 2),
) -> ConfigSpace:
    """根据 Module 生成默认配置空间。

    - conv / gemm 的 tile 候选来自 AutoSchedule 的公共因子；
    - ``cores``、``split_dims``、``db_depths`` 由调用方显式开启；
    - 图中不存在的算子维度不会产生 knob。
    """
    base = base or ScheduleOptions()
    if not cores or any(int(c) <= 0 for c in cores):
        raise TuneError(f"cores 候选非法: {cores}")
    if not db_depths or any(int(d) <= 0 for d in db_depths):
        raise TuneError(f"db_depths 候选非法: {db_depths}")

    axes = dict(_build_axes(module, AutoScheduleOptions(base=base)))
    knobs: list[Knob] = []
    if "conv" in axes:
        knobs.append(
            Knob("conv_oc_tile", _with_base(axes["conv"], base.conv_oc_tile))
        )
    if "gemm" in axes:
        knobs.append(
            Knob("gemm_n_tile", _with_base(axes["gemm"], base.gemm_n_tile))
        )
    knobs.append(Knob("cores", tuple(sorted(set(int(c) for c in cores)))))
    if max(cores) > 1:
        knobs.append(Knob("split_dim", tuple(sorted(set(split_dims)))))
    knobs.append(Knob("db_depth", tuple(sorted(set(int(d) for d in db_depths)))))
    return ConfigSpace(tuple(knobs))


__all__ = [
    "ConfigSpace",
    "Knob",
    "autotune_space",
    "default_config",
]
