"""AutoTune 调优器：网格 / 随机 / 坐标下降三种策略。

与 :mod:`nest.tile.autoschedule` 的区别：

- AutoSchedule 只有固定的两个 knob，且用静态代价模型一次遍历；
- AutoTune 的配置空间可插拔（tile / 多核 / 双缓冲），测量由 Runner 决定，
  结果写入 JSONL 日志，支持断点续跑与确定性复现。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ..errors import TuneError
from ..ir import Module
from ..tile.schedule import ScheduleOptions
from .build import BuildResult, build_module
from .log import TrialRecord, TuneLog, make_record
from .runner import Runner, StaticRunner
from .space import ConfigSpace, autotune_space, default_config

_STRATEGIES = ("grid", "random", "greedy")


@dataclass(frozen=True)
class TuneOptions:
    """调优参数。"""

    strategy: str = "grid"
    max_trials: int = 256
    seed: int = 0
    log_path: str | Path | None = None
    resume: bool = True
    patience: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGIES:
            raise TuneError(f"非法 strategy: {self.strategy!r}，可选 {_STRATEGIES}")
        if self.max_trials <= 0:
            raise TuneError("max_trials 必须为正数")
        if self.patience < 0:
            raise TuneError("patience 不能为负")


@dataclass
class TuneResult:
    """一次调优的结果。"""

    space: ConfigSpace
    strategy: str
    best: TrialRecord | None
    best_build: BuildResult | None
    records: list[TrialRecord]
    log: TuneLog

    @property
    def best_config(self) -> dict | None:
        return None if self.best is None else dict(self.best.config)

    @property
    def best_cost(self) -> float | None:
        return None if self.best is None else float(self.best.cost)

    def summary_text(self, limit: int = 10) -> str:
        feasible = [record for record in self.records if record.ok]
        feasible.sort(key=lambda record: record.sort_key)
        lines = [
            f"AutoTune[{self.strategy}]: {len(self.records)} trials, "
            f"{len(feasible)} feasible",
        ]
        if self.best is not None:
            lines.append(
                f"BEST entity={self.best.entity} cost={self.best.cost:.0f} "
                f"config={self.best.config}"
            )
        else:
            lines.append("BEST 无可行配置")
        lines.extend(
            f"  entity={record.entity} cost={record.cost:.0f} config={record.config}"
            for record in feasible[:limit]
        )
        return "\n".join(lines)


def _fresh_log(path: str | Path | None, resume: bool) -> TuneLog:
    if path is None:
        return TuneLog()
    path = Path(path)
    if not resume:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return TuneLog(path)


def tune(
    module: Module,
    space: ConfigSpace | None = None,
    runner: Runner | None = None,
    *,
    base: ScheduleOptions | None = None,
    options: TuneOptions | None = None,
) -> TuneResult:
    """在配置空间中搜索最优配置。"""
    base = base or ScheduleOptions()
    opts = options or TuneOptions()
    runner = runner or StaticRunner()
    space = space or autotune_space(module, base)

    log = _fresh_log(opts.log_path, opts.resume)
    loaded = {record.entity: record for record in log.records} if opts.resume else {}
    builds: dict[tuple[int, ...], BuildResult] = {}
    records: list[TrialRecord] = []
    seen: set[tuple[int, ...]] = set()
    budget = {"left": opts.max_trials}

    def evaluate(entity: tuple[int, ...]) -> TrialRecord | None:
        if opts.resume and entity in loaded:
            return loaded[entity]
        if entity in seen or budget["left"] <= 0:
            return None
        budget["left"] -= 1
        seen.add(entity)
        config = space.decode(entity)
        try:
            build = build_module(module, config, base, entity=entity)
        except Exception as exc:  # noqa: BLE001 - 不可行配置不能中断调优
            record = make_record(
                entity,
                config,
                None,
                error=f"{type(exc).__name__}: {exc}",
                runner=runner.name,
            )
        else:
            builds[entity] = build
            measurement = runner.run(build)
            record = make_record(
                entity,
                config,
                measurement.cost,
                error=measurement.error,
                detail=measurement.detail,
                runner=runner.name,
            )
        log.append(record)
        records.append(record)
        return record

    no_improve = 0
    best_cost = float("inf")

    def observe(record: TrialRecord | None) -> None:
        nonlocal no_improve, best_cost
        if record is None:
            return
        if record.ok and record.cost < best_cost:
            best_cost = float(record.cost)
            no_improve = 0
        else:
            no_improve += 1

    if opts.strategy == "grid":
        for entity in space.enumerate():
            if budget["left"] <= 0:
                break
            if opts.resume and entity in loaded:
                continue
            observe(evaluate(entity))
            if opts.patience and no_improve >= opts.patience:
                break

    elif opts.strategy == "random":
        rng = random.Random(opts.seed)
        sample_count = min(space.size, opts.max_trials)
        for entity in space.sample_distinct(rng, sample_count):
            if budget["left"] <= 0:
                break
            if opts.resume and entity in loaded:
                continue
            observe(evaluate(entity))
            if opts.patience and no_improve >= opts.patience:
                break

    else:  # greedy
        current = space.default_entity(default_config(base))
        current_record = evaluate(current)
        observe(current_record)
        current_cost = (
            float(current_record.cost)
            if current_record is not None and current_record.ok
            else float("inf")
        )
        for _ in range(2):
            improved = False
            for index, knob in enumerate(space.knobs):
                if len(knob.values) <= 1:
                    continue
                best_index = current[index]
                best_value_cost = current_cost
                for value_index in range(len(knob.values)):
                    if value_index == current[index]:
                        continue
                    if budget["left"] <= 0:
                        break
                    candidate = list(current)
                    candidate[index] = value_index
                    candidate = tuple(candidate)
                    record = evaluate(candidate)
                    observe(record)
                    if record is not None and record.ok and record.cost < best_value_cost:
                        best_value_cost = float(record.cost)
                        best_index = value_index
                if best_index != current[index]:
                    current = tuple(
                        best_index if position == index else value
                        for position, value in enumerate(current)
                    )
                    current_cost = best_value_cost
                    improved = True
            if not improved or budget["left"] <= 0:
                break

    all_records = log.records
    feasible = [record for record in all_records if record.ok]
    best = min(feasible, key=lambda record: record.sort_key) if feasible else None

    best_build = builds.get(best.entity) if best is not None else None
    if best is not None and best_build is None:
        try:
            best_build = build_module(
                module, space.decode(best.entity), base, entity=best.entity
            )
        except Exception:  # noqa: BLE001 - 历史最优在当前链路不可重建时保留记录
            best_build = None

    return TuneResult(
        space=space,
        strategy=opts.strategy,
        best=best,
        best_build=best_build,
        records=all_records,
        log=log,
    )


__all__ = ["TuneOptions", "TuneResult", "tune"]
