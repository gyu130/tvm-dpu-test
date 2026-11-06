"""Ansor-lite：sketch 枚举 + annotation 进化搜索 + 学习型代价模型。

流程（对应 Ansor 论文四步）：

1. ``generate_sketches`` 枚举结构（模板序列 / 融合 / 多核 / 双缓冲）；
2. 每个 sketch 的 annotation = tile 因子下标向量；
3. 进化搜索：用学习型模型给种群排序，只实测 top-k；
4. 模型用实测数据增量训练，最终结论仍以实测值为准。

在 nest 中，annotation 只覆盖调度模板真正支持的轴（conv 输出通道、
gemm 输出 N 维），多级空间 tiling 需要先扩展 ``schedule.py`` 模板。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ..errors import TuneError
from ..ir import Module
from ..tile.schedule import ScheduleOptions
from .build import BuildResult
from .learn_cost import LearnedCostModel, extract_features
from .log import TrialRecord, TuneLog, make_record
from .runner import Runner, StaticRunner
from .sketch import Sketch, generate_sketches

_MAX_ATTEMPTS_PER_GENERATION = 10


@dataclass(frozen=True)
class AnsorOptions:
    """Ansor 搜索参数。"""

    population: int = 32
    generations: int = 8
    top_k: int = 4
    mutate_prob: float = 0.3
    crossover_prob: float = 0.5
    seed: int = 0
    max_trials: int = 64
    min_train: int = 4
    log_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.population <= 0:
            raise TuneError("population 必须为正数")
        if self.generations <= 0:
            raise TuneError("generations 必须为正数")
        if self.top_k <= 0:
            raise TuneError("top_k 必须为正数")
        if not 0.0 <= self.mutate_prob <= 1.0:
            raise TuneError("mutate_prob 必须在 [0, 1]")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise TuneError("crossover_prob 必须在 [0, 1]")
        if self.max_trials <= 0:
            raise TuneError("max_trials 必须为正数")
        if self.min_train < 2:
            raise TuneError("min_train 至少为 2")


@dataclass
class SketchResult:
    sketch: Sketch
    best: TrialRecord | None
    best_build: BuildResult | None
    records: list[TrialRecord]

    def summary_text(self) -> str:
        if self.best is None:
            return f"{self.sketch.name}: 无可行测量"
        return (
            f"{self.sketch.name}: entity={self.best.entity} "
            f"cost={self.best.cost:.0f} config={self.best.config}"
        )


@dataclass
class AnsorResult:
    sketches: list[SketchResult]
    best_sketch: Sketch | None
    best: TrialRecord | None
    best_build: BuildResult | None
    records: list[TrialRecord]
    log: TuneLog
    model: LearnedCostModel

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
            f"Ansor: {len(self.sketches)} sketches, {len(self.records)} trials, "
            f"{len(feasible)} feasible",
            self.model.summary_text(),
        ]
        if self.best is not None and self.best_sketch is not None:
            lines.append(
                f"BEST sketch={self.best_sketch.name} entity={self.best.entity} "
                f"cost={self.best.cost:.0f} config={self.best.config}"
            )
        else:
            lines.append("BEST 无可行配置")
        lines.extend(item.summary_text() for item in self.sketches)
        lines.extend(
            f"  {record.config} -> {record.cost:.0f}" for record in feasible[:limit]
        )
        return "\n".join(lines)


def _mutate(state: tuple[int, ...], space, rng: random.Random, probability: float):
    genes = list(state)
    for index, knob in enumerate(space.knobs):
        if len(knob.values) > 1 and rng.random() < probability:
            genes[index] = rng.randrange(len(knob.values))
    return tuple(genes)


def _next_population(
    elites: list[tuple[int, ...]],
    population_size: int,
    space,
    rng: random.Random,
    options: AnsorOptions,
) -> list[tuple[int, ...]]:
    population = list(elites)
    seen = set(population)
    attempts = 0
    limit = population_size * _MAX_ATTEMPTS_PER_GENERATION
    while len(population) < population_size and attempts < limit:
        attempts += 1
        if len(elites) >= 2 and rng.random() < options.crossover_prob:
            first, second = rng.sample(elites, 2)
            child = tuple(
                first[index] if rng.random() < 0.5 else second[index]
                for index in range(len(first))
            )
        else:
            child = _mutate(rng.choice(elites), space, rng, options.mutate_prob)
        if child in seen:
            continue
        seen.add(child)
        population.append(child)
    return population


def ansor_tune(
    module: Module,
    *,
    sketches: list[Sketch] | None = None,
    runner: Runner | None = None,
    base: ScheduleOptions | None = None,
    options: AnsorOptions | None = None,
) -> AnsorResult:
    """执行 Ansor-lite 搜索。"""
    base = base or ScheduleOptions()
    options = options or AnsorOptions()
    runner = runner or StaticRunner()
    sketches = sketches or generate_sketches(module, base)
    if not sketches:
        raise TuneError("Ansor 没有可用 sketch")

    rng = random.Random(options.seed)
    log = TuneLog(options.log_path)
    model = LearnedCostModel()
    training_x: list[list[float]] = []
    training_y: list[float] = []
    records: list[TrialRecord] = []
    measured_all: list[tuple[TrialRecord, Sketch, BuildResult | None]] = []
    budget = {"left": options.max_trials}
    results: list[SketchResult] = []

    for sketch in sketches:
        prepared = sketch.prepare(module)
        space = sketch.space
        cache: dict[tuple[int, ...], tuple[BuildResult | None, str | None]] = {}
        measured: dict[tuple[int, ...], TrialRecord] = {}

        def get_build(entity: tuple[int, ...]):
            if entity not in cache:
                try:
                    cache[entity] = (
                        sketch.build(prepared, entity, base),
                        None,
                    )
                except Exception as exc:  # noqa: BLE001 - 结构不可行直接剪枝
                    cache[entity] = (None, f"{type(exc).__name__}: {exc}")
            return cache[entity]

        def measure(entity: tuple[int, ...]) -> TrialRecord | None:
            if entity in measured:
                return measured[entity]
            if budget["left"] <= 0:
                return None
            build, error = get_build(entity)
            config = sketch.config(entity)
            if build is None:
                record = make_record(
                    entity,
                    config,
                    None,
                    error=error or "build failed",
                    runner=runner.name,
                    detail={"sketch": sketch.name},
                )
            else:
                budget["left"] -= 1
                measurement = runner.run(build)
                detail = dict(measurement.detail)
                detail["sketch"] = sketch.name
                record = make_record(
                    entity,
                    config,
                    measurement.cost,
                    error=measurement.error,
                    detail=detail,
                    runner=runner.name,
                )
                if record.ok:
                    training_x.append(extract_features(build))
                    training_y.append(float(record.cost))
            log.append(record)
            records.append(record)
            measured[entity] = record
            measured_all.append((record, sketch, build))
            return record

        if space.size <= options.population:
            population = space.all_entities()
        else:
            population = space.sample_distinct(rng, options.population)

        for _ in range(options.generations):
            if budget["left"] <= 0:
                break
            scored: list[tuple[float, tuple[int, ...]]] = []
            for state in population:
                build, _error = get_build(state)
                if build is None:
                    continue
                if model.ready and model.samples >= options.min_train:
                    score = model.predict(extract_features(build))
                else:
                    score = rng.random()
                scored.append((score, state))
            if not scored:
                break
            scored.sort(key=lambda item: (item[0], item[1]))

            measured_now = 0
            for _score, state in scored:
                if measured_now >= options.top_k or budget["left"] <= 0:
                    break
                if state in measured:
                    continue
                measure(state)
                measured_now += 1

            if len(training_x) >= options.min_train:
                model.fit(training_x, training_y)

            if space.size <= 1:
                break
            elite_size = max(2, options.population // 4)
            elites = [state for _score, state in scored[:elite_size]]
            population = _next_population(
                elites, options.population, space, rng, options
            )

        sketch_records = list(measured.values())
        feasible = [record for record in sketch_records if record.ok]
        best = min(feasible, key=lambda record: record.sort_key) if feasible else None
        best_build = cache[best.entity][0] if best is not None else None
        results.append(
            SketchResult(
                sketch=sketch,
                best=best,
                best_build=best_build,
                records=sketch_records,
            )
        )

    feasible_all = [record for record in records if record.ok]
    if not feasible_all:
        raise TuneError(
            f"Ansor 未找到可行配置：{len(records)} 次尝试，"
            f"{len(results)} 个 sketch"
        )
    best_record, best_sketch, best_build = min(
        (item for item in measured_all if item[0].ok),
        key=lambda item: item[0].sort_key,
    )
    return AnsorResult(
        sketches=results,
        best_sketch=best_sketch,
        best=best_record,
        best_build=best_build,
        records=records,
        log=log,
        model=model,
    )


__all__ = ["AnsorOptions", "AnsorResult", "SketchResult", "ansor_tune"]
