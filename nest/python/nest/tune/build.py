"""调优用的构建链路封装：config -> 可测量产物。

把已有的 ``schedule_module -> validate_program -> plan_memory ->
plan_double_buffer -> lower_program -> encode_items`` 串成一个对象，
供各类 Runner 消费。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import TuneError, TileValidationError
from ..ir import Module
from ..tile.encoder import EncoderResult, encode_items
from ..tile.ir import Program
from ..tile.lower import lower_program
from ..tile.memory import MemoryPlan, plan_memory
from ..tile.multicore import MultiCoreOptions, schedule_multicore
from ..tile.pipeline import DoubleBufferPlan, plan_double_buffer
from ..tile.schedule import ScheduleOptions, schedule_module


@dataclass(frozen=True)
class BuildResult:
    """一次配置的完整构建产物。"""

    config: dict
    entity: tuple[int, ...]
    cores: int
    programs: tuple[Program, ...]
    lowered: tuple[Program, ...]
    plans: tuple[MemoryPlan, ...]
    double_buffer: tuple[DoubleBufferPlan, ...]
    encoded: tuple[EncoderResult, ...]
    sram_peak: int

    @property
    def words(self) -> int:
        return sum(len(item) for item in self.encoded)

    def summary_text(self) -> str:
        lines = [
            f"BuildResult: cores={self.cores} words={self.words} "
            f"sram_peak={self.sram_peak}B",
            f"  config={self.config}",
        ]
        for plan in self.plans:
            lines.append(
                f"  stages={len(plan.program.stages)} dram={plan.dram_peak}B "
                f"sram={plan.sram_peak}B"
            )
        return "\n".join(lines)


def _flatten_stage_items(program: Program) -> list:
    items: list = []
    for stage in program.stages:
        items.extend(stage.prologue)
        items.extend(stage.body)
    return items


def _single_core(
    module: Module,
    options: ScheduleOptions,
    db_depth: int,
) -> tuple[tuple[Program, ...], tuple[MemoryPlan, ...], tuple[DoubleBufferPlan, ...]]:
    program = schedule_module(module, options=options)
    plan = plan_memory(
        program, sram_bytes=options.sram_bytes, alignment=options.alignment
    )
    double = plan_double_buffer(plan.program) if db_depth >= 2 else None
    return (plan.program,), (plan,), ((double,) if double is not None else ())


def _multi_core(
    module: Module,
    options: ScheduleOptions,
    cores: int,
    split_dim: str,
    db_depth: int,
) -> tuple[tuple[Program, ...], tuple[MemoryPlan, ...], tuple[DoubleBufferPlan, ...]]:
    multi = schedule_multicore(
        module,
        MultiCoreOptions(
            cores=cores,
            split_dim=split_dim,
            schedule_options=options,
            double_buffer=db_depth >= 2,
        ),
    )
    multi.validate()
    return tuple(multi.programs), tuple(multi.memory), tuple(multi.double_buffer)


def build_module(
    module: Module,
    config: dict,
    base: ScheduleOptions | None = None,
    *,
    entity: tuple[int, ...] = (),
) -> BuildResult:
    """按配置构建可测量产物；任何不可行配置都以异常形式抛出。"""
    base = base or ScheduleOptions()
    cores = int(config.get("cores", 1))
    split_dim = str(config.get("split_dim", "auto"))
    db_depth = int(config.get("db_depth", 1))
    if cores <= 0:
        raise TuneError(f"cores 必须为正数: {cores}")
    if db_depth <= 0:
        raise TuneError(f"db_depth 必须为正数: {db_depth}")

    options = ScheduleOptions(
        sram_bytes=base.sram_bytes,
        alignment=base.alignment,
        conv_oc_tile=int(config.get("conv_oc_tile", base.conv_oc_tile)),
        gemm_n_tile=int(config.get("gemm_n_tile", base.gemm_n_tile)),
        bind_core=base.bind_core,
    )

    if cores == 1:
        programs, plans, doubles = _single_core(module, options, db_depth)
    else:
        programs, plans, doubles = _multi_core(
            module, options, cores, split_dim, db_depth
        )

    peaks = []
    for index, plan in enumerate(plans):
        extra = 0
        if doubles and index < len(doubles):
            extra = doubles[index].extra_bytes
        peaks.append(plan.sram_peak + extra)
    sram_peak = max(peaks) if peaks else 0
    if sram_peak > options.sram_bytes:
        raise TileValidationError(
            f"SRAM 峰值 {sram_peak}B（含双缓冲影子 buffer）超出预算 "
            f"{options.sram_bytes}B"
        )

    lowered = tuple(lower_program(program) for program in programs)
    encoded = tuple(encode_items(_flatten_stage_items(program)) for program in lowered)

    return BuildResult(
        config=dict(config),
        entity=tuple(entity),
        cores=cores,
        programs=programs,
        lowered=lowered,
        plans=plans,
        double_buffer=doubles,
        encoded=encoded,
        sram_peak=sram_peak,
    )


__all__ = ["BuildResult", "build_module"]
