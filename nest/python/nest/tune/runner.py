"""测量后端抽象与静态指令级估算。

Runner 的约定：**永远不抛异常**。构建失败由 tuner 捕获；测量失败必须通过
``Measurement.error`` 返回，保证一轮调优不会因为单个配置中断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..tile.autoschedule import CostModel
from ..tile.ir import (
    ArithOp,
    BranchOp,
    Copy,
    LabelOp,
    Loop,
    MemOp,
    Program,
)
from .build import BuildResult

_MUL_OPCODES = frozenset({"MUL", "DIV"})


@dataclass(frozen=True)
class Measurement:
    """一次测量的结果。"""

    cost: float
    error: str | None = None
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and math.isfinite(self.cost)

    def to_dict(self) -> dict:
        return {
            "cost": None if not math.isfinite(self.cost) else round(self.cost, 2),
            "error": self.error,
            "detail": dict(self.detail),
        }


class Runner(Protocol):
    """测量后端。"""

    name: str

    def run(self, build: BuildResult) -> Measurement: ...


def _region_elements(region) -> int:
    count = 1
    for extent in region.extents:
        count *= int(extent)
    return count


def _copy_bytes(copy: Copy, buffer_map) -> int:
    if copy.src_region is not None:
        name, region = copy.src, copy.src_region
    elif copy.dst_region is not None:
        name, region = copy.dst, copy.dst_region
    else:
        name, region = copy.src, None
    buffer = buffer_map.get(name)
    if buffer is None:
        return 0
    if region is not None:
        elements = _region_elements(region)
    else:
        elements = 1
        for dim in buffer.shape:
            elements *= int(dim)
    return elements * np.dtype(buffer.dtype).itemsize


def _stats(items, buffer_map, multiplier: int = 1) -> tuple[int, int, int, int]:
    """返回 (指令数, 乘除指令数, 访存指令数, DMA 字节数)，按循环迭代数放大。"""
    instructions = mul_instructions = mem_ops = dma_bytes = 0
    for item in items:
        if isinstance(item, Loop) and item.body is not None:
            sub = _stats(item.body, buffer_map, multiplier * item.iterations)
            instructions += sub[0]
            mul_instructions += sub[1]
            mem_ops += sub[2]
            dma_bytes += sub[3]
        elif isinstance(item, ArithOp):
            instructions += multiplier
            if item.opcode in _MUL_OPCODES:
                mul_instructions += multiplier
        elif isinstance(item, MemOp):
            instructions += multiplier
            mem_ops += multiplier
        elif isinstance(item, BranchOp):
            instructions += multiplier
        elif isinstance(item, LabelOp):
            continue
        elif isinstance(item, Copy):
            dma_bytes += multiplier * _copy_bytes(item, buffer_map)
    return instructions, mul_instructions, mem_ops, dma_bytes


def _stage_stats(stage, buffer_map) -> tuple[int, int, int, int]:
    """统计一个 stage：prologue 只执行一次，body 按外层 tile 循环迭代数放大。"""
    iterations = 1
    for loop in stage.loops:
        iterations *= loop.iterations
    once = _stats(stage.prologue, buffer_map)
    repeated = _stats(stage.body, buffer_map, iterations)
    return tuple(a + b for a, b in zip(once, repeated))


def program_stats(program: Program) -> dict:
    """统计一个 Program 的指令/DMA 规模。"""
    buffer_map = program.buffer_map()
    instructions = mul = mem_ops = dma = 0
    for stage in program.stages:
        sub = _stage_stats(stage, buffer_map)
        instructions += sub[0]
        mul += sub[1]
        mem_ops += sub[2]
        dma += sub[3]
    return {
        "stages": len(program.stages),
        "instructions": instructions,
        "mul_instructions": mul,
        "mem_ops": mem_ops,
        "dma_bytes": dma,
    }


def _stage_cost(
    stage, cost: CostModel, double_buffer: bool, buffer_map: dict
) -> tuple[float, dict]:
    instructions, mul, mem_ops, dma = _stage_stats(stage, buffer_map)
    compute_cycles = (
        mul / cost.mul_ops_per_cycle
        + (instructions - mul) / cost.alu_ops_per_cycle
    )
    dma_cycles = dma / cost.dma_bytes_per_cycle
    if double_buffer:
        latency = max(compute_cycles, dma_cycles)
    else:
        latency = compute_cycles + dma_cycles
    return latency, {
        "stage": stage.name,
        "instructions": instructions,
        "mul_instructions": mul,
        "mem_ops": mem_ops,
        "dma_bytes": dma,
        "compute_cycles": round(compute_cycles, 2),
        "dma_cycles": round(dma_cycles, 2),
    }


class StaticRunner:
    """基于 lower 后指令流与 DMA 字节的静态周期估算。

    与 AutoSchedule 的 ``CostModel`` 共用硬件参数，但统计口径不同：
    这里从真实 lower 产物出发，循环迭代数会放大每条指令，因此能反映
    tile 变小带来的额外循环开销。
    """

    name = "static"

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.cost_model = cost_model or CostModel()

    def run(self, build: BuildResult) -> Measurement:
        try:
            double = bool(build.double_buffer)
            per_core: list[float] = []
            stages: list[dict] = []
            dma_total = 0
            for program in build.lowered:
                core_cycles = 0.0
                buffer_map = program.buffer_map()
                for stage in program.stages:
                    latency, detail = _stage_cost(
                        stage, self.cost_model, double, buffer_map
                    )
                    core_cycles += latency
                    stages.append(detail)
                    dma_total += detail["dma_bytes"]
                per_core.append(core_cycles)
            cost = max(per_core) if per_core else math.inf
            return Measurement(
                cost=cost,
                detail={
                    "per_core_cycles": [round(item, 2) for item in per_core],
                    "sram_peak": build.sram_peak,
                    "words": build.words,
                    "dma_bytes": dma_total,
                    "stages": stages,
                },
            )
        except Exception as exc:  # noqa: BLE001 - Runner 不能中断调优
            return Measurement(cost=math.inf, error=f"{type(exc).__name__}: {exc}")


__all__ = [
    "Measurement",
    "Runner",
    "StaticRunner",
    "program_stats",
]
