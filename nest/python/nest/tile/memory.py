"""基础 DRAM/SRAM 内存规划。

策略：
- DRAM：所有 global buffer 生命周期覆盖整个程序，按声明顺序做确定性线性分配；
- SRAM：以 stage 为生命周期单位做线性扫描分配；stage 之间复用地址，
  stage 内的 buffer 互不重叠；
- 输出确定性的分配表与峰值报告；SRAM 超限时直接失败并给出修正建议。
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import TileValidationError
from .ir import Buffer, Program, align_up


def _fmt_bytes(size: int) -> str:
    return f"{size} 字节 ({size / 1024 / 1024:.2f} MiB)"


def _free_block(free: list[tuple[int, int]], block: tuple[int, int]) -> list[tuple[int, int]]:
    free.append(block)
    free.sort()
    merged: list[tuple[int, int]] = []
    for start, length in free:
        if length == 0:
            continue
        if merged and merged[-1][0] + merged[-1][1] == start:
            merged[-1] = (merged[-1][0], merged[-1][1] + length)
        else:
            merged.append((start, length))
    return merged


def _take_block(
    free: list[tuple[int, int]], size: int
) -> tuple[int, list[tuple[int, int]]] | None:
    """best-fit：选择能容纳 size 的最小空闲块，平局取地址更低者。"""
    if size == 0:
        return 0, free
    best_index = None
    for index, (start, length) in enumerate(free):
        if length < size:
            continue
        if best_index is None:
            best_index = index
            continue
        best_start, best_length = free[best_index]
        if length < best_length or (length == best_length and start < best_start):
            best_index = index
    if best_index is None:
        return None
    start, length = free.pop(best_index)
    if length > size:
        free.append((start + size, length - size))
        free.sort()
    return start, free


class Allocation:
    """一条 buffer 分配记录。"""

    __slots__ = ("name", "scope", "address", "nbytes", "alignment", "stage", "lifetime")

    def __init__(
        self,
        name: str,
        scope: str,
        address: int,
        nbytes: int,
        alignment: int,
        stage: str | None,
        lifetime: tuple[int, int],
    ) -> None:
        self.name = name
        self.scope = scope
        self.address = address
        self.nbytes = nbytes
        self.alignment = alignment
        self.stage = stage
        self.lifetime = lifetime

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scope": self.scope,
            "address": self.address,
            "nbytes": self.nbytes,
            "alignment": self.alignment,
            "stage": self.stage,
            "lifetime": list(self.lifetime),
        }

    def __repr__(self) -> str:
        return (
            f"Allocation({self.name!r}, {self.scope}, 0x{self.address:x}, "
            f"{self.nbytes}, {self.stage})"
        )


class MemoryPlan:
    """DRAM/SRAM 分配结果。"""

    def __init__(
        self,
        program: Program,
        dram: list[Allocation],
        sram: list[Allocation],
        dram_peak: int,
        sram_peak: int,
        sram_bytes: int,
        peak_stage: str | None,
    ) -> None:
        self.program = program
        self.dram = dram
        self.sram = sram
        self.dram_peak = dram_peak
        self.sram_peak = sram_peak
        self.sram_bytes = sram_bytes
        self.peak_stage = peak_stage

    def to_dict(self) -> dict:
        return {
            "dram_peak": self.dram_peak,
            "sram_peak": self.sram_peak,
            "sram_budget": self.sram_bytes,
            "sram_peak_stage": self.peak_stage,
            "dram": [item.to_dict() for item in self.dram],
            "sram": [item.to_dict() for item in self.sram],
        }

    def summary_text(self) -> str:
        lines = [
            f"DRAM 峰值: {_fmt_bytes(self.dram_peak)}，buffer 数: {len(self.dram)}",
            f"SRAM 峰值: {_fmt_bytes(self.sram_peak)} / 预算 "
            f"{_fmt_bytes(self.sram_bytes)}",
        ]
        if self.peak_stage:
            lines.append(f"SRAM 峰值所在 stage: {self.peak_stage}")
        return "\n".join(lines)


def plan_memory(
    program: Program, *, sram_bytes: int, alignment: int
) -> MemoryPlan:
    if sram_bytes <= 0:
        raise TileValidationError("sram_bytes 必须为正数")

    # ---- DRAM ----
    dram: list[Allocation] = []
    dram_addresses: dict[str, int] = {}
    offset = 0
    for buffer in program.global_buffers:
        address = align_up(offset, max(buffer.alignment, alignment))
        dram.append(
            Allocation(
                buffer.name,
                "global",
                address,
                buffer.nbytes,
                buffer.alignment,
                None,
                (0, len(program.stages)),
            )
        )
        dram_addresses[buffer.name] = address
        offset = address + buffer.nbytes
    dram_peak = offset

    # ---- SRAM ----
    sram: list[Allocation] = []
    sram_addresses: dict[str, int] = {}
    free: list[tuple[int, int]] = []
    high_water = 0
    peak_stage: str | None = None
    peak_size = 0

    for stage_index, stage in enumerate(program.stages):
        stage_allocations: list[Allocation] = []
        for buffer in stage.buffers:
            size = align_up(buffer.nbytes, max(buffer.alignment, alignment))
            taken = _take_block(free, size)
            if taken is None:
                address = align_up(high_water, max(buffer.alignment, alignment))
                high_water = address + size
            else:
                address, free = taken
            allocation = Allocation(
                buffer.name,
                buffer.scope,
                address,
                buffer.nbytes,
                buffer.alignment,
                stage.name,
                (stage_index, stage_index + 1),
            )
            stage_allocations.append(allocation)
            sram.append(allocation)
            sram_addresses[buffer.name] = address
        if high_water > peak_size:
            peak_size = high_water
            peak_stage = stage.name
        for allocation in stage_allocations:
            block_size = align_up(allocation.nbytes, max(allocation.alignment, alignment))
            free = _free_block(free, (allocation.address, block_size))

    if high_water > sram_bytes:
        raise TileValidationError(
            "SRAM 容量超限："
            f"峰值 {_fmt_bytes(high_water)}（stage {peak_stage}），"
            f"预算 {_fmt_bytes(sram_bytes)}。\n"
            f"建议：增大 --sram-bytes 到至少 {high_water}，"
            "或减小 --conv-oc-tile / --gemm-n-tile 以缩小单个 stage 的 buffer"
        )

    # ---- 回填地址 ----
    addressed_globals = [
        replace(buffer, address=dram_addresses[buffer.name])
        for buffer in program.global_buffers
    ]
    addressed_stages = [
        replace(
            stage,
            buffers=tuple(
                replace(buffer, address=sram_addresses[buffer.name])
                for buffer in stage.buffers
            ),
        )
        for stage in program.stages
    ]
    addressed_program = Program(
        name=program.name,
        global_buffers=addressed_globals,
        stages=addressed_stages,
    )

    return MemoryPlan(
        program=addressed_program,
        dram=dram,
        sram=sram,
        dram_peak=dram_peak,
        sram_peak=high_water,
        sram_bytes=sram_bytes,
        peak_stage=peak_stage,
    )


__all__ = ["Allocation", "MemoryPlan", "plan_memory"]
