"""DMA 双缓冲（ping-pong）计划。

M2 在 NestTile IR 层给出可校验的双缓冲方案：
- 只有带 tile 循环、且 body 内存在异步 DMA 装载的 stage 才可双缓冲；
- 每个装载 buffer 派生一个影子 buffer，额外 SRAM = nbytes * (depth-1)；
- 校验 token 配对、影子 buffer 命名与 SRAM 增量。

真正的“按迭代奇偶切换 buffer”的指令序列留给 NestNPU lowering。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import TileValidationError
from .ir import Copy, Program, Sync


@dataclass(frozen=True)
class DoubleBufferEntry:
    stage: str
    buffer: str
    shadow: str
    token: str
    nbytes: int
    depth: int

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "buffer": self.buffer,
            "shadow": self.shadow,
            "token": self.token,
            "nbytes": self.nbytes,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class DoubleBufferPlan:
    entries: tuple[DoubleBufferEntry, ...]
    extra_bytes: int
    depth: int

    def validate(self, program: Program) -> None:
        if self.depth < 2:
            raise TileValidationError(f"双缓冲 depth 必须 >= 2: {self.depth}")
        stages = {stage.name: stage for stage in program.stages}
        all_names = {buffer.name for buffer in program.global_buffers}
        for stage in program.stages:
            all_names.update(buffer.name for buffer in stage.buffers)
        seen_buffers = set()
        for entry in self.entries:
            stage = stages.get(entry.stage)
            if stage is None:
                raise TileValidationError(
                    f"双缓冲计划引用了不存在的 stage: {entry.stage}"
                )
            buffers = {buffer.name: buffer for buffer in stage.buffers}
            if entry.buffer not in buffers:
                raise TileValidationError(
                    f"stage {entry.stage} 没有 buffer {entry.buffer}"
                )
            if entry.shadow in all_names:
                raise TileValidationError(
                    f"双缓冲影子 buffer 名称冲突: {entry.shadow}"
                )
            if (entry.stage, entry.buffer) in seen_buffers:
                raise TileValidationError(
                    f"双缓冲条目重复: {entry.stage}/{entry.buffer}"
                )
            seen_buffers.add((entry.stage, entry.buffer))
            items = tuple(stage.prologue) + tuple(stage.body)
            loads = [
                item for item in items
                if isinstance(item, Copy) and item.async_token == entry.token
            ]
            waits = [
                item for item in items
                if isinstance(item, Sync)
                and item.kind == "dma_wait"
                and item.token == entry.token
            ]
            if not loads:
                raise TileValidationError(
                    f"stage {entry.stage} 的 token {entry.token} 没有 DMA 装载"
                )
            if not waits:
                raise TileValidationError(
                    f"stage {entry.stage} 的 token {entry.token} 没有 dma_wait"
                )
            if entry.nbytes != buffers[entry.buffer].nbytes:
                raise TileValidationError(
                    f"双缓冲条目 {entry.buffer} 的 nbytes 与 buffer 不一致"
                )
        total = sum(entry.nbytes * (entry.depth - 1) for entry in self.entries)
        if total != self.extra_bytes:
            raise TileValidationError(
                f"双缓冲额外 SRAM 汇总不一致: {total} vs {self.extra_bytes}"
            )
        if not self.entries and self.extra_bytes != 0:
            raise TileValidationError("空双缓冲计划不应有额外 SRAM")

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "extra_bytes": self.extra_bytes,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def summary_text(self) -> str:
        if not self.entries:
            return "DMA 双缓冲: 无符合条件的 stage"
        lines = [
            f"DMA 双缓冲 depth={self.depth}，"
            f"额外 SRAM {self.extra_bytes} 字节，条目 {len(self.entries)}"
        ]
        for entry in self.entries:
            lines.append(
                f"  {entry.stage}: {entry.buffer} <-> {entry.shadow} "
                f"({entry.nbytes} 字节, token={entry.token})"
            )
        return "\n".join(lines)


def plan_double_buffer(program: Program, *, depth: int = 2) -> DoubleBufferPlan:
    """为带循环的异步 DMA 装载生成确定性双缓冲计划。"""
    if depth < 2:
        raise TileValidationError(f"双缓冲 depth 必须 >= 2: {depth}")
    existing = {buffer.name for buffer in program.global_buffers}
    for stage in program.stages:
        existing.update(buffer.name for buffer in stage.buffers)
    entries = []
    for stage in program.stages:
        if not stage.loops:
            continue
        buffers = {buffer.name: buffer for buffer in stage.buffers}
        seen = set()
        index = 0
        for item in stage.body:
            if not isinstance(item, Copy) or item.async_token is None:
                continue
            buffer = buffers.get(item.dst)
            if buffer is None or buffer.scope == "global":
                continue
            if buffer.name in seen:
                continue
            seen.add(buffer.name)
            shadow = f"{buffer.name}.db{index}"
            while shadow in existing:
                index += 1
                shadow = f"{buffer.name}.db{index}"
            index += 1
            entries.append(
                DoubleBufferEntry(
                    stage=stage.name,
                    buffer=buffer.name,
                    shadow=shadow,
                    token=item.async_token,
                    nbytes=buffer.nbytes,
                    depth=depth,
                )
            )
            existing.add(shadow)
    extra = sum(entry.nbytes * (depth - 1) for entry in entries)
    plan = DoubleBufferPlan(entries=tuple(entries), extra_bytes=extra, depth=depth)
    plan.validate(program)
    return plan


__all__ = ["DoubleBufferEntry", "DoubleBufferPlan", "plan_double_buffer"]
