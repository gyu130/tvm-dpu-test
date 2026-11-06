"""NestTile 程序的可读打印。"""

from __future__ import annotations

from .ir import (
    ArithOp,
    AtomicOp,
    BranchOp,
    Compute,
    Copy,
    EndOp,
    LabelOp,
    Loop,
    MemManageOp,
    Program,
    QueueOp,
    Sync,
)


def _print_item(item, indent: str) -> str:
    if isinstance(item, Copy):
        token = f" async={item.async_token}" if item.async_token else ""
        region = ""
        if item.src_region is not None:
            region = f" src_region={item.src_region.to_dict()}"
        if item.dst_region is not None:
            region = f" dst_region={item.dst_region.to_dict()}"
        return f"{indent}dma.copy {item.src} -> {item.dst}{region}{token}"
    if isinstance(item, Compute):
        token = ""
        if item.inputs:
            token += f" ({', '.join(item.inputs)})"
        return (
            f"{indent}compute {item.canonical_name}{token} -> "
            f"({', '.join(item.outputs)})"
        )
    if isinstance(item, Sync):
        if item.kind == "dma_wait":
            return f"{indent}sync.dma_wait {item.token}"
        if item.kind == "instruction_barrier":
            return f"{indent}sync.instruction_barrier  # DPU ISB"
        if item.kind == "core_barrier":
            return f"{indent}sync.core_barrier  # DPU ISBC"
        return f"{indent}sync.barrier"
    if isinstance(item, Loop):
        header = (
            f"{indent}loop {item.var} axis={item.axis} "
            f"extent={item.extent} tile={item.tile}"
        )
        if item.body is None:
            return header
        lines = [header + ":"]
        inner = indent + "  "
        for sub in item.body:
            lines.append(_print_item(sub, inner))
        lines.append(f"{indent}end_loop {item.var}")
        return "\n".join(lines)
    if isinstance(item, ArithOp):
        return f"{indent}{item}"
    if isinstance(item, BranchOp):
        return f"{indent}{item}"
    if isinstance(item, LabelOp):
        return f"{indent}{item}"
    if isinstance(item, AtomicOp):
        return f"{indent}{item}"
    if isinstance(item, QueueOp):
        return f"{indent}{item}"
    if isinstance(item, MemManageOp):
        return f"{indent}{item}"
    if isinstance(item, EndOp):
        return f"{indent}{item}"
    return f"{indent}{item!r}"


def print_program(program: Program) -> str:
    lines = [f"nest.tile program @{program.name}"]
    for buffer in program.global_buffers:
        lines.append(f"  global {buffer}")
    for stage in program.stages:
        lines.append(f"  stage {stage.name} [template={stage.template}]")
        for loop in stage.loops:
            lines.append(
                f"    loop {loop.var} axis={loop.axis} "
                f"extent={loop.extent} tile={loop.tile}"
            )
        for buffer in stage.buffers:
            lines.append(f"    {buffer}")
        if stage.prologue:
            lines.append("    prologue:")
            for item in stage.prologue:
                lines.append(_print_item(item, "      "))
        lines.append("    body:")
        for item in stage.body:
            lines.append(_print_item(item, "      "))
    return "\n".join(lines)


__all__ = ["print_program"]
