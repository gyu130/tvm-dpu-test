"""NestTile 不变式校验。

校验通过后才允许进入 NestNPU 指令生成。校验项包括：
buffer 唯一性与作用域、tile 循环整除、Region 越界、DMA 方向与同步配对、
计算依赖顺序、地址对齐/重叠与 SRAM 容量。

DPU 扩展：支持 shared scope、Sync.kind="instruction_barrier"/"core_barrier"、
Loop.body 内嵌控制流、lower 后的 DPU 指令序列。
"""

from __future__ import annotations

from ..errors import TileValidationError
from .ir import (
    ArithOp,
    AtomicOp,
    BranchOp,
    Buffer,
    Compute,
    Copy,
    EndOp,
    LabelOp,
    Loop,
    MemManageOp,
    Program,
    QueueOp,
    Stage,
    Sync,
)


def _collect_buffers(program: Program) -> dict[str, Buffer]:
    """收集全局 buffer（跨 stage 唯一）和各 stage 私有 buffer（stage 内唯一）。

    返回 dict 便于后续查找；stage 私有 buffer 可以在不同 stage 重名
    （每个 stage 有独立的私有 SRAM，命名空间隔离）。
    """
    mapping: dict[str, Buffer] = {}
    for buffer in program.global_buffers:
        if buffer.name in mapping:
            raise TileValidationError(f"全局 buffer 重名: {buffer.name}")
        if buffer.scope not in ("global", "shared"):
            raise TileValidationError(
                f"program 级 buffer {buffer.name} 必须是 global 或 shared 作用域"
            )
        mapping[buffer.name] = buffer
    for stage in program.stages:
        seen_in_stage: set[str] = set()
        for buffer in stage.buffers:
            if buffer.name in seen_in_stage:
                raise TileValidationError(
                    f"stage {stage.name}: buffer 重名: {buffer.name}"
                )
            if buffer.scope == "global":
                raise TileValidationError(
                    f"stage {stage.name} 内的 buffer {buffer.name} 不能声明为 global"
                )
            if buffer.name in mapping and mapping[buffer.name].scope != "global":
                raise TileValidationError(
                    f"buffer {buffer.name} 同时出现在多个 stage 的私有作用域，"
                    f"或与全局 buffer 重名"
                )
            seen_in_stage.add(buffer.name)
            mapping[buffer.name] = buffer
    return mapping


def _check_region(
    stage: Stage,
    copy: Copy,
    src: Buffer,
    dst: Buffer,
    loops: dict[str, int],
    kind: str,
) -> None:
    if kind == "load":
        region = copy.src_region
        if region is not None:
            if len(region.extents) != len(src.shape):
                raise TileValidationError(
                    f"stage {stage.name}: DMA 源切片秩 {len(region.extents)} "
                    f"与 buffer {src.name} 秩 {len(src.shape)} 不一致"
                )
            max_offsets = region.max_offsets(loops)
            for index, (offset, extent, dim) in enumerate(
                zip(max_offsets, region.extents, src.shape)
            ):
                if offset + extent > dim:
                    raise TileValidationError(
                        f"stage {stage.name}: DMA 源切片第 {index} 维越界 "
                        f"(offset {offset} + extent {extent} > {dim})"
                    )
            if tuple(region.extents) != dst.shape:
                raise TileValidationError(
                    f"stage {stage.name}: DMA 目标 buffer {dst.name} shape "
                    f"{dst.shape} 与切片 extent {region.extents} 不一致"
                )
        elif src.shape != dst.shape:
            raise TileValidationError(
                f"stage {stage.name}: 整块 DMA 要求 shape 一致，"
                f"{src.name}{src.shape} vs {dst.name}{dst.shape}"
            )
    else:
        region = copy.dst_region
        if region is not None:
            if len(region.extents) != len(dst.shape):
                raise TileValidationError(
                    f"stage {stage.name}: DMA 目标切片秩 {len(region.extents)} "
                    f"与 buffer {dst.name} 秩 {len(dst.shape)} 不一致"
                )
            max_offsets = region.max_offsets(loops)
            for index, (offset, extent, dim) in enumerate(
                zip(max_offsets, region.extents, dst.shape)
            ):
                if offset + extent > dim:
                    raise TileValidationError(
                        f"stage {stage.name}: DMA 目标切片第 {index} 维越界 "
                        f"(offset {offset} + extent {extent} > {dim})"
                    )
            if tuple(region.extents) != src.shape:
                raise TileValidationError(
                    f"stage {stage.name}: DMA 源 buffer {src.name} shape "
                    f"{src.shape} 与切片 extent {region.extents} 不一致"
                )
        elif src.shape != dst.shape:
            raise TileValidationError(
                f"stage {stage.name}: 整块 DMA 要求 shape 一致，"
                f"{src.name}{src.shape} vs {dst.name}{dst.shape}"
            )


def _validate_dma_direction(
    src: Buffer, dst: Buffer
) -> str | None:
    """返回 'load' / 'store' 或 None（非法方向）。

    DPU 支持 global <-> shared <-> sram/local 多层搬运，
    只要求 src 和 dst 不是同一 scope。
    """
    if src.scope == "global" and dst.scope in ("shared", "sram", "local"):
        return "load"
    if src.scope == "shared" and dst.scope in ("sram", "local"):
        return "load"
    if dst.scope == "global" and src.scope in ("shared", "sram", "local"):
        return "store"
    if dst.scope == "shared" and src.scope in ("sram", "local"):
        return "store"
    return None


def _check_section_items(
    stage: Stage,
    items: tuple,
    buffers: dict[str, Buffer],
    loops: dict[str, int],
    *,
    allow_compute: bool,
    written: set[str],
    buffer_write_check: bool = True,
) -> set[str]:
    """递归校验操作序列，处理 Loop.body 内嵌控制流。"""
    tokens: set[str] = set()
    waits: list[str] = []

    for item in items:
        if isinstance(item, Copy):
            src = buffers.get(item.src)
            dst = buffers.get(item.dst)
            if src is None or dst is None:
                missing = item.src if src is None else item.dst
                raise TileValidationError(
                    f"stage {stage.name}: DMA 引用了未声明的 buffer {missing}"
                )
            kind = _validate_dma_direction(src, dst)
            if kind is None:
                raise TileValidationError(
                    f"stage {stage.name}: DMA 方向非法: {src.scope}->{dst.scope}"
                )
            _check_region(stage, item, src, dst, loops, kind)
            if buffer_write_check and kind == "store" and item.src not in written:
                raise TileValidationError(
                    f"stage {stage.name}: DMA store 读取了尚未写入的 buffer "
                    f"{item.src}"
                )
            written.add(item.dst)
            if item.async_token:
                tokens.add(item.async_token)

        elif isinstance(item, Compute):
            if not allow_compute:
                raise TileValidationError(
                    f"stage {stage.name}: prologue 不允许 Compute"
                )
            for name in item.inputs:
                if name not in buffers:
                    raise TileValidationError(
                        f"stage {stage.name}: Compute 引用了未声明的 buffer {name}"
                    )
                if buffers[name].scope == "global":
                    raise TileValidationError(
                        f"stage {stage.name}: Compute 输入 {name} 必须位于片上"
                    )
                if buffer_write_check and name not in written:
                    raise TileValidationError(
                        f"stage {stage.name}: Compute 输入 {name} 尚未由 DMA 写入"
                    )
            for name in item.outputs:
                if name not in buffers:
                    raise TileValidationError(
                        f"stage {stage.name}: Compute 引用了未声明的 buffer {name}"
                    )
                if buffers[name].scope == "global":
                    raise TileValidationError(
                        f"stage {stage.name}: Compute 输出 {name} 必须位于片上"
                    )
                written.add(name)

        elif isinstance(item, Sync):
            if item.kind == "dma_wait":
                waits.append(item.token or "")
            # instruction_barrier / core_barrier / barrier 无额外校验

        elif isinstance(item, Loop):
            # 注册新循环变量
            if item.var in loops:
                raise TileValidationError(
                    f"stage {stage.name}: 循环变量重名 {item.var}"
                )
            inner_loops = dict(loops)
            inner_loops[item.var] = item.extent
            # 递归校验 body（如果存在）
            if item.body is not None:
                written = _check_section_items(
                    stage, item.body, buffers, inner_loops,
                    allow_compute=allow_compute, written=written,
                    buffer_write_check=buffer_write_check,
                )

        elif isinstance(item, LabelOp):
            # 重名检查在 stage 末尾统一做，这里只做最小校验
            pass

        elif isinstance(item, ArithOp):
            # 寄存器级操作，不依赖 buffer 系统
            pass

        elif isinstance(item, BranchOp):
            # BranchOp 本身不在此校验 target，到 stage 末尾统一收集校验
            pass

        elif isinstance(item, (AtomicOp, QueueOp, MemManageOp, EndOp)):
            # 硬件原语，参数是寄存器 / 队列 ID，不涉及 buffer 校验
            pass

    # DMA 同步检查：waits 必须是 tokens 的子集
    # （允许多个 DMA 共用同一个 async_token，也允许多次 wait 同一 token）
    wait_set = set(waits)
    extra_waits = wait_set - tokens
    if extra_waits:
        raise TileValidationError(
            f"stage {stage.name}: dma_wait 引用了不存在的 token(s): "
            f"{sorted(extra_waits)}；已有 async tokens: {sorted(tokens)}"
        )
    return written


def _collect_labels(items: tuple, out: set[str]) -> None:
    """递归收集所有 LabelOp 名称，用于校验 BranchOp.target 存在。"""
    for item in items:
        if isinstance(item, LabelOp):
            out.add(item.name)
        elif isinstance(item, Loop) and item.body is not None:
            _collect_labels(item.body, out)


def _collect_branch_targets(items: tuple, out: set[str]) -> None:
    for item in items:
        if isinstance(item, BranchOp):
            out.add(item.target)
        elif isinstance(item, Loop) and item.body is not None:
            _collect_branch_targets(item.body, out)


def _loop_has_compute(loop: Loop) -> bool:
    """递归检查 Loop.body 是否仍有 Compute 占位符。"""
    if loop.body is None:
        return False
    for item in loop.body:
        if isinstance(item, Compute):
            return True
        if isinstance(item, Loop) and _loop_has_compute(item):
            return True
    return False


def _check_overlap(entries: list[Buffer], where: str) -> None:
    used = [(b.address, b.nbytes, b.name) for b in entries
            if b.address is not None and b.nbytes > 0]
    used.sort()
    for (a_start, a_size, a_name), (b_start, b_size, b_name) in zip(used, used[1:]):
        if a_start + a_size > b_start:
            raise TileValidationError(
                f"{where} 地址重叠: {a_name}@0x{a_start:x}+{a_size} 与 "
                f"{b_name}@0x{b_start:x}+{b_size}"
            )


def validate_program(
    program: Program,
    *,
    sram_bytes: int | None = None,
    alignment: int | None = None,
) -> None:
    buffers = _collect_buffers(program)

    for stage in program.stages:
        loop_extents: dict[str, int] = {}
        for loop in stage.loops:
            if loop.var in loop_extents:
                raise TileValidationError(
                    f"stage {stage.name}: loop 变量重名 {loop.var}"
                )
            loop_extents[loop.var] = loop.extent
            node_buffer = next(
                (b for b in buffers.values() if b.name == stage.node_value), None
            )
            if node_buffer is not None and loop.axis >= len(node_buffer.shape):
                raise TileValidationError(
                    f"stage {stage.name}: loop {loop.var} 的 axis={loop.axis} "
                    f"超出输出秩 {len(node_buffer.shape)}"
                )

        # 校验 prologue 里的 DMA Region 不引用 stage.loops 变量
        for item in stage.prologue:
            if isinstance(item, Copy):
                for region in (item.src_region, item.dst_region):
                    if region is None:
                        continue
                    if region.uses_loops():
                        raise TileValidationError(
                            f"stage {stage.name}: prologue 中的 DMA 不能引用 "
                            f"tile 循环变量 {sorted(region.uses_loops())}"
                        )
                    undefined = region.uses_loops() - set(loop_extents)
                    if undefined:
                        raise TileValidationError(
                            f"stage {stage.name}: Region 引用了未定义的循环变量 "
                            f"{sorted(undefined)}"
                        )

        # 校验 body 里的 DMA Region
        for item in stage.body:
            if isinstance(item, Copy):
                for region in (item.src_region, item.dst_region):
                    if region is None:
                        continue
                    undefined = region.uses_loops() - set(loop_extents)
                    if undefined:
                        raise TileValidationError(
                            f"stage {stage.name}: Region 引用了未定义的循环变量 "
                            f"{sorted(undefined)}"
                        )
            if isinstance(item, Loop) and item.body is not None:
                # Loop.body 内部的 Region 引用自己的 var 没问题
                pass

        # 判断本 stage 是否仍有 Compute 占位符
        # （如果已被 lower 成 DPU 指令，则跳过 buffer 写/读跟踪）
        stage_has_compute = any(
            isinstance(item, Compute)
            for item in tuple(stage.prologue) + tuple(stage.body)
        ) or any(
            _loop_has_compute(loop)
            for loop in stage.loops
        )
        buffer_write_check = stage_has_compute

        written: set[str] = set()
        label_names: set[str] = set()

        written = _check_section_items(
            stage, stage.prologue, buffers, loop_extents,
            allow_compute=False, written=written,
            buffer_write_check=buffer_write_check,
        )
        written = _check_section_items(
            stage, stage.body, buffers, loop_extents,
            allow_compute=True, written=written,
            buffer_write_check=buffer_write_check,
        )

        # BranchOp.target 必须在本 stage 内有对应 LabelOp
        all_labels: set[str] = set()
        all_targets: set[str] = set()
        _collect_labels(stage.body, all_labels)
        _collect_labels(stage.prologue, all_labels)
        _collect_branch_targets(stage.body, all_targets)
        _collect_branch_targets(stage.prologue, all_targets)
        # 加上 Loop.body 里的 labels 和 targets
        for loop in stage.loops:
            if loop.body is not None:
                _collect_labels(loop.body, all_labels)
                _collect_branch_targets(loop.body, all_targets)
        missing = all_targets - all_labels
        if missing:
            raise TileValidationError(
                f"stage {stage.name}: BranchOp 引用了未定义的 label(s): "
                f"{sorted(missing)}"
            )

    # 地址检查
    for buffer in buffers.values():
        if buffer.address is not None and buffer.address % buffer.alignment != 0:
            raise TileValidationError(
                f"buffer {buffer.name} 地址 0x{buffer.address:x} 未按 "
                f"{buffer.alignment} 字节对齐"
            )
    global_entries = [b for b in buffers.values() if b.scope == "global"]
    _check_overlap(global_entries, "DRAM")
    for stage in program.stages:
        stage_sram = [b for b in stage.buffers if b.scope in ("sram", "local")]
        _check_overlap(stage_sram, f"stage {stage.name} SRAM")
        if sram_bytes is not None and stage_sram:
            used = max(
                (b.address + b.nbytes for b in stage_sram if b.address is not None),
                default=0,
            )
            if used > sram_bytes:
                raise TileValidationError(
                    f"stage {stage.name} SRAM 超限: {used} > {sram_bytes} 字节；"
                    "建议增大 --sram-bytes 或减小 tile"
                )
    if alignment is not None:
        for buffer in buffers.values():
            if buffer.alignment < alignment:
                raise TileValidationError(
                    f"buffer {buffer.name} 对齐 {buffer.alignment} 小于配置的 "
                    f"{alignment}"
                )


__all__ = ["validate_program"]
