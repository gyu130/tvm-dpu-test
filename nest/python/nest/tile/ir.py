"""NestTile 调度 IR。

一个 Program 由顺序执行的 Stage 组成；每个 Stage 对应 NestGraph 中的一个
计算节点，用显式 buffer、tile 循环、DMA Copy、Compute 与 Sync 表达。

设计约束：
- buffer 必须显式声明作用域（global / shared / sram / local / register）、
  形状、layout、对齐；
- DMA 的切片使用 RegionSpec 表达，tile 变量通过 TileRef 引用；
- 指令生成前必须先通过 tile.validate 的校验。

DPU 适配（方案 A）：
- 高层 Compute 先占位，后续 ``lower.py`` 中的 DPU 指令集 pass 把它展开
  成 ArithOp / BranchOp / AtomicOp / QueueOp / LabelOp 序列；
- 新增 Loop.body，允许表达 Compute 内部的多层 tile 循环；
- 新增 Sync.kind="instruction_barrier"（DPU ISB）和 "core_barrier"（ISBC）；
- Buffer.scope 新增 "shared" 对应 MALCSMB 管理的核间共享内存。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import TileValidationError
from ..ir import SourceSpan

SCOPES = ("global", "shared", "sram", "local", "register")


def _prod(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise TileValidationError(f"alignment 必须为正数: {alignment}")
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Buffer:
    """显式声明的 buffer。"""

    name: str
    scope: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    alignment: int = 64
    address: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise TileValidationError("Buffer.name 不能为空")
        if self.scope not in SCOPES:
            raise TileValidationError(f"非法 buffer 作用域: {self.scope}")
        if not self.dtype:
            raise TileValidationError(f"buffer {self.name} 缺少 dtype")
        if self.layout is None:
            raise TileValidationError(f"buffer {self.name} 缺少 layout")
        for dim in self.shape:
            if int(dim) < 0:
                raise TileValidationError(
                    f"buffer {self.name} 存在负数维度: {self.shape}"
                )
        if self.alignment <= 0:
            raise TileValidationError(
                f"buffer {self.name} 的 alignment 必须为正数"
            )
        if self.address is not None:
            if self.address < 0:
                raise TileValidationError(f"buffer {self.name} 地址不能为负")
            if self.address % self.alignment != 0:
                raise TileValidationError(
                    f"buffer {self.name} 地址 0x{self.address:x} 未按 "
                    f"{self.alignment} 字节对齐"
                )

    @property
    def nbytes(self) -> int:
        return _prod(self.shape) * np.dtype(self.dtype).itemsize

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scope": self.scope,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "alignment": self.alignment,
            "address": self.address,
            "nbytes": self.nbytes,
        }

    def __str__(self) -> str:
        where = f"@0x{self.address:x}" if self.address is not None else "@unallocated"
        return (
            f"%{self.name}: {self.scope} {self.dtype}"
            f"[{','.join(str(d) for d in self.shape)}]({self.layout}) {where}"
        )


@dataclass(frozen=True)
class TileRef:
    """tile 循环变量的一次引用：offset = base + loop_index * tile。

    ``base`` 用于多核切分：每个核的循环从全局区间的 ``core * chunk`` 开始，
    但循环变量本身仍从 0 迭代，因此需要显式的基址。
    """

    loop: str
    tile: int
    base: int = 0

    def __post_init__(self) -> None:
        if not self.loop:
            raise TileValidationError("TileRef.loop 不能为空")
        if self.tile <= 0:
            raise TileValidationError(f"TileRef.tile 必须为正数: {self.tile}")
        if self.base < 0:
            raise TileValidationError(f"TileRef.base 不能为负: {self.base}")

    def evaluate(self, env: dict[str, int]) -> int:
        if self.loop not in env:
            raise TileValidationError(
                f"TileRef 引用了未定义的循环变量: {self.loop}"
            )
        return self.base + env[self.loop] * self.tile

    def to_dict(self) -> dict:
        data = {"loop": self.loop, "tile": self.tile}
        if self.base:
            data["base"] = self.base
        return data


@dataclass(frozen=True)
class RegionSpec:
    """buffer 的矩形切片（元素偏移 + extent）。"""

    offsets: tuple
    extents: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.extents):
            raise TileValidationError(
                f"RegionSpec 秩不一致: offsets={self.offsets}, extents={self.extents}"
            )
        for extent in self.extents:
            if int(extent) <= 0:
                raise TileValidationError(f"RegionSpec extent 必须为正数: {extent}")
        for offset in self.offsets:
            if isinstance(offset, TileRef):
                continue
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise TileValidationError(f"非法 RegionSpec offset: {offset!r}")

    def evaluate(self, env: dict[str, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        offsets = tuple(
            offset.evaluate(env) if isinstance(offset, TileRef) else offset
            for offset in self.offsets
        )
        return offsets, tuple(int(e) for e in self.extents)

    def max_offsets(self, loop_extents: dict[str, int]) -> tuple[int, ...]:
        offsets = []
        for offset in self.offsets:
            if isinstance(offset, TileRef):
                extent = loop_extents.get(offset.loop)
                if extent is None:
                    raise TileValidationError(
                        f"TileRef 引用了未定义的循环变量: {offset.loop}"
                    )
                iterations = max(extent // offset.tile, 1)
                offsets.append(offset.base + (iterations - 1) * offset.tile)
            else:
                offsets.append(offset)
        return tuple(offsets)

    def uses_loops(self) -> set[str]:
        return {o.loop for o in self.offsets if isinstance(o, TileRef)}

    def to_dict(self) -> dict:
        return {
            "offsets": [
                o.to_dict() if isinstance(o, TileRef) else o for o in self.offsets
            ],
            "extents": list(self.extents),
        }


@dataclass(frozen=True)
class Copy:
    """DMA 拷贝：global/shared <-> sram/local。"""

    src: str
    dst: str
    span: SourceSpan
    src_region: RegionSpec | None = None
    dst_region: RegionSpec | None = None
    async_token: str | None = None

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise TileValidationError(f"Copy 的源和目标相同: {self.src}")
        if self.src_region is not None and self.dst_region is not None:
            raise TileValidationError(
                f"Copy {self.src}->{self.dst} 不能同时指定源和目标的切片"
            )

    def to_dict(self) -> dict:
        return {
            "kind": "copy",
            "src": self.src,
            "dst": self.dst,
            "src_region": self.src_region.to_dict() if self.src_region else None,
            "dst_region": self.dst_region.to_dict() if self.dst_region else None,
            "async_token": self.async_token,
            "span": self.span.to_dict(),
        }


# ---------------------------------------------------------------------------
# Sync — 扩展支持 DPU ISB / ISBC
# ---------------------------------------------------------------------------


_VALID_SYNC_KINDS = ("dma_wait", "barrier", "instruction_barrier", "core_barrier")


@dataclass(frozen=True)
class Sync:
    """同步：DMA 完成等待、核内 barrier、DPU ISB 或核间 ISBC。"""

    kind: str
    token: str | None
    span: SourceSpan

    def __post_init__(self) -> None:
        if self.kind not in _VALID_SYNC_KINDS:
            raise TileValidationError(f"非法 Sync.kind: {self.kind}")
        if self.kind == "dma_wait" and not self.token:
            raise TileValidationError("dma_wait 必须携带 token")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "token": self.token,
            "span": self.span.to_dict(),
        }


# ---------------------------------------------------------------------------
# Compute — 高层占位，由 lower.py 展开成 DPU 指令序列
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Compute:
    """tile 级计算占位符，后续 lower pass 会把它展开成 DPU 基础指令。"""

    domain: str
    op_type: str
    canonical_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict
    span: SourceSpan

    def __post_init__(self) -> None:
        if not self.op_type or not self.canonical_name:
            raise TileValidationError("Compute 缺少 op_type / canonical_name")
        if not self.outputs:
            raise TileValidationError("Compute 必须至少有一个输出 buffer")

    def to_dict(self) -> dict:
        from ..ir.nodes import _jsonable

        return {
            "kind": "compute",
            "domain": self.domain,
            "op_type": self.op_type,
            "canonical_name": self.canonical_name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": _jsonable(self.attrs),
            "span": self.span.to_dict(),
        }


# ---------------------------------------------------------------------------
# DPU 指令集原语 —— 方案 A：lower.py 把 Compute 展开成这些类型
# ---------------------------------------------------------------------------

_VALID_ALU_OPCODES = frozenset(
    {
        # ALU_I
        "ADDI", "ADDCI", "SUBI", "SUBBI", "ORI", "ANDI", "XORI", "MOVI",
        # ALU_R
        "ADD", "ADDC", "SUB", "SUBB", "OR", "AND", "XOR", "MOVRR",
        "SLL", "SRL", "SLLI", "SRLI", "BITTZR", "BITCNT", "BITCLR", "BITSET",
        # ALU_RB
        "MUL", "DIV",
        # MOV 类
        "MOV",
    }
)


@dataclass(frozen=True)
class ArithOp:
    """DPU ALU 算术/逻辑指令（ALU_I / ALU_R / ALU_RB / MOV）。"""

    opcode: str
    rd: str
    rs1: str
    rs2: str | int | None = None
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.opcode not in _VALID_ALU_OPCODES:
            raise TileValidationError(
                f"非法 ArithOp.opcode: {self.opcode}；"
                f"合法值: {sorted(_VALID_ALU_OPCODES)}"
            )
        if not self.rd:
            raise TileValidationError("ArithOp 缺少 rd")
        if not self.rs1:
            raise TileValidationError(f"ArithOp {self.opcode} 缺少 rs1")

    def to_dict(self) -> dict:
        return {
            "kind": "arith",
            "opcode": self.opcode,
            "rd": self.rd,
            "rs1": self.rs1,
            "rs2": self.rs2,
        }

    def __str__(self) -> str:
        op = f"{self.opcode:<8s}"
        if self.rs2 is None:
            return f"  {op} {self.rd}, {self.rs1}"
        return f"  {op} {self.rd}, {self.rs1}, {self.rs2}"


_VALID_BRANCH_OPCODES = frozenset(
    {
        "BEQI", "BLTI", "BGTI", "BNEI", "BERRI", "BQMI", "BQHI",
        "BEQ", "BLT", "BGT", "BNE", "BQM", "BQH",
        "B", "BI",
    }
)


@dataclass(frozen=True)
class BranchOp:
    """DPU 分支/跳转指令（C_ 类）。

    条件分支 opcode 要求 rs1；无条件跳转（B, BI）rs1=None。
    rs2 可以是寄存器名 (str) 或 立即数 (int)。
    """

    opcode: str
    target: str
    rs1: str | None = None
    rs2: str | int | None = None
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.opcode not in _VALID_BRANCH_OPCODES:
            raise TileValidationError(
                f"非法 BranchOp.opcode: {self.opcode}；"
                f"合法值: {sorted(_VALID_BRANCH_OPCODES)}"
            )
        if not self.target:
            raise TileValidationError("BranchOp 缺少 target 标签")

    def to_dict(self) -> dict:
        return {
            "kind": "branch",
            "opcode": self.opcode,
            "target": self.target,
            "rs1": self.rs1,
            "rs2": self.rs2,
        }

    def __str__(self) -> str:
        op = f"{self.opcode:<8s}"
        if self.rs1 is None:
            return f"  {op} -> {self.target}"
        if self.rs2 is None:
            return f"  {op} {self.rs1} -> {self.target}"
        return f"  {op} {self.rs1}, {self.rs2} -> {self.target}"


@dataclass(frozen=True)
class LabelOp:
    """基本块标签，配合 BranchOp 使用。"""

    name: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise TileValidationError("LabelOp 缺少 name")

    def to_dict(self) -> dict:
        return {"kind": "label", "name": self.name}

    def __str__(self) -> str:
        return f"{self.name}:"


_VALID_ATOMIC_OPCODES = frozenset({"ATOMADDTLE", "ATOMADDTBE"})


@dataclass(frozen=True)
class AtomicOp:
    """DPU 原子操作（ISM 类）。多核 Conv 累加的关键硬件原语。"""

    opcode: str
    addr: str
    value: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.opcode not in _VALID_ATOMIC_OPCODES:
            raise TileValidationError(
                f"非法 AtomicOp.opcode: {self.opcode}；"
                f"合法值: {sorted(_VALID_ATOMIC_OPCODES)}"
            )
        if not self.addr or not self.value:
            raise TileValidationError("AtomicOp 必须同时给出 addr 和 value")

    def to_dict(self) -> dict:
        return {
            "kind": "atomic",
            "opcode": self.opcode,
            "addr": self.addr,
            "value": self.value,
        }

    def __str__(self) -> str:
        return f"  {self.opcode:<10s} [{self.addr}], {self.value}"


_VALID_QUEUE_OPCODES = frozenset({"PUSHQ", "POPQ", "POPQAL", "POPOPH"})


@dataclass(frozen=True)
class QueueOp:
    """DPU 核间队列操作（ISM 类）。"""

    opcode: str
    queue_id: int
    reg: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.opcode not in _VALID_QUEUE_OPCODES:
            raise TileValidationError(
                f"非法 QueueOp.opcode: {self.opcode}；"
                f"合法值: {sorted(_VALID_QUEUE_OPCODES)}"
            )
        if self.queue_id < 0:
            raise TileValidationError(f"QueueOp queue_id 不能为负: {self.queue_id}")
        if not self.reg:
            raise TileValidationError("QueueOp 缺少 reg")

    def to_dict(self) -> dict:
        return {
            "kind": "queue",
            "opcode": self.opcode,
            "queue_id": self.queue_id,
            "reg": self.reg,
        }

    def __str__(self) -> str:
        return f"  {self.opcode:<8s} q{self.queue_id}, {self.reg}"


_VALID_MEMMAN_OPCODES = frozenset({"MALCSMB", "FREESMB", "REFSMB", "REGSMB"})


@dataclass(frozen=True)
class MemManageOp:
    """DPU 共享内存管理（ISM 类）。"""

    opcode: str
    rd: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.opcode not in _VALID_MEMMAN_OPCODES:
            raise TileValidationError(
                f"非法 MemManageOp.opcode: {self.opcode}；"
                f"合法值: {sorted(_VALID_MEMMAN_OPCODES)}"
            )
        if not self.rd:
            raise TileValidationError("MemManageOp 缺少 rd")

    def to_dict(self) -> dict:
        return {
            "kind": "memmanage",
            "opcode": self.opcode,
            "rd": self.rd,
        }

    def __str__(self) -> str:
        return f"  {self.opcode:<8s} {self.rd}"


_VALID_MEM_OPCODES = frozenset({"LOADM", "STOREM"})


@dataclass(frozen=True)
class MemOp:
    """DPU 内存操作（ISM 类 LOADM/STOREM）。

    LOADM rd, addr_reg : rd = *(uint*)addr_reg
    STOREM rd, addr_reg: *(uint*)addr_reg = rd
    """

    opcode: str
    rd: str
    addr_reg: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.opcode not in _VALID_MEM_OPCODES:
            raise TileValidationError(
                f"非法 MemOp.opcode: {self.opcode}；"
                f"合法值: {sorted(_VALID_MEM_OPCODES)}"
            )
        if not self.rd or not self.addr_reg:
            raise TileValidationError("MemOp 必须同时给出 rd 和 addr_reg")

    def to_dict(self) -> dict:
        return {"kind": "mem", "opcode": self.opcode, "rd": self.rd, "addr_reg": self.addr_reg}

    def __str__(self) -> str:
        return f"  {self.opcode:<8s} {self.rd}, [{self.addr_reg}]"


@dataclass(frozen=True)
class EndOp:
    """DPU END 指令 — 程序终止。"""

    span: SourceSpan | None = None

    def to_dict(self) -> dict:
        return {"kind": "end"}

    def __str__(self) -> str:
        return "  END"


# ---------------------------------------------------------------------------
# Loop — 扩展 body 支持 Compute 内部的多层 tile 循环
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Loop:
    """tile 循环：extent 必须是 tile 的整数倍。

    body 为 None 时表示由 Schedule 模板控制的外层循环（Stage.loops 持有）；
    body 非 None 时表示 lower 后内嵌在 Stage.body 里的循环，可嵌套 ArithOp /
    BranchOp / 内层 Loop 等。
    """

    var: str
    axis: int
    extent: int
    tile: int
    body: tuple | None = None

    def __post_init__(self) -> None:
        if not self.var:
            raise TileValidationError("Loop.var 不能为空")
        if self.axis < 0:
            raise TileValidationError(f"loop {self.var} 的 axis 不能为负")
        if self.extent <= 0:
            raise TileValidationError(f"loop {self.var} 的 extent 必须为正数")
        if self.tile <= 0:
            raise TileValidationError(f"loop {self.var} 的 tile 必须为正数")
        if self.extent % self.tile != 0:
            raise TileValidationError(
                f"loop {self.var} 不整除: extent={self.extent}, tile={self.tile}"
            )

    @property
    def iterations(self) -> int:
        return self.extent // self.tile

    def to_dict(self) -> dict:
        data = {
            "var": self.var,
            "axis": self.axis,
            "extent": self.extent,
            "tile": self.tile,
            "iterations": self.iterations,
        }
        if self.body is not None:
            data["body"] = [
                item.to_dict() if hasattr(item, "to_dict") else {"kind": "repr", "repr": str(item)}
                for item in self.body
            ]
        return data


# ---------------------------------------------------------------------------
# Stage / Program
# ---------------------------------------------------------------------------


# Stage.body 和 Stage.prologue 允许的所有操作类型
_STAGE_ITEM_TYPES = (
    Copy, Compute, Sync, Loop,               # 原类型
    ArithOp, BranchOp, LabelOp,              # DPU 指令
    AtomicOp, QueueOp, MemManageOp, EndOp,   # DPU 指令
    MemOp,                                   # DPU LOADM/STOREM
)


@dataclass(frozen=True)
class Stage:
    """一个计算节点的 tile 程序。"""

    name: str
    template: str
    node_value: str
    loops: tuple[Loop, ...]
    prologue: tuple
    body: tuple
    buffers: tuple[Buffer, ...]
    span: SourceSpan
    core: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise TileValidationError("Stage.name 不能为空")
        if self.core < 0:
            raise TileValidationError(
                f"stage {self.name} 的 core 不能为负: {self.core}"
            )
        if not self.buffers:
            raise TileValidationError(f"stage {self.name} 没有声明任何 buffer")
        for loop in self.loops:
            if not isinstance(loop, Loop):
                raise TileValidationError(f"stage {self.name} 的 loop 类型非法")
        for item in self.prologue:
            if not isinstance(item, _STAGE_ITEM_TYPES):
                raise TileValidationError(
                    f"stage {self.name} 的 prologue 不允许 {type(item).__name__}"
                )
            if isinstance(item, Compute):
                raise TileValidationError(
                    f"stage {self.name} 的 prologue 不允许 Compute"
                )
        for item in self.body:
            if not isinstance(item, _STAGE_ITEM_TYPES):
                raise TileValidationError(
                    f"stage {self.name} 的 body 出现非法操作: {type(item).__name__}"
                )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "template": self.template,
            "node_value": self.node_value,
            "loops": [loop.to_dict() for loop in self.loops],
            "prologue": [
                item.to_dict() if hasattr(item, "to_dict") else {"repr": str(item)}
                for item in self.prologue
            ],
            "body": [
                item.to_dict() if hasattr(item, "to_dict") else {"repr": str(item)}
                for item in self.body
            ],
            "buffers": [buffer.to_dict() for buffer in self.buffers],
            "span": self.span.to_dict(),
            "core": self.core,
        }


@dataclass
class Program:
    """单核 tile 程序。"""

    name: str
    global_buffers: list[Buffer]
    stages: list[Stage]

    def buffer_map(self) -> dict[str, Buffer]:
        mapping: dict[str, Buffer] = {}
        for buffer in self.global_buffers:
            mapping[buffer.name] = buffer
        for stage in self.stages:
            for buffer in stage.buffers:
                mapping[buffer.name] = buffer
        return mapping

    def with_buffers(self, global_buffers: list[Buffer], stages: list[Stage]) -> "Program":
        return Program(self.name, global_buffers, stages)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "global_buffers": [buffer.to_dict() for buffer in self.global_buffers],
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def __str__(self) -> str:
        from .printer import print_program

        return print_program(self)


def replace_buffer_address(buffer: Buffer, address: int) -> Buffer:
    return replace(buffer, address=address)


__all__ = [
    # 原类型
    "Buffer", "Compute", "Copy", "Loop", "Program", "RegionSpec",
    "SCOPES", "Stage", "Sync", "TileRef",
    "align_up", "replace_buffer_address",
    # DPU 新类型
    "ArithOp", "BranchOp", "LabelOp", "AtomicOp", "QueueOp",
    "MemManageOp", "EndOp", "MemOp",
]
