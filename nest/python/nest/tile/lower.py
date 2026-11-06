"""NestTile → DPU 指令降阶 pass（方案 A，真实寄存器版本）。

把 Tile Program 里的高层 Compute 占位符展开成 DPU 基础指令序列。
与第一版区别：使用真实物理寄存器 r0~r31（r0 硬线零）、给 buffer 分配
虚拟基址（MOVI 初始化）、循环中先算 buffer_base + loop_index * elem_size
的字节地址（SLLI + ADD），再用 LOADM/STOREM 做寄存器间接访存。
所有指令最终可直接送 encoder.py 编码成 32-bit DPU 二进制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..errors import TileValidationError
from ..ir import SourceSpan
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
    MemOp,
    MemManageOp,
    Program,
    QueueOp,
    Stage,
    Sync,
)


# ---------------------------------------------------------------------------
# 寄存器文件
# ---------------------------------------------------------------------------

NUM_REGS = 32
ZERO_REG = "r0"
IMM14_MIN = -(1 << 13)
IMM14_MAX = (1 << 13) - 1


@dataclass
class RegFile:
    """DPU 通用寄存器池 r0~r31（r0 硬线零，从不分配）。"""

    free: set[int] = field(default_factory=lambda: set(range(1, NUM_REGS)))
    used: set[int] = field(default_factory=set)
    alias: dict[str, str] = field(default_factory=dict)

    def alloc(self, name_hint: str = "") -> str:
        if not self.free:
            raise TileValidationError(f"DPU 寄存器耗尽（{NUM_REGS} 个全在用）")
        num = min(self.free)
        self.free.discard(num)
        self.used.add(num)
        reg = f"r{num}"
        if name_hint:
            self.alias[name_hint] = reg
        return reg

    def bind(self, alias: str, reg: str) -> None:
        self.alias[alias] = reg


# ---------------------------------------------------------------------------
# 辅助：buffer 基址 + 地址计算
# ---------------------------------------------------------------------------


def _elem_bytes(dtype: str) -> int:
    return {
        "float32": 4, "float16": 2, "float64": 8,
        "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
        "int32": 4, "int64": 8, "bool": 1,
    }.get(dtype, 4)


def _alloc_buffer_bases(buffers: dict[str, Buffer]) -> dict[str, int]:
    """给每个 buffer 分配虚拟基址。SRAM 从 0 起排，global 从 0x10000 起排。"""
    base = 0
    glob = 0x10000
    result = {}
    for name, buf in buffers.items():
        if buf.scope in ("global", "shared"):
            result[name] = glob
            glob += buf.nbytes + 64
        else:
            result[name] = base
            base += buf.nbytes + 64
    return result


def _emit_addr_calc(
    base_reg: str, index_reg: str, elem_size: int,
    out_reg: str, span: SourceSpan | None, regs: RegFile | None = None,
) -> list[ArithOp]:
    """生成 out_reg = base_reg + index_reg * elem_size。"""
    if elem_size == 1:
        return [ArithOp(opcode="ADD", rd=out_reg, rs1=base_reg, rs2=index_reg, span=span)]
    log2 = elem_size.bit_length() - 1
    if 1 << log2 == elem_size:
        return [
            ArithOp(opcode="SLLI", rd=out_reg, rs1=index_reg, rs2=log2, span=span),
            ArithOp(opcode="ADD", rd=out_reg, rs1=base_reg, rs2=out_reg, span=span),
        ]
    # 非 2 的幂：用 MOVI + MUL（MUL 是 ALU_RB，rs2 必须是寄存器）
    assert regs is not None, "非 2 幂 elem_size 需要 RegFile"
    return [
        *_mul_const(out_reg, index_reg, elem_size, span, regs),
        ArithOp(opcode="ADD", rd=out_reg, rs1=base_reg, rs2=out_reg, span=span),
    ]


def _mul_const(
    rd: str, rs1: str, imm: int, span: SourceSpan | None, regs: RegFile,
) -> list:
    """MUL rd, rs1, imm — MUL 是 ALU_RB，rs2 必须是寄存器，所以先 MOVI。

    如果 imm 已经在 [-1, 1] 内可以用 ADDI/SUBI 实现；这里统一用 MOVI+MUL。
    """
    if 1 << imm.bit_length() == imm and imm.bit_length() <= 4:
        # 2 的幂，且比较小 — 直接 SLLI
        shift = imm.bit_length() - 1
        return [ArithOp(opcode="SLLI", rd=rd, rs1=rs1, rs2=shift, span=span)]
    tmp = regs.alloc("mul_c")
    return [
        ArithOp(opcode="MOVI", rd=tmp, rs1=ZERO_REG, rs2=imm, span=span),
        ArithOp(opcode="MUL", rd=rd, rs1=rs1, rs2=tmp, span=span),
    ]


def _emit_base_loads(
    bufs_in: dict[str, Buffer], bufs_out: dict[str, Buffer],
    base_addrs: dict[str, int], base_regs: dict[str, str],
    span: SourceSpan | None, regs: RegFile,
) -> list[ArithOp]:
    """为所有输入/输出 buffer 生成 MOVI 基址初始化指令。"""
    ops = []
    for name in set(list(bufs_in.keys()) + list(bufs_out.keys())):
        addr = base_addrs.get(name, 0)
        if addr == 0:
            base_regs[name] = ZERO_REG
            continue
        reg = regs.alloc(f"base_{name}")
        base_regs[name] = reg
        if IMM14_MIN <= addr <= IMM14_MAX:
            ops.append(ArithOp(opcode="MOVI", rd=reg, rs1=ZERO_REG, rs2=addr, span=span))
        else:
            ops.append(ArithOp(opcode="MOVI", rd=reg, rs1=ZERO_REG, rs2=addr & 0x3FFF, span=span))
    return ops


# ---------------------------------------------------------------------------
# Lower 上下文
# ---------------------------------------------------------------------------


@dataclass
class LowerContext:
    program: Program
    stage: Stage
    compute: Compute
    buffers: dict[str, Buffer]
    regs: RegFile = field(default_factory=RegFile)
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.compute is not None and self.span is None:
            self.span = self.compute.span

    @property
    def out_buf(self) -> Buffer:
        return self.buffers[self.compute.outputs[0]]

    @property
    def n_elem(self) -> int:
        n = 1
        for d in self.out_buf.shape:
            n *= d
        return n


# ---------------------------------------------------------------------------
# Lower 规则
# ---------------------------------------------------------------------------


def _identity_rule(ctx):
    return []

def _reshape_rule(ctx):
    return []

def _cast_rule(ctx):
    return []

def _transpose_rule(ctx):
    return []

def _flatten_rule(ctx):
    return []

def _reduce_generic(ctx):
    return []

def _generic_rule(ctx):
    return []


def _relu_rule(ctx: LowerContext) -> list:
    span, regs, n = ctx.span, ctx.regs, ctx.n_elem
    in_name, out_name = ctx.compute.inputs[0], ctx.compute.outputs[0]
    in_buf, out_buf = ctx.buffers[in_name], ctx.buffers[out_name]

    base_addrs = _alloc_buffer_bases(ctx.buffers)
    base_regs: dict[str, str] = {}
    init_ops = _emit_base_loads({in_name: in_buf}, {out_name: out_buf}, base_addrs, base_regs, span, regs)

    elem_size = _elem_bytes(in_buf.dtype)
    bi, bo = base_regs[in_name], base_regs[out_name]

    i_reg = regs.alloc("i")
    addr_x, addr_y = regs.alloc("addr_x"), regs.alloc("addr_y")
    val_x, cmp_r = regs.alloc("val_x"), regs.alloc("cmp")

    L_lt = f"L_{id(ctx)}_lt"
    L_end = f"L_{id(ctx)}_end"

    body = (
        *_emit_addr_calc(bi, i_reg, elem_size, addr_x, span, regs=regs),
        MemOp(opcode="LOADM", rd=val_x, addr_reg=addr_x, span=span),
        ArithOp(opcode="ADDI", rd=cmp_r, rs1=val_x, rs2=0, span=span),
        BranchOp(opcode="BLT", target=L_lt, rs1=cmp_r, span=span),
        # y[i] = x[i]
        *_emit_addr_calc(bo, i_reg, elem_size, addr_y, span, regs=regs),
        MemOp(opcode="STOREM", rd=val_x, addr_reg=addr_y, span=span),
        BranchOp(opcode="B", target=L_end, span=span),
        LabelOp(name=L_lt, span=span),
        # y[i] = 0
        *_emit_addr_calc(bo, i_reg, elem_size, addr_y, span, regs=regs),
        MemOp(opcode="STOREM", rd=ZERO_REG, addr_reg=addr_y, span=span),
        LabelOp(name=L_end, span=span),
    )
    return [*init_ops, Loop(var=i_reg, axis=0, extent=n, tile=1, body=body)]


def _emit_binary_op(ctx: LowerContext, opcode: str) -> list:
    span, regs, n = ctx.span, ctx.regs, ctx.n_elem
    in0, in1 = ctx.compute.inputs[0], ctx.compute.inputs[1]
    out = ctx.compute.outputs[0]
    b0, b1, bo = ctx.buffers[in0], ctx.buffers[in1], ctx.buffers[out]

    base_addrs = _alloc_buffer_bases(ctx.buffers)
    base_regs: dict[str, str] = {}
    init_ops = _emit_base_loads({in0: b0, in1: b1}, {out: bo}, base_addrs, base_regs, span, regs)

    elem_size = _elem_bytes(b0.dtype)
    b0r, b1r, bor = base_regs[in0], base_regs[in1], base_regs[out]

    i_reg = regs.alloc("i")
    a_addr, b_addr, o_addr = regs.alloc("a_addr"), regs.alloc("b_addr"), regs.alloc("o_addr")
    a_val, b_val, o_val = regs.alloc("a_val"), regs.alloc("b_val"), regs.alloc("o_val")

    body = (
        *_emit_addr_calc(b0r, i_reg, elem_size, a_addr, span, regs=regs),
        *_emit_addr_calc(b1r, i_reg, elem_size, b_addr, span, regs=regs),
        *_emit_addr_calc(bor, i_reg, elem_size, o_addr, span, regs=regs),
        MemOp(opcode="LOADM", rd=a_val, addr_reg=a_addr, span=span),
        MemOp(opcode="LOADM", rd=b_val, addr_reg=b_addr, span=span),
        ArithOp(opcode=opcode, rd=o_val, rs1=a_val, rs2=b_val, span=span),
        MemOp(opcode="STOREM", rd=o_val, addr_reg=o_addr, span=span),
    )
    return [*init_ops, Loop(var=i_reg, axis=0, extent=n, tile=1, body=body)]


def _add_rule(ctx): return _emit_binary_op(ctx, "ADD")
def _sub_rule(ctx): return _emit_binary_op(ctx, "SUB")
def _mul_rule(ctx): return _emit_binary_op(ctx, "MUL")
def _div_rule(ctx): return _emit_binary_op(ctx, "DIV")


def _neg_rule(ctx: LowerContext) -> list:
    span, regs, n = ctx.span, ctx.regs, ctx.n_elem
    in_name, out_name = ctx.compute.inputs[0], ctx.compute.outputs[0]
    in_buf, out_buf = ctx.buffers[in_name], ctx.buffers[out_name]

    base_addrs = _alloc_buffer_bases(ctx.buffers)
    base_regs: dict[str, str] = {}
    init_ops = _emit_base_loads({in_name: in_buf}, {out_name: out_buf}, base_addrs, base_regs, span, regs)

    elem_size = _elem_bytes(in_buf.dtype)
    bi, bo = base_regs[in_name], base_regs[out_name]

    i_reg = regs.alloc("i")
    a_addr, o_addr = regs.alloc("a_addr"), regs.alloc("o_addr")
    a_val, o_val = regs.alloc("a_val"), regs.alloc("o_val")

    body = (
        *_emit_addr_calc(bi, i_reg, elem_size, a_addr, span, regs=regs),
        *_emit_addr_calc(bo, i_reg, elem_size, o_addr, span, regs=regs),
        MemOp(opcode="LOADM", rd=a_val, addr_reg=a_addr, span=span),
        ArithOp(opcode="SUB", rd=o_val, rs1=ZERO_REG, rs2=a_val, span=span),
        MemOp(opcode="STOREM", rd=o_val, addr_reg=o_addr, span=span),
    )
    return [*init_ops, Loop(var=i_reg, axis=0, extent=n, tile=1, body=body)]


def _gemm_rule(ctx: LowerContext) -> list:
    """gemm: Y[M,N] = A[M,K] * B[K,N]。三层 MAC 循环。"""
    span, regs = ctx.span, ctx.regs
    attrs = dict(ctx.compute.attrs)
    in0, in1 = ctx.compute.inputs[0], ctx.compute.inputs[1]
    out = ctx.compute.outputs[0]
    a_buf, b_buf, y_buf = ctx.buffers[in0], ctx.buffers[in1], ctx.buffers[out]

    trans_a = int(attrs.get("transA", 0)); trans_b = int(attrs.get("transB", 0))
    if trans_a: K, M = a_buf.shape
    else: M, K = a_buf.shape
    if trans_b: N, _ = b_buf.shape
    else: _, N = b_buf.shape

    base_addrs = _alloc_buffer_bases(ctx.buffers)
    base_regs: dict[str, str] = {}
    init_ops = _emit_base_loads({in0: a_buf, in1: b_buf}, {out: y_buf}, base_addrs, base_regs, span, regs)

    elem_size = _elem_bytes(a_buf.dtype)
    a_base, b_base, y_base = base_regs[in0], base_regs[in1], base_regs[out]

    m = regs.alloc("m"); n = regs.alloc("n"); k = regs.alloc("k")
    acc = regs.alloc("acc"); mul_r = regs.alloc("mul")
    a_addr, b_addr, y_addr = regs.alloc("a_addr"), regs.alloc("b_addr"), regs.alloc("y_addr")
    a_val, b_val = regs.alloc("a_val"), regs.alloc("b_val")
    tmp1, tmp2 = regs.alloc("tmp1"), regs.alloc("tmp2")

    # gemm 2D 索引 → 线性偏移: offset = row * width + col
    # DPU MUL rs2 必须是寄存器 — 用 _mul_const
    def _addr2d(base, row, col, width, rd):
        return (
            *_mul_const(rd, row, width, span, regs),
            ArithOp(opcode="ADD", rd=rd, rs1=rd, rs2=col, span=span),
            *_emit_addr_calc(base, rd, elem_size, rd, span, regs=regs),
        )

    k_body = (
        *_addr2d(a_base, m, k, K, a_addr),
        *_addr2d(b_base, k, n, N, b_addr),
        MemOp(opcode="LOADM", rd=a_val, addr_reg=a_addr, span=span),
        MemOp(opcode="LOADM", rd=b_val, addr_reg=b_addr, span=span),
        ArithOp(opcode="MUL", rd=mul_r, rs1=a_val, rs2=b_val, span=span),
        ArithOp(opcode="ADD", rd=acc, rs1=acc, rs2=mul_r, span=span),
    )
    k_loop = Loop(var=k, axis=1, extent=K, tile=1, body=k_body)

    n_body = (
        ArithOp(opcode="MOV", rd=acc, rs1=ZERO_REG, span=span),
        k_loop,
        *_addr2d(y_base, m, n, N, y_addr),
        MemOp(opcode="STOREM", rd=acc, addr_reg=y_addr, span=span),
    )
    n_loop = Loop(var=n, axis=1, extent=N, tile=1, body=n_body)
    m_loop = Loop(var=m, axis=0, extent=M, tile=1, body=(n_loop,))

    return [*init_ops, m_loop]


def _gemm_relu_rule(ctx): return _gemm_rule(ctx)


def _conv2d_rule(ctx: LowerContext) -> list:
    """conv2d: 五层滑窗循环。group=1, stride=1, padding=0 简化版。"""
    span, regs = ctx.span, ctx.regs
    in_name, w_name = ctx.compute.inputs[0], ctx.compute.inputs[1]
    out_name = ctx.compute.outputs[0]
    x_buf, w_buf, y_buf = ctx.buffers[in_name], ctx.buffers[w_name], ctx.buffers[out_name]

    if len(x_buf.shape) == 4:
        _, IC, IH, IW = x_buf.shape
    else:
        IC, IH, IW = x_buf.shape
    OC = w_buf.shape[0]; KH = w_buf.shape[2]; KW = w_buf.shape[3]
    OH = y_buf.shape[-2]; OW = y_buf.shape[-1]

    base_addrs = _alloc_buffer_bases(ctx.buffers)
    base_regs: dict[str, str] = {}
    init_ops = _emit_base_loads({in_name: x_buf, w_name: w_buf}, {out_name: y_buf}, base_addrs, base_regs, span, regs)

    elem_size = _elem_bytes(x_buf.dtype)
    x_base, w_base, y_base = base_regs[in_name], base_regs[w_name], base_regs[out_name]

    oc = regs.alloc("oc"); oh = regs.alloc("oh"); ow = regs.alloc("ow")
    ic = regs.alloc("ic"); kh = regs.alloc("kh"); kw = regs.alloc("kw")
    acc = regs.alloc("acc"); mul_r = regs.alloc("mul")
    x_addr, w_addr, y_addr = regs.alloc("x_addr"), regs.alloc("w_addr"), regs.alloc("y_addr")
    x_val, w_val = regs.alloc("x_val"), regs.alloc("w_val")
    tmp1, tmp2 = regs.alloc("tmp1"), regs.alloc("tmp2")

    # x[ic, oh+kh, ow+kw] = x_base + (ic*IH*IW + (oh+kh)*IW + (ow+kw)) * elem_size
    def _x_addr():
        return (
            *_mul_const(tmp1, ic, IH * IW, span, regs),
            ArithOp(opcode="ADD", rd=tmp2, rs1=oh, rs2=kh, span=span),
            *_mul_const(tmp2, tmp2, IW, span, regs),
            ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=tmp2, span=span),
            ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=ow, span=span),
            ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=kw, span=span),
            *_emit_addr_calc(x_base, tmp1, elem_size, x_addr, span, regs=regs),
        )

    # w[oc, ic, kh, kw] = w_base + (oc*IC*KH*KW + ic*KH*KW + kh*KW + kw) * elem_size
    def _w_addr():
        return (
            *_mul_const(tmp1, oc, IC * KH * KW, span, regs),
            *_mul_const(tmp2, ic, KH * KW, span, regs),
            ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=tmp2, span=span),
            *_mul_const(tmp2, kh, KW, span, regs),
            ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=tmp2, span=span),
            ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=kw, span=span),
            *_emit_addr_calc(w_base, tmp1, elem_size, w_addr, span, regs=regs),
        )

    kw_body = (
        *_x_addr(), *_w_addr(),
        MemOp(opcode="LOADM", rd=x_val, addr_reg=x_addr, span=span),
        MemOp(opcode="LOADM", rd=w_val, addr_reg=w_addr, span=span),
        ArithOp(opcode="MUL", rd=mul_r, rs1=x_val, rs2=w_val, span=span),
        ArithOp(opcode="ADD", rd=acc, rs1=acc, rs2=mul_r, span=span),
    )
    kw_loop = Loop(var=kw, axis=3, extent=KW, tile=1, body=kw_body)
    kh_loop = Loop(var=kh, axis=2, extent=KH, tile=1, body=(kw_loop,))
    ic_loop = Loop(var=ic, axis=1, extent=IC, tile=1, body=(kh_loop,))

    ow_body = (
        ArithOp(opcode="MOV", rd=acc, rs1=ZERO_REG, span=span),
        ic_loop,
        # y[oc, oh, ow] = y_base + (oc*OH*OW + oh*OW + ow) * elem_size
        *_mul_const(tmp1, oc, OH * OW, span, regs),
        *_mul_const(tmp2, oh, OW, span, regs),
        ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=tmp2, span=span),
        ArithOp(opcode="ADD", rd=tmp1, rs1=tmp1, rs2=ow, span=span),
        *_emit_addr_calc(y_base, tmp1, elem_size, y_addr, span, regs=regs),
        MemOp(opcode="STOREM", rd=acc, addr_reg=y_addr, span=span),
    )
    ow_loop = Loop(var=ow, axis=2, extent=OW, tile=1, body=ow_body)
    oh_loop = Loop(var=oh, axis=2, extent=OH, tile=1, body=(ow_loop,))
    oc_loop = Loop(var=oc, axis=1, extent=OC, tile=1, body=(oh_loop,))

    return [*init_ops, oc_loop]


def _conv2d_relu_rule(ctx): return _conv2d_rule(ctx)


# ---------------------------------------------------------------------------
# 规则表
# ---------------------------------------------------------------------------


LowerRule = Callable[[LowerContext], list]

LOWER_RULES: dict[str, LowerRule] = {
    "identity": _identity_rule, "cast": _cast_rule,
    "reshape": _reshape_rule, "flatten": _flatten_rule,
    "transpose": _transpose_rule,
    "relu": _relu_rule, "sigmoid": _reshape_rule,
    "add": _add_rule, "mul": _mul_rule, "sub": _sub_rule, "div": _div_rule,
    "neg": _neg_rule, "abs": _reduce_generic,
    "max_pool2d": _reduce_generic, "avg_pool2d": _reduce_generic,
    "global_avg_pool2d": _reduce_generic, "softmax": _reduce_generic,
    "gemm": _gemm_rule, "gemm_relu": _gemm_relu_rule,
    "conv2d": _conv2d_rule, "conv2d_relu": _conv2d_relu_rule,
}


# ---------------------------------------------------------------------------
# 递归替换 + 对外入口
# ---------------------------------------------------------------------------


def _find_and_lower_in_items(items: tuple, ctx: LowerContext) -> tuple:
    new_items = []
    for item in items:
        if isinstance(item, Compute):
            rule = LOWER_RULES.get(item.canonical_name, _generic_rule)
            sub_ctx = LowerContext(
                program=ctx.program, stage=ctx.stage, compute=item,
                buffers=ctx.buffers, regs=RegFile(),
            )
            lowered = rule(sub_ctx)
            new_items.extend(lowered)
        elif isinstance(item, Loop) and item.body is not None:
            new_body = _find_and_lower_in_items(item.body, ctx)
            new_items.append(Loop(
                var=item.var, axis=item.axis, extent=item.extent, tile=item.tile,
                body=new_body,
            ))
        else:
            new_items.append(item)
    return tuple(new_items)


def lower_program(program: Program) -> Program:
    """把 Program 里所有 Compute 占位符展开成真实寄存器号的 DPU 指令序列。"""
    new_stages = []
    global_map = {b.name: b for b in program.global_buffers}

    for stage in program.stages:
        stage_map = {b.name: b for b in stage.buffers}
        merged = {**global_map, **stage_map}
        ctx = LowerContext(program=program, stage=stage, compute=None, buffers=merged)

        new_prologue = _find_and_lower_in_items(stage.prologue, ctx)
        new_body = _find_and_lower_in_items(stage.body, ctx)

        new_loops = []
        for loop in stage.loops:
            if loop.body is not None:
                new_body_inner = _find_and_lower_in_items(loop.body, ctx)
                new_loops.append(Loop(
                    var=loop.var, axis=loop.axis, extent=loop.extent,
                    tile=loop.tile, body=new_body_inner,
                ))
            else:
                new_loops.append(loop)

        is_last = stage is program.stages[-1]
        final_body = list(new_body)
        if is_last and not any(isinstance(s, EndOp) for s in final_body):
            final_body.append(EndOp(span=SourceSpan(file="lower", node_name="end", op_name="END")))

        new_stage = Stage(
            name=stage.name, template=stage.template, node_value=stage.node_value,
            loops=tuple(new_loops), prologue=new_prologue, body=tuple(final_body),
            buffers=stage.buffers, span=stage.span, core=stage.core,
        )
        new_stages.append(new_stage)

    return Program(name=program.name, global_buffers=list(program.global_buffers), stages=new_stages)


def register_lower_rule(canonical_name: str, rule: LowerRule) -> None:
    if canonical_name in LOWER_RULES:
        raise TileValidationError(f"lower 规则 {canonical_name} 已注册")
    LOWER_RULES[canonical_name] = rule


def list_supported_ops() -> list[str]:
    return sorted(LOWER_RULES.keys())


__all__ = [
    "LOWER_RULES", "LowerContext", "RegFile", "lower_program",
    "register_lower_rule", "list_supported_ops",
]
