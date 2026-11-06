"""DPU 指令编码引擎（32-bit 固定长指令字）。

DPU 指令格式：
┌──────────┬──────────────────────────────────────────────┐
│ bit 31:29│ bit 28:0                                     │
│ opcode  │ 各指令类自定义布局                              │
└──────────┴──────────────────────────────────────────────┘

opcode 大类编码：
  NOP  0b000  ALU_I 0b001  ALU_R 0b010  ALU_RB 0b011
  MOV  0b100  ISM   0b101  C_    0b110  END   0b111

寄存器：r0~r31（5 bit 编码）。r0 硬线零。
立即数：ALU_I imm=14 bit 有符号；ISM imm=13 bit 无符号；C_ imm=13 bit 有符号跳转偏移。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable

from ..errors import TileValidationError

# ---------------------------------------------------------------------------
# opcode 大类常量
# ---------------------------------------------------------------------------

OP_NOP = 0b000
OP_ALU_I = 0b001
OP_ALU_R = 0b010
OP_ALU_RB = 0b011
OP_MOV = 0b100
OP_ISM = 0b101
OP_CTRL = 0b110
OP_END = 0b111

# ---------------------------------------------------------------------------
# 各指令子类 subtype 编码（5 bit 或 6 bit）
# ---------------------------------------------------------------------------

ALU_I_SUB = {
    "ADDI": 0b00000, "ADDCI": 0b00001, "SUBI": 0b00010, "SUBBI": 0b00011,
    "ORI": 0b00100, "ANDI": 0b00101, "XORI": 0b00110, "MOVI": 0b00111,
}

ALU_R_SUB = {
    "ADD": 0b00000, "ADDC": 0b00001, "SUB": 0b00010, "SUBB": 0b00011,
    "OR": 0b00100, "AND": 0b00101, "XOR": 0b00110, "MOVRR": 0b00111,
    "SLL": 0b01000, "SRL": 0b01001, "SLLI": 0b01010, "SRLI": 0b01011,
    "BITTZR": 0b01100, "BITCNT": 0b01101, "BITCLR": 0b01110, "BITSET": 0b01111,
}

ALU_RB_SUB = {
    "MUL": 0b00000, "DIV": 0b00001,
}

MOV_SUB = {
    "MOV": 0b00000, "CKSUM": 0b00001, "HASH": 0b00010,
}

ISM_SUB = {
    "LOADT": 0b000000, "LOADTLCK": 0b000001, "STORET": 0b000010, "STORETULCK": 0b000011,
    "ATOMADDTLE": 0b000100, "ATOMADDTBE": 0b000101,
    "INSERTT": 0b000110, "DELETET": 0b000111, "QUERYT": 0b001000,
    "LOADM": 0b001001, "STOREM": 0b001010,
    "MALCSMB": 0b001011, "FREESMB": 0b001100, "REFSMB": 0b001101, "REGSMB": 0b001110,
    "PUSHQ": 0b001111, "POPQ": 0b010000, "POPQAL": 0b010001, "POPOPH": 0b010010,
}

CTRL_SUB = {
    "BI": 0b000000, "BEQI": 0b000001, "BERRI": 0b000010, "BLTI": 0b000011,
    "BQMI": 0b000100, "B": 0b000101, "BEQ": 0b000110, "BLT": 0b000111,
    "BQM": 0b001000, "BNEI": 0b001001, "BGTI": 0b001010, "BQHI": 0b001011,
    "BNE": 0b001100, "BGT": 0b001101, "BQH": 0b001110,
    "ISB": 0b001111, "ISBC": 0b010000,
}


# ---------------------------------------------------------------------------
# 寄存器号解析
# ---------------------------------------------------------------------------


def _parse_reg(reg: str) -> int:
    """从 'r0'/'r31' 解析出寄存器号 0~31。"""
    if not isinstance(reg, str) or not reg.startswith("r"):
        raise TileValidationError(f"寄存器名格式非法: {reg!r} (应为 r0~r31)")
    try:
        num = int(reg[1:])
    except ValueError:
        raise TileValidationError(f"寄存器名格式非法: {reg!r} (应为 r0~r31)")
    if num < 0 or num > 31:
        raise TileValidationError(f"寄存器号超出范围: {num} (应为 0~31)")
    return num


# ---------------------------------------------------------------------------
# 编码函数
# ---------------------------------------------------------------------------


def _sign_extend(value: int, bits: int) -> int:
    """有符号整数 → 指定 bit 宽的无符号二进制表示。"""
    mask = (1 << bits) - 1
    return value & mask


def encode_alu_i(opcode: str, rd: str, rs1: str, imm: int) -> int:
    """ALU_I 类：rd = rs1 op imm。

    字段：op=001 | subtype(5) | rd(5) | rs1(5) | imm(14)
    """
    sub = ALU_I_SUB[opcode]
    rd_num = _parse_reg(rd)
    rs1_num = _parse_reg(rs1)
    imm14 = _sign_extend(imm, 14)
    word = (OP_ALU_I << 29) | (sub << 24) | (rd_num << 19) | (rs1_num << 14) | imm14
    return word


def encode_alu_r(opcode: str, rd: str, rs1: str, rs2) -> int:
    """ALU_R 类：大部分 rs2 是寄存器，移位类（SLLI/SRLI）rs2 是立即数 shift count。"""
    sub = ALU_R_SUB[opcode]
    rd_num = _parse_reg(rd)
    rs1_num = _parse_reg(rs1)
    if opcode in ("SLLI", "SRLI"):
        # 移位立即数 — 放进 rs2 字段的低 5 bit
        rs2_num = (int(rs2) & 0x1F) if rs2 is not None else 0
    else:
        rs2_num = _parse_reg(rs2)
    word = (OP_ALU_R << 29) | (sub << 24) | (rd_num << 19) | (rs1_num << 14) | (rs2_num << 9)
    return word


def encode_alu_rb(opcode: str, rd: str, rs1: str, rs2: str) -> int:
    """ALU_RB 类（MUL/DIV）。

    字段：op=011 | subtype(5) | rd(5) | rs1(5) | rs2(5) | 0(9)
    """
    sub = ALU_RB_SUB[opcode]
    rd_num = _parse_reg(rd)
    rs1_num = _parse_reg(rs1)
    rs2_num = _parse_reg(rs2)
    word = (OP_ALU_RB << 29) | (sub << 24) | (rd_num << 19) | (rs1_num << 14) | (rs2_num << 9)
    return word


def encode_mov(opcode: str, rd: str, rs1: str) -> int:
    """MOV 类。

    字段：op=100 | subtype(5) | rd(5) | rs1(5) | 0(14)
    """
    sub = MOV_SUB[opcode]
    rd_num = _parse_reg(rd)
    rs1_num = _parse_reg(rs1)
    word = (OP_MOV << 29) | (sub << 24) | (rd_num << 19) | (rs1_num << 14)
    return word


def encode_ism(opcode: str, *, rd: str | None = None, rs1: str | None = None,
               imm: int = 0) -> int:
    """ISM 类（内存/队列/共享管理）。

    字段：op=101 | subtype(6) | rd(5) | rs1(5) | imm(13)

    不同子类型字段含义不同：
      LOADM/STOREM: rd=data_reg, rs1=addr_reg, imm=0
      ATOMADDTLE/B: rd=data_reg, rs1=addr_reg, imm=0
      PUSHQ:        rs1=queue_id, rd=data_reg
      POPQ:         rs1=queue_id, rd=data_reg
      MALCSMB/FREESMB: rd=result_reg, imm=size
      ISB/ISBC:     只用 subtype
    """
    sub = ISM_SUB[opcode]
    rd_num = _parse_reg(rd) if rd is not None else 0
    rs1_num = _parse_reg(rs1) if rs1 is not None else 0
    imm13 = imm & 0x1FFF
    word = (OP_ISM << 29) | (sub << 23) | (rd_num << 18) | (rs1_num << 13) | imm13
    return word


def encode_ctrl(opcode: str, *, rs1: str | None = None, rs2: str | int | None = None,
                offset: int = 0) -> int:
    """控制类（分支/同步）。

    字段：op=110 | subtype(6) | rs1(5) | rs2(5/立即数) | offset(13)

    BEQI/BEQ/BNEI/BNE: rs1 和 rs2 都要
    BLT/BLTI/BGT/BGTI: rs1 和 rs2 都要（或 rs2 是立即数）
    B/BI:              只需要 offset（rs1/rs2=0）
    ISB/ISBC:          只需要 subtype（rs1/rs2/offset=0）
    """
    sub = CTRL_SUB[opcode]
    rs1_num = _parse_reg(rs1) if rs1 is not None else 0
    if rs2 is None:
        rs2_num = 0
    elif isinstance(rs2, int):
        rs2_num = rs2 & 0x1F  # 立即数的低 5 bit 放到 rs2 字段
    else:
        rs2_num = _parse_reg(rs2)
    offset13 = _sign_extend(offset, 13)
    word = (OP_CTRL << 29) | (sub << 23) | (rs1_num << 18) | (rs2_num << 13) | offset13
    return word


def encode_nop() -> int:
    return OP_NOP << 29


def encode_end() -> int:
    return OP_END << 29


# ---------------------------------------------------------------------------
# 反汇编辅助
# ---------------------------------------------------------------------------

_OP_NAME = {
    OP_NOP: "NOP", OP_ALU_I: "ALU_I", OP_ALU_R: "ALU_R", OP_ALU_RB: "ALU_RB",
    OP_MOV: "MOV", OP_ISM: "ISM", OP_CTRL: "CTRL", OP_END: "END",
}

_ALU_I_REV = {v: k for k, v in ALU_I_SUB.items()}
_ALU_R_REV = {v: k for k, v in ALU_R_SUB.items()}
_ALU_RB_REV = {v: k for k, v in ALU_RB_SUB.items()}
_MOV_REV = {v: k for k, v in MOV_SUB.items()}
_ISM_REV = {v: k for k, v in ISM_SUB.items()}
_CTRL_REV = {v: k for k, v in CTRL_SUB.items()}


def disassemble(word: int) -> str:
    """32-bit 指令字 → 可读反汇编。"""
    op = (word >> 29) & 0x7
    if op == OP_END:
        return "END"
    if op == OP_NOP:
        return "NOP"
    if op == OP_ALU_I:
        sub = (word >> 24) & 0x1F; rd = (word >> 19) & 0x1F
        rs1 = (word >> 14) & 0x1F; imm = word & 0x3FFF
        if imm & 0x2000: imm -= 0x4000  # 符号扩展
        name = _ALU_I_REV.get(sub, f"?ALU_I_{sub}")
        return f"{name} r{rd}, r{rs1}, {imm}"
    if op == OP_ALU_R:
        sub = (word >> 24) & 0x1F; rd = (word >> 19) & 0x1F
        rs1 = (word >> 14) & 0x1F; rs2 = (word >> 9) & 0x1F
        name = _ALU_R_REV.get(sub, f"?ALU_R_{sub}")
        return f"{name} r{rd}, r{rs1}, r{rs2}"
    if op == OP_ALU_RB:
        sub = (word >> 24) & 0x1F; rd = (word >> 19) & 0x1F
        rs1 = (word >> 14) & 0x1F; rs2 = (word >> 9) & 0x1F
        name = _ALU_RB_REV.get(sub, f"?ALU_RB_{sub}")
        return f"{name} r{rd}, r{rs1}, r{rs2}"
    if op == OP_MOV:
        sub = (word >> 24) & 0x1F; rd = (word >> 19) & 0x1F
        rs1 = (word >> 14) & 0x1F
        name = _MOV_REV.get(sub, f"?MOV_{sub}")
        return f"{name} r{rd}, r{rs1}"
    if op == OP_ISM:
        sub = (word >> 23) & 0x3F; rd = (word >> 18) & 0x1F
        rs1 = (word >> 13) & 0x1F; imm = word & 0x1FFF
        name = _ISM_REV.get(sub, f"?ISM_{sub}")
        return f"{name} r{rd}, r{rs1}, imm={imm}"
    if op == OP_CTRL:
        sub = (word >> 23) & 0x3F; rs1 = (word >> 18) & 0x1F
        rs2 = (word >> 13) & 0x1F; offset = word & 0x1FFF
        if offset & 0x1000: offset -= 0x2000  # 符号扩展
        name = _CTRL_REV.get(sub, f"?CTRL_{sub}")
        return f"{name} r{rs1}, r{rs2}, offset={offset}"
    return f"ILLEGAL(0x{word:08x})"


# ---------------------------------------------------------------------------
# Encoder 主入口
# ---------------------------------------------------------------------------


@dataclass
class EncoderResult:
    words: list[int] = field(default_factory=list)
    disasm: list[str] = field(default_factory=list)

    def to_bytes(self, endian: str = "little") -> bytes:
        """导出为 DPU 二进制（4 字节对齐）。"""
        order = "<I" if endian == "little" else ">I"
        return b"".join(struct.pack(order, w) for w in self.words)

    def __len__(self) -> int:
        return len(self.words)

    def summary(self) -> str:
        lines = [f"DPU Encoder Result: {len(self.words)}  words ({len(self.words)*4} bytes)"]
        for i, (w, s) in enumerate(zip(self.words, self.disasm)):
            lines.append(f"  [{i:3d}] 0x{w:08x}  {s}")
        return "\n".join(lines)


def encode_arith(opcode: str, rd: str, rs1: str, rs2=None) -> int:
    """统一入口：根据 opcode 自动分派到对应的编码函数。"""
    if opcode in ALU_I_SUB:
        if rs2 is None:
            raise TileValidationError(f"{opcode} 需要立即数 rs2")
        return encode_alu_i(opcode, rd, rs1, rs2 if isinstance(rs2, int) else int(rs2))
    if opcode in ALU_R_SUB:
        if rs2 is None:
            raise TileValidationError(f"{opcode} 需要 rs2")
        return encode_alu_r(opcode, rd, rs1, rs2)
    if opcode in ALU_RB_SUB:
        if rs2 is None:
            raise TileValidationError(f"{opcode} 需要 rs2")
        return encode_alu_rb(opcode, rd, rs1, rs2)
    if opcode in MOV_SUB:
        if rs2 is not None:
            # MOV 子类忽略第三个操作数
            pass
        return encode_mov(opcode, rd, rs1)
    raise TileValidationError(f"未知 ALU/MOV opcode: {opcode}")


def encode_items(items: tuple | list) -> EncoderResult:
    """把一组 IR 操作（ArithOp/BranchOp/EndOp 等）编码成指令字列表。"""
    result = EncoderResult()

    def _enc_recursive(ops):
        from .ir import (
            ArithOp, BranchOp, LabelOp, EndOp, MemOp, Loop,
        )
        for op in ops:
            if isinstance(op, LabelOp):
                # Label 本身不编码，只是反汇编标记
                pass
            elif isinstance(op, ArithOp):
                word = encode_arith(op.opcode, op.rd, op.rs1, op.rs2)
                result.words.append(word)
                result.disasm.append(str(op))
            elif isinstance(op, BranchOp):
                word = encode_ctrl(op.opcode, rs1=op.rs1, rs2=op.rs2, offset=0)
                # 简化版：offset 暂时写 0，链接阶段填
                result.words.append(word)
                result.disasm.append(str(op))
            elif isinstance(op, MemOp):
                word = encode_ism(op.opcode, rd=op.rd, rs1=op.addr_reg, imm=0)
                result.words.append(word)
                result.disasm.append(str(op))
            elif isinstance(op, EndOp):
                word = encode_end()
                result.words.append(word)
                result.disasm.append(str(op))
            elif isinstance(op, Loop) and op.body is not None:
                _enc_recursive(op.body)

    _enc_recursive(items)
    return result


__all__ = [
    # 常量
    "OP_NOP", "OP_ALU_I", "OP_ALU_R", "OP_ALU_RB", "OP_MOV", "OP_ISM", "OP_CTRL", "OP_END",
    "ALU_I_SUB", "ALU_R_SUB", "ALU_RB_SUB", "MOV_SUB", "ISM_SUB", "CTRL_SUB",
    # 编码函数
    "encode_alu_i", "encode_alu_r", "encode_alu_rb", "encode_mov",
    "encode_ism", "encode_ctrl", "encode_nop", "encode_end",
    "encode_arith", "encode_items",
    # 反汇编
    "disassemble",
    # 结果容器
    "EncoderResult",
]
