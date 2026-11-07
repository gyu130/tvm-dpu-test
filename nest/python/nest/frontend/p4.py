"""P4 前端：将 P4 风格的报文处理程序转换为 TIR PrimFunc。

设计目标
--------
P4 的三阶段流水线（parser / match-action / deparser）与 TIR 的命令式模型天然对齐：
- parser   ->  从输入 pkt buffer 按偏移提取字段（BufferLoad）
- control  ->  条件分支 + 算术/位运算（IfThenElse + PrimExpr）
- deparser ->  将修改后的字段写回报文 buffer（BufferStore）

本模块提供一套声明式 API 描述 P4 程序，并在 ``to_tir`` 中生成 TVMScript
源码、经 TVM 脚本解析器产出 ``tirx.PrimFunc``，可直接送入 DPU 后端编译。

支持的 P4 子集
--------------
- 字节对齐的 header 字段（bit 宽度为 8 的倍数）
- parser：按序 extract + transition select（精确匹配 + default）
- table：单字段精确匹配，entries 列表 + default action
- action：字段赋值（SetField）、计数器累加（IncrementCounter）
- 表达式：FieldRef / IntConst / Add / Sub / And / Or / Shl / Shr
- deparser：按序 emit header
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tvm
from tvm.script import from_source


class Expr:
    """动作体表达式基类，支持运算符重载。"""

    def __add__(self, other): return Add(self, _wrap(other))
    def __sub__(self, other): return Sub(self, _wrap(other))
    def __and__(self, other): return And(self, _wrap(other))
    def __or__(self, other): return Or(self, _wrap(other))
    def __lshift__(self, other): return Shl(self, _wrap(other))
    def __rshift__(self, other): return Shr(self, _wrap(other))


def _wrap(x) -> "Expr":
    if isinstance(x, Expr):
        return x
    if isinstance(x, int):
        return IntConst(x)
    raise TypeError(f"不支持的表达式操作数: {x!r}")


# ---------------------------------------------------------------------------
# 数据模型：header / field
# ---------------------------------------------------------------------------

@dataclass
class Field:
    """一个 header 字段，bit_width 必须是 8 的倍数（字节对齐）。"""

    name: str
    bit_width: int

    @property
    def byte_width(self) -> int:
        assert self.bit_width % 8 == 0, f"字段 {self.name} 必须字节对齐"
        return self.bit_width // 8


@dataclass
class Header:
    """一个协议头，字段按网络字节序（大端）排列。"""

    name: str
    fields: list[Field]

    def __post_init__(self) -> None:
        self._field_offsets: dict[str, int] = {}
        offset = 0
        for f in self.fields:
            self._field_offsets[f.name] = offset
            offset += f.byte_width
        self.byte_width = offset

    def __getattr__(self, item: str) -> "FieldRef":
        # 仅在 fields 中查找，避免干扰 dataclass 自动属性
        fields = self.__dict__.get("fields")
        if fields is None:
            raise AttributeError(item)
        for f in fields:
            if f.name == item:
                return FieldRef(self, f)
        raise AttributeError(f"Header {self.name} 无字段 {item}")

    def field_offset(self, field_name: str) -> int:
        return self._field_offsets[field_name]


@dataclass
class FieldRef(Expr):
    """对某个 header 字段的引用，用于表达式和赋值。"""

    header: Header
    field: Field

    @property
    def byte_offset(self) -> int:
        # 绝对字节偏移 = header 在报文中的偏移 + 字段在 header 内的偏移
        return self.header.pkt_offset + self.header.field_offset(self.field.name)

    @property
    def byte_width(self) -> int:
        return self.field.byte_width


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

@dataclass
class State:
    """parser 状态：extract 一个 header，然后 transition select。"""

    name: str
    extract: Header | None = None
    transitions: dict[int, str] = field(default_factory=dict)  # key(字段值) -> next state
    default: str = "accept"  # 默认下一状态名或 "accept"/"reject"


@dataclass
class Parser:
    states: list[State]
    start: str = "start"


# ---------------------------------------------------------------------------
# 动作表达式与动作语句
# ---------------------------------------------------------------------------

@dataclass
class IntConst(Expr):
    value: int


@dataclass
class Add(Expr): a: Expr; b: Expr
@dataclass
class Sub(Expr): a: Expr; b: Expr
@dataclass
class And(Expr): a: Expr; b: Expr
@dataclass
class Or(Expr): a: Expr; b: Expr
@dataclass
class Shl(Expr): a: Expr; b: Expr
@dataclass
class Shr(Expr): a: Expr; b: Expr


class Stmt:
    """动作体语句基类。"""


@dataclass
class SetField(Stmt):
    target: FieldRef
    value: Expr


@dataclass
class IncrementCounter(Stmt):
    counter: str
    amount: int = 1


# ---------------------------------------------------------------------------
# Table / Control / Deparser
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    """表项：匹配值 -> 动作体（语句列表）。"""

    key: int
    action: list[Stmt]


@dataclass
class Table:
    name: str
    key: FieldRef
    entries: list[Entry]
    default: list[Stmt] | None = None  # None 表示空动作


@dataclass
class Control:
    tables: list[Table]


@dataclass
class Deparser:
    emit: list[Header]


# ---------------------------------------------------------------------------
# P4 程序
# ---------------------------------------------------------------------------

class P4Program:
    """一个完整的 P4 程序，可编译为 TIR PrimFunc。"""

    def __init__(
        self,
        *,
        headers: list[Header],
        parser: Parser,
        control: Control,
        deparser: Deparser,
        pkt_len: int = 1500,
        counters: dict[str, int] | None = None,
    ) -> None:
        self.headers = headers
        self.parser = parser
        self.control = control
        self.deparser = deparser
        self.pkt_len = pkt_len
        self.counters = counters or {}
        # 为每个 header 计算其在报文中的绝对字节偏移
        offset = 0
        for hdr in headers:
            hdr.pkt_offset = offset
            offset += hdr.byte_width
        self.total_hdr_bytes = offset
        # 计数器索引表
        self.counter_index = {name: i for i, name in enumerate(self.counters)}

    # -- 表达式 -> TVMScript 字符串 ------------------------------------------

    def _emit_expr(self, e: Expr) -> str:
        if isinstance(e, IntConst):
            return str(e.value)
        if isinstance(e, FieldRef):
            return self._emit_field_read(e)
        if isinstance(e, Add):
            return f"({self._emit_expr(e.a)} + {self._emit_expr(e.b)})"
        if isinstance(e, Sub):
            return f"({self._emit_expr(e.a)} - {self._emit_expr(e.b)})"
        if isinstance(e, And):
            return f"T.bitwise_and({self._emit_expr(e.a)}, {self._emit_expr(e.b)})"
        if isinstance(e, Or):
            return f"T.bitwise_or({self._emit_expr(e.a)}, {self._emit_expr(e.b)})"
        if isinstance(e, Shl):
            return f"T.shift_left({self._emit_expr(e.a)}, {self._emit_expr(e.b)})"
        if isinstance(e, Shr):
            return f"T.shift_right({self._emit_expr(e.a)}, {self._emit_expr(e.b)})"
        raise TypeError(f"未知表达式: {e!r}")

    def _emit_field_read(self, ref: FieldRef) -> str:
        """以大端序读字段，返回 int32 表达式。"""
        off = ref.byte_offset
        n = ref.byte_width
        parts = []
        for i in range(n):
            shift = 8 * (n - 1 - i)
            parts.append(f"T.shift_left(T.cast(hdr[{off + i}], \"int32\"), {shift})")
        # 用 T.bitwise_or 组合（TVMScript 不重载 | 运算符）
        expr = parts[0]
        for p in parts[1:]:
            expr = f"T.bitwise_or({expr}, {p})"
        return f"({expr})"

    def _emit_field_write(self, ref: FieldRef, val_expr: str) -> list[str]:
        """将 int32 表达式以大端序写入字段。"""
        off = ref.byte_offset
        n = ref.byte_width
        stmts = []
        for i in range(n):
            shift = 8 * (n - 1 - i)
            if i == n - 1:
                byte_expr = f"T.cast(({val_expr}) & 0xFF, \"uint8\")"
            else:
                byte_expr = (
                    f"T.cast(T.bitwise_and(T.shift_right({val_expr}, {shift}), 0xFF), \"uint8\")"
                )
            stmts.append(f"hdr[{off + i}] = {byte_expr}")
        return stmts

    # -- 动作体 -> TVMScript 语句 --------------------------------------------

    def _emit_action(self, stmts: list[Stmt]) -> list[str]:
        out: list[str] = []
        for s in stmts:
            if isinstance(s, SetField):
                val = self._emit_expr(s.value)
                out.extend(self._emit_field_write(s.target, val))
            elif isinstance(s, IncrementCounter):
                idx = self.counter_index[s.counter]
                out.append(f"counters[{idx}] = counters[{idx}] + {s.amount}")
            else:
                raise TypeError(f"未知语句: {s!r}")
        return out

    # -- 生成完整 TVMScript --------------------------------------------------

    def to_tir_source(self, func_name: str = "p4_main") -> str:
        lines: list[str] = []
        lines.append("@T.prim_func")
        num_counters = len(self.counters)
        lines.append(
            f"def {func_name}(pkt: T.Buffer(({self.pkt_len},), \"uint8\"), "
            f"counters: T.Buffer(({num_counters},), \"int32\")):"
        )
        # 局部 header scratch buffer（解析后字段存于此，动作修改，deparser 写回）
        lines.append(f"    hdr = T.alloc_buffer(({self.total_hdr_bytes},), \"uint8\")")

        # ---- parser: 把所有声明的 header 字节从 pkt 拷到 hdr ----
        # （简化：按序拷贝所有 header 区域；真实 parser 按路径 extract，此处
        #   假设报文包含全部声明头，与常见单栈协议场景一致。）
        if self.total_hdr_bytes > 0:
            lines.append("    for i in T.serial(%d):" % self.total_hdr_bytes)
            lines.append("        hdr[i] = pkt[i]")

        # ---- match-action: 逐表生成 if-elif 链 ----
        for table in self.control.tables:
            key_expr = self._emit_field_read(table.key)
            first = True
            for entry in table.entries:
                kw = "if" if first else "elif"
                first = False
                lines.append(f"    {kw} ({key_expr}) == {entry.key}:")
                body = self._emit_action(entry.action)
                if body:
                    for b in body:
                        lines.append(f"        {b}")
                else:
                    lines.append("        pass")
            if table.default is not None:
                lines.append("    else:")
                body = self._emit_action(table.default)
                if body:
                    for b in body:
                        lines.append(f"        {b}")
                else:
                    lines.append("        pass")

        # ---- deparser: 把 hdr 写回 pkt ----
        if self.total_hdr_bytes > 0:
            lines.append("    for i in T.serial(%d):" % self.total_hdr_bytes)
            lines.append("        pkt[i] = hdr[i]")

        return "\n".join(lines)

    def to_tir(self, func_name: str = "p4_main"):
        """编译为 TIR PrimFunc。"""
        src = self.to_tir_source(func_name=func_name)
        # from_source 解析 TVMScript 源码
        parsed = from_source(src)
        # from_source 对单个 @T.prim_func 返回 PrimFunc
        return parsed


# ---------------------------------------------------------------------------
# 便捷构造：经典 IPv4 TTL 递减 + 计数
# ---------------------------------------------------------------------------

def build_ipv4_ttl_decrement(pkt_len: int = 1500) -> P4Program:
    """构造一个经典 P4 程序：解析以太网/IPv4，递减 TTL，累加计数，写回报文。"""
    eth = Header("ethernet", [
        Field("dst_addr", 48),
        Field("src_addr", 48),
        Field("ethertype", 16),
    ])
    ipv4 = Header("ipv4", [
        Field("version_ihl", 8),
        Field("tos", 8),
        Field("total_len", 16),
        Field("id", 16),
        Field("flags_frag", 16),
        Field("ttl", 8),
        Field("protocol", 8),
        Field("checksum", 16),
        Field("src_addr", 32),
        Field("dst_addr", 32),
    ])

    parser = Parser([
        State("start", extract=eth, transitions={0x0800: "parse_ipv4"}, default="accept"),
        State("parse_ipv4", extract=ipv4, default="accept"),
    ])

    ttl_decrement_action = [
        SetField(ipv4.ttl, ipv4.ttl - 1),
        IncrementCounter("ipv4_counter", 1),
    ]

    table = Table(
        name="ipv4_ttl_table",
        key=eth.ethertype,
        entries=[Entry(0x0800, ttl_decrement_action)],
        default=None,
    )

    return P4Program(
        headers=[eth, ipv4],
        parser=parser,
        control=Control([table]),
        deparser=Deparser([eth, ipv4]),
        pkt_len=pkt_len,
        counters={"ipv4_counter": 0},
    )
