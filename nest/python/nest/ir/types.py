"""NestGraph 张量类型。

M0 约束：
- 只支持静态 shape；符号维必须在导入阶段被显式输入规格替换，否则报错。
- dtype、layout 必须显式给出，不允许隐式推断为某个默认值。
- 量化张量（int8/uint8/int16/uint16）必须携带 QuantInfo；浮点张量用 quant=None 表示未量化。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..errors import GraphValidationError

# 量化计算域中的整数 dtype：这类张量必须带 QuantInfo。
QUANTIZED_DTYPES = frozenset({"int8", "uint8", "int16", "uint16"})

# M0 支持的 dtype。M1 起再按后端能力扩展。
SUPPORTED_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "int64",
        "bool",
    }
)

# 通用张量的显式 layout。空间算子使用 NCHW / OIHW 等具名 layout。
LINEAR_LAYOUT = "linear"


@dataclass(frozen=True)
class SourceSpan:
    """节点来源，用于错误定位与精度归因。

    ONNX 没有统一的行号信息，``line`` 可以为 None；
    ``node_name`` / ``op_name`` 仍必须保留，保证错误可定位。
    """

    file: str
    node_name: str | None = None
    op_name: str | None = None
    line: int | None = None

    def __str__(self) -> str:
        parts = [self.file]
        if self.node_name:
            parts.append(str(self.node_name))
        if self.op_name:
            parts.append(str(self.op_name))
        if self.line is not None:
            parts.append(f"line {self.line}")
        return ":".join(parts)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "node_name": self.node_name,
            "op_name": self.op_name,
            "line": self.line,
        }

    @staticmethod
    def from_dict(data: dict) -> "SourceSpan":
        return SourceSpan(
            file=data["file"],
            node_name=data.get("node_name"),
            op_name=data.get("op_name"),
            line=data.get("line"),
        )


@dataclass(frozen=True)
class QuantInfo:
    """量化元数据。

    M0 只要求字段完整可表达，不执行 PTQ。字段不允许被静默默认：
    构造时 mode 必填，量化张量必须给出 ``bits`` 与 ``axis``。
    """

    mode: str  # "float" / "int" / "fake_quant"
    bits: int = 32
    scale: tuple[float, ...] | None = None
    zero_point: tuple[int, ...] | None = None
    axis: int | None = None
    rounding: str = "nearest_even"
    saturation: bool = True
    accum_bits: int | None = None
    # per-block 量化时的块长度；None 表示 per-tensor / per-channel。
    block_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("float", "int", "fake_quant"):
            raise GraphValidationError(f"非法量化模式: {self.mode}")
        if self.bits <= 0:
            raise GraphValidationError(f"非法量化位宽: {self.bits}")
        if self.mode == "int":
            if self.scale is None:
                raise GraphValidationError("量化张量必须给出 scale")
            if self.axis is None:
                raise GraphValidationError("量化张量必须给出 per-tensor/per-channel 的 axis")
            if self.zero_point is None:
                raise GraphValidationError("量化张量必须给出 zero_point")
        if self.rounding not in ("nearest_even", "nearest_away", "truncate"):
            raise GraphValidationError(f"非法舍入模式: {self.rounding}")
        if self.block_size is not None and self.block_size <= 0:
            raise GraphValidationError(
                f"非法 per-block 块长度: {self.block_size}"
            )

    @staticmethod
    def float(dtype: str = "float32") -> "QuantInfo":
        """浮点计算域，显式表示“未量化”。"""
        return QuantInfo(mode="float", bits=16 if dtype == "float16" else 32)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "bits": self.bits,
            "scale": list(self.scale) if self.scale is not None else None,
            "zero_point": list(self.zero_point) if self.zero_point is not None else None,
            "axis": self.axis,
            "rounding": self.rounding,
            "saturation": self.saturation,
            "accum_bits": self.accum_bits,
            "block_size": self.block_size,
        }

    @staticmethod
    def from_dict(data: dict | None) -> "QuantInfo | None":
        if data is None:
            return None
        return QuantInfo(
            mode=data["mode"],
            bits=data.get("bits", 32),
            scale=tuple(data["scale"]) if data.get("scale") is not None else None,
            zero_point=(
                tuple(data["zero_point"]) if data.get("zero_point") is not None else None
            ),
            axis=data.get("axis"),
            rounding=data.get("rounding", "nearest_even"),
            saturation=data.get("saturation", True),
            accum_bits=data.get("accum_bits"),
            block_size=data.get("block_size"),
        )


@dataclass(frozen=True)
class TensorType:
    """NestGraph 张量类型。"""

    dtype: str
    shape: tuple[int, ...]
    layout: str
    quant: QuantInfo | None = None
    memory_hint: str | None = None
    # 扩展元数据：不参与语义，只用于前端/调试保留信息。
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.dtype:
            raise GraphValidationError("TensorType.dtype 不能为空")
        if self.dtype not in SUPPORTED_DTYPES:
            raise GraphValidationError(f"M0 不支持 dtype: {self.dtype}")
        if not self.layout:
            raise GraphValidationError(
                "TensorType.layout 必须显式给出（通用张量使用 'linear'）"
            )
        if len(self.shape) > 8:
            raise GraphValidationError(f"不支持的张量秩: {len(self.shape)}")
        for dim in self.shape:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise GraphValidationError(
                    f"M0 只支持静态 shape，得到非法维度: {dim!r}"
                )
            if dim < 0:
                raise GraphValidationError(f"张量维度不能为负: {dim}")
        if self.dtype in QUANTIZED_DTYPES and self.quant is None:
            raise GraphValidationError(
                f"量化 dtype {self.dtype} 必须显式携带 QuantInfo"
            )
        if self.quant is not None and self.quant.mode == "int" and (
            self.dtype not in QUANTIZED_DTYPES
        ):
            raise GraphValidationError(
                f"int 量化标注只允许用于量化 dtype，当前 dtype={self.dtype}"
            )

    @property
    def is_quantized(self) -> bool:
        return self.quant is not None and self.quant.mode == "int"

    def with_shape(self, shape: tuple[int, ...]) -> "TensorType":
        return replace(self, shape=tuple(shape))

    def with_layout(self, layout: str) -> "TensorType":
        return replace(self, layout=layout)

    def to_dict(self) -> dict:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "quant": self.quant.to_dict() if self.quant is not None else None,
            "memory_hint": self.memory_hint,
            "extra": {k: v for k, v in self.extra},
        }

    @staticmethod
    def from_dict(data: dict) -> "TensorType":
        return TensorType(
            dtype=data["dtype"],
            shape=tuple(data["shape"]),
            layout=data["layout"],
            quant=QuantInfo.from_dict(data.get("quant")),
            memory_hint=data.get("memory_hint"),
            extra=tuple(sorted(data.get("extra", {}).items())),
        )

    def __str__(self) -> str:
        shape = ",".join(str(d) for d in self.shape)
        quant = ""
        if self.is_quantized:
            quant = f",q{self.quant.bits}"
        return f"{self.dtype}[{shape}]({self.layout}{quant})"
