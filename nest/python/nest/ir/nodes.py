"""NestGraph 节点定义。

M0 节点类型：
- ConstantNode：权重 / 常量张量。
- CallNode：可映射到 NestGraph 的算子调用（standard 或 custom）。
- FallbackNode：无法映射到 NPU、需要在标量核执行的算子。

所有节点必须携带 source span；fallback 节点还必须显式携带 backend 与原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import GraphValidationError
from .types import SourceSpan, TensorType


def _jsonable(value):
    """把属性值转换为可 JSON 序列化的结构。"""
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _from_jsonable(value):
    if isinstance(value, dict):
        if "__ndarray__" in value:
            payload = value["__ndarray__"]
            return np.asarray(payload["data"], dtype=payload["dtype"]).reshape(
                payload["shape"]
            )
        if "__bytes__" in value:
            return bytes.fromhex(value["__bytes__"])
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class Value:
    """SSA 值：张量名 + 类型。"""

    name: str
    type: TensorType

    def __post_init__(self) -> None:
        if not self.name:
            raise GraphValidationError("Value.name 不能为空")

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type.to_dict()}

    @staticmethod
    def from_dict(data: dict) -> "Value":
        return Value(name=data["name"], type=TensorType.from_dict(data["type"]))


@dataclass
class Node:
    """节点基类。"""

    inputs: list[str]
    outputs: list[Value]
    span: SourceSpan
    attrs: dict = field(default_factory=dict)

    kind: str = "node"

    def __post_init__(self) -> None:
        if not self.outputs:
            raise GraphValidationError(f"{type(self).__name__} 必须至少有一个输出")
        seen = set()
        for out in self.outputs:
            if out.name in seen:
                raise GraphValidationError(f"节点输出名重复: {out.name}")
            seen.add(out.name)
        if not isinstance(self.span, SourceSpan):
            raise GraphValidationError(f"{type(self).__name__} 缺少 SourceSpan")

    @property
    def output_names(self) -> list[str]:
        return [v.name for v in self.outputs]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "inputs": list(self.inputs),
            "outputs": [v.to_dict() for v in self.outputs],
            "span": self.span.to_dict(),
            "attrs": _jsonable(self.attrs),
        }


@dataclass
class ConstantNode(Node):
    """常量张量节点。"""

    value: np.ndarray | None = None
    kind: str = "constant"

    def __post_init__(self) -> None:
        if self.inputs:
            raise GraphValidationError("ConstantNode 不能有输入")
        if self.value is None:
            raise GraphValidationError("ConstantNode 缺少 value")
        super().__post_init__()
        self.value = np.asarray(self.value)
        if len(self.outputs) != 1:
            raise GraphValidationError("M0 ConstantNode 只支持单输出")
        out = self.outputs[0]
        if tuple(self.value.shape) != out.type.shape:
            raise GraphValidationError(
                f"常量 {out.name} 的实际 shape {self.value.shape} "
                f"与声明 {out.type.shape} 不一致"
            )
        if str(self.value.dtype) != out.type.dtype:
            raise GraphValidationError(
                f"常量 {out.name} 的 dtype {self.value.dtype} "
                f"与声明 {out.type.dtype} 不一致"
            )

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["value"] = _jsonable(self.value)
        return data

    @staticmethod
    def from_dict(data: dict) -> "ConstantNode":
        return ConstantNode(
            inputs=[],
            outputs=[Value.from_dict(v) for v in data["outputs"]],
            span=SourceSpan.from_dict(data["span"]),
            attrs=_from_jsonable(data.get("attrs", {})),
            value=_from_jsonable(data["value"]),
        )


@dataclass
class CallNode(Node):
    """可映射算子调用。"""

    domain: str = ""
    op_type: str = ""
    canonical_name: str = ""
    kind: str = "call"

    def __post_init__(self) -> None:
        if not self.op_type:
            raise GraphValidationError("CallNode 缺少 op_type")
        if not self.canonical_name:
            raise GraphValidationError(f"CallNode {self.op_type} 缺少 canonical_name")
        super().__post_init__()

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "domain": self.domain,
                "op_type": self.op_type,
                "canonical_name": self.canonical_name,
            }
        )
        return data

    @staticmethod
    def from_dict(data: dict) -> "CallNode":
        return CallNode(
            inputs=list(data["inputs"]),
            outputs=[Value.from_dict(v) for v in data["outputs"]],
            span=SourceSpan.from_dict(data["span"]),
            attrs=_from_jsonable(data.get("attrs", {})),
            domain=data.get("domain", ""),
            op_type=data["op_type"],
            canonical_name=data["canonical_name"],
        )


@dataclass
class FallbackNode(Node):
    """无法映射到 NPU 的算子。

    ``backend`` 必须是显式的标量执行后端；``reason`` 必须说明为什么无法映射。
    """

    domain: str = ""
    op_type: str = ""
    backend: str = ""
    reason: str = ""
    kind: str = "fallback"

    def __post_init__(self) -> None:
        if not self.op_type:
            raise GraphValidationError("FallbackNode 缺少 op_type")
        if not self.backend:
            raise GraphValidationError(
                f"FallbackNode {self.op_type} 必须显式指定 backend"
            )
        if not self.reason:
            raise GraphValidationError(
                f"FallbackNode {self.op_type} 必须显式给出 fallback 原因"
            )
        super().__post_init__()

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "domain": self.domain,
                "op_type": self.op_type,
                "backend": self.backend,
                "reason": self.reason,
            }
        )
        return data

    @staticmethod
    def from_dict(data: dict) -> "FallbackNode":
        return FallbackNode(
            inputs=list(data["inputs"]),
            outputs=[Value.from_dict(v) for v in data["outputs"]],
            span=SourceSpan.from_dict(data["span"]),
            attrs=_from_jsonable(data.get("attrs", {})),
            domain=data.get("domain", ""),
            op_type=data["op_type"],
            backend=data["backend"],
            reason=data["reason"],
        )


NODE_TYPES = {
    "constant": ConstantNode,
    "call": CallNode,
    "fallback": FallbackNode,
}


def node_from_dict(data: dict) -> Node:
    kind = data.get("kind")
    if kind not in NODE_TYPES:
        raise GraphValidationError(f"未知节点类型: {kind}")
    return NODE_TYPES[kind].from_dict(data)
