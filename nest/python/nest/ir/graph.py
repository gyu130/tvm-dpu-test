"""NestGraph 图与模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import GraphValidationError
from .nodes import CallNode, ConstantNode, FallbackNode, Node, Value, node_from_dict


@dataclass
class Graph:
    """SSA 形式的 NestGraph。

    - ``inputs`` / ``outputs`` 为 Value；
    - ``nodes`` 必须已按拓扑序排列；
    - 所有节点输入必须引用已定义的值。
    """

    inputs: list[Value]
    outputs: list[Value]
    nodes: list[Node]
    _tensors: dict[str, Value] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def tensors(self) -> dict[str, Value]:
        return dict(self._tensors)

    def validate(self) -> None:
        self._tensors = {}
        for value in self.inputs:
            if value.name in self._tensors:
                raise GraphValidationError(f"重复的图输入: {value.name}")
            self._tensors[value.name] = value

        fallback_count = 0
        for index, node in enumerate(self.nodes):
            for name in node.inputs:
                if name not in self._tensors:
                    raise GraphValidationError(
                        f"节点 #{index} ({node.kind}) 引用了未定义的值: {name}"
                    )
            for value in node.outputs:
                if value.name in self._tensors:
                    raise GraphValidationError(f"SSA 冲突，值被重复定义: {value.name}")
                self._tensors[value.name] = value
            if isinstance(node, ConstantNode):
                pass
            elif isinstance(node, CallNode):
                if not node.canonical_name:
                    raise GraphValidationError(f"节点 #{index} 缺少规范算子名")
            elif isinstance(node, FallbackNode):
                fallback_count += 1
            else:
                raise GraphValidationError(f"节点 #{index} 类型未知: {type(node)}")

        if not self.outputs:
            raise GraphValidationError("图缺少输出")
        for value in self.outputs:
            if value.name not in self._tensors:
                raise GraphValidationError(f"图输出未定义: {value.name}")

        self.fallback_count = fallback_count

    def node_map(self) -> dict[str, Node]:
        """值名 -> 生产它的节点。"""
        mapping: dict[str, Node] = {}
        for node in self.nodes:
            for name in node.output_names:
                mapping[name] = node
        return mapping

    def to_dict(self) -> dict:
        return {
            "inputs": [v.to_dict() for v in self.inputs],
            "outputs": [v.to_dict() for v in self.outputs],
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @staticmethod
    def from_dict(data: dict) -> "Graph":
        return Graph(
            inputs=[Value.from_dict(v) for v in data["inputs"]],
            outputs=[Value.from_dict(v) for v in data["outputs"]],
            nodes=[node_from_dict(n) for n in data["nodes"]],
        )


@dataclass
class Module:
    """NestCompiler 模块：一张图 + 模块级元数据。"""

    name: str
    graph: Graph
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise GraphValidationError("Module.name 不能为空")
        self.graph.validate()

    @property
    def fallback_nodes(self) -> list[FallbackNode]:
        return [n for n in self.graph.nodes if isinstance(n, FallbackNode)]

    def to_dict(self) -> dict:
        return {
            "format": "nest.module.v0",
            "name": self.name,
            "metadata": self.metadata,
            "graph": self.graph.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def from_dict(data: dict) -> "Module":
        if data.get("format") != "nest.module.v0":
            raise GraphValidationError(
                f"不支持的模块格式: {data.get('format')!r}"
            )
        return Module(
            name=data["name"],
            graph=Graph.from_dict(data["graph"]),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def load(path: str | Path) -> "Module":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Module.from_dict(data)

    def __str__(self) -> str:
        from .printer import print_module

        return print_module(self)
