"""图优化 pass 的公共基础设施。

约定：
- 每个 pass 接收 Module，返回新的 Module（绝不原地修改输入）；
- pass 失败时抛异常，调用方持有的输入图保持原样，不产生部分修改状态；
- 新增节点必须继承最近的有效 source span。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from ..ir import CallNode, ConstantNode, FallbackNode, Graph, Module, Node, Value
from ..ir.nodes import _jsonable


@dataclass
class PassOutput:
    """单个 pass 的执行结果。"""

    module: Module
    removed: int = 0
    added: int = 0
    folded: int = 0
    fused: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class PassStats:
    """pass 统计，用于编译报告。"""

    name: str
    nodes_before: int
    nodes_after: int
    removed: int = 0
    added: int = 0
    folded: int = 0
    fused: int = 0
    elapsed_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "removed": self.removed,
            "added": self.added,
            "folded": self.folded,
            "fused": self.fused,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "notes": list(self.notes),
        }


def rebuild_module(module: Module, nodes: list[Node], outputs: list[Value]) -> Module:
    """用新的节点/输出构造模块，保留模块名与元数据。"""
    graph = Graph(inputs=module.graph.inputs, outputs=outputs, nodes=nodes)
    return Module(name=module.name, graph=graph, metadata=dict(module.metadata))


def resolve_name(mapping: dict[str, str], name: str) -> str:
    """解析替换链，避免 a->b->c 只替换一层。"""
    seen = set()
    while name in mapping:
        if name in seen:
            raise ValueError(f"替换映射出现环: {mapping}")
        seen.add(name)
        name = mapping[name]
    return name


def substitute_inputs(node: Node, mapping: dict[str, str]) -> Node:
    """改写节点的输入名；常量节点没有输入，原样返回。"""
    if isinstance(node, ConstantNode) or not mapping or not node.inputs:
        return node
    return replace(
        node, inputs=[resolve_name(mapping, name) for name in node.inputs]
    )


def remap_outputs(
    module: Module, mapping: dict[str, str]
) -> list[Value]:
    """把图输出中已被替换的值映射到仍然存在的值。"""
    tensors = module.graph.tensors
    outputs = []
    for value in module.graph.outputs:
        target = resolve_name(mapping, value.name)
        if target == value.name:
            outputs.append(value)
        else:
            outputs.append(tensors[target])
    return outputs


def freeze_attrs(attrs: dict) -> str:
    """属性冻结为可比较的字符串，用于 CSE 签名。"""
    return json.dumps(_jsonable(attrs), sort_keys=True, ensure_ascii=False)


def consumer_counts(module: Module) -> dict[str, int]:
    """统计每个值被引用的次数（包含图输出）。"""
    counts: dict[str, int] = {}
    for node in module.graph.nodes:
        for name in node.inputs:
            counts[name] = counts.get(name, 0) + 1
    for value in module.graph.outputs:
        counts[value.name] = counts.get(value.name, 0) + 1
    return counts


def node_kind_label(node: Node) -> str:
    if isinstance(node, CallNode):
        return node.canonical_name
    if isinstance(node, FallbackNode):
        return f"fallback:{node.op_type}"
    return node.kind
