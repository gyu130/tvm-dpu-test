"""NestGraph 文本打印，便于 diff 与人工检查。"""

from __future__ import annotations

from .graph import Module
from .nodes import CallNode, ConstantNode, FallbackNode, Node
from .types import TensorType


def _fmt_type(t: TensorType) -> str:
    return str(t)


def _fmt_attrs(attrs: dict) -> str:
    if not attrs:
        return ""
    items = []
    for key in sorted(attrs):
        value = attrs[key]
        if isinstance(value, list) and len(value) > 8:
            value = f"[{len(value)} elems]"
        items.append(f"{key}={value}")
    return " {" + ", ".join(items) + "}"


def _fmt_node(node: Node) -> str:
    outs = ", ".join(f"%{v.name}: {_fmt_type(v.type)}" for v in node.outputs)
    ins = ", ".join(f"%{name}" for name in node.inputs)
    if isinstance(node, ConstantNode):
        return (
            f"  {outs} = constant(dtype={node.value.dtype}, "
            f"shape={tuple(node.value.shape)})"
        )
    if isinstance(node, CallNode):
        op = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        return f"  {outs} = nest.{node.canonical_name}({ins})  // {op}{_fmt_attrs(node.attrs)}"
    if isinstance(node, FallbackNode):
        op = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        return (
            f"  {outs} = fallback[{node.backend}]({ins})  "
            f"// {op}, reason={node.reason}{_fmt_attrs(node.attrs)}"
        )
    return f"  {outs} = {node.kind}({ins})"


def print_module(module: Module) -> str:
    graph = module.graph
    lines = [f"module @{module.name} {{"]
    for key in sorted(module.metadata):
        lines.append(f"  // {key}: {module.metadata[key]}")
    lines.append("  graph {")
    for value in graph.inputs:
        lines.append(f"    input %{value.name}: {_fmt_type(value.type)}")
    for node in graph.nodes:
        lines.append(_fmt_node(node))
    for value in graph.outputs:
        lines.append(f"    output %{value.name}: {_fmt_type(value.type)}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
