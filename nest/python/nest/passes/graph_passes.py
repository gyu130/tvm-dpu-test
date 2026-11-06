"""NestGraph 基础优化 pass。

包含 canonicalize、常量折叠、DCE、CSE、代数简化与 layout 校验。
全部 pass 均保持 source span，且重复执行结果稳定（幂等）。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..errors import GraphValidationError, PassError
from ..ir import (
    LINEAR_LAYOUT,
    CallNode,
    ConstantNode,
    Module,
    Node,
    Value,
)
from ..registry import GLOBAL_REGISTRY, OpRegistry
from .base import (
    PassOutput,
    freeze_attrs,
    rebuild_module,
    remap_outputs,
    resolve_name,
    substitute_inputs,
)

_UNARY_CANONICAL = frozenset(
    {"relu", "sigmoid", "tanh", "exp", "sqrt", "abs", "neg", "cos"}
)
_BINARY_CANONICAL = frozenset({"add", "sub", "mul", "div"})


def _kernel_outputs(raw, node: Node) -> list[np.ndarray]:
    if isinstance(raw, np.ndarray):
        outputs = [raw]
    elif isinstance(raw, (list, tuple)):
        outputs = [np.asarray(item) for item in raw]
    else:
        raise PassError(
            f"常量折叠时 {node.op_type} 的 kernel 返回非法类型: {type(raw)}"
        )
    if len(outputs) != len(node.outputs):
        raise PassError(
            f"常量折叠时节点 {node.output_names} 的 kernel 返回 {len(outputs)} "
            f"个张量，但声明 {len(node.outputs)} 个输出"
        )
    for value, array in zip(node.outputs, outputs):
        if tuple(array.shape) != value.type.shape:
            raise PassError(
                f"常量折叠结果 {value.name} shape {array.shape} "
                f"与声明 {value.type.shape} 不一致"
            )
        if str(array.dtype) != value.type.dtype:
            raise PassError(
                f"常量折叠结果 {value.name} dtype {array.dtype} "
                f"与声明 {value.type.dtype} 不一致"
            )
    return outputs


# ---------------------------------------------------------------------------
# canonicalize：消除 Identity，统一表示
# ---------------------------------------------------------------------------


def canonicalize(module: Module, options=None) -> PassOutput:
    mapping: dict[str, str] = {}
    nodes: list[Node] = []
    removed = 0
    for node in module.graph.nodes:
        if isinstance(node, CallNode) and node.canonical_name == "identity":
            src = resolve_name(mapping, node.inputs[0])
            for out in node.outputs:
                mapping[out.name] = src
            removed += 1
            continue
        nodes.append(substitute_inputs(node, mapping))
    outputs = remap_outputs(module, mapping)
    return PassOutput(
        rebuild_module(module, nodes, outputs),
        removed=removed,
        notes=[f"消除 Identity: {removed}"] if removed else [],
    )


# ---------------------------------------------------------------------------
# constant_fold：仅折叠 Call，fallback 永不折叠
# ---------------------------------------------------------------------------


def constant_fold(
    module: Module, options=None, registry: OpRegistry = GLOBAL_REGISTRY
) -> PassOutput:
    constants: dict[str, np.ndarray] = {}
    nodes: list[Node] = []
    folded = 0
    notes: list[str] = []
    for node in module.graph.nodes:
        if isinstance(node, ConstantNode):
            constants[node.outputs[0].name] = np.asarray(node.value)
            nodes.append(node)
            continue
        if isinstance(node, CallNode) and all(n in constants for n in node.inputs):
            if len(node.outputs) != 1:
                # ConstantNode 目前只支持单输出；保留调用节点，交由后续阶段处理。
                notes.append(f"跳过多输出常量折叠: {node.op_type}")
                nodes.append(node)
                continue
            spec = registry.lookup(node.domain, node.op_type)
            if spec is None or spec.kernel is None:
                notes.append(
                    f"跳过折叠（无 kernel 或未注册）: {node.op_type}"
                )
                nodes.append(node)
                continue
            raw = spec.kernel(
                [constants[name] for name in node.inputs], dict(node.attrs)
            )
            array = _kernel_outputs(raw, node)[0]
            constants[node.outputs[0].name] = array
            nodes.append(
                ConstantNode(
                    inputs=[],
                    outputs=node.outputs,
                    span=node.span,
                    attrs=dict(node.attrs),
                    value=array,
                )
            )
            folded += 1
            continue
        nodes.append(node)
    return PassOutput(
        rebuild_module(module, nodes, module.graph.outputs),
        removed=folded,
        added=folded,
        folded=folded,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DCE
# ---------------------------------------------------------------------------


def dce(module: Module, options=None) -> PassOutput:
    live = {value.name for value in module.graph.outputs}
    kept_reversed: list[Node] = []
    for node in reversed(module.graph.nodes):
        if any(name in live for name in node.output_names):
            kept_reversed.append(node)
            live.update(node.inputs)
    nodes = list(reversed(kept_reversed))
    removed = len(module.graph.nodes) - len(nodes)
    return PassOutput(
        rebuild_module(module, nodes, module.graph.outputs),
        removed=removed,
        notes=[f"删除死节点: {removed}"] if removed else [],
    )


# ---------------------------------------------------------------------------
# CSE
# ---------------------------------------------------------------------------


def _call_signature(node: CallNode) -> tuple:
    return (
        node.canonical_name,
        node.domain,
        node.op_type,
        tuple(node.inputs),
        freeze_attrs(node.attrs),
        tuple(str(value.type) for value in node.outputs),
    )


def cse(module: Module, options=None) -> PassOutput:
    seen: dict[tuple, CallNode] = {}
    mapping: dict[str, str] = {}
    nodes: list[Node] = []
    removed = 0
    for node in module.graph.nodes:
        rewritten = substitute_inputs(node, mapping)
        if isinstance(rewritten, CallNode) and not isinstance(rewritten, ConstantNode):
            signature = _call_signature(rewritten)
            previous = seen.get(signature)
            if previous is not None:
                for old, kept in zip(rewritten.outputs, previous.outputs):
                    mapping[old.name] = kept.name
                removed += 1
                continue
            seen[signature] = rewritten
        nodes.append(rewritten)
    outputs = remap_outputs(module, mapping)
    return PassOutput(
        rebuild_module(module, nodes, outputs),
        removed=removed,
        notes=[f"合并冗余计算: {removed}"] if removed else [],
    )


# ---------------------------------------------------------------------------
# 代数简化：Add 0 / Sub 0 / Mul 1 / Div 1
# ---------------------------------------------------------------------------


def _is_zero(array: np.ndarray) -> bool:
    return bool(np.all(array == 0))


def _is_one(array: np.ndarray) -> bool:
    return bool(np.all(array == 1))


def simplify(module: Module, options=None) -> PassOutput:
    constants = {
        node.outputs[0].name: np.asarray(node.value)
        for node in module.graph.nodes
        if isinstance(node, ConstantNode)
    }
    tensors = module.graph.tensors
    mapping: dict[str, str] = {}
    nodes: list[Node] = []
    removed = 0

    for node in module.graph.nodes:
        rewritten = substitute_inputs(node, mapping)
        if isinstance(rewritten, CallNode) and len(rewritten.outputs) == 1:
            identities = _simplify_candidate(rewritten, constants, tensors)
            if identities is not None:
                mapping[rewritten.outputs[0].name] = identities
                removed += 1
                continue
        nodes.append(rewritten)

    outputs = remap_outputs(module, mapping)
    return PassOutput(
        rebuild_module(module, nodes, outputs),
        removed=removed,
        notes=[f"代数简化: {removed}"] if removed else [],
    )


def _simplify_candidate(
    node: CallNode,
    constants: dict[str, np.ndarray],
    tensors: dict[str, Value],
) -> str | None:
    name = node.canonical_name
    if name not in ("add", "sub", "mul", "div") or len(node.inputs) != 2:
        return None
    left, right = node.inputs
    left_const = left in constants
    right_const = right in constants
    if left_const == right_const:
        # 两个常量由常量折叠处理；两个非常量没有恒等规则。
        return None
    if name == "sub" and left_const:
        return None
    if name == "div" and left_const:
        return None
    if name == "add" and not _is_zero(constants[right if right_const else left]):
        return None
    if name == "sub" and not _is_zero(constants[right]):
        return None
    if name == "mul" and not _is_one(constants[right if right_const else left]):
        return None
    if name == "div" and not _is_one(constants[right]):
        return None

    # 保留非常量输入：left 是常量时保留 right，否则保留 left。
    keep = right if left_const else left
    # 只有在元素数完全一致（无广播导致的 shape 变化）时才允许删除。
    out_type = node.outputs[0].type
    in_type = tensors[keep].type
    if (
        out_type.dtype != in_type.dtype
        or out_type.shape != in_type.shape
        or out_type.layout != in_type.layout
    ):
        return None
    return keep


# ---------------------------------------------------------------------------
# layout 校验
# ---------------------------------------------------------------------------


def _require_layout(node: Node, actual: str, expected: str, what: str) -> None:
    if actual != expected:
        raise GraphValidationError(
            f"layout 校验失败: {node.canonical_name} ({what}) 期望 {expected}，"
            f"实际 {actual} @ {node.span}"
        )


def validate_layouts(module: Module, options=None) -> PassOutput:
    tensors = module.graph.tensors
    checked = 0
    for node in module.graph.nodes:
        if not isinstance(node, CallNode):
            continue
        inputs = [tensors[name].type for name in node.inputs]
        outputs = [value.type for value in node.outputs]
        name = node.canonical_name
        checked += 1
        if name in ("conv2d", "conv2d_relu"):
            _require_layout(node, inputs[0].layout, "NCHW", "输入")
            _require_layout(node, inputs[1].layout, "OIHW", "权重")
            if len(inputs) > 2:
                _require_layout(node, inputs[2].layout, "C", "bias")
            _require_layout(node, outputs[0].layout, "NCHW", "输出")
        elif name in ("max_pool2d", "avg_pool2d", "global_avg_pool2d"):
            _require_layout(node, inputs[0].layout, "NCHW", "输入")
            _require_layout(node, outputs[0].layout, "NCHW", "输出")
        elif name == "batch_norm":
            for index, t in enumerate(inputs[1:], start=1):
                _require_layout(node, t.layout, "C", f"参数 {index}")
            _require_layout(node, outputs[0].layout, inputs[0].layout, "输出")
        elif name in ("gemm", "matmul", "gemm_relu"):
            for index, t in enumerate(inputs):
                _require_layout(node, t.layout, LINEAR_LAYOUT, f"输入 {index}")
            _require_layout(node, outputs[0].layout, LINEAR_LAYOUT, "输出")
        elif name in ("flatten", "reshape", "transpose", "concat"):
            _require_layout(node, outputs[0].layout, LINEAR_LAYOUT, "输出")
        elif name in _UNARY_CANONICAL:
            _require_layout(node, outputs[0].layout, inputs[0].layout, "输出")
        elif name in _BINARY_CANONICAL:
            allowed = {inputs[0].layout, inputs[1].layout, LINEAR_LAYOUT}
            if outputs[0].layout not in allowed:
                raise GraphValidationError(
                    f"layout 校验失败: {name} 输出 layout {outputs[0].layout} "
                    f"不在 {sorted(allowed)} 中 @ {node.span}"
                )
        elif name in ("softmax", "cast", "clip", "identity"):
            _require_layout(node, outputs[0].layout, inputs[0].layout, "输出")
    return PassOutput(
        module,
        notes=[f"校验 Call 节点 layout: {checked}"],
    )
