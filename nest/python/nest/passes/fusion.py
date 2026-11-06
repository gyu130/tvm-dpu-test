"""算子融合 pass。

M1 支持：
- conv + batch_norm：把 BN 折叠进卷积权重与偏置；
- conv + relu：生成内部融合算子 conv2d_relu；
- gemm + relu：生成内部融合算子 gemm_relu。

融合前提：语义等价、dtype/layout 兼容、中间结果只有一个消费者。
未融合的原因会写入 pass 报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import PassError
from ..ir import CallNode, ConstantNode, Module, Node, TensorType, Value
from .base import PassOutput, consumer_counts, rebuild_module

INTERNAL_DOMAIN = "nest.internal"
DEFAULT_PATTERNS = ("conv_bn", "conv_relu", "gemm_relu")


@dataclass
class _Plan:
    pattern: str
    removed: set[int] = field(default_factory=set)
    replacements: dict[int, list[Node]] = field(default_factory=dict)


def _is_call(node: Node, canonical: str) -> bool:
    return isinstance(node, CallNode) and node.canonical_name == canonical


def _constant_map(module: Module) -> dict[str, np.ndarray]:
    return {
        node.outputs[0].name: np.asarray(node.value)
        for node in module.graph.nodes
        if isinstance(node, ConstantNode)
    }


def _unique_name(base: str, existing: set[str]) -> str:
    name = base
    index = 1
    while name in existing:
        name = f"{base}_{index}"
        index += 1
    existing.add(name)
    return name


def _consumer_index(nodes: list[Node], value_name: str) -> int | None:
    for index, node in enumerate(nodes):
        if value_name in node.inputs:
            return index
    return None


def _find_conv_bn(module: Module, nodes: list[Node]) -> _Plan | None:
    tensors = module.graph.tensors
    constants = _constant_map(module)
    consumers = consumer_counts(module)
    existing = set(tensors)

    for index, conv in enumerate(nodes):
        if not _is_call(conv, "conv2d"):
            continue
        conv_out = conv.outputs[0].name
        if consumers.get(conv_out, 0) != 1:
            continue
        consumer_index = _consumer_index(nodes, conv_out)
        if consumer_index is None:
            continue
        bn = nodes[consumer_index]
        if not _is_call(bn, "batch_norm") or bn.inputs[0] != conv_out:
            continue
        if len(bn.inputs) != 5 or len(conv.inputs) < 2:
            continue
        params = [constants.get(name) for name in bn.inputs[1:5]]
        if any(p is None for p in params):
            continue
        weight = constants.get(conv.inputs[1])
        if weight is None:
            continue
        bias = constants.get(conv.inputs[2]) if len(conv.inputs) > 2 else None
        if len(conv.inputs) > 2 and bias is None:
            continue

        scale, beta, mean, var = params
        epsilon = float(bn.attrs.get("epsilon", 1e-5))
        channels = weight.shape[0]
        if scale.shape != (channels,) or beta.shape != (channels,) \
                or mean.shape != (channels,) or var.shape != (channels,):
            continue
        out_types = [value.type for value in bn.outputs]
        if len(out_types) != 1:
            continue
        conv_type = conv.outputs[0].type
        bn_type = out_types[0]
        if (
            conv_type.dtype != bn_type.dtype
            or conv_type.shape != bn_type.shape
            or conv_type.layout != bn_type.layout
        ):
            continue

        folded_w = weight * (scale / np.sqrt(var + epsilon)).reshape(
            -1, *([1] * (weight.ndim - 1))
        )
        base_bias = np.zeros((channels,), dtype=weight.dtype) if bias is None else bias
        folded_b = (base_bias - mean) * (scale / np.sqrt(var + epsilon)) + beta
        folded_w = folded_w.astype(weight.dtype, copy=False)
        folded_b = folded_b.astype(weight.dtype, copy=False)

        weight_value = tensors[conv.inputs[1]]
        w_name = _unique_name(f"{conv_out}_bn_w", existing)
        b_name = _unique_name(f"{conv_out}_bn_b", existing)
        b_type = (
            tensors[conv.inputs[2]].type
            if len(conv.inputs) > 2
            else TensorType(
                dtype=weight_value.type.dtype,
                shape=(channels,),
                layout="C",
            )
        )
        w_const = ConstantNode(
            inputs=[],
            outputs=[Value(name=w_name, type=weight_value.type)],
            span=conv.span,
            value=folded_w,
        )
        b_const = ConstantNode(
            inputs=[],
            outputs=[Value(name=b_name, type=b_type)],
            span=conv.span,
            value=folded_b,
        )
        fused = replace(
            conv,
            inputs=[conv.inputs[0], w_name, b_name],
            outputs=[bn.outputs[0]],
        )
        return _Plan(
            pattern="conv_bn",
            removed={index, consumer_index},
            replacements={index: [w_const, b_const, fused]},
        )
    return None


def _find_activation_fusion(
    module: Module, nodes: list[Node], pattern: str
) -> _Plan | None:
    producer_canonical, fused_domain, fused_op, fused_canonical = {
        "conv_relu": ("conv2d", INTERNAL_DOMAIN, "conv2d_relu", "conv2d_relu"),
        "gemm_relu": ("gemm", INTERNAL_DOMAIN, "gemm_relu", "gemm_relu"),
    }[pattern]
    consumers = consumer_counts(module)

    for index, producer in enumerate(nodes):
        if not _is_call(producer, producer_canonical):
            continue
        out_name = producer.outputs[0].name
        if consumers.get(out_name, 0) != 1:
            continue
        consumer_index = _consumer_index(nodes, out_name)
        if consumer_index is None:
            continue
        relu = nodes[consumer_index]
        if not _is_call(relu, "relu") or relu.inputs[0] != out_name:
            continue
        if len(relu.outputs) != 1:
            continue
        fused = CallNode(
            inputs=list(producer.inputs),
            outputs=[relu.outputs[0]],
            span=producer.span,
            attrs=dict(producer.attrs),
            domain=fused_domain,
            op_type=fused_op,
            canonical_name=fused_canonical,
        )
        return _Plan(
            pattern=pattern,
            removed={index, consumer_index},
            replacements={index: [fused]},
        )
    return None


def _find_plan(module: Module, patterns: tuple[str, ...]) -> _Plan | None:
    nodes = module.graph.nodes
    if "conv_bn" in patterns:
        plan = _find_conv_bn(module, nodes)
        if plan is not None:
            return plan
    for pattern in ("conv_relu", "gemm_relu"):
        if pattern in patterns:
            plan = _find_activation_fusion(module, nodes, pattern)
            if plan is not None:
                return plan
    return None


def _apply_plan(module: Module, plan: _Plan) -> Module:
    nodes: list[Node] = []
    for index, node in enumerate(module.graph.nodes):
        if index in plan.replacements:
            nodes.extend(plan.replacements[index])
        elif index in plan.removed:
            continue
        else:
            nodes.append(node)
    return rebuild_module(module, nodes, module.graph.outputs)


def _collect_rejections(module: Module, patterns: tuple[str, ...]) -> list[str]:
    """对仍存在的 conv+bn / conv+relu / gemm+relu 候选记录未融合原因。"""
    tensors = module.graph.tensors
    constants = _constant_map(module)
    consumers = consumer_counts(module)
    notes: list[str] = []
    nodes = module.graph.nodes

    for node in nodes:
        if isinstance(node, CallNode) and node.canonical_name in ("conv2d", "gemm"):
            out_name = node.outputs[0].name
            consumer_index = _consumer_index(nodes, out_name)
            if consumer_index is None:
                continue
            consumer = nodes[consumer_index]
            if not isinstance(consumer, CallNode):
                continue
            if node.canonical_name == "conv2d" and consumer.canonical_name == "batch_norm" \
                    and "conv_bn" in patterns:
                reasons = []
                if consumers.get(out_name, 0) != 1:
                    reasons.append("中间结果存在多个消费者")
                if len(consumer.inputs) != 5 or any(
                    name not in constants for name in consumer.inputs[1:5]
                ):
                    reasons.append("BN 参数不是常量")
                conv_type = node.outputs[0].type
                bn_type = consumer.outputs[0].type
                if (
                    conv_type.dtype != bn_type.dtype
                    or conv_type.shape != bn_type.shape
                    or conv_type.layout != bn_type.layout
                ):
                    reasons.append("Conv/BN 输出类型不兼容")
                if reasons:
                    notes.append(
                        f"{out_name}: 未融合 conv+bn（{'; '.join(reasons)}）"
                    )
            elif consumer.canonical_name == "relu":
                pattern = "conv_relu" if node.canonical_name == "conv2d" else "gemm_relu"
                if pattern not in patterns:
                    continue
                if consumers.get(out_name, 0) != 1:
                    notes.append(f"{out_name}: 未融合 {pattern}（中间结果存在多个消费者）")
    return notes


def fuse(module: Module, options=None) -> PassOutput:
    patterns = tuple(getattr(options, "fusion_patterns", DEFAULT_PATTERNS))
    unknown = [p for p in patterns if p not in DEFAULT_PATTERNS]
    if unknown:
        raise PassError(f"未知融合 pattern: {unknown}")

    current = module
    fused = 0
    removed = 0
    added = 0
    hit_patterns: dict[str, int] = {}
    while True:
        plan = _find_plan(current, patterns)
        if plan is None:
            break
        current = _apply_plan(current, plan)
        fused += 1
        removed += len(plan.removed)
        added += sum(len(v) for v in plan.replacements.values())
        hit_patterns[plan.pattern] = hit_patterns.get(plan.pattern, 0) + 1

    notes = _collect_rejections(current, patterns)
    if hit_patterns:
        notes.insert(
            0,
            "融合命中: "
            + ", ".join(f"{k}={v}" for k, v in sorted(hit_patterns.items())),
        )
    return PassOutput(
        current,
        removed=removed,
        added=added,
        fused=fused,
        notes=notes,
    )
