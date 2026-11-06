"""NumPy 参考执行器（M0）。

用途：
- 作为导入后的语义基准，验证 NestGraph 转换是否保持模型数值；
- 为 fallback 节点提供标量核执行路径（host CPU / RISC-V 标量核语义等价）。

执行原则：任何缺少 kernel 的节点都必须显式报错，不能静默跳过。
"""

from __future__ import annotations

import numpy as np

from ..errors import MissingKernelError, MissingScalarImplementationError
from ..ir import CallNode, ConstantNode, FallbackNode, Graph, Module, Node
from ..registry import GLOBAL_REGISTRY, OpRegistry


def _normalize_kernel_output(raw, node: Node) -> list[np.ndarray]:
    if isinstance(raw, np.ndarray):
        outputs = [raw]
    elif isinstance(raw, (list, tuple)):
        outputs = [np.asarray(item) for item in raw]
    else:
        raise MissingKernelError(
            f"算子 {node.kind}:{getattr(node, 'op_type', '?')} 的 kernel "
            f"返回非法类型: {type(raw)}"
        )
    if len(outputs) != len(node.outputs):
        raise MissingKernelError(
            f"节点 {node.output_names} 的 kernel 返回 {len(outputs)} 个张量，"
            f"但声明 {len(node.outputs)} 个输出"
        )
    return outputs


def _check_tensor(name: str, value: np.ndarray, node: Node) -> np.ndarray:
    declared = next(v.type for v in node.outputs if v.name == name)
    if tuple(value.shape) != declared.shape:
        raise MissingKernelError(
            f"节点 {node.output_names} 输出 {name} shape {value.shape} "
            f"与声明的 {declared.shape} 不一致"
        )
    if str(value.dtype) != declared.dtype:
        raise MissingKernelError(
            f"节点 {node.output_names} 输出 {name} dtype {value.dtype} "
            f"与声明的 {declared.dtype} 不一致"
        )
    return value


def _kernel_for(node: Node, registry: OpRegistry, local_kernels: dict):
    if isinstance(node, FallbackNode):
        key = (node.domain, node.op_type, node.backend)
        kernel = local_kernels.get(key) or registry.get_fallback_kernel(*key)
        if kernel is not None:
            return kernel
        spec = registry.lookup(node.domain, node.op_type)
        if spec is not None and spec.kernel is not None:
            return spec.kernel
        raise MissingScalarImplementationError(
            f"fallback 节点 {node.domain + '::' if node.domain else ''}"
            f"{node.op_type} (backend={node.backend}) 缺少标量实现；"
            f"请用 register_fallback_kernel 注册，或为该算子提供转换规则。"
            f" source={node.span}"
        )
    if isinstance(node, CallNode):
        spec = registry.lookup(node.domain, node.op_type)
        if spec is None or spec.kernel is None:
            raise MissingKernelError(
                f"算子 {node.domain + '::' if node.domain else ''}"
                f"{node.op_type} 缺少 NumPy kernel。source={node.span}"
            )
        return spec.kernel
    raise MissingKernelError(f"节点类型不支持执行: {type(node)}")


def run_graph(
    graph: Graph,
    inputs: dict[str, np.ndarray],
    *,
    registry: OpRegistry = GLOBAL_REGISTRY,
    fallback_kernels: dict | None = None,
    tensor_sink: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """按拓扑序执行 NestGraph，返回图输出。"""
    expected = {value.name: value.type for value in graph.inputs}
    missing = sorted(set(expected) - set(inputs))
    if missing:
        raise ValueError(f"缺少输入: {missing}")
    extra = sorted(set(inputs) - set(expected))
    if extra:
        raise ValueError(f"多余的输入: {extra}")

    env: dict[str, np.ndarray] = {}
    local = fallback_kernels or {}

    for name, declared in expected.items():
        array = np.asarray(inputs[name])
        if tuple(array.shape) != declared.shape:
            raise ValueError(
                f"输入 {name} shape {array.shape} 与声明 {declared.shape} 不一致"
            )
        if str(array.dtype) != declared.dtype:
            raise ValueError(
                f"输入 {name} dtype {array.dtype} 与声明 {declared.dtype} 不一致"
            )
        env[name] = array

    for node in graph.nodes:
        if isinstance(node, ConstantNode):
            env[node.outputs[0].name] = np.asarray(node.value)
            continue
        kernel = _kernel_for(node, registry, local)
        args = [env[name] for name in node.inputs]
        raw = kernel(args, dict(node.attrs))
        outputs = _normalize_kernel_output(raw, node)
        for value, array in zip(node.outputs, outputs):
            env[value.name] = _check_tensor(value.name, array, node)
            if tensor_sink is not None:
                tensor_sink[value.name] = env[value.name]

    result = {}
    for value in graph.outputs:
        result[value.name] = env[value.name]
    return result


def run_module(
    module: Module,
    inputs: dict[str, np.ndarray],
    *,
    registry: OpRegistry = GLOBAL_REGISTRY,
    fallback_kernels: dict | None = None,
    tensor_sink: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    return run_graph(
        module.graph,
        inputs,
        registry=registry,
        fallback_kernels=fallback_kernels,
        tensor_sink=tensor_sink,
    )
