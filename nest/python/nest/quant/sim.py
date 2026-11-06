"""QDQ 假量化仿真执行器。

每个节点：
1. 按 TensorType.quant 对输入做 quantize→dequantize；
2. 用浮点 kernel 计算；
3. 对输出再做 quantize→dequantize，并逐层记录张量。

这样既能复用 M0/M1 的数值 kernel，又能暴露每一层的量化误差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import MissingKernelError, MissingScalarImplementationError
from ..ir import CallNode, ConstantNode, FallbackNode, Module, TensorType
from ..registry import GLOBAL_REGISTRY, OpRegistry
from ..runtime import run_module
from .qmath import dequantize, quantize_dequantize


@dataclass(frozen=True)
class TraceEntry:
    name: str
    op_type: str
    canonical_name: str
    node_index: int
    span: object
    value: np.ndarray
    dtype: str
    quant_mode: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "canonical_name": self.canonical_name,
            "node_index": self.node_index,
            "span": self.span.to_dict() if self.span is not None else None,
            "dtype": self.dtype,
            "quant_mode": self.quant_mode,
            "shape": list(self.value.shape),
        }


@dataclass
class SimulationResult:
    outputs: dict[str, np.ndarray]
    tensors: dict[str, np.ndarray]
    traces: tuple[TraceEntry, ...]

    def tensor_order(self) -> list[str]:
        return [entry.name for entry in self.traces]


def _normalize(raw, expected: int) -> list[np.ndarray]:
    if isinstance(raw, np.ndarray):
        outputs = [raw]
    elif isinstance(raw, (list, tuple)):
        outputs = [np.asarray(item) for item in raw]
    else:
        raise MissingKernelError(f"kernel 返回非法类型: {type(raw)}")
    if len(outputs) != expected:
        raise MissingKernelError(
            f"kernel 返回 {len(outputs)} 个张量，但节点声明 {expected} 个输出"
        )
    return outputs


def _kernel_for(node, registry: OpRegistry, local: dict):
    if isinstance(node, FallbackNode):
        key = (node.domain, node.op_type, node.backend)
        kernel = local.get(key) or registry.get_fallback_kernel(*key)
        if kernel is not None:
            return kernel
        spec = registry.lookup(node.domain, node.op_type)
        if spec is not None and spec.kernel is not None:
            return spec.kernel
        raise MissingScalarImplementationError(
            f"fallback 节点 {node.domain + '::' if node.domain else ''}"
            f"{node.op_type} (backend={node.backend}) 缺少标量实现；"
            f"source={node.span}"
        )
    spec = registry.lookup(node.domain, node.op_type)
    if spec is None or spec.kernel is None:
        raise MissingKernelError(
            f"算子 {node.domain + '::' if node.domain else ''}"
            f"{node.op_type} 缺少 NumPy kernel。source={node.span}"
        )
    return spec.kernel


def _qdq_input(value: np.ndarray, tensor_type: TensorType) -> np.ndarray:
    quant = tensor_type.quant
    if quant is None or quant.mode != "int":
        return value
    return quantize_dequantize(
        value.astype(np.float64), quant, tensor_type.dtype, out_dtype="float32"
    )


def _qdq_output(value: np.ndarray, tensor_type: TensorType) -> np.ndarray:
    return _qdq_input(value, tensor_type)


def run_quantized(
    module: Module,
    inputs: dict[str, np.ndarray],
    *,
    registry: OpRegistry = GLOBAL_REGISTRY,
    fallback_kernels: dict | None = None,
) -> SimulationResult:
    """在带量化标注的模块上执行 QDQ 假量化。"""
    graph = module.graph
    types = graph.tensors
    expected = {value.name: value for value in graph.inputs}
    missing = sorted(set(expected) - set(inputs))
    if missing:
        raise ValueError(f"缺少输入: {missing}")
    extra = sorted(set(inputs) - set(expected))
    if extra:
        raise ValueError(f"多余的输入: {extra}")

    env: dict[str, np.ndarray] = {}
    traces: list[TraceEntry] = []
    local = fallback_kernels or {}

    for name, value in expected.items():
        array = np.asarray(inputs[name])
        if tuple(array.shape) != value.type.shape:
            raise ValueError(
                f"输入 {name} shape {array.shape} 与声明 {value.type.shape} 不一致"
            )
        if np.issubdtype(array.dtype, np.integer):
            quant = value.type.quant
            if quant is None or quant.mode != "int":
                raise ValueError(
                    f"输入 {name} 是整数张量，但声明类型没有 int 量化参数"
                )
            float_value = dequantize(array, quant, value.type.shape)
        else:
            float_value = _qdq_input(array, value.type)
        env[name] = float_value
        traces.append(
            TraceEntry(
                name=name,
                op_type="input",
                canonical_name="input",
                node_index=-1,
                span=None,
                value=env[name],
                dtype=value.type.dtype,
                quant_mode=(
                    value.type.quant.mode if value.type.quant is not None else "none"
                ),
            )
        )

    node_index = 0
    for node in graph.nodes:
        current = node_index
        node_index += 1
        if isinstance(node, ConstantNode):
            value = node.outputs[0]
            array = np.asarray(node.value)
            if value.type.quant is not None and value.type.quant.mode == "int":
                if np.issubdtype(array.dtype, np.integer):
                    float_value = dequantize(
                        array, value.type.quant, value.type.shape
                    )
                else:
                    float_value = quantize_dequantize(
                        array.astype(np.float64),
                        value.type.quant,
                        value.type.dtype,
                        out_dtype="float32",
                    )
            else:
                float_value = array
            env[value.name] = np.asarray(float_value, dtype=np.float64)
            traces.append(
                TraceEntry(
                    name=value.name,
                    op_type="constant",
                    canonical_name="constant",
                    node_index=current,
                    span=node.span,
                    value=env[value.name],
                    dtype=value.type.dtype,
                    quant_mode=(
                        value.type.quant.mode
                        if value.type.quant is not None
                        else "none"
                    ),
                )
            )
            continue

        kernel = _kernel_for(node, registry, local)
        args = [
            _qdq_input(env[name], types[name].type) for name in node.inputs
        ]
        raw = kernel(args, dict(node.attrs))
        outputs = _normalize(raw, len(node.outputs))
        for value, array in zip(node.outputs, outputs):
            array = np.asarray(array)
            if tuple(array.shape) != value.type.shape:
                raise MissingKernelError(
                    f"节点 {node.output_names} 输出 {value.name} shape "
                    f"{array.shape} 与声明 {value.type.shape} 不一致"
                )
            float_value = _qdq_output(array.astype(np.float64), value.type)
            env[value.name] = np.asarray(float_value, dtype=np.float64)
            traces.append(
                TraceEntry(
                    name=value.name,
                    op_type=node.op_type,
                    canonical_name=getattr(node, "canonical_name", node.op_type),
                    node_index=current,
                    span=node.span,
                    value=env[value.name],
                    dtype=value.type.dtype,
                    quant_mode=(
                        value.type.quant.mode
                        if value.type.quant is not None
                        else "none"
                    ),
                )
            )

    result = {value.name: env[value.name] for value in graph.outputs}
    return SimulationResult(outputs=result, tensors=dict(env), traces=tuple(traces))


def run_reference_tensors(
    module: Module,
    inputs: dict[str, np.ndarray],
    *,
    registry: OpRegistry = GLOBAL_REGISTRY,
    fallback_kernels: dict | None = None,
) -> dict[str, np.ndarray]:
    """参考执行器 + 图输入 + 常量，得到逐层 golden 张量。"""
    sink: dict[str, np.ndarray] = {}
    run_module(
        module,
        inputs,
        registry=registry,
        fallback_kernels=fallback_kernels,
        tensor_sink=sink,
    )
    tensors: dict[str, np.ndarray] = {}
    for value in module.graph.inputs:
        tensors[value.name] = np.asarray(inputs[value.name])
    for node in module.graph.nodes:
        if isinstance(node, ConstantNode):
            tensors[node.outputs[0].name] = np.asarray(node.value)
    tensors.update(sink)
    return tensors


__all__ = [
    "TraceEntry",
    "SimulationResult",
    "run_quantized",
    "run_reference_tensors",
]
