"""PTQ 量化：把量化参数写入 NestGraph 张量类型。

原则：
- 量化参数只来自校准结果或常量权重本身，不由后端猜测；
- 缺少校准数据、范围退化（scale=0）、NPU 不支持的模式都显式失败；
- 混合精度回退（float_ops / per_op.float）在报告中逐条记录。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from ..errors import QuantizationError
from ..ir import (
    CallNode,
    ConstantNode,
    FallbackNode,
    Graph,
    Module,
    QuantInfo,
    TensorType,
    Value,
)
from ..registry import GLOBAL_REGISTRY, OpRegistry
from .calibrate import CalibrationResult
from .config import QuantConfig
from .qmath import (
    compute_scale_zero_point,
    dtype_bits,
    dequantize,
    quantize,
)

_CONV_OPS = frozenset({"conv2d", "conv2d_relu"})
_GEMM_OPS = frozenset({"gemm", "gemm_relu"})
_WEIGHT_OPS = _CONV_OPS | _GEMM_OPS
# 仅改变形状、不改变数值分布的算子：per-tensor 量化参数可直接传播。
_SHAPE_ONLY = frozenset({"identity", "flatten", "reshape"})


@dataclass(frozen=True)
class QuantizationDecision:
    tensor: str
    op: str
    kind: str
    dtype: str
    granularity: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "tensor": self.tensor,
            "op": self.op,
            "kind": self.kind,
            "dtype": self.dtype,
            "granularity": self.granularity,
            "reason": self.reason,
        }


@dataclass
class QuantizationResult:
    module: Module
    decisions: tuple[QuantizationDecision, ...]
    config: QuantConfig
    config_hash: str
    calibration_hash: str
    float_ops: frozenset[str]

    def quant_params(self) -> list[dict]:
        types = self.module.graph.tensors
        order = [value.name for value in self.module.graph.inputs]
        for node in self.module.graph.nodes:
            order.extend(value.name for value in node.outputs)
        params = []
        for name in order:
            tensor_type = types[name].type
            quant = tensor_type.quant
            entry = {
                "tensor": name,
                "dtype": tensor_type.dtype,
                "shape": list(tensor_type.shape),
                "layout": tensor_type.layout,
                "mode": quant.mode if quant is not None else None,
                "axis": quant.axis if quant is not None else None,
                "scale": list(quant.scale) if quant and quant.scale else None,
                "zero_point": (
                    list(quant.zero_point) if quant and quant.zero_point else None
                ),
                "block_size": quant.block_size if quant is not None else None,
                "rounding": quant.rounding if quant is not None else None,
                "saturation": quant.saturation if quant is not None else None,
                "accum_bits": quant.accum_bits if quant is not None else None,
            }
            params.append(entry)
        return params

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "config_hash": self.config_hash,
            "calibration_hash": self.calibration_hash,
            "float_ops": sorted(self.float_ops),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "quant_params": self.quant_params(),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)


def _weight_axis(node: CallNode, config: QuantConfig) -> int:
    if config.weight_axis is not None:
        return config.weight_axis
    if node.canonical_name in _CONV_OPS:
        return 0
    if node.canonical_name in _GEMM_OPS:
        return 0 if int(node.attrs.get("transB", 0)) else 1
    return 0


def _weight_scale_zero_point(
    array: np.ndarray,
    node: CallNode,
    config: QuantConfig,
) -> tuple[int, tuple[float, ...], tuple[int, ...], int | None]:
    dtype = config.weight_dtype
    scheme = config.weight_scheme
    granularity = config.weight_granularity
    axis = _weight_axis(node, config)
    if granularity != "per_tensor" and array.ndim == 0:
        raise QuantizationError(
            f"权重 {node.op_type} 是标量，不能使用 {granularity} 量化"
        )
    if granularity != "per_tensor" and not (0 <= axis < array.ndim):
        raise QuantizationError(
            f"权重量化轴 {axis} 超出秩 {array.ndim}"
        )

    if granularity == "per_tensor":
        lo, hi = float(np.min(array)), float(np.max(array))
        scale, zero_point = compute_scale_zero_point(lo, hi, dtype, scheme)
        return 0, (scale,), (zero_point,), None

    moved = np.moveaxis(array, axis, 0)
    dim = int(moved.shape[0])
    blocks = None
    if granularity == "per_channel":
        lo = np.min(moved.reshape(dim, -1), axis=1)
        hi = np.max(moved.reshape(dim, -1), axis=1)
    else:
        if dim % config.block_size != 0:
            raise QuantizationError(
                f"per-block 块长 {config.block_size} 不能整除权重轴 {dim}"
            )
        blocks = dim // config.block_size
        reshaped = moved.reshape((blocks, config.block_size) + moved.shape[1:])
        reduce_axes = tuple(range(1, reshaped.ndim))
        lo = np.min(reshaped, axis=reduce_axes)
        hi = np.max(reshaped, axis=reduce_axes)
    scales = []
    zero_points = []
    for index, (low, high) in enumerate(zip(lo.reshape(-1), hi.reshape(-1))):
        scale, zero_point = compute_scale_zero_point(
            float(low), float(high), dtype, scheme
        )
        scales.append(scale)
        zero_points.append(zero_point)
    return axis, tuple(scales), tuple(zero_points), blocks


@dataclass
class _ValuePlan:
    kind: str
    quant: QuantInfo
    dtype: str


class _Annotator:
    def __init__(
        self,
        module: Module,
        calibration: CalibrationResult,
        config: QuantConfig,
        registry: OpRegistry,
    ) -> None:
        self.module = module
        self.calibration = calibration
        self.config = config
        self.registry = registry
        self.types = module.graph.tensors
        self.new_types: dict[str, TensorType] = {}
        self.new_constants: dict[str, np.ndarray] = {}
        self.decisions: list[QuantizationDecision] = []
        self.consumers: dict[str, list[tuple[object, int]]] = {}
        for node in module.graph.nodes:
            for position, name in enumerate(node.inputs):
                self.consumers.setdefault(name, []).append((node, position))
        self.float_ops = set(config.float_ops)
        for node in module.graph.nodes:
            if isinstance(node, CallNode):
                op_config = config.for_op(node.canonical_name)
                if op_config.op_is_float(node.canonical_name):
                    self.float_ops.add(node.canonical_name)
                else:
                    spec = registry.lookup(node.domain, node.op_type)
                    if spec is None:
                        self.float_ops.add(node.canonical_name)
                    elif not spec.supports_npu(
                        node.attrs,
                        [self.types[name].type for name in node.inputs],
                    ):
                        self.float_ops.add(node.canonical_name)

    # ---- 角色判定 ----
    def _role(self, name: str) -> tuple[str, object | None]:
        quant_consumer = None
        for node, position in self.consumers.get(name, []):
            if isinstance(node, FallbackNode):
                continue
            if isinstance(node, CallNode):
                if node.canonical_name in self.float_ops:
                    continue
                if node.canonical_name in _WEIGHT_OPS and position == 1:
                    return "weight", node
                if node.canonical_name in _WEIGHT_OPS and position == 2:
                    return "bias", node
                quant_consumer = node
        if quant_consumer is not None:
            return "activation", quant_consumer
        return "float", None

    # ---- 量化参数 ----
    def _float_type(self, tensor_type: TensorType) -> TensorType:
        bits = 16 if tensor_type.dtype == "float16" else 32
        return replace(tensor_type, quant=QuantInfo(mode="float", bits=bits))

    def _activation_plan(
        self,
        name: str,
        tensor_type: TensorType,
        op: str,
        config: QuantConfig,
        *,
        weight: bool = False,
    ) -> _ValuePlan:
        dtype = config.weight_dtype if weight else config.activation_dtype
        scheme = config.weight_scheme if weight else config.activation_scheme
        granularity = (
            config.weight_granularity if weight else config.activation_granularity
        )
        bits = dtype_bits(dtype)
        if granularity == "per_block":
            raise QuantizationError(
                f"张量 {name} 使用 per-block 激活量化，但 M2 校准只提供 "
                "per-tensor / per-channel 范围；请改用权重 per-block 或激活 "
                "per-channel"
            )
        if granularity == "per_tensor":
            if name not in self.calibration.ranges:
                raise QuantizationError(
                    f"张量 {name} 缺少校准数据，无法量化（op={op}）"
                )
            lo, hi = self.calibration.ranges[name]
            scale, zero_point = compute_scale_zero_point(lo, hi, dtype, scheme)
            quant = QuantInfo(
                mode="int",
                bits=bits,
                scale=(scale,),
                zero_point=(zero_point,),
                axis=0,
                rounding=config.rounding,
                saturation=config.saturation,
                accum_bits=config.accum_bits,
            )
        else:
            axis = config.activation_channel_axis
            if name not in self.calibration.channel_ranges:
                raise QuantizationError(
                    f"张量 {name} 没有 per-channel 校准范围（op={op}）"
                )
            channel_lo, channel_hi = self.calibration.channel_ranges[name]
            if axis >= len(tensor_type.shape):
                raise QuantizationError(
                    f"张量 {name} 的 per-channel 轴 {axis} 超出秩 "
                    f"{len(tensor_type.shape)}"
                )
            if channel_lo.shape[0] != tensor_type.shape[axis]:
                raise QuantizationError(
                    f"张量 {name} 的逐通道校准长度 {channel_lo.shape[0]} 与 "
                    f"轴 {axis} 维度 {tensor_type.shape[axis]} 不一致"
                )
            scales = []
            zero_points = []
            for low, high in zip(channel_lo, channel_hi):
                scale, zero_point = compute_scale_zero_point(
                    float(low), float(high), dtype, scheme
                )
                scales.append(scale)
                zero_points.append(zero_point)
            quant = QuantInfo(
                mode="int",
                bits=bits,
                scale=tuple(scales),
                zero_point=tuple(zero_points),
                axis=axis,
                rounding=config.rounding,
                saturation=config.saturation,
                accum_bits=config.accum_bits,
            )
        return _ValuePlan(kind="int", quant=quant, dtype=dtype)

    def _weight_plan(
        self, name: str, tensor_type: TensorType, node: CallNode, config: QuantConfig
    ) -> _ValuePlan:
        if self._is_constant(name):
            array = self._constant_array(name)
            dtype = config.weight_dtype
            axis, scale, zero_point, block_size = _weight_scale_zero_point(
                array, node, config
            )
            quant = QuantInfo(
                mode="int",
                bits=dtype_bits(dtype),
                scale=scale,
                zero_point=zero_point,
                axis=axis,
                rounding=config.rounding,
                saturation=config.saturation,
                accum_bits=config.accum_bits,
                block_size=block_size,
            )
            return _ValuePlan(kind="int", quant=quant, dtype=dtype)
        return self._activation_plan(
            name, tensor_type, node.canonical_name, config, weight=True
        )

    def _is_constant(self, name: str) -> bool:
        return any(
            isinstance(node, ConstantNode)
            and node.outputs[0].name == name
            for node in self.module.graph.nodes
        )

    def _constant_array(self, name: str) -> np.ndarray:
        for node in self.module.graph.nodes:
            if isinstance(node, ConstantNode) and node.outputs[0].name == name:
                return np.asarray(node.value)
        raise QuantizationError(f"{name} 不是常量张量")

    # ---- 主流程 ----
    def run(self) -> None:
        for value in self.module.graph.inputs:
            self._annotate_value(value.name, value.type, producer_op="input")
        for node in self.module.graph.nodes:
            if isinstance(node, ConstantNode):
                self._annotate_constant(node)
            elif isinstance(node, FallbackNode):
                for value in node.outputs:
                    self.new_types[value.name] = self._float_type(value.type)
                    self.decisions.append(
                        QuantizationDecision(
                            tensor=value.name,
                            op=node.op_type,
                            kind="float",
                            dtype=value.type.dtype,
                            granularity="per_tensor",
                            reason="fallback 节点保持浮点",
                        )
                    )
            elif isinstance(node, CallNode):
                self._annotate_call(node)
            else:
                raise QuantizationError(f"未知节点类型: {type(node)}")
        self._bias_correction()

    def _annotate_value(
        self, name: str, tensor_type: TensorType, producer_op: str
    ) -> None:
        if name in self.new_types:
            return
        role, node = self._role(name)
        config = self.config
        if isinstance(node, CallNode):
            config = self.config.for_op(node.canonical_name)
        if role == "float":
            self.new_types[name] = self._float_type(tensor_type)
            self.decisions.append(
                QuantizationDecision(
                    tensor=name,
                    op=producer_op,
                    kind="float",
                    dtype=tensor_type.dtype,
                    granularity="per_tensor",
                    reason="显式浮点回退",
                )
            )
            return
        if role == "bias":
            self.new_types[name] = self._float_type(tensor_type)
            self.decisions.append(
                QuantizationDecision(
                    tensor=name,
                    op=producer_op,
                    kind="bias",
                    dtype=tensor_type.dtype,
                    granularity="per_tensor",
                    reason="bias 保持浮点（int32 累加结果在 kernel 内显式相加）",
                )
            )
            return
        if role == "weight" and isinstance(node, CallNode):
            plan = self._weight_plan(name, tensor_type, node, config)
        else:
            plan = self._activation_plan(
                name, tensor_type, producer_op, config
            )
        self.new_types[name] = replace(
            tensor_type, dtype=plan.dtype, quant=plan.quant
        )
        self.decisions.append(
            QuantizationDecision(
                tensor=name,
                op=producer_op,
                kind=role,
                dtype=plan.dtype,
                granularity=(
                    config.weight_granularity
                    if role == "weight"
                    else config.activation_granularity
                ),
                reason="量化",
            )
        )

    def _annotate_constant(self, node: ConstantNode) -> None:
        value = node.outputs[0]
        self._annotate_value(value.name, value.type, producer_op="constant")
        tensor_type = self.new_types[value.name]
        if tensor_type.quant is None or tensor_type.quant.mode != "int":
            return
        original = self._constant_array(value.name)
        quantized = quantize(
            original.astype(np.float64), tensor_type.quant, tensor_type.dtype
        )
        self.new_constants[value.name] = quantized.astype(
            tensor_type.dtype, copy=False
        )

    def _annotate_call(self, node: CallNode) -> None:
        config = self.config.for_op(node.canonical_name)
        is_float = node.canonical_name in self.float_ops
        dtype = config.activation_dtype
        source_type = self.new_types[node.inputs[0]]
        propagated = (
            not is_float
            and node.canonical_name in _SHAPE_ONLY
            and source_type.quant is not None
            and source_type.quant.mode == "int"
            and len(source_type.quant.scale or ()) == 1
        )
        for position, value in enumerate(node.outputs):
            if is_float:
                self.new_types[value.name] = self._float_type(value.type)
                self.decisions.append(
                    QuantizationDecision(
                        tensor=value.name,
                        op=node.op_type,
                        kind="float",
                        dtype=value.type.dtype,
                        granularity="per_tensor",
                        reason="混合精度回退（用户配置或 NPU 不支持）",
                    )
                )
                continue
            if propagated:
                self.new_types[value.name] = replace(
                    value.type, dtype=dtype, quant=source_type.quant
                )
                self.decisions.append(
                    QuantizationDecision(
                        tensor=value.name,
                        op=node.op_type,
                        kind="activation",
                        dtype=dtype,
                        granularity="per_tensor",
                        reason="量化参数沿形状变换传播",
                    )
                )
                continue
            plan = self._activation_plan(
                value.name, value.type, node.canonical_name, config
            )
            self.new_types[value.name] = replace(
                value.type, dtype=plan.dtype, quant=plan.quant
            )
            self.decisions.append(
                QuantizationDecision(
                    tensor=value.name,
                    op=node.op_type,
                    kind="activation",
                    dtype=plan.dtype,
                    granularity=config.activation_granularity,
                    reason="量化",
                )
            )

    def _bias_correction(self) -> None:
        if not self.config.bias_correction:
            return
        for node in self.module.graph.nodes:
            if not isinstance(node, CallNode) or node.canonical_name not in _WEIGHT_OPS:
                continue
            if len(node.inputs) < 3:
                continue
            weight_name, bias_name = node.inputs[1], node.inputs[2]
            weight_type = self.new_types.get(weight_name)
            if weight_type is None or weight_type.quant is None:
                continue
            if weight_type.quant.mode != "int":
                continue
            if bias_name not in self.new_constants:
                bias_value = self._constant_array(bias_name)
                if not np.issubdtype(bias_value.dtype, np.floating):
                    continue
                original = self._constant_array(weight_name)
                quant = weight_type.quant
                dequantized = dequantize(
                    self.new_constants[weight_name], quant, original.shape
                )
                axis = quant.axis or 0
                reduce_axes = tuple(
                    index for index in range(original.ndim) if index != axis
                )
                error = np.mean(original - dequantized, axis=reduce_axes)
                corrected = bias_value + error.reshape(bias_value.shape)
                self.new_constants[bias_name] = corrected.astype(
                    bias_value.dtype, copy=False
                )
                self.decisions.append(
                    QuantizationDecision(
                        tensor=bias_name,
                        op=node.op_type,
                        kind="bias_correction",
                        dtype=str(bias_value.dtype),
                        granularity="per_tensor",
                        reason="权重量化误差按输出通道均值补偿到 bias",
                    )
                )


def _rebuild_module(
    module: Module,
    new_types: dict[str, TensorType],
    new_constants: dict[str, np.ndarray],
) -> Module:
    inputs = [Value(name=value.name, type=new_types[value.name])
              for value in module.graph.inputs]
    outputs = [Value(name=value.name, type=new_types[value.name])
               for value in module.graph.outputs]
    nodes = []
    for node in module.graph.nodes:
        node_outputs = [
            Value(name=value.name, type=new_types[value.name])
            for value in node.outputs
        ]
        if isinstance(node, ConstantNode):
            name = node.outputs[0].name
            value = new_constants.get(name, node.value)
            nodes.append(
                ConstantNode(
                    inputs=[],
                    outputs=node_outputs,
                    span=node.span,
                    attrs=dict(node.attrs),
                    value=value,
                )
            )
        elif isinstance(node, CallNode):
            nodes.append(
                CallNode(
                    inputs=list(node.inputs),
                    outputs=node_outputs,
                    span=node.span,
                    attrs=dict(node.attrs),
                    domain=node.domain,
                    op_type=node.op_type,
                    canonical_name=node.canonical_name,
                )
            )
        elif isinstance(node, FallbackNode):
            nodes.append(
                FallbackNode(
                    inputs=list(node.inputs),
                    outputs=node_outputs,
                    span=node.span,
                    attrs=dict(node.attrs),
                    domain=node.domain,
                    op_type=node.op_type,
                    backend=node.backend,
                    reason=node.reason,
                )
            )
        else:
            raise QuantizationError(f"未知节点类型: {type(node)}")
    return Module(
        name=module.name,
        graph=Graph(inputs=inputs, outputs=outputs, nodes=nodes),
        metadata=dict(module.metadata),
    )


def quantize_module(
    module: Module,
    calibration: CalibrationResult,
    config: QuantConfig | None = None,
    *,
    registry: OpRegistry = GLOBAL_REGISTRY,
) -> QuantizationResult:
    """执行 PTQ，返回带量化标注与定点常量权重的新模块。"""
    config = config or QuantConfig()
    annotator = _Annotator(module, calibration, config, registry)
    annotator.run()
    for value in module.graph.inputs:
        if value.name not in annotator.new_types:
            raise QuantizationError(f"图输入 {value.name} 未完成量化标注")
    quantized = _rebuild_module(
        module, annotator.new_types, annotator.new_constants
    )
    return QuantizationResult(
        module=quantized,
        decisions=tuple(annotator.decisions),
        config=config,
        config_hash=config.hash(),
        calibration_hash=calibration.calibration_hash,
        float_ops=frozenset(annotator.float_ops),
    )


__all__ = ["QuantizationDecision", "QuantizationResult", "quantize_module"]
