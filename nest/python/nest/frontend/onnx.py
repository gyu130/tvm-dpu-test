"""ONNX 前端（M0）。

职责：
- 识别 opset 与 domain，校验版本支持范围；
- 把 initializer / Constant 转换为 ConstantNode；
- 按“custom -> standard -> fallback”顺序分派算子；
- 输出 shape/dtype 优先来自模型 value_info，缺失时由注册的推导规则补齐；
- fallback 支持 mark / error 两种策略，错误信息包含算子名与 source span。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx

from ..errors import ModelImportError, ShapeInferenceError, UnsupportedOperatorError
from ..ir import (
    LINEAR_LAYOUT,
    CallNode,
    ConstantNode,
    FallbackNode,
    Graph,
    Module,
    SourceSpan,
    TensorType,
    Value,
)
from ..ir.types import QUANTIZED_DTYPES
from ..registry import GLOBAL_REGISTRY, OpRegistry, OpSpec
from ..report import FallbackEntry, ImportReport
from .onnx_util import (
    ONNX_DTYPE_TO_NAME,
    convert_attrs,
    initializer_to_numpy,
    numpy_dtype_name,
    value_info_type,
)

STANDARD_OPSET_MIN = 7
STANDARD_OPSET_MAX = 23
FALLBACK_MARK = "mark"
FALLBACK_ERROR = "error"


@dataclass
class ImportOptions:
    fallback_policy: str = FALLBACK_MARK
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    module_name: str | None = None
    registry: OpRegistry = field(default_factory=lambda: GLOBAL_REGISTRY)

    def __post_init__(self) -> None:
        if self.fallback_policy not in (FALLBACK_MARK, FALLBACK_ERROR):
            raise ModelImportError(
                f"非法 fallback_policy: {self.fallback_policy!r}"
            )


@dataclass
class ImportResult:
    module: Module
    report: ImportReport


def _load_model(model) -> tuple[onnx.ModelProto, str]:
    if isinstance(model, onnx.ModelProto):
        return model, "<in-memory>"
    if isinstance(model, bytes):
        return onnx.load_from_string(model), "<bytes>"
    path = Path(model)
    if not path.exists():
        raise ModelImportError(f"模型文件不存在: {path}")
    try:
        return onnx.load(str(path)), str(path)
    except Exception as exc:  # onnx 内部异常类型不稳定
        raise ModelImportError(f"无法解析 ONNX 模型 {path}: {exc}") from exc


def _collect_opsets(model: onnx.ModelProto) -> dict[str, int]:
    opsets: dict[str, int] = {}
    for item in model.opset_import:
        domain = item.domain or ""
        if domain in opsets:
            raise ModelImportError(f"重复的 opset_import: {domain or 'ai.onnx'}")
        opsets[domain] = int(item.version)
    if "" not in opsets and "ai.onnx" in opsets:
        opsets = {("" if k == "ai.onnx" else k): v for k, v in opsets.items()}
    if "" not in opsets:
        raise ModelImportError("模型缺少标准域 opset_import（ai.onnx）")
    standard = opsets[""]
    if standard < STANDARD_OPSET_MIN or standard > STANDARD_OPSET_MAX:
        raise ModelImportError(
            f"标准域 opset={standard} 不在 M0 支持范围 "
            f"[{STANDARD_OPSET_MIN}, {STANDARD_OPSET_MAX}]"
        )
    for domain, version in opsets.items():
        if domain and domain != "ai.onnx.ml" and version <= 0:
            raise ModelImportError(f"自定义 domain {domain} 的 opset 非法: {version}")
    return opsets


def _default_input_layout(rank: int) -> str:
    # ONNX 不携带 layout。M0 采用显式约定：4D 模型输入按 NCHW，其余按 linear。
    return "NCHW" if rank == 4 else LINEAR_LAYOUT


def _constant_layouts(nodes, initializer_names: set[str]) -> dict[str, str]:
    """根据消费算子的参数位置，为常量声明显式 layout。"""
    layouts = {name: LINEAR_LAYOUT for name in initializer_names}
    for node in nodes:
        op = node.op_type
        for index, name in enumerate(node.input):
            if name not in initializer_names:
                continue
            if op == "Conv" and index == 1:
                layouts[name] = "OIHW"
            elif op == "Conv" and index == 2:
                layouts[name] = "C"
            elif op in ("BatchNormalization", "MaxPool", "AveragePool") and index >= 1:
                layouts[name] = "C"
    return layouts


def _make_value_type(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    layout: str,
    where: str,
) -> TensorType:
    if dtype in QUANTIZED_DTYPES:
        raise ModelImportError(
            f"{where}: 量化 dtype {dtype} 需要显式量化参数，M0 尚未实现量化（M2）"
        )
    try:
        return TensorType(dtype=dtype, shape=shape, layout=layout)
    except Exception as exc:
        raise ModelImportError(f"{where}: 生成张量类型失败: {exc}") from exc


def _attr_repr(attrs: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(attrs.items()))


def _span(model_file: str, node, op_type: str, domain: str = "") -> SourceSpan:
    return SourceSpan(
        file=model_file,
        node_name=node.name or None,
        op_name=f"{domain + '::' if domain else ''}{op_type}",
    )


def _resolve_types(
    out_name: str,
    spec: OpSpec | None,
    attrs: dict,
    input_types: list[TensorType],
    vi_map: dict,
    model_file: str,
) -> TensorType:
    inferred = None
    if spec is not None and spec.infer_types is not None:
        try:
            inferred_list = spec.infer_types(attrs, input_types)
        except ShapeInferenceError as exc:
            raise ModelImportError(
                f"{out_name}: 算子 {spec.op_type} 的类型推导失败: {exc}"
            ) from exc
        if inferred_list:
            inferred = inferred_list[0]

    vi = vi_map.get(out_name)
    declared = value_info_type(vi) if vi is not None else None
    if inferred is not None and declared is not None:
        if inferred.dtype != declared[0] or inferred.shape != declared[1]:
            raise ModelImportError(
                f"{out_name}: 推导结果 {inferred} 与模型声明 "
                f"{declared[0]}{declared[1]} 不一致"
            )
        return inferred
    if inferred is not None:
        return inferred
    if declared is not None:
        return _make_value_type(
            out_name, declared[0], declared[1], LINEAR_LAYOUT, f"{out_name} @ {model_file}"
        )
    raise ModelImportError(
        f"{out_name} @ {model_file}: 无法确定输出 shape/dtype。"
        "模型缺少 value_info，且没有注册可用的推导规则"
    )


def import_onnx_model(model, options: ImportOptions | None = None) -> ImportResult:
    options = options or ImportOptions()
    registry = options.registry
    proto, model_file = _load_model(model)
    opsets = _collect_opsets(proto)
    standard_opset = opsets[""]

    graph = proto.graph
    initializers = {init.name: initializer_to_numpy(init) for init in graph.initializer}
    init_names = set(initializers)

    report = ImportReport(
        frontend="onnx",
        model_file=model_file,
        module_name=options.module_name or graph.name or Path(model_file).stem,
        opset=dict(opsets),
    )

    # ---- 图输入 ----
    inputs: list[Value] = []
    tensor_types: dict[str, TensorType] = {}
    for vi in graph.input:
        if vi.name in init_names:
            continue
        declared = value_info_type(vi)
        if declared is None:
            if vi.name not in options.input_shapes:
                raise ModelImportError(
                    f"图输入 {vi.name} 含动态维或缺少类型；"
                    "M0 只支持静态 shape，请用 input_shapes 显式指定"
                )
            dtype = ONNX_DTYPE_TO_NAME.get(vi.type.tensor_type.elem_type)
            if dtype is None:
                raise ModelImportError(f"图输入 {vi.name} 缺少 dtype")
            shape = tuple(options.input_shapes[vi.name])
        else:
            dtype, shape = declared
            if vi.name in options.input_shapes:
                shape = tuple(options.input_shapes[vi.name])
        layout = _default_input_layout(len(shape))
        ttype = _make_value_type(vi.name, dtype, shape, layout, f"图输入 {vi.name}")
        value = Value(name=vi.name, type=ttype)
        inputs.append(value)
        tensor_types[vi.name] = ttype
        report.inputs.append(
            {
                "name": vi.name,
                "dtype": dtype,
                "shape": list(shape),
                "layout": layout,
            }
        )

    # ---- value_info 索引 ----
    vi_map = {vi.name: vi for vi in graph.value_info}
    vi_map.update({vi.name: vi for vi in graph.output})

    const_layouts = _constant_layouts(graph.node, init_names)

    nodes = []
    node_counts: dict[str, int] = {"constant": 0, "standard": 0, "custom": 0,
                                   "fallback": 0}
    op_counts: dict[str, int] = {}
    fallback_entries: list[FallbackEntry] = []
    unsupported: list[tuple[str, str]] = []

    # ---- initializer -> ConstantNode（先定义，保证 SSA） ----
    for name in sorted(init_names):
        array = initializers[name]
        dtype = numpy_dtype_name(array)
        ttype = _make_value_type(
            name, dtype, tuple(array.shape), const_layouts[name], f"initializer {name}"
        )
        span = SourceSpan(file=model_file, node_name=name, op_name="Initializer")
        nodes.append(
            ConstantNode(
                inputs=[],
                outputs=[Value(name=name, type=ttype)],
                span=span,
                value=array,
            )
        )
        tensor_types[name] = ttype
        node_counts["constant"] += 1

    def define_outputs(
        out_names: list[str],
        out_types: list[TensorType],
        node_desc: str,
    ) -> list[Value]:
        values = []
        for out_name, out_type in zip(out_names, out_types):
            if out_name in tensor_types:
                raise ModelImportError(
                    f"{node_desc}: 输出名与已定义张量冲突: {out_name}"
                )
            tensor_types[out_name] = out_type
            values.append(Value(name=out_name, type=out_type))
        return values

    # ---- 逐节点转换 ----
    for node in graph.node:
        domain = node.domain or ""
        op_type = node.op_type
        span = _span(model_file, node, op_type, domain)
        where = str(span)
        out_names = [name for name in node.output if name]
        in_names = [name for name in node.input if name]
        op_key = f"{domain + '::' if domain else ''}{op_type}"
        op_counts[op_key] = op_counts.get(op_key, 0) + 1

        if not out_names:
            report.warnings.append(f"{where}: 算子没有输出，已跳过")
            continue
        for in_name in in_names:
            if in_name not in tensor_types:
                raise ModelImportError(f"{where}: 引用了未定义的输入 {in_name}")
        input_types = [tensor_types[name] for name in in_names]
        attrs, subgraphs = convert_attrs(node)

        # ONNX Constant 直接落成 ConstantNode。
        if domain in ("", "ai.onnx") and op_type == "Constant":
            if subgraphs:
                raise ModelImportError(f"{where}: Constant 不允许包含子图属性")
            value = attrs.get("value")
            if not isinstance(value, np.ndarray):
                raise ModelImportError(
                    f"{where}: M0 只支持 tensor 形式的 Constant，"
                    f"实际属性: {_attr_repr(attrs)}"
                )
            if len(out_names) != 1:
                raise ModelImportError(f"{where}: Constant 只支持单输出")
            ttype = _make_value_type(
                out_names[0], numpy_dtype_name(value), tuple(value.shape),
                LINEAR_LAYOUT, where,
            )
            nodes.append(
                ConstantNode(
                    inputs=[],
                    outputs=[Value(name=out_names[0], type=ttype)],
                    span=span,
                    attrs={},
                    value=value,
                )
            )
            tensor_types[out_names[0]] = ttype
            node_counts["constant"] += 1
            continue

        spec = registry.lookup(domain, op_type)
        categories = "custom" if registry.is_custom(domain, op_type) else "standard"

        # Reshape 的 shape 来自常量输入；导入期把常量值固化为属性，便于类型推导。
        if op_type == "Reshape" and "shape" not in attrs:
            shape_input = node.input[1] if len(node.input) > 1 else ""
            if shape_input in initializers:
                attrs["shape"] = initializers[shape_input].astype(np.int64).tolist()

        # 归一化属性；自定义算子的 attrs_normalizer 也走同一入口。
        if spec is not None and spec.normalize_attrs is not None:
            try:
                attrs = spec.normalize_attrs(attrs, standard_opset)
            except ShapeInferenceError as exc:
                raise ModelImportError(f"{where}: 属性归一化失败: {exc}") from exc

        reason = None
        backend = spec.fallback_backend if spec is not None else "scalar_host"
        supported = True
        if subgraphs:
            supported = False
            attrs["subgraph_attrs"] = subgraphs
            reason = f"子图属性无法展开: {subgraphs}"
        if spec is None:
            supported = False
            reason = reason or "未注册的算子，无标准/自定义转换规则"
        elif not spec.supports_npu(attrs, input_types):
            supported = False
            if registry.is_custom(domain, op_type):
                reason = reason or "客户自定义算子未声明 NPU 映射"
            else:
                reason = reason or "算子不满足 NPU 支持条件"

        # 解析输出类型。
        out_types: list[TensorType] = []
        for index, out_name in enumerate(out_names):
            if supported and spec is not None and index > 0:
                # M0 所有 Call 算子的 NumPy kernel 都是单输出。
                raise ModelImportError(
                    f"{where}: M0 不支持 {op_key} 的多输出 "
                    f"(输出 {out_names})"
                )
            if spec is not None:
                # 即使进入 fallback，只要注册了推导规则就优先使用，
                # 避免要求模型为非标准算子额外提供 value_info。
                out_types.append(
                    _resolve_types(
                        out_name, spec, attrs, input_types, vi_map, model_file
                    )
                )
                continue
            vi = vi_map.get(out_name)
            declared = value_info_type(vi) if vi is not None else None
            if declared is None:
                raise ModelImportError(
                    f"{where}: fallback 输出 {out_name} 缺少静态 shape/dtype。"
                    "无法推导时导入失败，避免生成语义不明的节点"
                )
            out_types.append(
                _make_value_type(
                    out_name, declared[0], declared[1], LINEAR_LAYOUT, where
                )
            )

        values = define_outputs(out_names, out_types, where)
        if supported:
            nodes.append(
                CallNode(
                    inputs=in_names,
                    outputs=values,
                    span=span,
                    attrs=attrs,
                    domain=domain,
                    op_type=op_type,
                    canonical_name=spec.canonical_name,
                )
            )
            node_counts[categories] += 1
        else:
            if options.fallback_policy == FALLBACK_ERROR:
                unsupported.append((op_key, str(span)))
            nodes.append(
                FallbackNode(
                    inputs=in_names,
                    outputs=values,
                    span=span,
                    attrs=attrs,
                    domain=domain,
                    op_type=op_type,
                    backend=backend,
                    reason=reason,
                )
            )
            node_counts["fallback"] += 1
            fallback_entries.append(
                FallbackEntry(
                    op=op_type,
                    domain=domain,
                    backend=backend,
                    reason=reason,
                    span=str(span),
                )
            )

    if options.fallback_policy == FALLBACK_ERROR and unsupported:
        raise UnsupportedOperatorError(unsupported)

    # ---- 图输出 ----
    outputs: list[Value] = []
    for vi in graph.output:
        if vi.name not in tensor_types:
            raise ModelImportError(f"图输出未定义: {vi.name}")
        ttype = tensor_types[vi.name]
        declared = value_info_type(vi)
        if declared is not None and ttype.dtype != declared[0]:
            raise ModelImportError(
                f"图输出 {vi.name} dtype 不一致: 推导 {ttype.dtype}, "
                f"模型声明 {declared[0]}"
            )
        outputs.append(Value(name=vi.name, type=ttype))

    nest_graph = Graph(inputs=inputs, outputs=outputs, nodes=nodes)
    module_name = options.module_name or graph.name or Path(model_file).stem
    module = Module(
        name=module_name,
        graph=nest_graph,
        metadata={
            "frontend": "onnx",
            "model_file": model_file,
            "opset": opsets,
            "fallback_policy": options.fallback_policy,
            "input_shapes_override": {
                k: list(v) for k, v in options.input_shapes.items()
            },
            "standard_opset_range": [STANDARD_OPSET_MIN, STANDARD_OPSET_MAX],
        },
    )

    report.module_name = module_name
    report.node_counts = {k: v for k, v in node_counts.items() if v}
    report.op_counts = dict(sorted(op_counts.items()))
    report.fallback = fallback_entries
    return ImportResult(module=module, report=report)
