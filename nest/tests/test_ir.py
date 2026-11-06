import json

import numpy as np
import pytest

from nest.errors import GraphValidationError
from nest.ir import (
    CallNode,
    ConstantNode,
    FallbackNode,
    Graph,
    Module,
    QuantInfo,
    SourceSpan,
    TensorType,
    Value,
)


def _f32(shape, layout="linear"):
    return TensorType("float32", tuple(shape), layout)


def _module() -> Module:
    x = Value("x", _f32([1, 4]))
    w = Value("w", _f32([3, 4]))
    span = SourceSpan(file="tiny.onnx", node_name="gemm_1", op_name="Gemm")
    const = ConstantNode(
        inputs=[],
        outputs=[w],
        span=SourceSpan(file="tiny.onnx", op_name="Initializer"),
        value=np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    h = Value("h", _f32([1, 3]))
    call = CallNode(
        inputs=["x", "w"],
        outputs=[h],
        span=span,
        attrs={"transB": 1},
        op_type="Gemm",
        canonical_name="gemm",
    )
    y = Value("y", _f32([1, 3]))
    fallback = FallbackNode(
        inputs=["h"],
        outputs=[y],
        span=SourceSpan(file="tiny.onnx", node_name="cos_1", op_name="Cos"),
        attrs={"min": np.array([0.0, 1.0], dtype=np.float32)},
        op_type="Cos",
        backend="scalar_host",
        reason="NPU 不支持三角函数",
    )
    graph = Graph(inputs=[x], outputs=[y], nodes=[const, call, fallback])
    return Module(name="tiny", graph=graph, metadata={"frontend": "onnx", "opset": {"": 17}})


def test_tensor_type_requires_explicit_layout():
    with pytest.raises(GraphValidationError):
        TensorType("float32", (1, 4), "")


def test_tensor_type_rejects_dynamic_shape():
    with pytest.raises(GraphValidationError):
        TensorType("float32", ("batch", 4), "linear")


def test_quantized_dtype_requires_quant_info():
    with pytest.raises(GraphValidationError):
        TensorType("int8", (1, 4), "linear")
    quant = QuantInfo(
        mode="int", bits=8, scale=(0.5,), zero_point=(0,), axis=0
    )
    t = TensorType("int8", (1, 4), "linear", quant=quant)
    assert t.is_quantized


def test_fallback_requires_backend_and_reason():
    x = Value("x", _f32([1, 4]))
    y = Value("y", _f32([1, 4]))
    with pytest.raises(GraphValidationError):
        FallbackNode(
            inputs=["x"], outputs=[y],
            span=SourceSpan(file="m.onnx", op_name="Mystery"),
            op_type="Mystery", backend="", reason="x",
        )
    with pytest.raises(GraphValidationError):
        FallbackNode(
            inputs=["x"], outputs=[y],
            span=SourceSpan(file="m.onnx", op_name="Mystery"),
            op_type="Mystery", backend="scalar_host", reason="",
        )


def test_graph_rejects_ssa_violation():
    x = Value("x", _f32([1, 4]))
    y = Value("x", _f32([1, 4]))
    const = ConstantNode(
        inputs=[], outputs=[y],
        span=SourceSpan(file="m.onnx", op_name="Initializer"),
        value=np.zeros((1, 4), dtype=np.float32),
    )
    with pytest.raises(GraphValidationError, match="重复定义"):
        Graph(inputs=[x], outputs=[y], nodes=[const])


def test_graph_rejects_use_before_def():
    x = Value("x", _f32([1, 4]))
    y = Value("y", _f32([1, 4]))
    call = CallNode(
        inputs=["missing"], outputs=[y],
        span=SourceSpan(file="m.onnx", op_name="Relu"),
        op_type="Relu", canonical_name="relu",
    )
    with pytest.raises(GraphValidationError, match="未定义"):
        Graph(inputs=[x], outputs=[y], nodes=[call])


def test_module_serialization_roundtrip():
    module = _module()
    data = json.loads(module.to_json())
    loaded = Module.from_dict(data)
    assert loaded.to_json() == module.to_json()
    assert loaded.metadata["opset"] == {"": 17}

    fallback = loaded.fallback_nodes[0]
    assert fallback.backend == "scalar_host"
    assert isinstance(fallback.attrs["min"], np.ndarray)
    np.testing.assert_array_equal(
        fallback.attrs["min"], np.array([0.0, 1.0], dtype=np.float32)
    )


def test_printer_contains_all_nodes():
    text = str(_module())
    assert "module @tiny" in text
    assert "constant(" in text
    assert "nest.gemm" in text
    assert "fallback[scalar_host]" in text
    assert "output %y" in text
