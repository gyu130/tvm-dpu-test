"""M1 图优化 pass 单元测试。"""

import json

import numpy as np
import pytest

from nest.errors import GraphValidationError, PartitionError
from nest.frontend import from_onnx
from nest.ir import (
    CallNode,
    ConstantNode,
    FallbackNode,
    Graph,
    Module,
    SourceSpan,
    TensorType,
    Value,
)
from nest.passes import (
    OptimizeOptions,
    analyze_partition,
    canonicalize,
    constant_fold,
    cse,
    dce,
    require_fully_npu,
    run_passes,
    simplify,
    validate_layouts,
)
from nest.runtime import run_module

from model_builders import make_cnn

SPAN = SourceSpan(file="unit.onnx", node_name="n", op_name="op")


def _value(name, shape, layout="linear", dtype="float32"):
    return Value(name=name, type=TensorType(dtype=dtype, shape=tuple(shape), layout=layout))


def _const(name, array, layout="linear"):
    array = np.asarray(array)
    return ConstantNode(
        inputs=[],
        outputs=[_value(name, array.shape, layout, str(array.dtype))],
        span=SPAN,
        value=array,
    )


def _call(op_type, canonical, inputs, outputs, domain="", attrs=None):
    return CallNode(
        inputs=list(inputs),
        outputs=outputs,
        span=SPAN,
        attrs=dict(attrs or {}),
        domain=domain,
        op_type=op_type,
        canonical_name=canonical,
    )


def _module(name, inputs, nodes, outputs):
    return Module(name=name, graph=Graph(inputs=list(inputs), outputs=list(outputs), nodes=list(nodes)))


def _run_one(pass_fn, module):
    output = pass_fn(module, OptimizeOptions())
    return output


def test_canonicalize_removes_identity():
    x = _value("x", (1, 4))
    identity = _call(
        "Identity", "identity", ["x"], [_value("i", (1, 4))]
    )
    relu = _call("Relu", "relu", ["i"], [_value("y", (1, 4))])
    module = _module("m", [x], [identity, relu], [relu.outputs[0]])

    output = _run_one(canonicalize, module)
    assert output.removed == 1
    assert len(output.module.graph.nodes) == 1
    assert output.module.graph.nodes[0].inputs == ["x"]
    assert output.module.graph.outputs[0].name == "y"


def test_constant_fold_and_fallback_not_folded():
    left = _const("a", np.arange(6, dtype=np.float32).reshape(2, 3))
    right = _const("b", np.ones((2, 3), dtype=np.float32))
    add = _call("Add", "add", ["a", "b"], [_value("c", (2, 3))])
    fallback = FallbackNode(
        inputs=["c"],
        outputs=[_value("d", (2, 3))],
        span=SPAN,
        domain="com.example",
        op_type="Mystery",
        backend="scalar_host",
        reason="未注册",
    )
    module = _module("m", [], [left, right, add, fallback], [fallback.outputs[0]])

    output = _run_one(constant_fold, module)
    assert output.folded == 1
    folded_node = output.module.graph.nodes[2]
    assert isinstance(folded_node, ConstantNode)
    assert np.allclose(folded_node.value, np.arange(6).reshape(2, 3) + 1)
    # fallback 节点保持原样，不得在 host 上提前执行。
    assert isinstance(output.module.graph.nodes[3], FallbackNode)


def test_dce_removes_dead_nodes():
    x = _value("x", (1, 4))
    keep = _call("Relu", "relu", ["x"], [_value("a", (1, 4))])
    dead = _call("Sigmoid", "sigmoid", ["x"], [_value("b", (1, 4))])
    module = _module("m", [x], [keep, dead], [keep.outputs[0]])

    output = _run_one(dce, module)
    assert output.removed == 1
    assert [node.output_names[0] for node in output.module.graph.nodes] == ["a"]


def test_cse_merges_duplicate_calls():
    x = _value("x", (1, 4))
    first = _call("Relu", "relu", ["x"], [_value("a", (1, 4))])
    second = _call("Relu", "relu", ["x"], [_value("b", (1, 4))])
    add = _call("Add", "add", ["a", "b"], [_value("out", (1, 4))])
    module = _module("m", [x], [first, second, add], [add.outputs[0]])

    output = _run_one(cse, module)
    assert output.removed == 1
    assert len(output.module.graph.nodes) == 2
    assert output.module.graph.nodes[1].inputs == ["a", "a"]


def test_simplify_removes_identity_arithmetic():
    x = _value("x", (1, 4))
    zero = _const("zero", np.zeros((1, 4), dtype=np.float32))
    one = _const("one", np.ones((1, 4), dtype=np.float32))
    add = _call("Add", "add", ["x", "zero"], [_value("y", (1, 4))])
    mul = _call("Mul", "mul", ["y", "one"], [_value("out", (1, 4))])
    module = _module("m", [x], [zero, one, add, mul], [mul.outputs[0]])

    output = _run_one(simplify, module)
    assert output.removed == 2
    assert output.module.graph.outputs[0].name == "x"


def test_layout_validation_rejects_bad_conv_layout():
    x = _value("x", (1, 1, 4, 4), "NCHW")
    weight = _value("w", (2, 1, 3, 3), "linear")
    conv = _call(
        "Conv", "conv2d", ["x", "w"], [_value("y", (1, 2, 2, 2), "NCHW")],
        attrs={"group": 1},
    )
    module = _module("m", [x, weight], [conv], [conv.outputs[0]])
    with pytest.raises(GraphValidationError, match="OIHW"):
        validate_layouts(module, OptimizeOptions())


def _conv_bn_graph(with_relu=False):
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((2, 1, 3, 3)).astype(np.float32)
    bias = rng.standard_normal((2,)).astype(np.float32)
    scale = rng.standard_normal((2,)).astype(np.float32) * 0.2 + 1.0
    beta = rng.standard_normal((2,)).astype(np.float32)
    mean = rng.standard_normal((2,)).astype(np.float32) * 0.1
    var = (rng.standard_normal((2,)) * 0.1 + 1.0).astype(np.float32)

    x = _value("x", (1, 1, 4, 4), "NCHW")
    nodes = [
        _const("w", weight, "OIHW"),
        _const("b", bias, "C"),
        _const("scale", scale, "C"),
        _const("beta", beta, "C"),
        _const("mean", mean, "C"),
        _const("var", var, "C"),
        _call(
            "Conv", "conv2d", ["x", "w", "b"], [_value("c", (1, 2, 4, 4), "NCHW")],
            attrs={"pads": [1, 1, 1, 1], "strides": [1, 1], "dilations": [1, 1],
                   "group": 1, "auto_pad": "NOTSET"},
        ),
        _call(
            "BatchNormalization", "batch_norm",
            ["c", "scale", "beta", "mean", "var"],
            [_value("bn", (1, 2, 4, 4), "NCHW")],
            attrs={"epsilon": 1e-5},
        ),
    ]
    output = "bn"
    if with_relu:
        nodes.append(_call("Relu", "relu", ["bn"], [_value("out", (1, 2, 4, 4), "NCHW")]))
        output = "out"
    output_value = next(
        value for node in nodes for value in node.outputs if value.name == output
    )
    return _module("m", [x], nodes, [output_value])


def test_conv_bn_fusion_matches_reference():
    module = _conv_bn_graph()
    x = np.random.default_rng(7).standard_normal((1, 1, 4, 4)).astype(np.float32)
    before = run_module(module, {"x": x})["bn"]

    optimized, stats = run_passes(module, OptimizeOptions())
    canonical = [n.canonical_name for n in optimized.graph.nodes if isinstance(n, CallNode)]
    assert "batch_norm" not in canonical
    assert any(s.fused == 1 for s in stats if s.name == "fusion")
    after = run_module(optimized, {"x": x})["bn"]
    assert np.allclose(before, after, atol=1e-5, rtol=1e-5)


def test_conv_bn_relu_fuses_to_conv_relu():
    module = _conv_bn_graph(with_relu=True)
    x = np.random.default_rng(8).standard_normal((1, 1, 4, 4)).astype(np.float32)
    optimized, _ = run_passes(module, OptimizeOptions())
    names = [n.canonical_name for n in optimized.graph.nodes if isinstance(n, CallNode)]
    assert names == ["conv2d_relu"]
    node = optimized.graph.nodes[-1]
    assert node.domain == "nest.internal"
    before = run_module(module, {"x": x})["out"]
    after = run_module(optimized, {"x": x})["out"]
    assert np.allclose(before, after, atol=1e-5, rtol=1e-5)


def test_gemm_relu_fusion():
    x = _value("x", (1, 4))
    weight = _const("w", np.ones((3, 4), dtype=np.float32))
    bias = _const("b", np.zeros((3,), dtype=np.float32))
    gemm = _call(
        "Gemm", "gemm", ["x", "w", "b"], [_value("h", (1, 3))],
        attrs={"transB": 1, "alpha": 1.0, "beta": 1.0, "transA": 0},
    )
    relu = _call("Relu", "relu", ["h"], [_value("out", (1, 3))])
    module = _module("m", [x], [weight, bias, gemm, relu], [relu.outputs[0]])
    optimized, _ = run_passes(module, OptimizeOptions())
    names = [n.canonical_name for n in optimized.graph.nodes if isinstance(n, CallNode)]
    assert names == ["gemm_relu"]
    assert optimized.graph.nodes[-1].domain == "nest.internal"

    values = np.random.default_rng(9).standard_normal((1, 4)).astype(np.float32)
    before = run_module(module, {"x": values})["out"]
    after = run_module(optimized, {"x": values})["out"]
    assert np.allclose(before, after)


def test_fusion_rejected_when_consumer_not_unique():
    module = _conv_bn_graph()
    conv_output = module.graph.tensors["c"]
    relu = _call("Relu", "relu", ["c"], [_value("r", (1, 2, 4, 4), "NCHW")])
    nodes = list(module.graph.nodes) + [relu]
    new_module = Module(
        name="m",
        graph=Graph(
            inputs=module.graph.inputs,
            outputs=[module.graph.tensors["bn"], relu.outputs[0]],
            nodes=nodes,
        ),
    )
    optimized, stats = run_passes(new_module, OptimizeOptions())
    fusion_stats = next(s for s in stats if s.name == "fusion")
    assert fusion_stats.fused == 0
    assert any("多个消费者" in note for note in fusion_stats.notes)


def test_pipeline_is_idempotent(tmp_path):
    path = make_cnn(tmp_path / "cnn.onnx")
    module = from_onnx(path).module
    first, _ = run_passes(module, OptimizeOptions())
    second, _ = run_passes(first, OptimizeOptions())
    assert json.dumps(first.graph.to_dict(), sort_keys=True) == json.dumps(
        second.graph.to_dict(), sort_keys=True
    )


def test_pipeline_stop_after_and_disable():
    module = _module(
        "m",
        [_value("x", (1, 4))],
        [_call("Relu", "relu", ["x"], [_value("y", (1, 4))])],
        [_value("y", (1, 4))],
    )
    _, stats = run_passes(module, OptimizeOptions(stop_after="constant_fold"))
    assert [s.name for s in stats] == ["canonicalize", "constant_fold"]
    _, stats = run_passes(module, OptimizeOptions(disabled=("fusion", "layout")))
    assert "fusion" not in [s.name for s in stats]


def test_partition_reports_fallback():
    x = _value("x", (1, 4))
    fallback = FallbackNode(
        inputs=["x"],
        outputs=[_value("y", (1, 4))],
        span=SPAN,
        domain="com.example",
        op_type="Mystery",
        backend="scalar_host",
        reason="未注册算子",
    )
    module = _module("m", [x], [fallback], [fallback.outputs[0]])
    report = analyze_partition(module)
    assert not report.is_fully_npu
    with pytest.raises(PartitionError, match="Mystery"):
        require_fully_npu(report)
