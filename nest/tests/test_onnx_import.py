import pytest

from nest.errors import ModelImportError, UnsupportedOperatorError
from nest.frontend import ImportOptions, from_onnx
from nest.ir import CallNode, FallbackNode
from nest.registry import register_custom

from model_builders import (
    make_bad_opset,
    make_cnn,
    make_custom_and_fallback,
    make_dynamic_batch,
    make_mlp,
    make_reshape,
    make_two_unknown_ops,
    make_unknown_op,
)


def _register_scale_shift(npu_support=True):
    register_custom(
        "com.example",
        "ScaleShift",
        canonical_name="custom.scale_shift",
        infer_types=lambda attrs, its: [its[0]],
        kernel=lambda inputs, attrs: [
            (inputs[0] * attrs["scale"] + attrs["shift"]).astype(inputs[0].dtype)
        ],
        npu_support=npu_support,
    )


def test_import_mlp_node_categories(tmp_path):
    path = make_mlp(tmp_path / "mlp.onnx")
    result = from_onnx(path)
    graph = result.module.graph

    kinds = [type(n).__name__ for n in graph.nodes]
    assert kinds.count("ConstantNode") == 4
    assert kinds.count("CallNode") == 3
    assert kinds.count("FallbackNode") == 0
    assert result.report.node_counts == {"constant": 4, "standard": 3}
    assert result.report.opset == {"": 17}
    assert result.report.fallback == []


def test_import_cnn_layouts(tmp_path):
    path = make_cnn(tmp_path / "cnn.onnx")
    result = from_onnx(path)
    conv = next(n for n in result.module.graph.nodes if isinstance(n, CallNode))
    assert conv.canonical_name == "conv2d"
    assert conv.outputs[0].type.layout == "NCHW"
    assert conv.inputs[1] == "conv_w"
    weight = result.module.graph.tensors["conv_w"]
    assert weight.type.layout == "OIHW"


def test_import_standard_unsupported_goes_fallback(tmp_path):
    _register_scale_shift()
    path = make_custom_and_fallback(tmp_path / "combo.onnx")
    result = from_onnx(path)
    fallbacks = [n for n in result.module.graph.nodes if isinstance(n, FallbackNode)]
    assert len(fallbacks) == 1
    assert fallbacks[0].op_type == "Cos"
    assert fallbacks[0].backend == "scalar_host"
    assert "NPU" in fallbacks[0].reason
    assert result.report.node_counts.get("custom") == 1
    assert result.report.node_counts.get("fallback") == 1


def test_import_dynamic_shape_requires_override(tmp_path):
    path = make_dynamic_batch(tmp_path / "dyn.onnx")
    with pytest.raises(ModelImportError, match="动态维"):
        from_onnx(path)

    result = from_onnx(path, input_shapes={"x": (1, 4)})
    assert result.module.graph.inputs[0].type.shape == (1, 4)


def test_import_rejects_unsupported_opset(tmp_path):
    path = make_bad_opset(tmp_path / "bad.onnx")
    with pytest.raises(ModelImportError, match="opset"):
        from_onnx(path)


def test_unknown_op_marked_as_fallback(tmp_path):
    path = make_unknown_op(tmp_path / "unknown.onnx")
    result = from_onnx(path)
    fallback = result.module.graph.nodes[0]
    assert isinstance(fallback, FallbackNode)
    assert fallback.backend == "scalar_host"
    assert fallback.domain == "com.example"
    assert result.report.fallback[0].op == "Mystery"
    assert "scalar_host" in result.report.summary_text()


def test_unknown_op_without_value_info_fails(tmp_path):
    path = make_unknown_op(tmp_path / "unknown.onnx", with_value_info=False)
    with pytest.raises(ModelImportError, match="缺少静态 shape"):
        from_onnx(path)


def test_fallback_error_policy_lists_all_unsupported(tmp_path):
    path = make_two_unknown_ops(tmp_path / "two.onnx")
    with pytest.raises(UnsupportedOperatorError) as exc_info:
        from_onnx(path, fallback_policy="error")
    message = str(exc_info.value)
    assert "MysteryA" in message
    assert "MysteryB" in message
    assert "mystery_a" in message
    assert len(exc_info.value.unsupported) == 2


def test_custom_op_without_npu_support_is_fallback(tmp_path):
    _register_scale_shift(npu_support=False)
    path = make_custom_and_fallback(tmp_path / "combo.onnx")
    result = from_onnx(path)
    fallbacks = [n for n in result.module.graph.nodes if isinstance(n, FallbackNode)]
    ops = sorted(n.op_type for n in fallbacks)
    assert ops == ["Cos", "ScaleShift"]
    assert result.report.node_counts == {
        "constant": 2,
        "standard": 1,
        "fallback": 2,
    }


def test_reshape_type_inference(tmp_path):
    path = make_reshape(tmp_path / "reshape.onnx")
    result = from_onnx(path)
    reshape = [
        n for n in result.module.graph.nodes
        if isinstance(n, CallNode) and n.op_type == "Reshape"
    ][0]
    assert reshape.outputs[0].type.shape == (1, 2, 2)
    assert reshape.attrs["shape"] == [1, 2, 2]


def test_options_reject_invalid_policy():
    with pytest.raises(ModelImportError):
        ImportOptions(fallback_policy="ignore")
