import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

from nest.errors import MissingScalarImplementationError
from nest.frontend import from_onnx
from nest.registry import register_custom, register_fallback_kernel
from nest.runtime import run_module

from model_builders import (
    make_cnn,
    make_custom_and_fallback,
    make_mlp,
    make_reshape,
    make_unknown_op,
)


def _ref(path, feed):
    session = ReferenceEvaluator(onnx.load(str(path)))
    return session.run(None, feed)


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


def test_mlp_matches_onnx_reference(tmp_path):
    path = make_mlp(tmp_path / "mlp.onnx")
    x = np.random.default_rng(3).standard_normal((1, 4)).astype(np.float32)
    expected = _ref(path, {"x": x})[0]

    result = from_onnx(path)
    actual = run_module(result.module, {"x": x})["out"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_cnn_matches_onnx_reference(tmp_path):
    path = make_cnn(tmp_path / "cnn.onnx")
    x = np.random.default_rng(4).standard_normal((1, 1, 8, 8)).astype(np.float32)
    expected = _ref(path, {"x": x})[0]

    result = from_onnx(path)
    actual = run_module(result.module, {"x": x})["out"]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_reshape_runs(tmp_path):
    path = make_reshape(tmp_path / "reshape.onnx")
    x = np.random.default_rng(5).standard_normal((1, 4)).astype(np.float32)
    expected = _ref(path, {"x": x})[0]
    actual = run_module(from_onnx(path).module, {"x": x})["out"]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_standard_custom_fallback_all_execute(tmp_path):
    _register_scale_shift(npu_support=True)
    path = make_custom_and_fallback(tmp_path / "combo.onnx")
    model = onnx.load(str(path))
    weights = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
    x = np.random.default_rng(6).standard_normal((1, 4)).astype(np.float32)
    expected = (np.cos(x @ weights["W"].T + weights["B"]) * 2.0 + 1.0).astype(
        np.float32
    )

    result = from_onnx(path)
    actual = run_module(result.module, {"x": x})["out"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    kinds = [type(n).__name__ for n in result.module.graph.nodes]
    assert "FallbackNode" in kinds  # Cos
    assert "CallNode" in kinds


def test_custom_op_fallback_executes_after_registration(tmp_path):
    _register_scale_shift(npu_support=False)
    path = make_custom_and_fallback(tmp_path / "combo.onnx")
    x = np.random.default_rng(7).standard_normal((1, 4)).astype(np.float32)
    result = from_onnx(path)
    actual = run_module(result.module, {"x": x})["out"]
    model = onnx.load(str(path))
    weights = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
    expected = (np.cos(x @ weights["W"].T + weights["B"]) * 2.0 + 1.0).astype(
        np.float32
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_missing_fallback_kernel_raises_then_registered_kernel_runs(tmp_path):
    path = make_unknown_op(tmp_path / "unknown.onnx")
    x = np.random.default_rng(8).standard_normal((1, 3)).astype(np.float32)
    result = from_onnx(path)

    with pytest.raises(MissingScalarImplementationError, match="scalar_host"):
        run_module(result.module, {"x": x})

    register_fallback_kernel(
        "com.example", "Mystery", "scalar_host",
        lambda inputs, attrs: [(inputs[0] + 1.0).astype(inputs[0].dtype)],
    )
    np.testing.assert_allclose(
        run_module(result.module, {"x": x})["out"], x + 1.0, rtol=1e-6
    )


def test_run_validates_input_shape_and_names(tmp_path):
    path = make_mlp(tmp_path / "mlp.onnx")
    module = from_onnx(path).module

    with pytest.raises(ValueError, match="缺少输入"):
        run_module(module, {})
    with pytest.raises(ValueError, match="多余的输入"):
        run_module(module, {"x": np.zeros((1, 4), np.float32),
                            "extra": np.zeros((1, 1), np.float32)})
    with pytest.raises(ValueError, match="shape"):
        run_module(module, {"x": np.zeros((2, 4), np.float32)})
    with pytest.raises(ValueError, match="dtype"):
        run_module(module, {"x": np.zeros((1, 4), np.float64)})
