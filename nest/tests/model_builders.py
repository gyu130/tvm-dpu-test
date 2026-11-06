"""测试用 tiny ONNX 模型构造器。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _save(model: onnx.ModelProto, path: Path) -> Path:
    onnx.save(model, str(path))
    return path


def make_mlp(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((8, 4)) * 0.5).astype(np.float32)
    b1 = (rng.standard_normal((8,)) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((3, 8)) * 0.5).astype(np.float32)
    b2 = (rng.standard_normal((3,)) * 0.1).astype(np.float32)
    initializers = [
        numpy_helper.from_array(w1, "W1"),
        numpy_helper.from_array(b1, "B1"),
        numpy_helper.from_array(w2, "W2"),
        numpy_helper.from_array(b2, "B2"),
    ]
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "B1"], ["h"], name="gemm_1", transB=1),
        helper.make_node("Relu", ["h"], ["r"], name="relu_1"),
        helper.make_node("Gemm", ["r", "W2", "B2"], ["out"], name="gemm_2", transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return _save(model, path)


def make_cnn(path: Path, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((4, 1, 3, 3)) * 0.3).astype(np.float32)
    b = (rng.standard_normal((4,)) * 0.1).astype(np.float32)
    wfc = (rng.standard_normal((10, 64)) * 0.2).astype(np.float32)
    bfc = (rng.standard_normal((10,)) * 0.1).astype(np.float32)
    initializers = [
        numpy_helper.from_array(w, "conv_w"),
        numpy_helper.from_array(b, "conv_b"),
        numpy_helper.from_array(wfc, "fc_w"),
        numpy_helper.from_array(bfc, "fc_b"),
    ]
    nodes = [
        helper.make_node(
            "Conv", ["x", "conv_w", "conv_b"], ["c"], name="conv_1",
            pads=[1, 1, 1, 1], strides=[1, 1],
        ),
        helper.make_node("Relu", ["c"], ["cr"], name="relu_1"),
        helper.make_node(
            "MaxPool", ["cr"], ["p"], name="pool_1",
            kernel_shape=[2, 2], strides=[2, 2],
        ),
        helper.make_node("Flatten", ["p"], ["flat"], name="flatten_1", axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["out"], name="fc_1",
                         transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_cnn",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 10])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return _save(model, path)


def make_reshape(path: Path) -> Path:
    shape = numpy_helper.from_array(np.array([1, 2, 2], dtype=np.int64), "shape")
    nodes = [
        helper.make_node("Relu", ["x"], ["r"], name="relu_1"),
        helper.make_node("Reshape", ["r", "shape"], ["out"], name="reshape_1"),
    ]
    graph = helper.make_graph(
        nodes,
        "reshape",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 2])],
        initializer=[shape],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return _save(model, path)


def make_dynamic_batch(path: Path) -> Path:
    nodes = [helper.make_node("Relu", ["x"], ["out"], name="relu_dyn")]
    graph = helper.make_graph(
        nodes,
        "dynamic_batch",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return _save(model, path)


def make_custom_and_fallback(path: Path, domain: str = "com.example") -> Path:
    """Gemm -> Cos(标准 fallback) -> ScaleShift(自定义算子)。"""
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((3, 4)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((3,)) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node("Gemm", ["x", "W", "B"], ["g"], name="gemm_1", transB=1),
        helper.make_node("Cos", ["g"], ["c"], name="cos_1"),
        helper.make_node(
            "ScaleShift", ["c"], ["out"], name="scale_shift_1",
            domain=domain, scale=2.0, shift=1.0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "custom_fallback",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        initializer=[
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(b, "B"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid(domain, 1),
        ],
    )
    model.ir_version = 8
    return _save(model, path)


def make_unknown_op(path: Path, domain: str = "com.example", with_value_info=True) -> Path:
    """未注册的自定义算子，用于验证 fallback 标记/报错。"""
    nodes = [helper.make_node("Mystery", ["x"], ["out"], name="mystery_1",
                              domain=domain)]
    output_shape = [1, 3] if with_value_info else ["dyn", 3]
    graph = helper.make_graph(
        nodes,
        "unknown_op",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid(domain, 1),
        ],
    )
    model.ir_version = 8
    return _save(model, path)


def make_two_unknown_ops(path: Path) -> Path:
    nodes = [
        helper.make_node("MysteryA", ["x"], ["a"], name="mystery_a",
                         domain="com.example"),
        helper.make_node("MysteryB", ["a"], ["out"], name="mystery_b",
                         domain="com.example"),
    ]
    graph = helper.make_graph(
        nodes,
        "two_unknown",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
    )
    # 中间值需要 value_info，否则无法确定 fallback 输出类型。
    graph.value_info.append(
        helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 3])
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.example", 1),
        ],
    )
    model.ir_version = 8
    return _save(model, path)


def make_bad_opset(path: Path, version: int = 30) -> Path:
    nodes = [helper.make_node("Relu", ["x"], ["out"], name="relu_1")]
    graph = helper.make_graph(
        nodes,
        "bad_opset",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", version)]
    )
    model.ir_version = 8
    return _save(model, path)
