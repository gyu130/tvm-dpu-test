"""M2 PTQ 量化、QDQ 仿真与逐层精度 Debug 测试。"""

import csv
import json

import numpy as np
import pytest

from nest.cli import main
from nest.errors import CalibrationError, QuantizationError
from nest.frontend import ImportOptions, import_onnx_model
from nest.ir import CallNode, ConstantNode
from nest.quant import (
    AccuracyThresholds,
    CalibrationSample,
    CalibrationSet,
    QuantConfig,
    build_accuracy_report,
    calibrate,
    make_calibration_set,
    quantize_module,
    run_quantized,
    run_reference_tensors,
)
from nest.quant.qmath import compute_scale_zero_point

from model_builders import make_cnn, make_mlp


def _import(path):
    result = import_onnx_model(str(path), options=ImportOptions())
    return result.module


def _pipeline(module, config=None):
    config = config or QuantConfig()
    calibration_set = make_calibration_set(module, config)
    calibration = calibrate(module, calibration_set, config)
    result = quantize_module(module, calibration, config)
    return result, calibration, calibration_set


def _call_nodes(module, canonical):
    return [
        node
        for node in module.graph.nodes
        if isinstance(node, CallNode) and node.canonical_name == canonical
    ]


def _tensor(module, name):
    return module.graph.tensors[name].type


# ---------------------------------------------------------------------------
# 标注正确性
# ---------------------------------------------------------------------------


def test_mlp_weight_axis_gemm_transb(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    result, _, _ = _pipeline(module)

    gemms = _call_nodes(module, "gemm")
    assert gemms
    for node in gemms:
        # transB=1 时权重形状为 (N, K)，量化轴必须是输出通道轴 0。
        weight = _tensor(result.module, node.inputs[1])
        assert weight.dtype == "int8"
        assert weight.quant is not None and weight.quant.mode == "int"
        assert weight.quant.axis == 0

    # 激活使用无符号 uint8，且带有显式 scale / zero_point。
    relu = _call_nodes(module, "relu")[0]
    activation = _tensor(result.module, relu.outputs[0].name)
    assert activation.dtype == "uint8"
    assert activation.quant is not None and activation.quant.mode == "int"
    assert len(activation.quant.scale) == 1
    assert len(activation.quant.zero_point) == 1


def test_cnn_conv_weight_per_channel_and_int_constant(tmp_path):
    module = _import(make_cnn(tmp_path / "cnn.onnx"))
    result, _, _ = _pipeline(module)

    conv = _call_nodes(module, "conv2d")[0]
    weight_name = conv.inputs[1]
    weight = _tensor(result.module, weight_name)
    assert weight.dtype == "int8"
    assert weight.quant is not None and weight.quant.mode == "int"
    assert weight.quant.axis == 0
    # per-channel：scale 长度等于输出通道数。
    assert len(weight.quant.scale) == weight.shape[0]

    constants = {
        node.outputs[0].name: node
        for node in result.module.graph.nodes
        if isinstance(node, ConstantNode)
    }
    quantized = constants[weight_name]
    assert quantized.value.dtype == np.int8
    assert quantized.value.shape == weight.shape


def test_npu_unsupported_op_falls_back_to_float(tmp_path):
    # 显式把 relu 退回浮点：该节点输出必须保持浮点量化模式。
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    result, _, _ = _pipeline(module, QuantConfig(float_ops=frozenset({"relu"})))

    assert "relu" in result.float_ops
    relu = _call_nodes(module, "relu")[0]
    activation = _tensor(result.module, relu.outputs[0].name)
    assert activation.quant is not None and activation.quant.mode == "float"
    decision = next(
        item for item in result.decisions
        if item.tensor == relu.outputs[0].name
    )
    assert decision.kind == "float"


def test_bias_correction_toggle(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    enabled, _, _ = _pipeline(module)
    assert any(item.kind == "bias_correction" for item in enabled.decisions)

    disabled, _, _ = _pipeline(module, QuantConfig(bias_correction=False))
    assert not any(item.kind == "bias_correction" for item in disabled.decisions)


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------


def test_quantization_is_reproducible(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    config = QuantConfig(seed=7, calibration_samples=6)

    first, first_calibration, _ = _pipeline(module, config)
    second, second_calibration, _ = _pipeline(module, config)

    assert first.config_hash == second.config_hash
    assert first.calibration_hash == second.calibration_hash
    assert first_calibration.calibration_hash == second_calibration.calibration_hash

    def scales(result):
        return {
            item["tensor"]: item["scale"]
            for item in result.quant_params()
            if item["mode"] == "int"
        }

    assert scales(first) == scales(second)

    other, _, _ = _pipeline(
        module, QuantConfig(seed=8, calibration_samples=6)
    )
    assert other.calibration_hash != first.calibration_hash


# ---------------------------------------------------------------------------
# 精度与首个发散层
# ---------------------------------------------------------------------------


def _report(module, quantized, calibration, sample, thresholds=None):
    reference = run_reference_tensors(module, sample)
    simulated = run_quantized(quantized.module, sample)
    report = build_accuracy_report(
        quantized.module,
        reference,
        simulated.tensors,
        config_hash=quantized.config_hash,
        calibration_hash=calibration.calibration_hash,
        seed=calibration.seed,
        thresholds=thresholds,
        reproduction="unit-test",
        source="unit.onnx",
    )
    return report, simulated


def test_qdq_accuracy_meets_thresholds(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    quantized, calibration, calibration_set = _pipeline(module)
    report, _ = _report(
        module, quantized, calibration, calibration_set.samples[0].inputs
    )

    assert report.first_divergent is None
    assert all(layer.cosine >= 0.99 for layer in report.layers)
    assert all(
        layer.sqnr_exact or layer.sqnr_db >= 20.0 for layer in report.layers
    )


def test_first_divergent_layer_is_located(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    quantized, calibration, calibration_set = _pipeline(module)
    sample = calibration_set.samples[0].inputs

    report, simulated = _report(module, quantized, calibration, sample)
    assert report.first_divergent is None

    # 选择范数最大的计算层注入误差，确保 golden 张量非全零。
    reference = run_reference_tensors(module, sample)
    computed = [
        layer for layer in report.layers
        if layer.op_type not in ("input", "constant")
    ]
    target = max(
        computed, key=lambda layer: float(np.linalg.norm(reference[layer.name]))
    ).name
    assert np.any(reference[target] != 0)

    corrupted = dict(simulated.tensors)
    corrupted[target] = np.zeros_like(corrupted[target])
    diverged = build_accuracy_report(
        quantized.module,
        reference,
        corrupted,
        config_hash=quantized.config_hash,
        calibration_hash=calibration.calibration_hash,
        seed=calibration.seed,
        source="unit.onnx",
    )

    assert diverged.first_divergent == target
    index = next(layer.index for layer in diverged.layers if layer.name == target)
    assert all(not layer.is_divergent for layer in diverged.layers[:index])
    assert diverged.aggregates["by_op"]
    assert diverged.aggregates["by_quant_mode"]


def test_report_serialization(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    quantized, calibration, calibration_set = _pipeline(module)
    report, _ = _report(
        module, quantized, calibration, calibration_set.samples[0].inputs
    )

    json_path = tmp_path / "accuracy.json"
    csv_path = tmp_path / "accuracy.csv"
    report.save_json(json_path)
    report.save_csv(csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config_hash"] == quantized.config_hash
    assert payload["first_divergent"] is None
    assert payload["layers"]

    with open(csv_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "index"
    assert len(rows) == len(report.layers) + 1


# ---------------------------------------------------------------------------
# 显式失败
# ---------------------------------------------------------------------------


def test_empty_calibration_set_is_rejected():
    with pytest.raises(CalibrationError, match="校准集为空"):
        CalibrationSet(samples=(), seed=0, strategy="normal")


def test_calibration_sample_rejects_nan():
    with pytest.raises(CalibrationError, match="NaN/Inf"):
        CalibrationSample({"x": np.array([np.nan], dtype=np.float32)})


def test_degenerate_calibration_range_is_rejected(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    inputs = {
        value.name: np.zeros(value.type.shape, dtype=value.type.dtype)
        for value in module.graph.inputs
    }
    calibration_set = CalibrationSet(
        (CalibrationSample(inputs),), seed=0, strategy="file"
    )
    with pytest.raises(CalibrationError, match="退化范围"):
        calibrate(module, calibration_set, QuantConfig())


def test_zero_scale_is_rejected():
    with pytest.raises(QuantizationError, match="scale 为 0"):
        compute_scale_zero_point(0.0, 0.0, "uint8", "asymmetric")


def test_symmetric_requires_signed_dtype():
    with pytest.raises(QuantizationError, match="有符号"):
        compute_scale_zero_point(-1.0, 1.0, "uint8", "symmetric")


def test_per_block_activation_is_rejected(tmp_path):
    module = _import(make_mlp(tmp_path / "mlp.onnx"))
    config = QuantConfig(activation_granularity="per_block")
    calibration_set = make_calibration_set(module, config)
    calibration = calibrate(module, calibration_set, config)
    with pytest.raises(QuantizationError, match="per-block"):
        quantize_module(module, calibration, config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_quantize_writes_report(tmp_path):
    model = make_mlp(tmp_path / "mlp.onnx")
    report_path = tmp_path / "quant.json"

    rc = main([
        "quantize", str(model),
        "--calibration-samples", "4",
        "--seed", "3",
        "--quant-report", str(report_path),
    ])
    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["config_hash"]
    assert payload["calibration_hash"]
    assert any(item["mode"] == "int" for item in payload["quant_params"])


def test_cli_debug_writes_accuracy_and_csv(tmp_path):
    model = make_mlp(tmp_path / "mlp.onnx")
    json_path = tmp_path / "accuracy.json"
    csv_path = tmp_path / "accuracy.csv"

    rc = main([
        "debug", str(model),
        "--accuracy-report", str(json_path),
        "--accuracy-csv", str(csv_path),
        "--fail-on-divergence",
    ])
    assert rc == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["first_divergent"] is None
    assert csv_path.exists()


def test_cli_debug_fails_on_divergence(tmp_path):
    model = make_mlp(tmp_path / "mlp.onnx")
    # 极低阈值必然触发发散，用于验证 --fail-on-divergence 的退出码。
    rc = main([
        "debug", str(model),
        "--activation-dtype", "uint8",
        "--min-cosine", "1.0",
        "--fail-on-divergence",
    ])
    assert rc == 1
