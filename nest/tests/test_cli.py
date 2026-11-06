import numpy as np

from nest import Module
from nest.cli import main

from model_builders import (
    make_custom_and_fallback,
    make_dynamic_batch,
    make_mlp,
    make_unknown_op,
)


def test_cli_import_prints_graph_and_saves_report(tmp_path, capsys):
    path = make_mlp(tmp_path / "mlp.onnx")
    report_path = tmp_path / "report.json"
    module_path = tmp_path / "mlp.nest.json"

    rc = main([
        "import", str(path),
        "--print-graph",
        "--report", str(report_path),
        "--save-module", str(module_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "导入报告" in out
    assert "nest.gemm" in out
    assert report_path.exists()
    module = Module.load(module_path)
    assert module.name == "mlp"


def test_cli_run_writes_outputs(tmp_path, capsys):
    path = make_mlp(tmp_path / "mlp.onnx")
    out_path = tmp_path / "out.npz"
    rc = main(["run", str(path), "--outputs", str(out_path)])
    assert rc == 0
    payload = np.load(out_path)
    assert payload["out"].shape == (1, 3)


def test_cli_error_policy_returns_nonzero(tmp_path, capsys):
    path = make_unknown_op(tmp_path / "unknown.onnx")
    rc = main(["import", str(path), "--fallback", "error"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Mystery" in err
    assert "mystery_1" in err


def test_cli_loads_custom_ops_module(tmp_path, capsys):
    plugin = tmp_path / "my_ops.py"
    plugin.write_text(
        "from nest import register_custom\n"
        "def register():\n"
        "    register_custom('com.example', 'ScaleShift',\n"
        "        canonical_name='custom.scale_shift',\n"
        "        infer_types=lambda attrs, its: [its[0]],\n"
        "        kernel=lambda ins, attrs: [\n"
        "            (ins[0] * attrs['scale'] + attrs['shift']).astype(ins[0].dtype)\n"
        "        ],\n"
        "        npu_support=True)\n",
        encoding="utf-8",
    )
    path = make_custom_and_fallback(tmp_path / "combo.onnx")
    out_path = tmp_path / "out.npz"
    rc = main([
        "run", str(path), "--custom-ops", str(plugin), "--outputs", str(out_path)
    ])
    assert rc == 0
    assert np.load(out_path)["out"].shape == (1, 3)


def test_cli_dynamic_shape_override(tmp_path):
    path = make_dynamic_batch(tmp_path / "dyn.onnx")
    out_path = tmp_path / "out.npz"
    rc = main([
        "run", str(path), "--input-shape", "x=1,4", "--outputs", str(out_path)
    ])
    assert rc == 0
    assert np.load(out_path)["out"].shape == (1, 4)
