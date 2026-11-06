"""M2 多核切分、核间同步、负载报告与 DMA 双缓冲测试。"""

import json
from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from nest.cli import main
from nest.errors import ScheduleError
from nest.ir import CallNode, ConstantNode, Graph, Module, SourceSpan, TensorType, Value
from nest.tile import (
    Compute,
    Copy,
    CoreBarrier,
    CoreSyncGraph,
    MultiCoreOptions,
    TileRef,
    plan_double_buffer,
    schedule_multicore,
    validate_program,
)

SPAN = SourceSpan(file="unit.onnx", node_name="n", op_name="op")


def _value(name, shape, layout="linear", dtype="float32"):
    return Value(
        name=name, type=TensorType(dtype=dtype, shape=tuple(shape), layout=layout)
    )


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
    return Module(
        name=name,
        graph=Graph(inputs=list(inputs), outputs=list(outputs), nodes=list(nodes)),
    )


def _conv_module(oc=4, ic=2, hw=8, k=3, pad=1):
    x = _value("x", (1, ic, hw, hw), "NCHW")
    w = _const(
        "w",
        np.arange(oc * ic * k * k, dtype=np.float32).reshape(oc, ic, k, k),
        "OIHW",
    )
    b = _const("b", np.zeros((oc,), dtype=np.float32), "C")
    oh = hw + 2 * pad - k + 1
    conv = _call(
        "Conv",
        "conv2d",
        ["x", "w", "b"],
        [_value("y", (1, oc, oh, oh), "NCHW")],
        attrs={
            "pads": [pad, pad, pad, pad],
            "strides": [1, 1],
            "dilations": [1, 1],
            "group": 1,
            "auto_pad": "NOTSET",
        },
    )
    return _module("m", [x], [w, b, conv], [conv.outputs[0]])


def _gemm_module(n=6, k=4):
    x = _value("x", (1, k))
    w = _const("w", np.arange(n * k, dtype=np.float32).reshape(n, k))
    b = _const("b", np.zeros((n,), dtype=np.float32))
    gemm = _call(
        "Gemm",
        "gemm",
        ["x", "w", "b"],
        [_value("y", (1, n))],
        attrs={"transB": 1, "alpha": 1.0, "beta": 1.0, "transA": 0},
    )
    return _module("m", [x], [w, b, gemm], [gemm.outputs[0]])


def _pool_module():
    x = _value("x", (1, 2, 8, 8), "NCHW")
    pool = _call(
        "MaxPool",
        "max_pool2d",
        ["x"],
        [_value("y", (1, 2, 4, 4), "NCHW")],
        attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
    )
    return _module("m", [x], [pool], [pool.outputs[0]])


def _flatten_module(batch=2):
    x = _value("x", (batch, 2, 2), "NCHW")
    flatten = _call(
        "Flatten",
        "flatten",
        ["x"],
        [_value("y", (batch, 4))],
        attrs={"axis": 1},
    )
    return _module("m", [x], [flatten], [flatten.outputs[0]])


def _stage_copy(stage, *, src=None, dst=None):
    for item in tuple(stage.prologue) + tuple(stage.body):
        if not isinstance(item, Copy):
            continue
        if src is not None and item.src != src:
            continue
        if dst is not None and item.dst != dst:
            continue
        return item
    raise AssertionError("未找到匹配的 DMA Copy")


def _stage_compute(stage):
    return next(item for item in stage.body if isinstance(item, Compute))


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------


def test_gemm_splits_output_n_across_two_cores():
    module = _gemm_module(n=6, k=4)
    result = schedule_multicore(module, MultiCoreOptions(cores=2))

    assert result.cores == 2
    plan = result.splits[0]
    assert (plan.template, plan.split_dim, plan.axis) == ("gemm", "n", 1)
    assert plan.extent == 6 and plan.chunk == 3
    assert plan.halo == ()

    for core, program in enumerate(result.programs):
        stage = program.stages[0]
        assert stage.core == core
        assert (stage.loops[0].extent, stage.loops[0].tile) == (3, 3)

        weight_load = _stage_copy(stage, src="w")
        offset = weight_load.src_region.offsets[0]
        assert isinstance(offset, TileRef) and offset.base == core * 3

        out_store = _stage_copy(stage, dst="y")
        out_offset = out_store.dst_region.offsets[1]
        assert isinstance(out_offset, TileRef) and out_offset.base == core * 3

        buffers = {buffer.name: buffer for buffer in stage.buffers}
        assert buffers[f"{stage.name}_y"].shape == (1, 3)
        assert buffers[f"{stage.name}_b"].shape == (3, 4)
        validate_program(program)

    result.validate()


def test_gemm_splits_n_across_three_cores():
    result = schedule_multicore(
        _gemm_module(n=6, k=4), MultiCoreOptions(cores=3)
    )
    plan = result.splits[0]
    assert plan.chunk == 2
    for program in result.programs:
        assert program.stages[0].loops[0].tile == 2
    result.validate()


def test_conv_splits_output_channels():
    result = schedule_multicore(
        _conv_module(oc=4, ic=2, hw=4), MultiCoreOptions(cores=2)
    )
    plan = result.splits[0]
    assert (plan.split_dim, plan.axis) == ("oc", 1)
    assert plan.chunk == 2

    for core, program in enumerate(result.programs):
        stage = program.stages[0]
        assert stage.core == core
        buffers = {buffer.name: buffer for buffer in stage.buffers}
        assert buffers[f"{stage.name}_w"].shape == (2, 2, 3, 3)
        assert buffers[f"{stage.name}_y"].shape == (1, 2, 4, 4)
        weight_load = _stage_copy(stage, src="w")
        assert weight_load.src_region.offsets[0].base == core * 2
        validate_program(program)
    result.validate()


def test_conv_spatial_split_sets_halo_and_pads():
    module = _conv_module(oc=2, ic=2, hw=8)
    result = schedule_multicore(
        module,
        MultiCoreOptions(cores=2, split_dim="h"),
    )
    plan = result.splits[0]
    assert (plan.split_dim, plan.axis) == ("h", 2)
    assert plan.chunk == 4
    assert len(plan.halo) == 2

    first, second = plan.halo
    assert (first.core, second.core) == (0, 1)
    assert (first.output_start, first.output_end) == (0, 4)
    assert (first.input_start, first.input_end) == (0, 5)
    assert (first.pad_before, first.pad_after) == (1, 0)
    assert (second.input_start, second.input_end) == (3, 8)
    assert (second.pad_before, second.pad_after) == (0, 1)

    for core, program in enumerate(result.programs):
        stage = program.stages[0]
        buffers = {buffer.name: buffer for buffer in stage.buffers}
        assert buffers[f"{stage.name}_x"].shape == (1, 2, 5, 8)
        assert buffers[f"{stage.name}_y"].shape == (1, 2, 4, 8)

        load = _stage_copy(stage, dst=f"{stage.name}_x")
        assert load.src_region.offsets[2] == (0, 3)[core]
        assert load.src_region.extents[2] == 5

        pads = _stage_compute(stage).attrs["pads"]
        assert pads == ([1, 1, 0, 1], [0, 1, 1, 1])[core]
        validate_program(program)
    result.validate()


def test_flatten_splits_batch_only():
    result = schedule_multicore(
        _flatten_module(batch=2),
        MultiCoreOptions(cores=2, split_dim="batch"),
    )
    plan = result.splits[0]
    assert (plan.split_dim, plan.chunk) == ("batch", 1)

    for core, program in enumerate(result.programs):
        stage = program.stages[0]
        load = _stage_copy(stage, dst=f"{stage.name}_in0")
        assert load.src_region.offsets[0] == core
        assert load.src_region.extents[0] == 1
        validate_program(program)
    result.validate()


# ---------------------------------------------------------------------------
# 显式失败
# ---------------------------------------------------------------------------


def test_indivisible_extent_is_rejected():
    with pytest.raises(ScheduleError, match="没有可被"):
        schedule_multicore(_gemm_module(n=5, k=4), MultiCoreOptions(cores=2))


def test_cores_exceed_extent_is_rejected():
    with pytest.raises(ScheduleError, match="没有可被"):
        schedule_multicore(_gemm_module(n=2, k=4), MultiCoreOptions(cores=4))


def test_explicit_axis_indivisible_is_rejected():
    with pytest.raises(ScheduleError, match="不能被 2 核整除"):
        schedule_multicore(
            _gemm_module(n=5, k=4),
            MultiCoreOptions(cores=2, split_dim="n"),
        )


def test_flatten_rejects_non_batch_split():
    with pytest.raises(ScheduleError, match="只支持按 batch 轴切分"):
        schedule_multicore(
            _flatten_module(batch=2),
            MultiCoreOptions(cores=2, split_dim="n"),
        )


def test_reduce_hw_split_requires_halo():
    with pytest.raises(ScheduleError, match="H/W 切分需要 halo"):
        schedule_multicore(
            _pool_module(),
            MultiCoreOptions(cores=2, split_dim="h"),
        )


def test_invalid_split_dim_is_rejected():
    with pytest.raises(ScheduleError, match="非法 split_dim"):
        MultiCoreOptions(cores=2, split_dim="channel")


# ---------------------------------------------------------------------------
# 核间同步：死锁 / 配对
# ---------------------------------------------------------------------------


def test_core_sync_graph_rejects_missing_barrier():
    result = schedule_multicore(_gemm_module(), MultiCoreOptions(cores=2))
    broken = CoreSyncGraph(
        core_count=2, barriers=result.sync.barriers[1:]
    )
    with pytest.raises(ScheduleError, match="barrier 数量"):
        broken.validate(result.programs)


def test_core_sync_graph_rejects_stage_mismatch():
    result = schedule_multicore(_gemm_module(), MultiCoreOptions(cores=2))
    first = replace(result.sync.barriers[0], stage_name="ghost")
    broken = CoreSyncGraph(
        core_count=2, barriers=(first,) + result.sync.barriers[1:]
    )
    with pytest.raises(ScheduleError, match="stage 顺序"):
        broken.validate(result.programs)


def test_core_sync_graph_rejects_incomplete_cores():
    barrier = CoreBarrier(index=0, stage_name="s0_gemm", cores=(0,))
    broken = CoreSyncGraph(core_count=2, barriers=(barrier,))
    with pytest.raises(ScheduleError, match="未覆盖全部核"):
        broken.validate()


# ---------------------------------------------------------------------------
# 负载、确定性与输出覆盖
# ---------------------------------------------------------------------------


def test_load_report_sums_to_single_core():
    module = _gemm_module(n=6, k=4)
    single = schedule_multicore(module, MultiCoreOptions(cores=1))
    multi = schedule_multicore(module, MultiCoreOptions(cores=2))

    assert single.cores == 1
    assert len(single.loads) == 1
    assert sum(load.macs for load in multi.loads) == single.loads[0].macs
    assert all(load.stages == len(single.programs[0].stages) for load in multi.loads)
    assert all(load.sram_peak > 0 for load in multi.loads)


def test_multicore_is_deterministic():
    options = MultiCoreOptions(cores=2, double_buffer=True)
    first = schedule_multicore(_gemm_module(), options).to_dict()
    second = schedule_multicore(_gemm_module(), options).to_dict()
    assert first == second


def test_output_regions_cover_all_tiles_without_overlap():
    result = schedule_multicore(_gemm_module(n=6, k=4), MultiCoreOptions(cores=2))
    covered = []
    for program in result.programs:
        stage = program.stages[0]
        store = _stage_copy(stage, dst="y")
        start = store.dst_region.offsets[1].base
        extent = store.dst_region.extents[1]
        covered.append((start, start + extent))
    covered.sort()
    assert covered == [(0, 3), (3, 6)]
    assert covered[0][1] <= covered[1][0]


# ---------------------------------------------------------------------------
# DMA 双缓冲
# ---------------------------------------------------------------------------


def test_double_buffer_plan_adds_shadow_and_extra_sram():
    program = schedule_multicore(
        _gemm_module(), MultiCoreOptions(cores=1)
    ).programs[0]
    plan = plan_double_buffer(program)

    assert plan.depth == 2
    assert plan.entries
    assert plan.extra_bytes == sum(entry.nbytes for entry in plan.entries)
    stage = program.stages[0]
    buffers = {buffer.name: buffer for buffer in stage.buffers}
    for entry in plan.entries:
        assert entry.stage == stage.name
        assert entry.buffer in buffers
        assert entry.nbytes == buffers[entry.buffer].nbytes
        assert entry.shadow.endswith("db0") or ".db" in entry.shadow
    plan.validate(program)


def test_multicore_double_buffer_covered_by_validate():
    result = schedule_multicore(
        _gemm_module(), MultiCoreOptions(cores=2, double_buffer=True)
    )
    assert len(result.double_buffer) == 2
    assert all(plan.entries for plan in result.double_buffer)
    result.validate()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_gemm_onnx(path, n=8, k=4, seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((n,)) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node("Gemm", ["x", "W", "B"], ["out"], name="gemm_1", transB=1)
    ]
    graph = helper.make_graph(
        nodes,
        "gemm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, k])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n])],
        initializer=[
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(b, "B"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_cli_compile_writes_tile_report(tmp_path):
    model = _make_gemm_onnx(tmp_path / "gemm.onnx")
    report_path = tmp_path / "tile.json"

    rc = main([
        "compile", str(model),
        "--cores", "2",
        "--double-buffer",
        "--tile-report", str(report_path),
    ])
    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["multicore"]["cores"] == 2
    assert len(payload["multicore"]["programs"]) == 2
    assert len(payload["multicore"]["double_buffer"]) == 2
    assert payload["multicore"]["sync"]["barriers"]
