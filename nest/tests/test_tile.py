"""M1 NestTile 调度、校验与内存规划测试。"""

from dataclasses import replace

import numpy as np
import pytest

from nest.errors import ScheduleError, TileValidationError
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
from nest.tile import (
    Buffer,
    Compute,
    Copy,
    Loop,
    Program,
    RegionSpec,
    ScheduleOptions,
    Stage,
    Sync,
    TileRef,
    plan_memory,
    print_program,
    schedule_module,
    validate_program,
)

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


def _conv_module(oc=2, ic=1, group=1, hw=4, k=3, pad=1):
    x = _value("x", (1, ic, hw, hw), "NCHW")
    w = _const("w", np.arange(oc * (ic // group) * k * k, dtype=np.float32).reshape(
        oc, ic // group, k, k
    ), "OIHW")
    b = _const("b", np.zeros((oc,), dtype=np.float32), "C")
    oh = hw + 2 * pad - k + 1
    conv = _call(
        "Conv", "conv2d", ["x", "w", "b"],
        [_value("y", (1, oc, oh, oh), "NCHW")],
        attrs={
            "pads": [pad, pad, pad, pad],
            "strides": [1, 1],
            "dilations": [1, 1],
            "group": group,
            "auto_pad": "NOTSET",
        },
    )
    return _module("m", [x], [w, b, conv], [conv.outputs[0]])


def _gemm_module(n=6, k=4):
    x = _value("x", (1, k))
    w = _const("w", np.arange(n * k, dtype=np.float32).reshape(n, k))
    b = _const("b", np.zeros((n,), dtype=np.float32))
    gemm = _call(
        "Gemm", "gemm", ["x", "w", "b"], [_value("y", (1, n))],
        attrs={"transB": 1, "alpha": 1.0, "beta": 1.0, "transA": 0},
    )
    return _module("m", [x], [w, b, gemm], [gemm.outputs[0]])


# ---------------------------------------------------------------------------
# 调度模板
# ---------------------------------------------------------------------------


def test_conv_tiles_output_channels():
    module = _conv_module(oc=6)
    program = schedule_module(module, ScheduleOptions(conv_oc_tile=4))

    stage = program.stages[0]
    assert stage.template == "conv"
    assert len(stage.loops) == 1
    loop = stage.loops[0]
    # 4 不整除 6，退化为不超过 4 且整除 6 的最大值 3。
    assert (loop.var, loop.axis, loop.extent, loop.tile) == ("oc", 1, 6, 3)
    assert loop.iterations == 2

    computes = [item for item in stage.body if isinstance(item, Compute)]
    assert len(computes) == 1
    assert computes[0].attrs["group"] == 1
    assert computes[0].outputs == (f"{stage.name}_y",)

    validate_program(program)


def test_conv_group_slices_input_per_group():
    module = _conv_module(oc=4, ic=4, group=2, hw=4)
    program = schedule_module(module, ScheduleOptions())

    stage = program.stages[0]
    assert stage.loops[0].extent == 4
    assert stage.loops[0].tile == 2
    # group>1 时输入切片按 group 逐 tile 加载，不能整体 hoist 到 prologue。
    assert stage.prologue == ()
    loads = [item for item in stage.body if isinstance(item, Copy) and item.src == "x"]
    assert len(loads) == 1
    region = loads[0].src_region
    assert region.extents == (1, 2, 4, 4)
    assert any(isinstance(offset, TileRef) and offset.tile == 2 for offset in region.offsets)
    validate_program(program)


def test_gemm_tiles_output_n_and_shapes_batch():
    module = _gemm_module(n=6, k=4)
    program = schedule_module(module, ScheduleOptions(gemm_n_tile=2))

    stage = program.stages[0]
    assert stage.template == "gemm"
    loop = stage.loops[0]
    assert (loop.var, loop.axis, loop.extent, loop.tile) == ("n", 1, 6, 2)
    assert loop.iterations == 3

    buffers = {buffer.name: buffer for buffer in stage.buffers}
    assert buffers[f"{stage.name}_b"].shape == (2, 4)
    assert buffers[f"{stage.name}_y"].shape == (1, 2)
    validate_program(program)


def test_generic_stage_loads_compute_stores():
    module = _conv_module(oc=2)
    relu = _call("Relu", "relu", ["y"], [_value("out", (1, 2, 4, 4), "NCHW")])
    module = _module(
        "m",
        module.graph.inputs,
        list(module.graph.nodes) + [relu],
        [relu.outputs[0]],
    )
    program = schedule_module(module, ScheduleOptions())

    conv_stage, relu_stage = program.stages
    assert conv_stage.template == "conv"
    assert relu_stage.template == "elementwise"
    assert relu_stage.loops == ()
    assert any(isinstance(item, Compute) for item in relu_stage.body)
    validate_program(program)


def test_schedule_rejects_fallback_node():
    x = _value("x", (1, 4))
    fallback = FallbackNode(
        inputs=["x"],
        outputs=[_value("y", (1, 4))],
        span=SPAN,
        domain="com.example",
        op_type="Mystery",
        backend="scalar_host",
        reason="未注册",
    )
    module = _module("m", [x], [fallback], [fallback.outputs[0]])
    with pytest.raises(ScheduleError, match="fallback"):
        schedule_module(module, ScheduleOptions())


# ---------------------------------------------------------------------------
# 校验：非法调度必须被拒绝
# ---------------------------------------------------------------------------


def test_validate_rejects_loop_axis_out_of_range():
    program = schedule_module(_conv_module(oc=2), ScheduleOptions())
    stage = program.stages[0]
    bad = replace(stage, loops=(Loop("oc", axis=9, extent=2, tile=2),))
    bad_program = Program(program.name, program.global_buffers, [bad])
    with pytest.raises(TileValidationError, match="超出输出秩"):
        validate_program(bad_program)


def test_validate_rejects_undefined_loop_variable():
    program = schedule_module(_conv_module(oc=2), ScheduleOptions())
    stage = program.stages[0]
    bad_copy = Copy(
        src="x",
        dst=f"{stage.name}_x",
        span=SPAN,
        src_region=RegionSpec(offsets=(0, TileRef("ghost", 2), 0, 0), extents=(1, 2, 4, 4)),
    )
    bad = replace(stage, body=(bad_copy,) + stage.body)
    bad_program = Program(program.name, program.global_buffers, [bad])
    with pytest.raises(TileValidationError, match="未定义的循环变量"):
        validate_program(bad_program)


def test_validate_rejects_unpaired_dma_sync():
    program = schedule_module(_conv_module(oc=2), ScheduleOptions())
    stage = program.stages[0]
    # 去掉最后一次 store 完成等待，异步 token 找不到配对的 dma_wait。
    body = tuple(
        item
        for item in stage.body
        if not (isinstance(item, Sync) and item.kind == "dma_wait"
                and item.token == f"{stage.name}.store")
    )
    bad = replace(stage, body=body)
    bad_program = Program(program.name, program.global_buffers, [bad])
    with pytest.raises(TileValidationError, match="同步不配对"):
        validate_program(bad_program)


def test_validate_rejects_missing_buffer():
    program = schedule_module(_conv_module(oc=2), ScheduleOptions())
    stage = program.stages[0]
    bad_copy = Copy(src="ghost", dst=f"{stage.name}_w", span=SPAN)
    bad = replace(stage, body=(bad_copy,) + stage.body)
    bad_program = Program(program.name, program.global_buffers, [bad])
    with pytest.raises(TileValidationError, match="未声明的 buffer"):
        validate_program(bad_program)


def test_validate_rejects_bad_dma_direction():
    program = schedule_module(_conv_module(oc=2), ScheduleOptions())
    stage = program.stages[0]
    # SRAM -> SRAM 不是合法 DMA。
    bad_copy = Copy(src=f"{stage.name}_x", dst=f"{stage.name}_w", span=SPAN)
    bad = replace(stage, body=(bad_copy,) + stage.body)
    bad_program = Program(program.name, program.global_buffers, [bad])
    with pytest.raises(TileValidationError, match="global<\\->片上"):
        validate_program(bad_program)


def test_validate_rejects_compute_before_load():
    program = schedule_module(_conv_module(oc=2), ScheduleOptions())
    stage = program.stages[0]
    # 删掉权重/偏置加载，Compute 会在脏 SRAM 上执行。
    body = tuple(
        item
        for item in stage.body
        if not (isinstance(item, Copy) and item.src in ("w", "b"))
    )
    bad = replace(stage, body=body)
    bad_program = Program(program.name, program.global_buffers, [bad])
    with pytest.raises(TileValidationError, match="尚未由 DMA 写入"):
        validate_program(bad_program)


def test_validate_rejects_sram_budget_overflow():
    program = schedule_module(_conv_module(oc=2), ScheduleOptions())
    # 容量校验基于已回填的地址；先用大预算规划，再用小预算校验。
    plan = plan_memory(program, sram_bytes=1024 * 1024, alignment=64)
    with pytest.raises(TileValidationError, match="SRAM 超限"):
        validate_program(plan.program, sram_bytes=64)


def test_buffer_rejects_unaligned_address():
    with pytest.raises(TileValidationError, match="未按"):
        Buffer(name="b", scope="sram", dtype="float32", shape=(4,), layout="linear",
               alignment=64, address=8)


# ---------------------------------------------------------------------------
# 内存规划
# ---------------------------------------------------------------------------


def _conv_relu_program():
    module = _conv_module(oc=2)
    relu = _call("Relu", "relu", ["y"], [_value("out", (1, 2, 4, 4), "NCHW")])
    module = _module(
        "m", module.graph.inputs, list(module.graph.nodes) + [relu], [relu.outputs[0]]
    )
    return schedule_module(module, ScheduleOptions())


def test_memory_plan_is_deterministic_and_aligned():
    program = _conv_relu_program()
    first = plan_memory(program, sram_bytes=1024 * 1024, alignment=64)
    second = plan_memory(program, sram_bytes=1024 * 1024, alignment=64)

    assert first.to_dict() == second.to_dict()
    for allocation in first.dram + first.sram:
        assert allocation.address % allocation.alignment == 0
    for buffer in first.program.global_buffers:
        assert buffer.address is not None
    for stage in first.program.stages:
        for buffer in stage.buffers:
            assert buffer.address is not None

    # 回填后的程序必须仍然通过校验（地址对齐、无重叠、容量足够）。
    validate_program(first.program, sram_bytes=first.sram_bytes, alignment=64)


def test_memory_plan_reuses_sram_across_stages():
    program = _conv_relu_program()
    plan = plan_memory(program, sram_bytes=1024 * 1024, alignment=64)

    conv_name, relu_name = (stage.name for stage in program.stages)
    conv_end = max(
        allocation.address + allocation.nbytes
        for allocation in plan.sram
        if allocation.stage == conv_name
    )
    relu_end = max(
        allocation.address + allocation.nbytes
        for allocation in plan.sram
        if allocation.stage == relu_name
    )
    # 第二个 stage 复用第一个 stage 释放的地址，不额外抬高峰值。
    assert relu_end <= conv_end
    assert plan.sram_peak == conv_end


def test_memory_plan_overflow_suggests_fix():
    program = _conv_relu_program()
    with pytest.raises(TileValidationError, match="增大 --sram-bytes 到至少"):
        plan_memory(program, sram_bytes=64, alignment=64)


def test_print_program_lists_stages_and_loops():
    program = schedule_module(_conv_module(oc=4), ScheduleOptions(conv_oc_tile=2))
    text = print_program(program)
    assert program.stages[0].name in text
    assert "oc" in text
    assert "compute" in text


def test_program_serializes_loops_and_regions():
    program = schedule_module(_gemm_module(n=6), ScheduleOptions(gemm_n_tile=2))
    data = program.to_dict()
    stage = data["stages"][0]
    assert stage["loops"][0]["tile"] == 2
    assert stage["loops"][0]["iterations"] == 3
    stores = [
        item for item in stage["body"]
        if item["kind"] == "copy" and item["dst"] == "y"
    ]
    assert stores and stores[0]["dst_region"]["offsets"][1] == {"loop": "n", "tile": 2}


def test_reconstruct_program_from_parts():
    # Stage/Program 都能由显式部分组装，避免隐式状态。
    buffer = Buffer(name="s0_x", scope="sram", dtype="float32", shape=(4,), layout="linear")
    stage = Stage(
        name="s0",
        template="generic",
        node_value="y",
        loops=(),
        prologue=(Copy(src="x", dst="s0_x", span=SPAN, async_token="s0.load"),),
        body=(Compute(
            domain="", op_type="Relu", canonical_name="relu",
            inputs=("s0_x",), outputs=("s0_x",), attrs={}, span=SPAN,
        ),),
        buffers=(buffer,),
        span=SPAN,
    )
    program = Program(
        "m",
        [Buffer(name="x", scope="global", dtype="float32", shape=(4,), layout="linear"),
         Buffer(name="y", scope="global", dtype="float32", shape=(4,), layout="linear")],
        [stage],
    )
    # 没有 Sync 时，prologue 的 x 拷贝缺少 dma_wait：必须被拒绝。
    with pytest.raises(TileValidationError, match="同步不配对"):
        validate_program(program)
