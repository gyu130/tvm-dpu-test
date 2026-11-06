"""AutoSchedule 搜索与代价模型测试。"""

import numpy as np
import pytest

from nest.errors import ScheduleError
from nest.ir import (
    CallNode,
    ConstantNode,
    Graph,
    Module,
    SourceSpan,
    TensorType,
    Value,
)
from nest.tile import (
    AutoScheduleOptions,
    CostModel,
    ScheduleOptions,
    auto_schedule,
    estimate_module,
    plan_memory,
    schedule_module,
    validate_program,
)

SPAN = SourceSpan(file="unit.onnx", node_name="n", op_name="op")


def _value(name, shape, layout="linear", dtype="float32"):
    return Value(name=name, type=TensorType(dtype=dtype, shape=tuple(shape), layout=layout))


def _const(name, shape, layout="linear", dtype="float32", seed=0):
    rng = np.random.default_rng(seed)
    return ConstantNode(
        inputs=[],
        outputs=[_value(name, shape, layout, dtype)],
        span=SPAN,
        value=rng.standard_normal(shape).astype(dtype),
    )


def _call(op_type, canonical, inputs, outputs, attrs=None):
    return CallNode(
        inputs=list(inputs),
        outputs=outputs,
        span=SPAN,
        attrs=dict(attrs or {}),
        domain="",
        op_type=op_type,
        canonical_name=canonical,
    )


def _conv_module(oc=64, ic=64, ih=8, iw=8, kh=3, kw=3):
    oh, ow = ih - kh + 1, iw - kw + 1
    x = _value("x", (1, ic, ih, iw), "NCHW")
    w = _const("w", (oc, ic, kh, kw), "OIHW")
    y = _value("y", (1, oc, oh, ow), "NCHW")
    node = _call(
        "Conv", "conv2d", ["x", "w"], [y],
        {"pads": [0, 0, 0, 0], "strides": [1, 1], "group": 1},
    )
    return Module(name="conv", graph=Graph(inputs=[x], outputs=[y], nodes=[w, node]))


def _gemm_module(m=1, k=4, n=64):
    a = _value("a", (m, k))
    b = _const("b", (n, k))
    y = _value("y", (m, n))
    node = _call("Gemm", "gemm", ["a", "b"], [y], {"transA": 0, "transB": 1})
    return Module(name="gemm", graph=Graph(inputs=[a], outputs=[y], nodes=[b, node]))


def _conv_gemm_module(oc=64, ic=64, ih=8, iw=8, n=64):
    oh, ow = ih - 2, iw - 2
    x = _value("x", (1, ic, ih, iw), "NCHW")
    w = _const("w", (oc, ic, 3, 3), "OIHW", seed=1)
    conv_y = _value("conv_y", (1, oc, oh, ow), "NCHW")
    flat = _value("flat", (1, oc * oh * ow))
    b = _const("b", (n, oc * oh * ow), layout="linear", seed=2)
    out = _value("out", (1, n))
    conv = _call(
        "Conv", "conv2d", ["x", "w"], [conv_y],
        {"pads": [0, 0, 0, 0], "strides": [1, 1], "group": 1},
    )
    reshape = _call("Flatten", "flatten", ["conv_y"], [flat], {"axis": 1})
    gemm = _call("Gemm", "gemm", ["flat", "b"], [out], {"transA": 0, "transB": 1})
    return Module(
        name="conv_gemm",
        graph=Graph(inputs=[x], outputs=[out], nodes=[w, conv, reshape, b, gemm]),
    )


# ---------------------------------------------------------------------------
# 基本行为
# ---------------------------------------------------------------------------


def test_conv_only_search_finds_feasible_and_valid_program():
    module = _conv_module()
    result = auto_schedule(module, AutoScheduleOptions(ScheduleOptions(sram_bytes=4 << 20)))

    assert result.best_trial.feasible
    assert result.best_plan.sram_peak <= result.best_options.sram_bytes
    validate_program(result.best_program)
    # 最优 tile 一般应大于最小候选，否则说明代价模型没起作用
    assert result.best_options.conv_oc_tile >= 1
    assert len(result.trials) > 1


def test_best_trial_is_min_score_among_feasible():
    module = _conv_module()
    result = auto_schedule(module, AutoScheduleOptions(ScheduleOptions(sram_bytes=4 << 20)))

    best_score = min(t.score for t in result.feasible_trials)
    assert result.best_trial.score == best_score
    # 同分时优先大 tile
    tied = [t for t in result.feasible_trials if t.score == best_score]
    assert result.best_trial.conv_oc_tile == max(t.conv_oc_tile for t in tied)


def test_small_budget_forces_smaller_tile():
    module = _conv_module()
    budget = 60 * 1024
    result = auto_schedule(module, AutoScheduleOptions(ScheduleOptions(sram_bytes=budget)))

    assert result.best_plan.sram_peak <= budget
    assert result.best_options.conv_oc_tile < 64
    # 大 tile 应当被标记为不可行
    big = [t for t in result.trials if t.conv_oc_tile == 64]
    assert big and not big[0].feasible


def test_gemm_only_module_keeps_conv_tile_default():
    """只有 gemm 的图不应把 gemm 候选误写进 conv_oc_tile。"""
    module = _gemm_module()
    base = ScheduleOptions(sram_bytes=4 << 20, conv_oc_tile=32, gemm_n_tile=8)
    result = auto_schedule(module, AutoScheduleOptions(base))

    assert result.best_options.conv_oc_tile == 32
    for trial in result.trials:
        assert trial.conv_oc_tile == 32
    gemm_tiles = {t.gemm_n_tile for t in result.trials}
    assert len(gemm_tiles) > 1
    assert all(64 % tile == 0 for tile in gemm_tiles)


def test_conv_only_module_keeps_gemm_tile_default():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20, conv_oc_tile=64, gemm_n_tile=8)
    result = auto_schedule(module, AutoScheduleOptions(base))

    assert result.best_options.gemm_n_tile == 8
    for trial in result.trials:
        assert trial.gemm_n_tile == 8


def test_infeasible_budget_raises():
    module = _conv_module()
    with pytest.raises(ScheduleError):
        auto_schedule(module, AutoScheduleOptions(ScheduleOptions(sram_bytes=1024)))


def test_deterministic():
    module = _conv_gemm_module()
    options = AutoScheduleOptions(ScheduleOptions(sram_bytes=8 << 20))
    first = auto_schedule(module, options)
    second = auto_schedule(module, options)
    assert first.best_options.conv_oc_tile == second.best_options.conv_oc_tile
    assert first.best_options.gemm_n_tile == second.best_options.gemm_n_tile
    assert first.best_trial.score == second.best_trial.score


# ---------------------------------------------------------------------------
# 代价模型
# ---------------------------------------------------------------------------


def test_estimate_has_positive_cycles_for_all_stages():
    module = _conv_gemm_module()
    estimate = estimate_module(module, ScheduleOptions(), CostModel())
    assert len(estimate.stages) == 3  # conv2d + flatten + gemm
    assert estimate.total_cycles > 0
    assert estimate.total_dma_bytes > 0
    for stage in estimate.stages:
        assert stage.compute_cycles > 0
        assert stage.dma_bytes > 0


def test_smaller_tile_increases_dma_bytes():
    module = _conv_module()
    big = estimate_module(module, ScheduleOptions(conv_oc_tile=64), CostModel())
    small = estimate_module(module, ScheduleOptions(conv_oc_tile=8), CostModel())
    assert small.total_dma_bytes > big.total_dma_bytes


def test_double_buffer_latency_is_max_not_sum():
    module = _conv_module()
    estimate = estimate_module(module, ScheduleOptions(conv_oc_tile=64), CostModel())
    stage = estimate.stages[0]
    assert stage.latency_cycles == max(stage.compute_cycles, stage.dma_cycles)


# ---------------------------------------------------------------------------
# 与既有链路一致
# ---------------------------------------------------------------------------


def test_best_program_matches_best_options():
    module = _conv_gemm_module()
    result = auto_schedule(module, AutoScheduleOptions(ScheduleOptions(sram_bytes=8 << 20)))

    direct = schedule_module(module, options=result.best_options)
    validate_program(direct)
    plan = plan_memory(
        direct, sram_bytes=result.best_options.sram_bytes, alignment=result.best_options.alignment
    )
    assert plan.sram_peak == result.best_plan.sram_peak
    assert [s.name for s in direct.stages] == [s.name for s in result.best_program.stages]
