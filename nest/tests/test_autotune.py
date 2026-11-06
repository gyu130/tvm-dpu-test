"""AutoTune 配置空间、构建、Runner、日志与调优器测试。"""

import numpy as np
import pytest

from nest.errors import TuneError
from nest.ir import (
    CallNode,
    ConstantNode,
    Graph,
    Module,
    SourceSpan,
    TensorType,
    Value,
)
from nest.tile import OP_END, ScheduleOptions, validate_program
from nest.tune import (
    Measurement,
    StaticRunner,
    TuneLog,
    TuneOptions,
    autotune_space,
    build_module,
    default_config,
    make_record,
    program_stats,
    tune,
)

SPAN = SourceSpan(file="unit.onnx", node_name="n", op_name="op")


# ---------------------------------------------------------------------------
# 图构造 helper
# ---------------------------------------------------------------------------


def _value(name, shape, layout="linear", dtype="float32"):
    return Value(
        name=name, type=TensorType(dtype=dtype, shape=tuple(shape), layout=layout)
    )


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


class _OracleRunner:
    """用配置的函数值伪造测量，便于验证调优器本身。"""

    name = "oracle"

    def __init__(self, fn):
        self._fn = fn
        self.calls = []

    def run(self, build):
        self.calls.append(tuple(build.entity))
        return Measurement(
            cost=float(self._fn(build.config)), detail={"config": dict(build.config)}
        )


# ---------------------------------------------------------------------------
# 配置空间
# ---------------------------------------------------------------------------


def test_space_encode_decode_roundtrip():
    module = _conv_module()
    space = autotune_space(module, ScheduleOptions(sram_bytes=4 << 20))

    assert [knob.name for knob in space.knobs] == [
        "conv_oc_tile", "cores", "db_depth",
    ]
    assert space.size == 12
    for entity in space.all_entities():
        assert space.encode(space.decode(entity)) == entity
    # 混合进制：最后一个 knob 变化最快
    assert space.entity_at(0) == (0, 0, 0)
    assert space.entity_at(1) == (0, 0, 1)

    with pytest.raises(TuneError):
        space.decode((len(space.knobs[0].values), 0, 0))


def test_autotune_space_contains_base_values_and_skips_absent_axes():
    module = _conv_module(oc=48)
    base = ScheduleOptions(sram_bytes=4 << 20, conv_oc_tile=48)
    space = autotune_space(module, base)

    names = [knob.name for knob in space.knobs]
    assert "gemm_n_tile" not in names
    conv = next(knob for knob in space.knobs if knob.name == "conv_oc_tile")
    assert 48 in conv.values
    assert all(48 % value == 0 for value in conv.values)
    assert space.decode(space.default_entity(default_config(base)))["conv_oc_tile"] == 48


def test_invalid_tune_options_raise():
    with pytest.raises(TuneError):
        TuneOptions(strategy="bogus")
    with pytest.raises(TuneError):
        TuneOptions(max_trials=0)
    with pytest.raises(TuneError):
        TuneOptions(patience=-1)


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------


def test_build_module_produces_valid_program_and_end_op():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    build = build_module(module, default_config(base, db_depth=1), base, entity=(0, 0, 0))

    assert build.cores == 1
    assert build.words > 0
    validate_program(build.programs[0])
    assert build.sram_peak <= base.sram_bytes
    # lower 只给最后一个 stage 追加 END
    assert build.encoded[-1].words[-1] == OP_END << 29


def test_double_buffer_increases_sram_peak():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    single = build_module(module, default_config(base, db_depth=1), base)
    double = build_module(module, default_config(base, db_depth=2), base)

    assert double.double_buffer and not single.double_buffer
    assert double.sram_peak > single.sram_peak


def test_infeasible_budget_raises():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=1024)
    with pytest.raises(Exception) as excinfo:
        build_module(module, default_config(base, db_depth=1), base)
    assert "SRAM" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_static_runner_reports_cost_and_detail():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    big = build_module(module, default_config(base, db_depth=1), base)
    small = build_module(
        module, {**default_config(base, db_depth=1), "conv_oc_tile": 8}, base
    )

    runner = StaticRunner()
    big_m = runner.run(big)
    small_m = runner.run(small)

    assert big_m.ok and small_m.ok
    assert big_m.cost > 0 and small_m.cost > 0
    for key in ("per_core_cycles", "sram_peak", "words", "dma_bytes", "stages"):
        assert key in big_m.detail
    assert big_m.detail["sram_peak"] == big.sram_peak
    assert big_m.detail["words"] == big.words
    assert big_m.detail["dma_bytes"] > 0
    # 同一份数据搬移总量与 tile 无关；小 tile 只增加每 tile 的额外开销
    assert big_m.detail["dma_bytes"] == small_m.detail["dma_bytes"]
    assert big_m.detail["sram_peak"] > small_m.detail["sram_peak"]
    assert small_m.cost >= big_m.cost


def test_program_stats_counts_instructions_and_dma():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    build = build_module(module, default_config(base, db_depth=1), base)

    stats = program_stats(build.lowered[0])
    assert stats["stages"] == 1
    assert stats["instructions"] > 0
    assert stats["mul_instructions"] > 0
    assert stats["dma_bytes"] > 0
    # conv 的 MAC 循环里每条 MUL 都伴随一次 ADD
    assert stats["mul_instructions"] <= stats["instructions"]


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


def test_log_roundtrip(tmp_path):
    path = tmp_path / "tune.jsonl"
    log = TuneLog(path)
    assert len(log) == 0

    log.append(make_record((0, 0, 0), {"conv_oc_tile": 1}, 100.0, runner="oracle"))
    log.append(make_record((1, 0, 0), {"conv_oc_tile": 2}, 50.0, runner="oracle"))

    assert path.exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    reloaded = TuneLog(path)
    assert len(reloaded) == 2
    assert reloaded.records[0].entity == (0, 0, 0)
    assert reloaded.records[1].cost == 50.0
    assert reloaded.best().entity == (1, 0, 0)
    assert reloaded.record_for((0, 0, 0)).cost == 100.0


# ---------------------------------------------------------------------------
# 调优器
# ---------------------------------------------------------------------------


def test_grid_tune_finds_oracle_optimum():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    space = autotune_space(module, base)
    oracle = _OracleRunner(lambda config: 10000.0 / float(config["conv_oc_tile"]))

    result = tune(
        module, space, oracle, base=base,
        options=TuneOptions(strategy="grid", max_trials=64),
    )

    assert result.best is not None
    assert result.best_config["conv_oc_tile"] == 64
    assert result.best_cost == pytest.approx(10000.0 / 64)
    assert len(result.records) == space.size
    assert result.best_build is not None


def test_random_tune_is_deterministic():
    module = _conv_gemm_module()
    base = ScheduleOptions(sram_bytes=8 << 20)
    space = autotune_space(module, base)
    oracle = _OracleRunner(
        lambda config: 100000.0
        / (float(config["conv_oc_tile"]) * float(config["gemm_n_tile"]))
    )
    options = TuneOptions(strategy="random", max_trials=6, seed=7)

    first = tune(module, space, oracle, base=base, options=options)
    second = tune(module, space, oracle, base=base, options=options)

    assert first.best.entity == second.best.entity
    assert [record.entity for record in first.records] == [
        record.entity for record in second.records
    ]


def test_tune_resume_skips_evaluated_entities(tmp_path):
    path = tmp_path / "resume.jsonl"
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    space = autotune_space(module, base)
    fn = lambda config: 10000.0 / float(config["conv_oc_tile"])  # noqa: E731

    first_runner = _OracleRunner(fn)
    first = tune(
        module, space, first_runner, base=base,
        options=TuneOptions(strategy="grid", max_trials=3, log_path=path, resume=False),
    )
    assert len(first_runner.calls) == 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3

    second_runner = _OracleRunner(fn)
    second = tune(
        module, space, second_runner, base=base,
        options=TuneOptions(strategy="grid", max_trials=3, log_path=path, resume=True),
    )
    evaluated = set(first_runner.calls)
    # 续跑不得重复评估历史 entity
    assert all(entity not in evaluated for entity in second_runner.calls)
    assert len(second.records) == 3 + len(second_runner.calls)
    # 续跑会看到更多历史记录，最优值只可能更好
    assert second.best_cost <= first.best_cost


def test_failed_build_is_recorded_without_raising():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=1024)
    space = autotune_space(module, base)

    result = tune(
        module, space, StaticRunner(), base=base,
        options=TuneOptions(strategy="grid", max_trials=64),
    )

    assert result.best is None
    assert result.best_build is None
    assert result.records
    assert all(record.error for record in result.records)
    assert all("SRAM" in record.error for record in result.records)


def test_greedy_improves_over_default():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20, conv_oc_tile=1)
    space = autotune_space(module, base)
    oracle = _OracleRunner(lambda config: 10000.0 / float(config["conv_oc_tile"]))

    result = tune(
        module, space, oracle, base=base, options=TuneOptions(strategy="greedy")
    )

    assert result.best_config["conv_oc_tile"] == 64
    assert result.best_cost == pytest.approx(10000.0 / 64)
    # 默认（tile=1）的代价被记录下来，说明坐标下降确实从默认点出发
    default_record = next(
        record for record in result.records if record.config["conv_oc_tile"] == 1
    )
    assert default_record.cost == pytest.approx(10000.0)
