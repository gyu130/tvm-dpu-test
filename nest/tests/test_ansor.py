"""Ansor sketch 枚举、GBDT 代价模型与进化搜索测试。"""

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
from nest.tile import ScheduleOptions
from nest.tune import (
    FEATURE_NAMES,
    AnsorOptions,
    GBDTRegressor,
    LearnedCostModel,
    Measurement,
    StaticRunner,
    ansor_tune,
    build_module,
    default_config,
    extract_features,
    generate_sketches,
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


def _conv_relu_module(oc=64, ic=64, ih=8, iw=8, kh=3, kw=3):
    oh, ow = ih - kh + 1, iw - kw + 1
    x = _value("x", (1, ic, ih, iw), "NCHW")
    w = _const("w", (oc, ic, kh, kw), "OIHW")
    conv_y = _value("conv_y", (1, oc, oh, ow), "NCHW")
    y = _value("y", (1, oc, oh, ow), "NCHW")
    conv = _call(
        "Conv", "conv2d", ["x", "w"], [conv_y],
        {"pads": [0, 0, 0, 0], "strides": [1, 1], "group": 1},
    )
    relu = _call("Relu", "relu", ["conv_y"], [y], {})
    return Module(
        name="conv_relu", graph=Graph(inputs=[x], outputs=[y], nodes=[w, conv, relu])
    )


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
    """用配置的函数值伪造测量，便于验证搜索本身。"""

    name = "oracle"

    def __init__(self, fn):
        self._fn = fn

    def run(self, build):
        return Measurement(
            cost=float(self._fn(build.config)), detail={"config": dict(build.config)}
        )


# ---------------------------------------------------------------------------
# sketch 枚举
# ---------------------------------------------------------------------------


def test_generate_sketches_enumerates_db_depths():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    sketches = generate_sketches(
        module, base, db_depths=(1, 2), fusion_patterns=((),)
    )

    assert len(sketches) == 2
    assert {sketch.db_depth for sketch in sketches} == {1, 2}
    for sketch in sketches:
        assert sketch.templates == ("conv",)
        assert sketch.ops == ("conv2d",)
        assert sketch.cores == 1
        assert [axis.name for axis in sketch.axes] == ["conv_oc_tile"]


def test_sketch_config_merges_structural_fields():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    sketch = next(
        item
        for item in generate_sketches(
            module, base, db_depths=(2,), fusion_patterns=((),)
        )
    )

    assert sketch.structural == {"cores": 1, "db_depth": 2}
    config = sketch.config((0,))
    assert config["cores"] == 1
    assert config["db_depth"] == 2
    assert config["conv_oc_tile"] == sketch.axes[0].candidates[0]
    assert sketch.space.size == len(sketch.axes[0].candidates)
    assert sketch.to_dict()["axes"][0]["name"] == "conv_oc_tile"


def test_generate_sketches_applies_conv_relu_fusion():
    module = _conv_relu_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    sketches = generate_sketches(
        module, base, db_depths=(1,), fusion_patterns=((), ("conv_relu",))
    )

    assert len(sketches) == 2
    plain = [item for item in sketches if not item.fusion_patterns]
    fused = [item for item in sketches if item.fusion_patterns]
    assert plain[0].ops == ("conv2d", "relu")
    assert fused[0].ops == ("conv2d_relu",)
    assert fused[0].templates == ("conv",)

    prepared = fused[0].prepare(module)
    build = fused[0].build(prepared, (0,), base)
    assert build.words > 0


def test_sketch_prepare_without_fusion_is_identity():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    sketch = generate_sketches(module, base, db_depths=(1,), fusion_patterns=((),))[0]

    assert sketch.prepare(module) is module


# ---------------------------------------------------------------------------
# 学习型代价模型
# ---------------------------------------------------------------------------


def test_gbdt_fits_nonlinear_relation():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(120, 3))
    y = X[:, 0] ** 2 + 2.0 * np.sin(X[:, 1]) + 0.5 * X[:, 2]

    model = GBDTRegressor(
        n_estimators=60, learning_rate=0.2, max_depth=3, min_samples_leaf=3
    )
    model.fit(X, y)
    prediction = model.predict(X)
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert 1.0 - ss_res / ss_tot > 0.8


def test_gbdt_rejects_bad_training_data():
    model = GBDTRegressor()
    with pytest.raises(TuneError):
        model.fit(np.zeros((3, 2)), np.zeros(4))
    with pytest.raises(TuneError):
        GBDTRegressor(n_estimators=0)


def test_learned_cost_model_save_load(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(40, 4))
    y = 10.0 ** (3.0 * X[:, 0] + 2.0 * X[:, 1])

    model = LearnedCostModel(
        GBDTRegressor(n_estimators=20, learning_rate=0.3, max_depth=3, min_samples_leaf=2)
    )
    model.fit(X.tolist(), y.tolist())
    assert model.ready
    before = [model.predict(row) for row in X[:3]]

    path = tmp_path / "cost_model.json"
    model.save(path)
    loaded = LearnedCostModel.load(path)
    assert loaded.ready
    after = [loaded.predict(row) for row in X[:3]]
    assert before == pytest.approx(after)


def test_extract_features_shape_and_finiteness():
    module = _conv_gemm_module()
    base = ScheduleOptions(sram_bytes=8 << 20)
    build = build_module(module, default_config(base), base)

    features = extract_features(build)
    assert len(features) == len(FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features)


# ---------------------------------------------------------------------------
# Ansor 搜索
# ---------------------------------------------------------------------------


def test_ansor_finds_oracle_optimum():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)
    sketches = generate_sketches(module, base, db_depths=(1,), fusion_patterns=((),))
    assert len(sketches) == 1
    oracle = _OracleRunner(lambda config: 10000.0 / float(config["conv_oc_tile"]))

    result = ansor_tune(
        module, sketches=sketches, runner=oracle, base=base,
        options=AnsorOptions(
            population=8, generations=3, top_k=8, seed=1, max_trials=8, min_train=2
        ),
    )

    assert result.best is not None
    assert result.best_config["conv_oc_tile"] == 64
    assert result.best_cost == pytest.approx(10000.0 / 64)
    assert result.best_build is not None
    assert result.best_sketch is sketches[0]
    assert result.model.samples > 0


def test_ansor_is_deterministic():
    module = _conv_gemm_module()
    base = ScheduleOptions(sram_bytes=8 << 20)
    sketches = generate_sketches(module, base, db_depths=(1,), fusion_patterns=((),))
    options = AnsorOptions(
        population=8, generations=3, top_k=2, seed=5, max_trials=10, min_train=2
    )

    first = ansor_tune(module, sketches=sketches, runner=StaticRunner(), base=base,
                       options=options)
    second = ansor_tune(module, sketches=sketches, runner=StaticRunner(), base=base,
                        options=options)

    assert first.best.entity == second.best.entity
    assert [record.entity for record in first.records] == [
        record.entity for record in second.records
    ]


def test_ansor_with_static_runner_returns_feasible_best():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=4 << 20)

    result = ansor_tune(
        module, runner=StaticRunner(), base=base,
        options=AnsorOptions(
            population=8, generations=3, top_k=3, seed=2, max_trials=12
        ),
    )

    assert result.best is not None
    assert result.best_cost > 0
    assert result.best_build is not None
    assert result.best_sketch is not None
    assert result.model.samples > 0
    assert all(record.ok for record in result.records)


def test_ansor_raises_when_no_feasible_config():
    module = _conv_module()
    base = ScheduleOptions(sram_bytes=1024)
    sketches = generate_sketches(module, base, db_depths=(1,), fusion_patterns=((),))

    with pytest.raises(TuneError):
        ansor_tune(
            module, sketches=sketches, base=base,
            options=AnsorOptions(
                population=8, generations=2, top_k=2, max_trials=4, min_train=2
            ),
        )


def test_invalid_ansor_options_raise():
    with pytest.raises(TuneError):
        AnsorOptions(population=0)
    with pytest.raises(TuneError):
        AnsorOptions(mutate_prob=1.5)
    with pytest.raises(TuneError):
        AnsorOptions(min_train=1)
