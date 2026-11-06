import numpy as np
import pytest

from nest.errors import RegistryError
from nest.ir import TensorType
from nest.registry import GLOBAL_REGISTRY, OpSpec, register_custom


def _t():
    return TensorType("float32", (1, 4), "linear")


def test_standard_ops_registered():
    assert GLOBAL_REGISTRY.lookup("", "Gemm") is not None
    assert GLOBAL_REGISTRY.lookup("ai.onnx", "Relu") is not None
    spec = GLOBAL_REGISTRY.lookup("", "Conv")
    assert spec.canonical_name == "conv2d"
    assert spec.quant_rule == "per_channel_int8"


def test_cos_is_standard_but_not_npu_supported():
    spec = GLOBAL_REGISTRY.lookup("", "Cos")
    assert spec is not None
    assert spec.supports_npu({}, [_t()]) is False


def test_custom_registration_requires_custom_domain():
    with pytest.raises(RegistryError):
        register_custom("", "MyOp")
    with pytest.raises(RegistryError):
        register_custom("ai.onnx", "MyOp")


def test_custom_registration_and_priority():
    spec = register_custom(
        "com.example", "Map",
        canonical_name="custom.map",
        infer_types=lambda attrs, its: [its[0]],
        kernel=lambda inputs, attrs: [inputs[0] * 2.0],
        npu_support=True,
    )
    assert GLOBAL_REGISTRY.lookup("com.example", "Map") is spec
    # 自定义域不会回落到标准表
    assert GLOBAL_REGISTRY.lookup("com.example", "Relu") is None


def test_duplicate_registration_requires_override():
    register_custom(
        "com.example", "Map",
        canonical_name="custom.map",
        infer_types=lambda attrs, its: [its[0]],
    )
    with pytest.raises(RegistryError):
        register_custom(
            "com.example", "Map",
            canonical_name="custom.map",
            infer_types=lambda attrs, its: [its[0]],
        )
    register_custom(
        "com.example", "Map",
        canonical_name="custom.map2",
        infer_types=lambda attrs, its: [its[0]],
        override=True,
    )
    assert GLOBAL_REGISTRY.lookup("com.example", "Map").canonical_name == "custom.map2"


def test_invalid_backend_rejected():
    with pytest.raises(RegistryError):
        register_custom(
            "com.example", "Map",
            fallback_backend="cuda",
        )
