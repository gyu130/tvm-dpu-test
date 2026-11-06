"""ONNX 与 NestGraph 之间的基础转换工具。"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..errors import ModelImportError

# ONNX TensorProto.DataType -> NestGraph dtype 名
ONNX_DTYPE_TO_NAME = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
}
NAME_TO_ONNX_DTYPE = {v: k for k, v in ONNX_DTYPE_TO_NAME.items()}


def onnx_dtype_name(elem_type: int) -> str:
    if elem_type not in ONNX_DTYPE_TO_NAME:
        raise ModelImportError(f"不支持的 ONNX 数据类型: {elem_type}")
    return ONNX_DTYPE_TO_NAME[elem_type]


def numpy_dtype_name(array: np.ndarray) -> str:
    name = str(array.dtype)
    if name not in NAME_TO_ONNX_DTYPE:
        raise ModelImportError(f"不支持的 NumPy dtype: {name}")
    return name


def value_info_type(vi) -> tuple[str, tuple[int, ...]] | None:
    """从 ValueInfoProto 提取 (dtype, 静态 shape)。

    动态维（dim_param 或未设置）返回 None，由调用方决定用输入规格覆盖或报错。
    """
    if vi.type.WhichOneof("value") != "tensor_type":
        raise ModelImportError(f"不支持非 Tensor 类型: {vi.name}")
    elem_type = vi.type.tensor_type.elem_type
    if elem_type == 0:
        return None
    dtype = onnx_dtype_name(elem_type)
    dims = vi.type.tensor_type.shape.dim
    if not dims:
        return (dtype, ())
    shape = []
    for dim in dims:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            return None
    return (dtype, tuple(shape))


def initializer_to_numpy(tensor) -> np.ndarray:
    arr = numpy_helper.to_array(tensor)
    return np.asarray(arr)


def convert_attrs(node) -> tuple[dict, list[str]]:
    """把 ONNX 属性转换为 Python 值。

    返回 ``(attrs, subgraph_attrs)``。子图属性不静默丢弃：名字会返回给调用方，
    由调用方决定 fallback 或报错。
    """
    attrs: dict = {}
    subgraphs: list[str] = []
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS:
            subgraphs.append(attr.name)
            continue
        value = helper.get_attribute_value(attr)
        if isinstance(value, TensorProto):
            value = initializer_to_numpy(value)
        elif isinstance(value, list) and value and isinstance(value[0], TensorProto):
            value = [initializer_to_numpy(v) for v in value]
        attrs[attr.name] = _normalize_scalar(value)
    return attrs, subgraphs


def _normalize_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_normalize_scalar(v) for v in value]
    return value
