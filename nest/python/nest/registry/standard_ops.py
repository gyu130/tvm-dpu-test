"""M0 标准算子表（ONNX 标准域）。

每个算子包含：
- attrs 归一化（补齐 ONNX 默认值）；
- 输出类型推导（静态 shape / layout）；
- NumPy 参考 kernel（host 标量核，供参考执行与 fallback 对拍）；
- NPU 支持标记（M0 不含真实 NPU 后端，仅表达可映射性）。

M0 支持的算子集合是刻意收敛的：只覆盖 tiny MLP / CNN 端到端所需的算子，
长尾算子走 fallback 或显式报错，避免“看起来支持、实际跑不通”。
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import ShapeInferenceError
from ..ir.types import LINEAR_LAYOUT, QUANTIZED_DTYPES, TensorType
from .op_registry import GLOBAL_REGISTRY, OpSpec

FLOAT_DTYPES = ("float16", "float32", "float64")

# 编译器内部融合算子的 domain。
INTERNAL_DOMAIN = "nest.internal"


def _type(dtype: str, shape, layout: str) -> TensorType:
    quant = None
    if dtype in QUANTIZED_DTYPES:
        raise ShapeInferenceError(f"M0 标准算子不产生量化 dtype: {dtype}")
    return TensorType(dtype=dtype, shape=tuple(int(d) for d in shape), layout=layout)


def _like(t: TensorType, shape=None, layout=None, dtype=None) -> TensorType:
    new_shape = tuple(shape) if shape is not None else t.shape
    new_layout = layout if layout is not None else t.layout
    new_dtype = dtype if dtype is not None else t.dtype
    quant = t.quant if new_dtype in QUANTIZED_DTYPES else None
    return TensorType(
        dtype=new_dtype, shape=tuple(new_shape), layout=new_layout, quant=quant
    )


def _require_float(t: TensorType, op: str) -> None:
    if t.dtype not in FLOAT_DTYPES:
        raise ShapeInferenceError(f"{op} 只支持浮点输入，得到 {t.dtype}")


def _norm_axis(axis: int, rank: int, op: str) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ShapeInferenceError(f"{op} 的 axis={axis} 超出秩 {rank}")
    return axis


# ---------------------------------------------------------------------------
# attrs 归一化
# ---------------------------------------------------------------------------


def _norm_conv(attrs: dict, opset: int | None) -> dict:
    out = dict(attrs)
    auto_pad = out.get("auto_pad", "NOTSET")
    if auto_pad not in ("NOTSET", ""):
        raise ShapeInferenceError(
            f"M0 暂不支持 Conv auto_pad={auto_pad}，请在前端固化 pads"
        )
    out["auto_pad"] = "NOTSET"
    out["pads"] = [int(v) for v in out.get("pads", [0, 0, 0, 0])]
    out["strides"] = [int(v) for v in out.get("strides", [1, 1])]
    out["dilations"] = [int(v) for v in out.get("dilations", [1, 1])]
    out["group"] = int(out.get("group", 1))
    if len(out["pads"]) != 4 or len(out["strides"]) != 2 or len(out["dilations"]) != 2:
        raise ShapeInferenceError("M0 Conv 只支持 2D 卷积")
    return out


def _norm_pool(attrs: dict, opset: int | None) -> dict:
    out = dict(attrs)
    if "kernel_shape" not in out:
        raise ShapeInferenceError("Pool 缺少 kernel_shape")
    out["kernel_shape"] = [int(v) for v in out["kernel_shape"]]
    out["strides"] = [int(v) for v in out.get("strides", [1, 1])]
    out["pads"] = [int(v) for v in out.get("pads", [0, 0, 0, 0])]
    out["ceil_mode"] = int(out.get("ceil_mode", 0))
    out["count_include_pad"] = int(out.get("count_include_pad", 0))
    out["dilations"] = [int(v) for v in out.get("dilations", [1, 1])]
    if len(out["kernel_shape"]) != 2:
        raise ShapeInferenceError("M0 Pool 只支持 2D")
    return out


def _norm_gemm(attrs: dict, opset: int | None) -> dict:
    out = dict(attrs)
    out["alpha"] = float(out.get("alpha", 1.0))
    out["beta"] = float(out.get("beta", 1.0))
    out["transA"] = int(out.get("transA", 0))
    out["transB"] = int(out.get("transB", 0))
    return out


def _norm_flatten(attrs: dict, opset: int | None) -> dict:
    out = dict(attrs)
    out["axis"] = int(out.get("axis", 1))
    return out


def _norm_transpose(attrs: dict, opset: int | None) -> dict:
    out = dict(attrs)
    if "perm" in out:
        out["perm"] = [int(v) for v in out["perm"]]
    return out


def _norm_softmax(attrs: dict, opset: int | None) -> dict:
    out = dict(attrs)
    default_axis = 1 if (opset is None or opset <= 12) else -1
    out["axis"] = int(out.get("axis", default_axis))
    return out


def _norm_bn(attrs: dict, opset: int | None) -> dict:
    out = dict(attrs)
    out["epsilon"] = float(out.get("epsilon", 1e-5))
    return out


def _norm_cast(attrs: dict, opset: int | None) -> dict:
    if "to" not in attrs:
        raise ShapeInferenceError("Cast 缺少 to 属性")
    return dict(attrs)


def _norm_clip(attrs: dict, opset: int | None) -> dict:
    return dict(attrs)


def _norm_passthrough(attrs: dict, opset: int | None) -> dict:
    return dict(attrs)


# ---------------------------------------------------------------------------
# 类型推导
# ---------------------------------------------------------------------------


def _infer_unary(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    _require_float(its[0], "unary")
    return [its[0]]


def _broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(np.broadcast_shapes(a, b))
    except ValueError as exc:
        raise ShapeInferenceError(f"广播失败: {a} vs {b}") from exc


def _infer_binary(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    a, b = its[0], its[1]
    _require_float(a, "binary")
    _require_float(b, "binary")
    shape = _broadcast_shape(a.shape, b.shape)
    dtype = a.dtype if a.dtype == b.dtype else str(np.result_type(a.dtype, b.dtype))
    layout = a.layout if a.layout == b.layout else LINEAR_LAYOUT
    return [_type(dtype, shape, layout)]


def _infer_matmul(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    a, b = its[0], its[1]
    _require_float(a, "MatMul")
    _require_float(b, "MatMul")
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ShapeInferenceError("M0 MatMul 只支持 rank>=2 的输入")
    batch = _broadcast_shape(a.shape[:-2], b.shape[:-2])
    shape = batch + (a.shape[-2], b.shape[-1])
    layout = LINEAR_LAYOUT
    return [_type(a.dtype, shape, layout)]


def _infer_gemm(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    a, b = its[0], its[1]
    _require_float(a, "Gemm")
    _require_float(b, "Gemm")
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ShapeInferenceError("Gemm 只支持 2D 输入")
    m = a.shape[1] if attrs["transA"] else a.shape[0]
    k = a.shape[0] if attrs["transA"] else a.shape[1]
    kb = b.shape[1] if attrs["transB"] else b.shape[0]
    n = b.shape[0] if attrs["transB"] else b.shape[1]
    if k != kb:
        raise ShapeInferenceError(f"Gemm 内维不一致: {k} vs {kb}")
    shape = (m, n)
    if len(its) == 3:
        c = its[2]
        if c.shape not in ((n,), (1, n), (m, n), (m, 1), ()):
            try:
                np.broadcast_shapes(c.shape, shape)
            except ValueError as exc:
                raise ShapeInferenceError(f"Gemm bias shape 非法: {c.shape}") from exc
    return [_type(a.dtype, shape, LINEAR_LAYOUT)]


def _conv_out_dim(size: int, pad0: int, pad1: int, k: int, stride: int, dil: int) -> int:
    eff = (k - 1) * dil + 1
    return (size + pad0 + pad1 - eff) // stride + 1


def _infer_conv_output(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    """仅推导 Conv 输出；权重/偏置类型由前端 value_info 提供。"""
    x = its[0]
    w = its[1]
    _require_float(x, "Conv")
    if len(x.shape) != 4 or len(w.shape) != 4:
        raise ShapeInferenceError("M0 Conv 只支持 4D NCHW 输入")
    group = attrs["group"]
    m, cg, kh, kw = w.shape
    c = x.shape[1]
    if c != cg * group or m % group != 0:
        raise ShapeInferenceError(
            f"Conv group={group} 与输入通道 {c} / 权重 {cg} 不匹配"
        )
    pt, pl, pb, pr = attrs["pads"]
    sh, sw = attrs["strides"]
    dh, dw = attrs["dilations"]
    oh = _conv_out_dim(x.shape[2], pt, pb, kh, sh, dh)
    ow = _conv_out_dim(x.shape[3], pl, pr, kw, sw, dw)
    if oh <= 0 or ow <= 0:
        raise ShapeInferenceError(f"Conv 输出尺寸非法: {oh}x{ow}")
    return [_type(x.dtype, (x.shape[0], m, oh, ow), "NCHW")]


def _pool_out_dim(
    size: int, pad0: int, pad1: int, k: int, stride: int, dil: int, ceil_mode: int
) -> int:
    eff = (k - 1) * dil + 1
    numerator = size + pad0 + pad1 - eff
    if ceil_mode:
        out = math.ceil(numerator / stride) + 1
        # ONNX 要求最后一个窗口起点必须落在输入（含 padding）范围内。
        if (out - 1) * stride >= size + pad0:
            out -= 1
        return out
    return numerator // stride + 1


def _infer_pool(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    x = its[0]
    _require_float(x, "Pool")
    if len(x.shape) != 4:
        raise ShapeInferenceError("M0 Pool 只支持 4D NCHW 输入")
    kh, kw = attrs["kernel_shape"]
    pt, pl, pb, pr = attrs["pads"]
    sh, sw = attrs["strides"]
    dh, dw = attrs["dilations"]
    oh = _pool_out_dim(x.shape[2], pt, pb, kh, sh, dh, attrs["ceil_mode"])
    ow = _pool_out_dim(x.shape[3], pl, pr, kw, sw, dw, attrs["ceil_mode"])
    if oh <= 0 or ow <= 0:
        raise ShapeInferenceError(f"Pool 输出尺寸非法: {oh}x{ow}")
    return [_type(x.dtype, (x.shape[0], x.shape[1], oh, ow), "NCHW")]


def _infer_global_avg_pool(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    x = its[0]
    _require_float(x, "GlobalAveragePool")
    if len(x.shape) != 4:
        raise ShapeInferenceError("M0 GlobalAveragePool 只支持 4D NCHW 输入")
    return [_type(x.dtype, (x.shape[0], x.shape[1], 1, 1), "NCHW")]


def _infer_bn(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    x = its[0]
    _require_float(x, "BatchNormalization")
    if len(x.shape) < 2:
        raise ShapeInferenceError("BatchNormalization 输入秩必须 >=2")
    c = x.shape[1]
    for idx, t in enumerate(its[1:], start=1):
        if t.shape != (c,):
            raise ShapeInferenceError(
                f"BatchNormalization 参数 {idx} shape {t.shape} 应为 ({c},)"
            )
    return [_like(x, layout=x.layout if x.layout != LINEAR_LAYOUT else LINEAR_LAYOUT)]


def _infer_flatten(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    x = its[0]
    axis = _norm_axis(attrs["axis"], len(x.shape), "Flatten")
    if axis == 0:
        return [_type(x.dtype, (1, int(np.prod(x.shape))), LINEAR_LAYOUT)]
    dim0 = int(np.prod(x.shape[:axis]))
    dim1 = int(np.prod(x.shape[axis:]))
    return [_type(x.dtype, (dim0, dim1), LINEAR_LAYOUT)]


def _infer_reshape(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    x = its[0]
    if "shape" not in attrs:
        raise ShapeInferenceError("Reshape 缺少常量 shape 输入")
    target = [int(v) for v in attrs["shape"]]
    allowzero = int(attrs.get("allowzero", 0))
    out_shape = []
    for i, dim in enumerate(target):
        if dim == 0 and not allowzero:
            if i >= len(x.shape):
                raise ShapeInferenceError(f"Reshape 0 维索引 {i} 超出输入秩")
            out_shape.append(x.shape[i])
        elif dim < -1:
            raise ShapeInferenceError(f"Reshape 非法维度 {dim}")
        else:
            out_shape.append(dim)
    if out_shape.count(-1) > 1:
        raise ShapeInferenceError("Reshape 最多允许一个 -1")
    if -1 in out_shape:
        known = int(np.prod([d for d in out_shape if d != -1]))
        total = int(np.prod(x.shape))
        if known == 0 or total % known != 0:
            raise ShapeInferenceError(f"Reshape 无法匹配元素数: {x.shape} -> {target}")
        out_shape[out_shape.index(-1)] = total // known
    if int(np.prod(out_shape)) != int(np.prod(x.shape)):
        raise ShapeInferenceError(f"Reshape 元素数不一致: {x.shape} -> {target}")
    return [_type(x.dtype, out_shape, LINEAR_LAYOUT)]


def _infer_transpose(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    x = its[0]
    rank = len(x.shape)
    perm = attrs.get("perm", list(reversed(range(rank))))
    if sorted(perm) != list(range(rank)):
        raise ShapeInferenceError(f"Transpose perm 非法: {perm}")
    return [_type(x.dtype, [x.shape[p] for p in perm], LINEAR_LAYOUT)]


def _infer_concat(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    if "axis" not in attrs:
        raise ShapeInferenceError("Concat 缺少 axis")
    rank = len(its[0].shape)
    axis = _norm_axis(int(attrs["axis"]), rank, "Concat")
    dtype = its[0].dtype
    if any(t.dtype != dtype for t in its):
        raise ShapeInferenceError("Concat 输入 dtype 不一致")
    if any(len(t.shape) != rank for t in its):
        raise ShapeInferenceError("Concat 输入秩不一致")
    shape = list(its[0].shape)
    shape[axis] = sum(t.shape[axis] for t in its)
    return [_type(dtype, shape, LINEAR_LAYOUT)]


def _infer_softmax(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    x = its[0]
    _require_float(x, "Softmax")
    _norm_axis(attrs["axis"], len(x.shape), "Softmax")
    return [_like(x)]


def _infer_cast(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    from ..frontend.onnx_util import ONNX_DTYPE_TO_NAME  # 延迟导入，避免循环

    to = int(attrs["to"])
    if to not in ONNX_DTYPE_TO_NAME:
        raise ShapeInferenceError(f"Cast 目标 dtype 不支持: {to}")
    dtype = ONNX_DTYPE_TO_NAME[to]
    if dtype in QUANTIZED_DTYPES:
        raise ShapeInferenceError("M0 Cast 不支持直接转换到 int8/uint8 等量化 dtype")
    return [_type(dtype, its[0].shape, its[0].layout)]


def _infer_clip(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    return [_like(its[0])]


def _infer_identity(attrs: dict, its: list[TensorType]) -> list[TensorType]:
    return [_like(its[0])]


# ---------------------------------------------------------------------------
# NumPy kernel
# ---------------------------------------------------------------------------


def _k_unary(fn):
    def kernel(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
        return [fn(inputs[0])]

    return kernel


def _k_binary(fn):
    def kernel(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
        return [fn(inputs[0], inputs[1])]

    return kernel


def _k_matmul(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    return [np.matmul(inputs[0], inputs[1])]


def _k_gemm(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    a = inputs[0]
    b = inputs[1]
    if attrs["transA"]:
        a = a.T
    if attrs["transB"]:
        b = b.T
    y = attrs["alpha"] * np.matmul(a, b)
    if len(inputs) == 3:
        y = y + attrs["beta"] * inputs[2]
    return [y.astype(np.result_type(a.dtype, b.dtype), copy=False)]


def _k_conv(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) == 3 else None
    n, c, h, wd = x.shape
    m, cg, kh, kw = w.shape
    group = attrs["group"]
    pt, pl, pb, pr = attrs["pads"]
    sh, sw = attrs["strides"]
    dh, dw = attrs["dilations"]
    oh = _conv_out_dim(h, pt, pb, kh, sh, dh)
    ow = _conv_out_dim(wd, pl, pr, kw, sw, dw)
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    out = np.zeros((n, m, oh, ow), dtype=np.float32)
    mg = m // group
    for oc in range(m):
        g = oc // mg
        acc = np.zeros((oh, ow), dtype=np.float32)
        for i in range(kh):
            h0 = i * dh
            h1 = h0 + sh * (oh - 1) + 1
            for j in range(kw):
                w0 = j * dw
                w1 = w0 + sw * (ow - 1) + 1
                patch = xp[:, g * cg : (g + 1) * cg, h0:h1:sh, w0:w1:sw]
                acc = acc + np.tensordot(w[oc, :, i, j], patch, axes=([0], [1]))
        if b is not None:
            acc = acc + b[oc]
        out[:, oc] = acc
    return [out.astype(x.dtype, copy=False)]


def _pool_patches(x, attrs, fill):
    kh, kw = attrs["kernel_shape"]
    pt, pl, pb, pr = attrs["pads"]
    sh, sw = attrs["strides"]
    dh, dw = attrs["dilations"]
    oh = _pool_out_dim(x.shape[2], pt, pb, kh, sh, dh, attrs["ceil_mode"])
    ow = _pool_out_dim(x.shape[3], pl, pr, kw, sw, dw, attrs["ceil_mode"])
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant",
                constant_values=fill)
    for i in range(kh):
        h0 = i * dh
        h1 = h0 + sh * (oh - 1) + 1
        for j in range(kw):
            w0 = j * dw
            w1 = w0 + sw * (ow - 1) + 1
            yield xp[:, :, h0:h1:sh, w0:w1:sw], i, j


def _k_maxpool(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    x = inputs[0]
    outs = []
    for patch, _, _ in _pool_patches(x, attrs, -np.inf):
        outs.append(patch)
    result = np.max(np.stack(outs, axis=0), axis=0)
    return [result.astype(x.dtype, copy=False)]


def _k_avgpool(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    x = inputs[0]
    kh, kw = attrs["kernel_shape"]
    pt, pl, pb, pr = attrs["pads"]
    sh, sw = attrs["strides"]
    dh, dw = attrs["dilations"]
    oh = _pool_out_dim(x.shape[2], pt, pb, kh, sh, dh, attrs["ceil_mode"])
    ow = _pool_out_dim(x.shape[3], pl, pr, kw, sw, dw, attrs["ceil_mode"])
    total = np.zeros((x.shape[0], x.shape[1], oh, ow), dtype=np.float32)
    divisor = np.zeros_like(total)
    # 统计有效位置（不含 padding），count_include_pad=1 时固定为 kernel 大小。
    valid = np.pad(
        np.ones(x.shape, dtype=np.float32),
        ((0, 0), (0, 0), (pt, pb), (pl, pr)),
        mode="constant",
    )
    for patch, i, j in _pool_patches(x, attrs, 0.0):
        total = total + patch.astype(np.float32)
        if attrs["count_include_pad"]:
            divisor = divisor + 1.0
        else:
            h0 = i * dh
            h1 = h0 + sh * (oh - 1) + 1
            w0 = j * dw
            w1 = w0 + sw * (ow - 1) + 1
            divisor = divisor + valid[:, :, h0:h1:sh, w0:w1:sw]
    return [(total / divisor).astype(x.dtype, copy=False)]


def _k_global_avg_pool(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    x = inputs[0]
    return [x.mean(axis=(2, 3), keepdims=True, dtype=np.float32).astype(
        x.dtype, copy=False
    )]


def _k_bn(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    x, scale, bias, mean, var = inputs[:5]
    shape = [1] * len(x.shape)
    shape[1] = x.shape[1]
    inv = 1.0 / np.sqrt(var.reshape(shape) + attrs["epsilon"])
    return [((x - mean.reshape(shape)) * inv * scale.reshape(shape)
             + bias.reshape(shape)).astype(x.dtype, copy=False)]


def _k_flatten(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    x = inputs[0]
    axis = attrs["axis"] if attrs["axis"] >= 0 else attrs["axis"] + x.ndim
    if axis == 0:
        return [x.reshape(1, -1)]
    return [x.reshape(int(np.prod(x.shape[:axis])), -1)]


def _k_reshape(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    shape = attrs.get("shape")
    if shape is None:
        shape = inputs[1].astype(np.int64).tolist()
    return [inputs[0].reshape(tuple(int(v) for v in shape))]


def _k_transpose(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    perm = attrs.get("perm")
    return [np.transpose(inputs[0], perm)]


def _k_concat(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    return [np.concatenate(inputs, axis=int(attrs["axis"]))]


def _k_softmax(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    x = inputs[0]
    axis = attrs["axis"]
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return [(e / np.sum(e, axis=axis, keepdims=True)).astype(x.dtype, copy=False)]


def _k_cast(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    from ..frontend.onnx_util import ONNX_DTYPE_TO_NAME

    dtype = ONNX_DTYPE_TO_NAME[int(attrs["to"])]
    return [inputs[0].astype(dtype)]


def _k_clip(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    lo = attrs.get("min", None)
    hi = attrs.get("max", None)
    if len(inputs) >= 2 and inputs[1] is not None:
        lo = float(inputs[1].reshape(-1)[0])
    if len(inputs) >= 3 and inputs[2] is not None:
        hi = float(inputs[2].reshape(-1)[0])
    return [np.clip(inputs[0], lo, hi)]


def _k_identity(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    return [inputs[0]]


def _k_cos(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    return [np.cos(inputs[0])]


def _k_conv_relu(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    return [np.maximum(_k_conv(inputs, attrs)[0], 0)]


def _k_gemm_relu(inputs: list[np.ndarray], attrs: dict) -> list[np.ndarray]:
    return [np.maximum(_k_gemm(inputs, attrs)[0], 0)]


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------


def _reg(
    op_type: str,
    canonical: str,
    infer,
    kernel,
    normalize=_norm_passthrough,
    *,
    npu_support=True,
    quant_rule="propagate",
) -> None:
    GLOBAL_REGISTRY.register(
        OpSpec(
            domain="",
            op_type=op_type,
            canonical_name=canonical,
            category="standard",
            infer_types=infer,
            kernel=kernel,
            normalize_attrs=normalize,
            npu_support=npu_support,
            fallback_backend="scalar_host",
            quant_rule=quant_rule,
        )
    )


def _reg_internal(op_type: str, canonical: str, infer, kernel) -> None:
    """注册编译器内部融合算子（domain=nest.internal）。"""
    GLOBAL_REGISTRY.register(
        OpSpec(
            domain=INTERNAL_DOMAIN,
            op_type=op_type,
            canonical_name=canonical,
            category="standard",
            infer_types=infer,
            kernel=kernel,
            normalize_attrs=_norm_passthrough,
            npu_support=True,
            fallback_backend="scalar_host",
        )
    )


_UNARY = {
    "Relu": lambda x: np.maximum(x, 0),
    "Sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "Tanh": np.tanh,
    "Exp": np.exp,
    "Sqrt": np.sqrt,
    "Abs": np.abs,
    "Neg": np.negative,
}
for _name, _impl in _UNARY.items():
    _reg(_name, _name.lower(), _infer_unary, _k_unary(_impl))
_reg("Identity", "identity", _infer_identity, _k_identity)

_BINARY = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}
for _name, _fn in _BINARY.items():
    _reg(_name, _name.lower(), _infer_binary, _k_binary(_fn))

_reg("MatMul", "matmul", _infer_matmul, _k_matmul)
_reg("Gemm", "gemm", _infer_gemm, _k_gemm, _norm_gemm, quant_rule="per_tensor_int8")
_reg("Conv", "conv2d", _infer_conv_output, _k_conv, _norm_conv,
     quant_rule="per_channel_int8")
_reg("MaxPool", "max_pool2d", _infer_pool, _k_maxpool, _norm_pool)
_reg("AveragePool", "avg_pool2d", _infer_pool, _k_avgpool, _norm_pool)
_reg("GlobalAveragePool", "global_avg_pool2d", _infer_global_avg_pool,
     _k_global_avg_pool)
_reg("BatchNormalization", "batch_norm", _infer_bn, _k_bn, _norm_bn,
     quant_rule="propagate")
_reg("Flatten", "flatten", _infer_flatten, _k_flatten, _norm_flatten)
_reg("Reshape", "reshape", _infer_reshape, _k_reshape)
_reg("Transpose", "transpose", _infer_transpose, _k_transpose, _norm_transpose)
_reg("Concat", "concat", _infer_concat, _k_concat)
_reg("Softmax", "softmax", _infer_softmax, _k_softmax, _norm_softmax)
_reg("Cast", "cast", _infer_cast, _k_cast, _norm_cast)
_reg("Clip", "clip", _infer_clip, _k_clip, _norm_clip)
_reg("Cos", "cos", _infer_unary, _k_cos, npu_support=False)

# 编译器内部融合算子：不参与 ONNX 导入，只在融合 pass 之后出现。
_reg_internal("conv2d_relu", "conv2d_relu", _infer_conv_output, _k_conv_relu)
_reg_internal("gemm_relu", "gemm_relu", _infer_gemm, _k_gemm_relu)

__all__ = ["FLOAT_DTYPES", "INTERNAL_DOMAIN"]
