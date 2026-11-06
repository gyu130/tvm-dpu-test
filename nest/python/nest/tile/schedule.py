"""单核 NestTile 调度模板。

M1 提供：
- conv（含 conv2d_relu）：按输出通道 tile；
- gemm（含 gemm_relu）：按输出 N 维 tile；
- elementwise / reduce / generic：整块加载、计算、写回。

所有模板都显式声明 SRAM buffer、DMA 与同步，不依赖隐式分配。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ScheduleError
from ..ir import CallNode, ConstantNode, FallbackNode, Module, SourceSpan, TensorType
from ..registry import GLOBAL_REGISTRY, OpRegistry
from .ir import Buffer, Compute, Copy, Loop, Program, RegionSpec, Stage, Sync, TileRef

DEFAULT_SRAM_BYTES = 16 * 1024 * 1024
DEFAULT_ALIGNMENT = 64
DEFAULT_CONV_OC_TILE = 64
DEFAULT_GEMM_N_TILE = 64

_REDUCE_OPS = frozenset({"max_pool2d", "avg_pool2d", "global_avg_pool2d", "softmax"})
_ELEMENTWISE_OPS = frozenset(
    {
        "relu",
        "sigmoid",
        "tanh",
        "exp",
        "sqrt",
        "abs",
        "neg",
        "cos",
        "add",
        "sub",
        "mul",
        "div",
        "clip",
        "cast",
        "identity",
    }
)
_CONV_OPS = frozenset({"conv2d", "conv2d_relu"})
_GEMM_OPS = frozenset({"gemm", "gemm_relu"})


@dataclass
class ScheduleOptions:
    """调度配置。M1 固定单核、静态 tile，不做 autotune。"""

    sram_bytes: int = DEFAULT_SRAM_BYTES
    alignment: int = DEFAULT_ALIGNMENT
    conv_oc_tile: int = DEFAULT_CONV_OC_TILE
    gemm_n_tile: int = DEFAULT_GEMM_N_TILE
    bind_core: int = 0

    def __post_init__(self) -> None:
        if self.sram_bytes <= 0:
            raise ScheduleError("sram_bytes 必须为正数")
        if self.alignment <= 0:
            raise ScheduleError("alignment 必须为正数")
        if self.conv_oc_tile <= 0 or self.gemm_n_tile <= 0:
            raise ScheduleError("tile 大小必须为正数")
        if self.bind_core < 0:
            raise ScheduleError("bind_core 不能为负")


def _divisor_tile(default: int, extent: int) -> int:
    """选择不超过 default 且整除 extent 的最大 tile。"""
    tile = min(default, extent)
    while tile > 1 and extent % tile != 0:
        tile -= 1
    return max(tile, 1)


def _unique_name(base: str, existing: set[str]) -> str:
    name = base
    index = 1
    while name in existing:
        name = f"{base}_{index}"
        index += 1
    existing.add(name)
    return name


def _buffer(
    base: str,
    dtype: str,
    shape: tuple[int, ...],
    layout: str,
    alignment: int,
    existing: set[str],
    scope: str = "sram",
) -> Buffer:
    return Buffer(
        name=_unique_name(base, existing),
        scope=scope,
        dtype=dtype,
        shape=tuple(int(d) for d in shape),
        layout=layout,
        alignment=alignment,
    )


def _value_types(module: Module) -> dict[str, TensorType]:
    return {name: value.type for name, value in module.graph.tensors.items()}


def _build_global_buffers(
    module: Module, alignment: int
) -> tuple[list[Buffer], dict[str, Buffer]]:
    buffers: list[Buffer] = []
    mapping: dict[str, Buffer] = {}
    existing: set[str] = set()
    for value in module.graph.inputs:
        buffer = _buffer(
            value.name, value.type.dtype, value.type.shape, value.type.layout,
            alignment, existing, scope="global",
        )
        buffers.append(buffer)
        mapping[value.name] = buffer
    for node in module.graph.nodes:
        if isinstance(node, FallbackNode):
            raise ScheduleError(
                f"节点 {node.op_type} 是 fallback，不能进入单核 NPU 调度 @ {node.span}"
            )
        if isinstance(node, ConstantNode):
            for value in node.outputs:
                buffer = _buffer(
                    value.name, value.type.dtype, value.type.shape, value.type.layout,
                    alignment, existing, scope="global",
                )
                buffers.append(buffer)
                mapping[value.name] = buffer
            continue
    # Call 的输出在第二遍补齐，保持“输入/常量/节点输出”的确定性顺序。
    for node in module.graph.nodes:
        if isinstance(node, CallNode):
            for value in node.outputs:
                buffer = _buffer(
                    value.name, value.type.dtype, value.type.shape, value.type.layout,
                    alignment, existing, scope="global",
                )
                buffers.append(buffer)
                mapping[value.name] = buffer
    return buffers, mapping


def _conv_stage(
    node: CallNode,
    index: int,
    types: dict[str, TensorType],
    options: ScheduleOptions,
    existing: set[str],
) -> Stage:
    stage_name = f"s{index}_{node.canonical_name}"
    x_type = types[node.inputs[0]]
    w_type = types[node.inputs[1]]
    out_type = node.outputs[0].type
    if len(x_type.shape) != 4 or len(w_type.shape) != 4:
        raise ScheduleError(
            f"{stage_name}: M1 conv 模板只支持 4D NCHW 输入/OIHW 权重 @ {node.span}"
        )
    n, channels = x_type.shape[0], x_type.shape[1]
    m, cg, kh, kw = w_type.shape
    group = int(node.attrs.get("group", 1))
    if group <= 0 or channels % group != 0 or m % group != 0:
        raise ScheduleError(f"{stage_name}: conv group={group} 不合法")
    mg = m // group
    cg = channels // group
    if group == 1:
        oc_tile = _divisor_tile(options.conv_oc_tile, m)
    else:
        # 子 tile 固定为一个 group，保证子卷积 group=1。
        oc_tile = mg
    oh, ow = out_type.shape[2], out_type.shape[3]
    dtype = x_type.dtype
    align = options.alignment

    x_shape = (n, cg, x_type.shape[2], x_type.shape[3]) if group > 1 else x_type.shape
    x_buf = _buffer(f"{stage_name}_x", dtype, x_shape, "NCHW", align, existing)
    w_buf = _buffer(f"{stage_name}_w", w_type.dtype, (oc_tile, cg, kh, kw), "OIHW",
                    align, existing)
    y_buf = _buffer(f"{stage_name}_y", out_type.dtype, (n, oc_tile, oh, ow), "NCHW",
                    align, existing)
    buffers = [x_buf, w_buf]

    load_token = f"{stage_name}.load"
    store_token = f"{stage_name}.store"
    span = node.span
    prologue: list = []
    body: list = []

    if group == 1:
        x_token = f"{stage_name}.x"
        prologue.append(
            Copy(src=node.inputs[0], dst=x_buf.name, span=span, async_token=x_token)
        )
        prologue.append(Sync(kind="dma_wait", token=x_token, span=span))
    else:
        body.append(
            Copy(
                src=node.inputs[0],
                dst=x_buf.name,
                span=span,
                src_region=RegionSpec(
                    offsets=(0, TileRef("oc", cg), 0, 0),
                    extents=(n, cg, x_type.shape[2], x_type.shape[3]),
                ),
                async_token=load_token,
            )
        )

    body.append(
        Copy(
            src=node.inputs[1],
            dst=w_buf.name,
            span=span,
            src_region=RegionSpec(
                offsets=(TileRef("oc", oc_tile), 0, 0, 0),
                extents=(oc_tile, cg, kh, kw),
            ),
            async_token=load_token,
        )
    )
    compute_inputs = [x_buf.name, w_buf.name]
    if len(node.inputs) > 2:
        b_type = types[node.inputs[2]]
        b_buf = _buffer(f"{stage_name}_b", b_type.dtype, (oc_tile,), "C", align,
                        existing)
        buffers.append(b_buf)
        body.append(
            Copy(
                src=node.inputs[2],
                dst=b_buf.name,
                span=span,
                src_region=RegionSpec(
                    offsets=(TileRef("oc", oc_tile),) , extents=(oc_tile,)
                ),
                async_token=load_token,
            )
        )
        compute_inputs.append(b_buf.name)

    buffers.append(y_buf)
    body.append(Sync(kind="dma_wait", token=load_token, span=span))
    sub_attrs = dict(node.attrs)
    sub_attrs["group"] = 1
    body.append(
        Compute(
            domain=node.domain,
            op_type=node.op_type,
            canonical_name=node.canonical_name,
            inputs=tuple(compute_inputs),
            outputs=(y_buf.name,),
            attrs=sub_attrs,
            span=span,
        )
    )
    body.append(Sync(kind="barrier", token=None, span=span))
    body.append(
        Copy(
            src=y_buf.name,
            dst=node.outputs[0].name,
            span=span,
            dst_region=RegionSpec(
                offsets=(0, TileRef("oc", oc_tile), 0, 0),
                extents=(n, oc_tile, oh, ow),
            ),
            async_token=store_token,
        )
    )
    body.append(Sync(kind="dma_wait", token=store_token, span=span))

    return Stage(
        name=stage_name,
        template="conv",
        node_value=node.outputs[0].name,
        loops=(Loop("oc", axis=1, extent=m, tile=oc_tile),),
        prologue=tuple(prologue),
        body=tuple(body),
        buffers=tuple(buffers),
        span=node.span,
    )


def _gemm_stage(
    node: CallNode,
    index: int,
    types: dict[str, TensorType],
    options: ScheduleOptions,
    existing: set[str],
) -> Stage:
    stage_name = f"s{index}_{node.canonical_name}"
    a_type = types[node.inputs[0]]
    b_type = types[node.inputs[1]]
    out_type = node.outputs[0].type
    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if len(a_type.shape) != 2 or len(b_type.shape) != 2:
        raise ScheduleError(f"{stage_name}: M1 gemm 模板只支持 2D 输入")
    m = a_type.shape[1] if trans_a else a_type.shape[0]
    k = a_type.shape[0] if trans_a else a_type.shape[1]
    n = b_type.shape[0] if trans_b else b_type.shape[1]
    k_b = b_type.shape[1] if trans_b else b_type.shape[0]
    if k != k_b:
        raise ScheduleError(f"{stage_name}: Gemm K 维不匹配: {k} vs {k_b}")
    if tuple(out_type.shape) != (m, n):
        raise ScheduleError(
            f"{stage_name}: Gemm 输出 shape {out_type.shape} 与推导 ({m},{n}) 不一致"
        )
    nt = _divisor_tile(options.gemm_n_tile, n)
    align = options.alignment
    dtype = a_type.dtype

    a_buf = _buffer(f"{stage_name}_a", dtype, a_type.shape, "linear", align, existing)
    b_shape = (nt, k) if trans_b else (k, nt)
    b_buf = _buffer(f"{stage_name}_b", b_type.dtype, b_shape, "linear", align, existing)
    y_buf = _buffer(f"{stage_name}_y", out_type.dtype, (m, nt), "linear", align, existing)
    buffers = [a_buf, b_buf, y_buf]

    load_token = f"{stage_name}.load"
    store_token = f"{stage_name}.store"
    span = node.span
    a_token = f"{stage_name}.a"
    prologue: list = [
        Copy(src=node.inputs[0], dst=a_buf.name, span=span, async_token=a_token),
        Sync(kind="dma_wait", token=a_token, span=span),
    ]
    b_region = (
        RegionSpec(offsets=(TileRef("n", nt), 0), extents=(nt, k))
        if trans_b
        else RegionSpec(offsets=(0, TileRef("n", nt)), extents=(k, nt))
    )
    body: list = [
        Copy(
            src=node.inputs[1],
            dst=b_buf.name,
            span=span,
            src_region=b_region,
            async_token=load_token,
        )
    ]
    compute_inputs = [a_buf.name, b_buf.name]
    if len(node.inputs) > 2:
        c_type = types[node.inputs[2]]
        c_buf = _buffer(f"{stage_name}_c", c_type.dtype, (nt,), "linear", align,
                        existing)
        buffers.append(c_buf)
        body.append(
            Copy(
                src=node.inputs[2],
                dst=c_buf.name,
                span=span,
                src_region=RegionSpec(
                    offsets=(TileRef("n", nt),), extents=(nt,)
                ),
                async_token=load_token,
            )
        )
        compute_inputs.append(c_buf.name)

    body.append(Sync(kind="dma_wait", token=load_token, span=span))
    body.append(
        Compute(
            domain=node.domain,
            op_type=node.op_type,
            canonical_name=node.canonical_name,
            inputs=tuple(compute_inputs),
            outputs=(y_buf.name,),
            attrs=dict(node.attrs),
            span=span,
        )
    )
    body.append(Sync(kind="barrier", token=None, span=span))
    body.append(
        Copy(
            src=y_buf.name,
            dst=node.outputs[0].name,
            span=span,
            dst_region=RegionSpec(
                offsets=(0, TileRef("n", nt)), extents=(m, nt)
            ),
            async_token=store_token,
        )
    )
    body.append(Sync(kind="dma_wait", token=store_token, span=span))

    return Stage(
        name=stage_name,
        template="gemm",
        node_value=node.outputs[0].name,
        loops=(Loop("n", axis=1, extent=n, tile=nt),),
        prologue=tuple(prologue),
        body=tuple(body),
        buffers=tuple(buffers),
        span=node.span,
    )


def _template_name(canonical: str) -> str:
    if canonical in _CONV_OPS:
        return "conv"
    if canonical in _GEMM_OPS:
        return "gemm"
    if canonical in _REDUCE_OPS:
        return "reduce"
    if canonical in _ELEMENTWISE_OPS:
        return "elementwise"
    return "generic"


def _generic_stage(
    node: CallNode,
    index: int,
    types: dict[str, TensorType],
    options: ScheduleOptions,
    existing: set[str],
) -> Stage:
    stage_name = f"s{index}_{node.canonical_name}"
    align = options.alignment
    span: SourceSpan = node.span
    buffers: list[Buffer] = []
    prologue: list = []
    body: list = []
    input_buffers: list[str] = []
    output_buffers: list[str] = []

    for position, name in enumerate(node.inputs):
        t = types[name]
        buffer = _buffer(
            f"{stage_name}_in{position}", t.dtype, t.shape, t.layout, align, existing
        )
        buffers.append(buffer)
        input_buffers.append(buffer.name)
        prologue.append(
            Copy(src=name, dst=buffer.name, span=span,
                 async_token=f"{stage_name}.load")
        )

    for position, value in enumerate(node.outputs):
        t = value.type
        buffer = _buffer(
            f"{stage_name}_out{position}", t.dtype, t.shape, t.layout, align,
            existing,
        )
        buffers.append(buffer)
        output_buffers.append(buffer.name)

    prologue.append(
        Sync(kind="dma_wait", token=f"{stage_name}.load", span=span)
    )
    body.append(
        Compute(
            domain=node.domain,
            op_type=node.op_type,
            canonical_name=node.canonical_name,
            inputs=tuple(input_buffers),
            outputs=tuple(output_buffers),
            attrs=dict(node.attrs),
            span=span,
        )
    )
    body.append(Sync(kind="barrier", token=None, span=span))
    store_token = f"{stage_name}.store"
    for position, value in enumerate(node.outputs):
        body.append(
            Copy(
                src=output_buffers[position],
                dst=value.name,
                span=span,
                async_token=store_token,
            )
        )
    body.append(Sync(kind="dma_wait", token=store_token, span=span))

    return Stage(
        name=stage_name,
        template=_template_name(node.canonical_name),
        node_value=node.outputs[0].name,
        loops=(),
        prologue=tuple(prologue),
        body=tuple(body),
        buffers=tuple(buffers),
        span=node.span,
    )


def schedule_module(
    module: Module,
    options: ScheduleOptions | None = None,
    registry: OpRegistry = GLOBAL_REGISTRY,
) -> Program:
    """为单个全 NPU 子图生成单核 NestTile 程序。"""
    options = options or ScheduleOptions()
    global_buffers, _ = _build_global_buffers(module, options.alignment)
    existing = {buffer.name for buffer in global_buffers}
    types = _value_types(module)
    stages: list[Stage] = []

    for index, node in enumerate(module.graph.nodes):
        if isinstance(node, ConstantNode):
            continue
        if not isinstance(node, CallNode):
            raise ScheduleError(
                f"节点类型 {type(node).__name__} 不支持单核调度 @ {node.span}"
            )
        spec = registry.lookup(node.domain, node.op_type)
        if spec is None or spec.kernel is None:
            raise ScheduleError(
                f"算子 {node.domain + '::' if node.domain else ''}{node.op_type} "
                f"缺少 kernel，无法生成 compute 指令 @ {node.span}"
            )
        if node.canonical_name in _CONV_OPS:
            stage = _conv_stage(node, index, types, options, existing)
        elif node.canonical_name in _GEMM_OPS:
            stage = _gemm_stage(node, index, types, options, existing)
        else:
            stage = _generic_stage(node, index, types, options, existing)
        stages.append(stage)

    return Program(name=module.name, global_buffers=global_buffers, stages=stages)


__all__ = ["DEFAULT_SRAM_BYTES", "ScheduleOptions", "schedule_module"]
