"""M2 多核切分。

在 M1 单核 Program 上按输出维度切分：
- Conv：oc / batch / H / W（H/W 显式计算 halo 与逐核 pads）；
- Gemm：n / batch(M)；
- elementwise / reduce / generic：batch / oc / n / h / w。

每个核得到独立的 Program（私有 SRAM），stage 之间插入全核 barrier；
输出切分方案、CoreSyncGraph 与确定性负载报告。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ..errors import ScheduleError
from ..ir import CallNode, Module, TensorType
from ..registry import GLOBAL_REGISTRY, OpRegistry
from .ir import Buffer, Compute, Copy, Loop, Program, RegionSpec, Stage, Sync, TileRef
from .memory import MemoryPlan, plan_memory
from .pipeline import DoubleBufferPlan, plan_double_buffer
from .schedule import ScheduleOptions, schedule_module
from .validate import validate_program

_SPLIT_DIMS = frozenset({"auto", "batch", "oc", "n", "h", "w"})
_CONV_OPS = frozenset({"conv2d", "conv2d_relu"})
_GEMM_OPS = frozenset({"gemm", "gemm_relu"})
_REDUCE_OPS = frozenset({"max_pool2d", "avg_pool2d", "global_avg_pool2d", "softmax"})
_RANK_CHANGING = frozenset({"flatten", "reshape"})


@dataclass(frozen=True)
class MultiCoreOptions:
    cores: int = 2
    split_dim: str = "auto"
    schedule_options: ScheduleOptions = field(default_factory=ScheduleOptions)
    double_buffer: bool = False

    def __post_init__(self) -> None:
        if self.cores < 1:
            raise ScheduleError(f"cores 必须 >= 1: {self.cores}")
        if self.split_dim not in _SPLIT_DIMS:
            raise ScheduleError(
                f"非法 split_dim: {self.split_dim}；可选 {sorted(_SPLIT_DIMS)}"
            )


@dataclass(frozen=True)
class HaloSpec:
    core: int
    axis: int
    output_start: int
    output_end: int
    input_start: int
    input_end: int
    pad_before: int
    pad_after: int

    def to_dict(self) -> dict:
        return {
            "core": self.core,
            "axis": self.axis,
            "output_start": self.output_start,
            "output_end": self.output_end,
            "input_start": self.input_start,
            "input_end": self.input_end,
            "pad_before": self.pad_before,
            "pad_after": self.pad_after,
        }


@dataclass(frozen=True)
class SplitPlan:
    stage: str
    template: str
    split_dim: str
    axis: int
    extent: int
    chunk: int
    cores: int
    halo: tuple[HaloSpec, ...] = ()

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "template": self.template,
            "split_dim": self.split_dim,
            "axis": self.axis,
            "extent": self.extent,
            "chunk": self.chunk,
            "cores": self.cores,
            "halo": [item.to_dict() for item in self.halo],
        }


@dataclass(frozen=True)
class CoreBarrier:
    index: int
    stage_name: str
    cores: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "stage_name": self.stage_name,
            "cores": list(self.cores),
        }


@dataclass(frozen=True)
class CoreSyncGraph:
    core_count: int
    barriers: tuple[CoreBarrier, ...]

    def validate(self, programs: list[Program] | None = None) -> None:
        if self.core_count < 1:
            raise ScheduleError("CoreSyncGraph.core_count 必须 >= 1")
        expected = tuple(range(self.core_count))
        for index, barrier in enumerate(self.barriers):
            if barrier.index != index:
                raise ScheduleError(
                    f"barrier 序号不连续: 期望 {index}，得到 {barrier.index}"
                )
            if barrier.cores != expected:
                raise ScheduleError(
                    f"barrier {index} 未覆盖全部核: {barrier.cores} != {expected}"
                )
        if programs is not None:
            if len(programs) != self.core_count:
                raise ScheduleError(
                    f"程序数量 {len(programs)} 与核数 {self.core_count} 不一致"
                )
            stage_counts = {len(program.stages) for program in programs}
            if stage_counts and stage_counts != {len(self.barriers)}:
                raise ScheduleError(
                    f"barrier 数量 {len(self.barriers)} 与 stage 数 "
                    f"{sorted(stage_counts)} 不一致（会导致核间死锁）"
                )
            for index, barrier in enumerate(self.barriers):
                for program in programs:
                    if program.stages[index].name != barrier.stage_name:
                        raise ScheduleError(
                            f"核 {program.stages[index].core} 的 stage 顺序与 "
                            f"barrier {index} 不一致: "
                            f"{program.stages[index].name} vs {barrier.stage_name}"
                        )

    def to_dict(self) -> dict:
        return {
            "core_count": self.core_count,
            "barriers": [barrier.to_dict() for barrier in self.barriers],
        }


@dataclass(frozen=True)
class CoreLoad:
    core: int
    stages: int
    tiles: int
    compute_elements: int
    macs: int
    sram_peak: int

    def to_dict(self) -> dict:
        return {
            "core": self.core,
            "stages": self.stages,
            "tiles": self.tiles,
            "compute_elements": self.compute_elements,
            "macs": self.macs,
            "sram_peak": self.sram_peak,
        }

    def summary_text(self) -> str:
        return (
            f"core {self.core}: stages={self.stages}, tiles={self.tiles}, "
            f"elements={self.compute_elements}, macs={self.macs}, "
            f"sram_peak={self.sram_peak}"
        )


@dataclass
class MultiCoreProgram:
    cores: int
    programs: list[Program]
    splits: tuple[SplitPlan, ...]
    memory: list[MemoryPlan]
    sync: CoreSyncGraph
    loads: tuple[CoreLoad, ...]
    double_buffer: tuple[DoubleBufferPlan, ...]

    def validate(self) -> None:
        if len(self.programs) != self.cores:
            raise ScheduleError(
                f"programs 数量 {len(self.programs)} 与核数 {self.cores} 不一致"
            )
        self.sync.validate(self.programs)
        for program in self.programs:
            validate_program(program)
        if self.double_buffer:
            if len(self.double_buffer) != self.cores:
                raise ScheduleError(
                    f"双缓冲计划数量 {len(self.double_buffer)} 与核数 "
                    f"{self.cores} 不一致"
                )
            for plan, program in zip(self.double_buffer, self.programs):
                plan.validate(program)

    def to_dict(self) -> dict:
        return {
            "cores": self.cores,
            "splits": [plan.to_dict() for plan in self.splits],
            "sync": self.sync.to_dict(),
            "loads": [load.to_dict() for load in self.loads],
            "memory": [plan.to_dict() for plan in self.memory],
            "double_buffer": [plan.to_dict() for plan in self.double_buffer],
            "programs": [program.to_dict() for program in self.programs],
        }

    def summary_text(self) -> str:
        lines = [f"多核程序: cores={self.cores}"]
        for plan in self.splits:
            halo = ""
            if plan.halo:
                halo = (
                    f", halo=[{plan.halo[0].pad_before},{plan.halo[0].pad_after}]"
                )
            lines.append(
                f"  {plan.stage}: {plan.template} split={plan.split_dim} "
                f"axis={plan.axis} extent={plan.extent} chunk={plan.chunk}{halo}"
            )
        for load in self.loads:
            lines.append(f"  {load.summary_text()}")
        lines.append(
            f"  barriers: {len(self.sync.barriers)} 个全核 barrier"
        )
        for plan in self.double_buffer:
            lines.append(f"  {plan.summary_text()}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------

def _compute(stage: Stage) -> Compute:
    for item in stage.body:
        if isinstance(item, Compute):
            return item
    raise ScheduleError(f"stage {stage.name} 缺少 Compute")


def _buffer(stage: Stage, name: str) -> Buffer:
    for buffer in stage.buffers:
        if buffer.name == name:
            return buffer
    raise ScheduleError(f"stage {stage.name} 缺少 buffer {name}")


def _replace_buffer(stage: Stage, updated: dict[str, Buffer]) -> Stage:
    if not updated:
        return stage
    return replace(
        stage,
        buffers=tuple(updated.get(buffer.name, buffer) for buffer in stage.buffers),
    )


def _shrink(buffer: Buffer, axis: int, extent: int) -> Buffer:
    shape = list(buffer.shape)
    shape[axis] = extent
    return replace(buffer, shape=tuple(shape), address=None)


def _map_region(
    region: RegionSpec | None,
    loop_var: str,
    old_tile: int,
    new_tile: int,
    base: int,
) -> tuple[RegionSpec | None, list[int]]:
    if region is None:
        return None, []
    offsets = []
    extents = []
    changed_dims = []
    for index, offset in enumerate(region.offsets):
        extent = region.extents[index]
        if (
            isinstance(offset, TileRef)
            and offset.loop == loop_var
            and offset.tile == old_tile
        ):
            if extent != old_tile:
                raise ScheduleError(
                    f"TileRef {loop_var} 的 extent {extent} 与 tile {old_tile} "
                    "不一致，无法多核切分"
                )
            offsets.append(replace(offset, tile=new_tile, base=offset.base + base))
            extents.append(new_tile)
            changed_dims.append(index)
        else:
            offsets.append(offset)
            extents.append(extent)
    return RegionSpec(offsets=tuple(offsets), extents=tuple(extents)), changed_dims


def _loop_split(stage: Stage, loop: Loop, chunk: int, core: int) -> Stage:
    old_tile = loop.tile
    new_tile = math.gcd(old_tile, chunk)
    base = core * chunk
    new_loop = Loop(loop.var, loop.axis, chunk, new_tile)
    loops = tuple(new_loop if item.var == loop.var else item for item in stage.loops)
    stage_buffers = {buffer.name for buffer in stage.buffers}
    updates: dict[str, Buffer] = {}
    prologue = []
    body = []
    for item in stage.prologue:
        if isinstance(item, Copy):
            src_region, src_dims = _map_region(
                item.src_region, loop.var, old_tile, new_tile, base
            )
            dst_region, dst_dims = _map_region(
                item.dst_region, loop.var, old_tile, new_tile, base
            )
            for axis in src_dims:
                if item.dst in stage_buffers:
                    current = updates.get(item.dst, _buffer(stage, item.dst))
                    updates[item.dst] = _shrink(current, axis, new_tile)
            for axis in dst_dims:
                if item.src in stage_buffers:
                    current = updates.get(item.src, _buffer(stage, item.src))
                    updates[item.src] = _shrink(current, axis, new_tile)
            prologue.append(replace(item, src_region=src_region, dst_region=dst_region))
        else:
            prologue.append(item)
    for item in stage.body:
        if isinstance(item, Copy):
            src_region, src_dims = _map_region(
                item.src_region, loop.var, old_tile, new_tile, base
            )
            dst_region, dst_dims = _map_region(
                item.dst_region, loop.var, old_tile, new_tile, base
            )
            for axis in src_dims:
                if item.dst in stage_buffers:
                    current = updates.get(item.dst, _buffer(stage, item.dst))
                    updates[item.dst] = _shrink(current, axis, new_tile)
            for axis in dst_dims:
                if item.src in stage_buffers:
                    current = updates.get(item.src, _buffer(stage, item.src))
                    updates[item.src] = _shrink(current, axis, new_tile)
            body.append(replace(item, src_region=src_region, dst_region=dst_region))
        else:
            body.append(item)
    stage = _replace_buffer(stage, updates)
    return replace(
        stage,
        loops=loops,
        prologue=tuple(prologue),
        body=tuple(body),
        core=core,
    )


def _conv_geometry(compute: Compute) -> tuple[list[int], list[int], list[int]]:
    pads = [int(v) for v in compute.attrs.get("pads", [0, 0, 0, 0])]
    strides = [int(v) for v in compute.attrs.get("strides", [1, 1])]
    dilations = [int(v) for v in compute.attrs.get("dilations", [1, 1])]
    return pads, strides, dilations


def _conv_spatial_split(
    stage: Stage,
    axis: int,
    chunk: int,
    core: int,
    out_shape: tuple[int, ...],
) -> tuple[Stage, HaloSpec | None]:
    compute = _compute(stage)
    x_buf = _buffer(stage, f"{stage.name}_x")
    y_buf = _buffer(stage, f"{stage.name}_y")
    w_buf = _buffer(stage, f"{stage.name}_w")
    x_shape = tuple(x_buf.shape)
    rank = len(x_shape)

    pads, strides, dilations = _conv_geometry(compute)
    output_start = core * chunk
    output_end = output_start + chunk
    pad_before = 0
    pad_after = 0
    if axis == 0:
        input_start = output_start
        input_end = output_end
        new_x_shape = x_shape[:0] + (chunk,) + x_shape[1:]
    elif axis in (2, 3):
        kernel = w_buf.shape[axis]
        position = 0 if axis == 2 else 1
        stride = strides[position]
        dilation = dilations[position]
        pad = pads[position]
        effective = (kernel - 1) * dilation + 1
        raw_start = output_start * stride - pad
        raw_end = (output_end - 1) * stride - pad + effective
        input_start = max(0, raw_start)
        input_end = min(x_shape[axis], raw_end)
        pad_before = max(0, -raw_start)
        pad_after = max(0, raw_end - x_shape[axis])
        if input_start >= input_end:
            raise ScheduleError(
                f"stage {stage.name} 核 {core} 的 halo 区间为空: "
                f"[{raw_start}, {raw_end})"
            )
        shape = list(x_shape)
        shape[axis] = input_end - input_start
        new_x_shape = tuple(shape)
    else:
        raise ScheduleError(f"conv 不支持在 axis={axis} 上做空间切分")

    new_x = _shrink(x_buf, axis, new_x_shape[axis])
    new_y = _shrink(y_buf, axis, chunk)

    x_region = None

    def transform(item):
        nonlocal x_region
        if isinstance(item, Copy) and item.dst == x_buf.name:
            if item.src_region is None:
                offsets = [0] * rank
                offsets[axis] = input_start
                x_region = RegionSpec(
                    offsets=tuple(offsets), extents=tuple(new_x_shape)
                )
            else:
                offsets = list(item.src_region.offsets)
                offsets[axis] = input_start
                x_region = RegionSpec(
                    offsets=tuple(offsets), extents=tuple(new_x_shape)
                )
            return replace(item, src_region=x_region)
        if isinstance(item, Copy) and item.src == y_buf.name:
            offsets = list(item.dst_region.offsets) if item.dst_region else []
            extents = list(item.dst_region.extents) if item.dst_region else []
            offsets[axis] = output_start
            extents[axis] = chunk
            return replace(
                item,
                dst_region=RegionSpec(
                    offsets=tuple(offsets), extents=tuple(extents)
                ),
            )
        return item

    new_prologue = [transform(item) for item in stage.prologue]
    new_body = [transform(item) for item in stage.body]

    if x_region is None:
        raise ScheduleError(f"stage {stage.name} 缺少输入 x 的 DMA 装载")

    if axis in (2, 3):
        pad_position = 0 if axis == 2 else 1
        after_position = 2 if axis == 2 else 3
        attrs = dict(compute.attrs)
        new_pads = list(pads)
        new_pads[pad_position] = pad_before
        new_pads[after_position] = pad_after
        attrs["pads"] = new_pads
        new_body = [
            replace(item, attrs=attrs) if isinstance(item, Compute) else item
            for item in new_body
        ]

    stage = _replace_buffer(stage, {x_buf.name: new_x, y_buf.name: new_y})
    stage = replace(
        stage,
        prologue=tuple(new_prologue),
        body=tuple(new_body),
        core=core,
    )
    halo = None
    if axis in (2, 3):
        halo = HaloSpec(
            core=core,
            axis=axis,
            output_start=output_start,
            output_end=output_end,
            input_start=input_start,
            input_end=input_end,
            pad_before=pad_before,
            pad_after=pad_after,
        )
    return stage, halo


def _gemm_batch_split(
    stage: Stage,
    chunk: int,
    core: int,
    out_shape: tuple[int, ...],
) -> Stage:
    compute = _compute(stage)
    a_buf = _buffer(stage, f"{stage.name}_a")
    y_buf = _buffer(stage, f"{stage.name}_y")
    trans_a = int(compute.attrs.get("transA", 0))
    a_axis = 1 if trans_a else 0
    new_a = _shrink(a_buf, a_axis, chunk)
    new_y = _shrink(y_buf, 0, chunk)

    def transform(item):
        if isinstance(item, Copy) and item.dst == a_buf.name:
            offsets = [0] * len(a_buf.shape)
            offsets[a_axis] = core * chunk
            return replace(
                item,
                src_region=RegionSpec(
                    offsets=tuple(offsets), extents=tuple(new_a.shape)
                ),
            )
        if isinstance(item, Copy) and item.src == y_buf.name:
            offsets = list(item.dst_region.offsets) if item.dst_region else []
            extents = list(item.dst_region.extents) if item.dst_region else []
            offsets[0] = core * chunk
            extents[0] = chunk
            return replace(
                item,
                dst_region=RegionSpec(
                    offsets=tuple(offsets), extents=tuple(extents)
                ),
            )
        return item

    new_prologue = [transform(item) for item in stage.prologue]
    new_body = [transform(item) for item in stage.body]
    stage = _replace_buffer(stage, {a_buf.name: new_a, y_buf.name: new_y})
    return replace(
        stage,
        prologue=tuple(new_prologue),
        body=tuple(new_body),
        core=core,
    )


def _input_split_axis(
    canonical: str,
    src_shape: tuple[int, ...],
    out_rank: int,
    axis: int,
    extent: int,
) -> int | None:
    if canonical in _RANK_CHANGING and axis == 0:
        return 0 if src_shape and src_shape[0] == extent else None
    if len(src_shape) == out_rank:
        candidate = axis
    elif len(src_shape) < out_rank:
        candidate = axis - (out_rank - len(src_shape))
    else:
        return None
    if candidate < 0 or candidate >= len(src_shape):
        return None
    if src_shape[candidate] != extent:
        return None
    return candidate


def _generic_split(
    stage: Stage,
    axis: int,
    chunk: int,
    core: int,
    out_shape: tuple[int, ...],
    types: dict[str, TensorType],
) -> Stage:
    compute = _compute(stage)
    canonical = compute.canonical_name
    out_rank = len(out_shape)
    extent = out_shape[axis]
    if canonical in _RANK_CHANGING and axis != 0:
        raise ScheduleError(
            f"stage {stage.name} 是 {canonical}，M2 只支持按 batch 轴切分"
        )
    if canonical in _REDUCE_OPS and axis in (2, 3):
        raise ScheduleError(
            f"stage {stage.name} 是 {canonical}，H/W 切分需要 halo，"
            "M2 仅支持 Conv 的 halo 切分"
        )
    buffers = {buffer.name: buffer for buffer in stage.buffers}
    updates: dict[str, Buffer] = {}

    def transform(item):
        if isinstance(item, Copy) and item.src in types and item.dst in buffers:
            src_shape = tuple(types[item.src].shape)
            src_axis = _input_split_axis(
                canonical, src_shape, out_rank, axis, extent
            )
            if src_axis is not None:
                new_buffer = _shrink(buffers[item.dst], src_axis, chunk)
                offsets = [0] * len(src_shape)
                offsets[src_axis] = core * chunk
                updates[item.dst] = new_buffer
                return replace(
                    item,
                    src_region=RegionSpec(
                        offsets=tuple(offsets), extents=tuple(new_buffer.shape)
                    ),
                )
        if (
            isinstance(item, Copy)
            and item.src in buffers
            and item.dst not in buffers
        ):
            buffer = buffers[item.src]
            if buffer.shape[axis] != extent:
                raise ScheduleError(
                    f"stage {stage.name} 输出 buffer {buffer.name} 的轴 {axis} "
                    f"维度 {buffer.shape[axis]} 与切分 extent {extent} 不一致"
                )
            new_buffer = _shrink(buffer, axis, chunk)
            offsets = [0] * len(new_buffer.shape)
            offsets[axis] = core * chunk
            updates[item.src] = new_buffer
            return replace(
                item,
                dst_region=RegionSpec(
                    offsets=tuple(offsets), extents=tuple(new_buffer.shape)
                ),
            )
        return item

    new_prologue = [transform(item) for item in stage.prologue]
    new_body = [transform(item) for item in stage.body]
    stage = _replace_buffer(stage, updates)
    return replace(
        stage,
        prologue=tuple(new_prologue),
        body=tuple(new_body),
        core=core,
    )


# --------------------------------------------------------------------------
# 切分方案
# --------------------------------------------------------------------------

def _explicit_axis(template: str, split_dim: str, rank: int) -> int:
    if split_dim == "batch":
        return 0
    if split_dim == "oc":
        if template != "conv" and rank < 2:
            raise ScheduleError(f"{template} 的 oc 切分需要秩 >= 2")
        return 1
    if split_dim == "n":
        if template == "gemm":
            return 1
        return rank - 1
    if split_dim == "h":
        if rank < 3:
            raise ScheduleError(f"{template} 的 h 切分需要秩 >= 3")
        return 2
    if split_dim == "w":
        if rank < 4:
            raise ScheduleError(f"{template} 的 w 切分需要秩 >= 4")
        return 3
    raise ScheduleError(f"非法 split_dim: {split_dim}")


def _choose_split(
    template: str, out_shape: tuple[int, ...], options: MultiCoreOptions
) -> tuple[str, int, int]:
    rank = len(out_shape)
    cores = options.cores

    def acceptable(axis: int) -> bool:
        if axis < 0 or axis >= rank:
            return False
        dim = out_shape[axis]
        return dim >= cores and dim % cores == 0

    if options.split_dim != "auto":
        axis = _explicit_axis(template, options.split_dim, rank)
        if axis >= rank:
            raise ScheduleError(
                f"{template} 输出秩 {rank} 不支持 axis={axis} 切分"
            )
        dim = out_shape[axis]
        if dim < cores:
            raise ScheduleError(
                f"{template} 在 {options.split_dim} 轴只有 {dim} 个 tile，"
                f"少于核数 {cores}"
            )
        if dim % cores != 0:
            raise ScheduleError(
                f"{template} 在 {options.split_dim} 轴维度 {dim} 不能被 "
                f"{cores} 核整除"
            )
        return options.split_dim, axis, dim // cores

    if template == "conv":
        candidates = [("oc", 1), ("batch", 0)]
    elif template == "gemm":
        candidates = [("n", 1), ("batch", 0)]
    else:
        candidates = [(name, axis) for name, axis in
                      [("batch", 0), ("oc", 1), ("n", rank - 1)]
                      if axis < rank and axis >= 0]
    for name, axis in candidates:
        if acceptable(axis):
            return name, axis, out_shape[axis] // cores
    raise ScheduleError(
        f"{template} 输出 shape {out_shape} 没有可被 {options.cores} 核整除的轴；"
        "请显式指定 split_dim 或调整核数"
    )


def _split_stage(
    stage: Stage,
    split_dim: str,
    axis: int,
    chunk: int,
    core: int,
    out_shape: tuple[int, ...],
    types: dict[str, TensorType],
) -> tuple[Stage, HaloSpec | None]:
    if stage.template in ("conv", "gemm"):
        loop = next((item for item in stage.loops if item.axis == axis), None)
        if loop is not None:
            if loop.extent != out_shape[axis]:
                raise ScheduleError(
                    f"stage {stage.name} 的 loop extent {loop.extent} 与输出轴 "
                    f"{out_shape[axis]} 不一致"
                )
            return _loop_split(stage, loop, chunk, core), None
    if stage.template == "conv":
        spatial, halo = _conv_spatial_split(stage, axis, chunk, core, out_shape)
        return spatial, halo
    if stage.template == "gemm":
        if axis != 0:
            raise ScheduleError(
                f"gemm stage {stage.name} 只支持 n 轴循环切分或 batch(M) 切分"
            )
        return _gemm_batch_split(stage, chunk, core, out_shape), None
    return (
        _generic_split(stage, axis, chunk, core, out_shape, types),
        None,
    )


# --------------------------------------------------------------------------
# 负载报告
# --------------------------------------------------------------------------

def _stage_metrics(stage: Stage, compute: Compute, buffers: dict[str, Buffer]) -> tuple[int, int, int]:
    iterations = 1
    for loop in stage.loops:
        iterations *= loop.iterations
    elements = 0
    for name in compute.outputs:
        elements += _prod(buffers[name].shape)
    total_elements = elements * iterations
    macs = total_elements
    if stage.template == "conv":
        weight = buffers[compute.inputs[1]]
        factor = _prod(tuple(weight.shape[1:]))
        macs = total_elements * factor
    elif stage.template == "gemm":
        b_buffer = buffers[compute.inputs[1]]
        trans_b = int(compute.attrs.get("transB", 0))
        k = b_buffer.shape[1] if trans_b else b_buffer.shape[0]
        macs = total_elements * k
    return iterations, total_elements, macs


def _prod(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def _load_report(
    core: int, program: Program, memory: MemoryPlan
) -> CoreLoad:
    tiles = 0
    elements = 0
    macs = 0
    for stage in program.stages:
        compute = _compute(stage)
        buffers = {buffer.name: buffer for buffer in stage.buffers}
        iterations, stage_elements, stage_macs = _stage_metrics(
            stage, compute, buffers
        )
        tiles += iterations
        elements += stage_elements
        macs += stage_macs
    return CoreLoad(
        core=core,
        stages=len(program.stages),
        tiles=tiles,
        compute_elements=elements,
        macs=macs,
        sram_peak=memory.sram_peak,
    )


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------

def schedule_multicore(
    module: Module,
    options: MultiCoreOptions | None = None,
    *,
    registry: OpRegistry = GLOBAL_REGISTRY,
) -> MultiCoreProgram:
    """生成多核 NestTile 程序；切分结果必须可复现。"""
    options = options or MultiCoreOptions()
    single = schedule_module(module, options.schedule_options, registry)
    if options.cores == 1:
        memory = plan_memory(
            single,
            sram_bytes=options.schedule_options.sram_bytes,
            alignment=options.schedule_options.alignment,
        )
        program = memory.program
        validate_program(program)
        barriers = tuple(
            CoreBarrier(index, stage.name, (0,))
            for index, stage in enumerate(program.stages)
        )
        sync = CoreSyncGraph(core_count=1, barriers=barriers)
        sync.validate([program])
        loads = (_load_report(0, program, memory),)
        double = (plan_double_buffer(program),) if options.double_buffer else ()
        return MultiCoreProgram(
            cores=1,
            programs=[program],
            splits=(),
            memory=[memory],
            sync=sync,
            loads=loads,
            double_buffer=double,
        )

    types = {
        name: value.type for name, value in module.graph.tensors.items()
    }
    splits: list[SplitPlan] = []
    per_core_stages: list[list[Stage]] = [[] for _ in range(options.cores)]
    for stage in single.stages:
        out_shape = tuple(types[stage.node_value].shape)
        split_dim, axis, chunk = _choose_split(stage.template, out_shape, options)
        halo_specs = []
        for core in range(options.cores):
            split_stage, halo = _split_stage(
                stage, split_dim, axis, chunk, core, out_shape, types
            )
            per_core_stages[core].append(split_stage)
            if halo is not None:
                halo_specs.append(halo)
        splits.append(
            SplitPlan(
                stage=stage.name,
                template=stage.template,
                split_dim=split_dim,
                axis=axis,
                extent=out_shape[axis],
                chunk=chunk,
                cores=options.cores,
                halo=tuple(halo_specs),
            )
        )

    programs: list[Program] = []
    memories: list[MemoryPlan] = []
    for core in range(options.cores):
        program = Program(
            name=f"{single.name}.core{core}",
            global_buffers=list(single.global_buffers),
            stages=per_core_stages[core],
        )
        memory = plan_memory(
            program,
            sram_bytes=options.schedule_options.sram_bytes,
            alignment=options.schedule_options.alignment,
        )
        programs.append(memory.program)
        memories.append(memory)
        validate_program(memory.program)

    barriers = tuple(
        CoreBarrier(index, program.stages[index].name,
                    tuple(range(options.cores)))
        for index in range(len(programs[0].stages))
    )
    sync = CoreSyncGraph(core_count=options.cores, barriers=barriers)
    sync.validate(programs)
    loads = tuple(
        _load_report(core, programs[core], memories[core])
        for core in range(options.cores)
    )
    double = (
        tuple(plan_double_buffer(program) for program in programs)
        if options.double_buffer
        else ()
    )
    result = MultiCoreProgram(
        cores=options.cores,
        programs=programs,
        splits=tuple(splits),
        memory=memories,
        sync=sync,
        loads=loads,
        double_buffer=double,
    )
    return result


__all__ = [
    "CoreBarrier",
    "CoreLoad",
    "CoreSyncGraph",
    "HaloSpec",
    "MultiCoreOptions",
    "MultiCoreProgram",
    "SplitPlan",
    "schedule_multicore",
]
