"""AutoSchedule：在静态代价模型下搜索最优 tile 配置。

M1 的 :class:`ScheduleOptions` 只有 ``conv_oc_tile`` 与 ``gemm_n_tile`` 两个可调维度，
本模块围绕这两个维度做搜索：

1. **搜索空间**：所有能整除对应维度的 tile 候选（每个算子维度最多保留若干个）；
2. **代价模型**：按运算量（MAC / ALU 周期）与 DMA 字节数估算每个 stage 的延迟，
   双缓冲下 stage 延迟取 ``max(compute, dma)``；
3. **硬约束**：候选必须能通过 ``schedule_module`` + ``validate_program``，
   且 ``plan_memory`` 的 SRAM 峰值不超过预算；
4. **输出**：最优配置、对应 Program、每个 trial 的代价与可行性。

搜索策略：候选组合数不超过 ``max_trials`` 时全量枚举，否则按索引均匀抽样（确定性）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import prod

from ..errors import ScheduleError
from ..ir import CallNode, ConstantNode, Module, TensorType
from .memory import MemoryPlan, plan_memory
from .schedule import ScheduleOptions, schedule_module
from .validate import validate_program

_CONV_OPS = frozenset({"conv2d", "conv2d_relu"})
_GEMM_OPS = frozenset({"gemm", "gemm_relu"})
_REDUCE_OPS = frozenset({"max_pool2d", "avg_pool2d", "global_avg_pool2d", "softmax"})


def _elem_size(dtype: str) -> int:
    return {
        "float32": 4, "float16": 2, "float64": 8,
        "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
        "int32": 4, "int64": 8, "bool": 1,
    }.get(dtype, 4)


def _effective_tile(default: int, extent: int) -> int:
    """与 schedule._divisor_tile 相同的规则：不超过 default 且整除 extent 的最大值。"""
    tile = min(default, extent)
    while tile > 1 and extent % tile != 0:
        tile -= 1
    return max(tile, 1)


# ---------------------------------------------------------------------------
# 代价模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """硬件代价模型参数（单位：每周期）。"""

    alu_ops_per_cycle: float = 4.0
    mul_ops_per_cycle: float = 1.0
    dma_bytes_per_cycle: float = 64.0
    double_buffer: bool = True

    def __post_init__(self) -> None:
        if self.alu_ops_per_cycle <= 0 or self.mul_ops_per_cycle <= 0:
            raise ScheduleError("CostModel 的 ALU/MUL 吞吐必须为正数")
        if self.dma_bytes_per_cycle <= 0:
            raise ScheduleError("CostModel 的 DMA 带宽必须为正数")


@dataclass(frozen=True)
class StageEstimate:
    stage: str
    compute_cycles: float
    dma_bytes: int
    dma_cycles: float

    @property
    def latency_cycles(self) -> float:
        return max(self.compute_cycles, self.dma_cycles)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "compute_cycles": round(self.compute_cycles, 2),
            "dma_bytes": self.dma_bytes,
            "dma_cycles": round(self.dma_cycles, 2),
            "latency_cycles": round(self.latency_cycles, 2),
        }

    def summary_text(self) -> str:
        return (
            f"{self.stage:<24s} compute={self.compute_cycles:>12.0f}  "
            f"dma={self.dma_bytes:>10d}B ({self.dma_cycles:>10.0f}c)  "
            f"latency={self.latency_cycles:>12.0f}c"
        )


@dataclass
class CostEstimate:
    stages: list[StageEstimate] = field(default_factory=list)

    @property
    def total_cycles(self) -> float:
        return sum(item.latency_cycles for item in self.stages)

    @property
    def total_dma_bytes(self) -> int:
        return sum(item.dma_bytes for item in self.stages)

    def to_dict(self) -> dict:
        return {
            "total_cycles": round(self.total_cycles, 2),
            "total_dma_bytes": self.total_dma_bytes,
            "stages": [item.to_dict() for item in self.stages],
        }

    def summary_text(self) -> str:
        lines = [item.summary_text() for item in self.stages]
        lines.append(
            f"{'TOTAL':<24s} {' ':>22s} {' ':>20s} latency={self.total_cycles:>12.0f}c"
        )
        return "\n".join(lines)


def _compute_cycles(macs: int, cost: CostModel) -> float:
    """MAC 拆成 MUL + ADD 两条 DPU 指令，返回总周期。"""
    mul_cycles = macs / cost.mul_ops_per_cycle
    add_cycles = macs / cost.alu_ops_per_cycle
    return mul_cycles + add_cycles


def _elementwise_cycles(elems: int, cost: CostModel) -> float:
    return elems / cost.alu_ops_per_cycle


def _stage_estimate(
    compute_cycles: float, dma_bytes: int, name: str, cost: CostModel
) -> StageEstimate:
    dma_cycles = dma_bytes / cost.dma_bytes_per_cycle
    item = StageEstimate(
        stage=name,
        compute_cycles=compute_cycles,
        dma_bytes=dma_bytes,
        dma_cycles=dma_cycles,
    )
    return item


def estimate_module(module: Module, options: ScheduleOptions, cost: CostModel) -> CostEstimate:
    """按静态代价模型估算整个 Module 的延迟（不依赖实际生成的 Program）。"""
    types: dict[str, TensorType] = {name: v.type for name, v in module.graph.tensors.items()}
    estimates: list[StageEstimate] = []

    for index, node in enumerate(module.graph.nodes):
        if isinstance(node, ConstantNode) or not isinstance(node, CallNode):
            continue
        name = f"s{index}_{node.canonical_name}"
        canonical = node.canonical_name

        if canonical in _CONV_OPS:
            x_type = types[node.inputs[0]]
            w_type = types[node.inputs[1]]
            out_type = node.outputs[0].type
            n, ic, ih, iw = x_type.shape[0], x_type.shape[1], x_type.shape[2], x_type.shape[3]
            oc, _, kh, kw = w_type.shape
            oh, ow = out_type.shape[2], out_type.shape[3]
            group = int(node.attrs.get("group", 1)) or 1
            elem = _elem_size(x_type.dtype)

            oc_eff = oc // group
            tile = _effective_tile(options.conv_oc_tile, oc_eff) if group == 1 else oc_eff
            iterations = (oc_eff + tile - 1) // tile * group

            macs = n * oc * oh * ow * ic * kh * kw
            compute = _compute_cycles(macs, cost)

            input_bytes = n * ic * ih * iw * elem
            weight_tile = tile * (ic // group) * kh * kw * elem
            out_tile = n * tile * oh * ow * elem
            dma_bytes = iterations * (input_bytes + weight_tile + out_tile)
            estimates.append(_stage_estimate(compute, dma_bytes, name, cost))

        elif canonical in _GEMM_OPS:
            a_type = types[node.inputs[0]]
            b_type = types[node.inputs[1]]
            trans_a = int(node.attrs.get("transA", 0))
            trans_b = int(node.attrs.get("transB", 0))
            m = a_type.shape[1] if trans_a else a_type.shape[0]
            k = a_type.shape[0] if trans_a else a_type.shape[1]
            n = b_type.shape[0] if trans_b else b_type.shape[1]
            elem = _elem_size(a_type.dtype)

            tile = _effective_tile(options.gemm_n_tile, n)
            iterations = (n + tile - 1) // tile

            macs = m * k * n
            compute = _compute_cycles(macs, cost)

            a_bytes = m * k * elem
            b_tile = k * tile * elem
            y_tile = m * tile * elem
            dma_bytes = iterations * (a_bytes + b_tile + y_tile)
            estimates.append(_stage_estimate(compute, dma_bytes, name, cost))

        else:
            out_type = node.outputs[0].type
            elems = prod(out_type.shape)
            elem = _elem_size(out_type.dtype)
            compute = _elementwise_cycles(elems, cost)
            if canonical in _REDUCE_OPS:
                # reduce 需要遍历输入多次，按输入规模放大
                in_type = types[node.inputs[0]]
                elems = prod(in_type.shape)
                compute = _elementwise_cycles(elems, cost)
            in_bytes = sum(prod(types[name].shape) * _elem_size(types[name].dtype)
                           for name in node.inputs if name in types)
            out_bytes = elems * elem
            dma_bytes = in_bytes + out_bytes
            estimates.append(_stage_estimate(compute, dma_bytes, name, cost))

    return CostEstimate(stages=estimates)


# ---------------------------------------------------------------------------
# 候选生成
# ---------------------------------------------------------------------------


def _divisors(total: int, min_tile: int, max_tile: int) -> list[int]:
    return [d for d in range(max(min_tile, 1), min(total, max_tile) + 1) if total % d == 0]


def _trim(candidates: list[int], limit: int) -> list[int]:
    """超过上限时均匀抽样，保留两端（最小/最大 tile）。"""
    if len(candidates) <= limit:
        return candidates
    if limit <= 2:
        return [candidates[0], candidates[-1]][:limit]
    step = (len(candidates) - 1) / (limit - 1)
    picked = sorted({candidates[round(i * step)] for i in range(limit)})
    return picked


def _common_divisors(dims: list[int], min_tile: int, max_tile: int) -> list[int]:
    """多个维度都可整除的 tile（全局 tile 会被每个算子复用）。"""
    if not dims:
        return []
    smallest = min(dims)
    candidates = [d for d in _divisors(smallest, min_tile, max_tile)
                  if all(dim % d == 0 for dim in dims)]
    if candidates:
        return candidates
    # 没有公共因子时退化为最小维度的因子
    return _divisors(smallest, min_tile, max_tile)


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoScheduleOptions:
    """AutoSchedule 配置。"""

    base: ScheduleOptions = field(default_factory=ScheduleOptions)
    max_trials: int = 64
    max_candidates_per_axis: int = 6
    min_tile: int = 1
    max_tile: int = 1024
    cost: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        if self.max_trials <= 0:
            raise ScheduleError("max_trials 必须为正数")
        if self.max_candidates_per_axis <= 0:
            raise ScheduleError("max_candidates_per_axis 必须为正数")
        if self.min_tile <= 0:
            raise ScheduleError("min_tile 必须为正数")
        if self.max_tile < self.min_tile:
            raise ScheduleError("max_tile 不能小于 min_tile")


@dataclass
class Trial:
    """一次 tile 配置的评估结果。"""

    conv_oc_tile: int
    gemm_n_tile: int
    feasible: bool
    estimate: CostEstimate
    sram_peak: int = 0
    reason: str = ""

    @property
    def score(self) -> float:
        return self.estimate.total_cycles

    @property
    def rank_key(self) -> tuple:
        """排序键：先比估计延迟，同分时优先大 tile（减少循环/传输次数），最后比 SRAM。"""
        return (round(self.score, 6), -self.conv_oc_tile, -self.gemm_n_tile, self.sram_peak)

    def to_dict(self) -> dict:
        return {
            "conv_oc_tile": self.conv_oc_tile,
            "gemm_n_tile": self.gemm_n_tile,
            "feasible": self.feasible,
            "score_cycles": round(self.score, 2),
            "sram_peak": self.sram_peak,
            "reason": self.reason,
            "estimate": self.estimate.to_dict(),
        }

    def summary_text(self) -> str:
        flag = "OK  " if self.feasible else "FAIL"
        reason = f"  ({self.reason})" if self.reason else ""
        return (
            f"[{flag}] conv_oc_tile={self.conv_oc_tile:<4d} gemm_n_tile={self.gemm_n_tile:<4d} "
            f"score={self.score:>12.0f}c  sram={self.sram_peak:>9d}B{reason}"
        )


@dataclass
class AutoScheduleResult:
    best_options: ScheduleOptions
    best_program: object
    best_plan: MemoryPlan
    best_trial: Trial
    trials: list[Trial]

    @property
    def feasible_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.feasible]

    def summary_text(self, limit: int = 10) -> str:
        lines = [
            f"AutoSchedule: {len(self.trials)} trials, "
            f"{len(self.feasible_trials)} feasible",
            f"BEST conv_oc_tile={self.best_options.conv_oc_tile} "
            f"gemm_n_tile={self.best_options.gemm_n_tile} "
            f"score={self.best_trial.score:.0f}c sram={self.best_plan.sram_peak}B",
            "",
        ]
        ranked = sorted(self.feasible_trials, key=lambda t: t.rank_key)
        if not ranked:
            ranked = self.trials
        lines.extend(item.summary_text() for item in ranked[:limit])
        return "\n".join(lines)


def _build_axes(
    module: Module, opts: AutoScheduleOptions
) -> list[tuple[str, list[int]]]:
    """返回 [(维度名, 候选 tile 列表)]，维度名 ∈ {"conv", "gemm"}。"""
    types = {name: v.type for name, v in module.graph.tensors.items()}
    conv_dims: list[int] = []
    gemm_dims: list[int] = []
    for node in module.graph.nodes:
        if not isinstance(node, CallNode):
            continue
        if node.canonical_name in _CONV_OPS:
            group = int(node.attrs.get("group", 1)) or 1
            conv_dims.append(types[node.inputs[0]].shape[1] // group)
        elif node.canonical_name in _GEMM_OPS:
            b_type = types[node.inputs[1]]
            trans_b = int(node.attrs.get("transB", 0))
            gemm_dims.append(b_type.shape[0] if trans_b else b_type.shape[1])

    axes: list[tuple[str, list[int]]] = []
    conv_candidates = _trim(
        _common_divisors(conv_dims, opts.min_tile, opts.max_tile),
        opts.max_candidates_per_axis,
    )
    gemm_candidates = _trim(
        _common_divisors(gemm_dims, opts.min_tile, opts.max_tile),
        opts.max_candidates_per_axis,
    )
    if conv_candidates:
        axes.append(("conv", conv_candidates))
    if gemm_candidates:
        axes.append(("gemm", gemm_candidates))
    return axes


def _configs(
    axes: list[tuple[str, list[int]]], opts: AutoScheduleOptions
) -> list[tuple[int, int]]:
    """把候选组合映射回 (conv_oc_tile, gemm_n_tile)。"""
    if not axes:
        return [(opts.base.conv_oc_tile, opts.base.gemm_n_tile)]

    combos = list(product(*[values for _, values in axes]))
    if len(combos) > opts.max_trials:
        # 候选过多：按索引均匀抽样，保证确定性
        step = (len(combos) - 1) / (opts.max_trials - 1) if opts.max_trials > 1 else 1
        combos = [combos[round(i * step)] for i in range(opts.max_trials)]
        combos = list(dict.fromkeys(combos))

    result: list[tuple[int, int]] = []
    for combo in combos:
        conv_tile = opts.base.conv_oc_tile
        gemm_tile = opts.base.gemm_n_tile
        for (kind, _), value in zip(axes, combo):
            if kind == "conv":
                conv_tile = value
            else:
                gemm_tile = value
        result.append((conv_tile, gemm_tile))
    return result


def _evaluate(
    module: Module,
    options: ScheduleOptions,
    opts: AutoScheduleOptions,
) -> tuple[Trial, object, MemoryPlan | None]:
    """生成 Program + 内存规划并评估；任何异常都记为不可行。"""
    estimate = estimate_module(module, options, opts.cost)
    program = None
    try:
        program = schedule_module(module, options=options)
        validate_program(program)
        plan = plan_memory(
            program, sram_bytes=options.sram_bytes, alignment=options.alignment
        )
    except Exception as exc:  # noqa: BLE001 - 任何失败都记为不可行
        return (
            Trial(
                conv_oc_tile=options.conv_oc_tile,
                gemm_n_tile=options.gemm_n_tile,
                feasible=False,
                estimate=estimate,
                reason=f"{type(exc).__name__}: {exc}",
            ),
            program,
            None,
        )

    feasible = plan.sram_peak <= options.sram_bytes
    reason = "" if feasible else (
        f"SRAM 峰值 {plan.sram_peak}B 超出预算 {options.sram_bytes}B"
    )
    trial = Trial(
        conv_oc_tile=options.conv_oc_tile,
        gemm_n_tile=options.gemm_n_tile,
        feasible=feasible,
        estimate=estimate,
        sram_peak=plan.sram_peak,
        reason=reason,
    )
    return trial, program, plan


def auto_schedule(
    module: Module,
    options: AutoScheduleOptions | None = None,
) -> AutoScheduleResult:
    """搜索最优 tile 配置并返回对应 Program 与内存规划。"""
    opts = options or AutoScheduleOptions()
    axes = _build_axes(module, opts)

    trials: list[Trial] = []
    best: tuple[Trial, ScheduleOptions, object, MemoryPlan] | None = None

    for conv_tile, gemm_tile in _configs(axes, opts):
        trial_options = ScheduleOptions(
            sram_bytes=opts.base.sram_bytes,
            alignment=opts.base.alignment,
            conv_oc_tile=conv_tile,
            gemm_n_tile=gemm_tile,
            bind_core=opts.base.bind_core,
        )
        trial, program, plan = _evaluate(module, trial_options, opts)
        trials.append(trial)
        if not trial.feasible or plan is None:
            continue
        if best is None or trial.rank_key < best[0].rank_key:
            best = (trial, trial_options, program, plan)

    if best is None:
        peaks = [t.sram_peak for t in trials if t.sram_peak]
        max_peak = max(peaks) if peaks else 0
        raise ScheduleError(
            f"AutoSchedule 未找到可行配置：共尝试 {len(trials)} 个候选，"
            f"SRAM 峰值最大约 {max_peak}B，预算 {opts.base.sram_bytes}B"
        )

    return AutoScheduleResult(
        best_options=best[1],
        best_program=best[2],
        best_plan=best[3],
        best_trial=best[0],
        trials=trials,
    )


__all__ = [
    "AutoScheduleOptions",
    "AutoScheduleResult",
    "CostEstimate",
    "CostModel",
    "StageEstimate",
    "Trial",
    "auto_schedule",
    "estimate_module",
]
