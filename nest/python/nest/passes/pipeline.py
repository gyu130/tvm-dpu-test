"""图优化 pass 流水线。

pass 可单独启停，支持 stop_after；每个 pass 返回新的 Module，
失败时不会留下部分修改的中间状态。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..errors import PassError
from ..ir import Module
from .base import PassOutput, PassStats, rebuild_module, substitute_inputs
from .fusion import DEFAULT_PATTERNS, fuse as _fuse
from .graph_passes import (
    canonicalize as _canonicalize,
    constant_fold as _constant_fold,
    cse as _cse,
    dce as _dce,
    simplify as _simplify,
    validate_layouts,
)

PassFn = Callable[[Module, "OptimizeOptions"], PassOutput]

DEFAULT_PASSES = (
    "canonicalize",
    "constant_fold",
    "simplify",
    "cse",
    "fusion",
    "dce",
    "layout",
)


@dataclass
class OptimizeOptions:
    """图优化选项。"""

    passes: tuple[str, ...] = DEFAULT_PASSES
    stop_after: str | None = None
    disabled: tuple[str, ...] = ()
    fusion_patterns: tuple[str, ...] = DEFAULT_PATTERNS

    def resolve(self) -> tuple[str, ...]:
        names = list(self.passes)
        unknown = [name for name in names if name not in _PASSES]
        if unknown:
            raise PassError(
                f"未知 pass: {unknown}；可选: {sorted(_PASSES)}"
            )
        if self.stop_after is not None:
            if self.stop_after not in names:
                raise PassError(
                    f"stop_after={self.stop_after!r} 不在启用 pass 列表中: {names}"
                )
            names = names[: names.index(self.stop_after) + 1]
        disabled = set(self.disabled)
        unknown_disabled = disabled - set(_PASSES)
        if unknown_disabled:
            raise PassError(f"未知 disabled pass: {sorted(unknown_disabled)}")
        return tuple(name for name in names if name not in disabled)


def _run_canonicalize(module, options):
    return _canonicalize(module, options)


def _run_constant_fold(module, options):
    return _constant_fold(module, options)


def _run_simplify(module, options):
    return _simplify(module, options)


def _run_cse(module, options):
    return _cse(module, options)


def _run_fusion(module, options):
    return _fuse(module, options)


def _run_dce(module, options):
    return _dce(module, options)


def _run_layout(module, options):
    return validate_layouts(module, options)


_PASSES: dict[str, PassFn] = {
    "canonicalize": _run_canonicalize,
    "constant_fold": _run_constant_fold,
    "simplify": _run_simplify,
    "cse": _run_cse,
    "fusion": _run_fusion,
    "dce": _run_dce,
    "layout": _run_layout,
}


def run_passes(
    module: Module, options: OptimizeOptions | None = None
) -> tuple[Module, list[PassStats]]:
    options = options or OptimizeOptions()
    stats: list[PassStats] = []
    current = module
    for name in options.resolve():
        before = len(current.graph.nodes)
        started = time.perf_counter()
        output = _PASSES[name](current, options)
        elapsed = (time.perf_counter() - started) * 1000.0
        current = output.module
        stats.append(
            PassStats(
                name=name,
                nodes_before=before,
                nodes_after=len(current.graph.nodes),
                removed=output.removed,
                added=output.added,
                folded=output.folded,
                fused=output.fused,
                elapsed_ms=elapsed,
                notes=list(output.notes),
            )
        )
    return current, stats


__all__ = [
    "DEFAULT_PASSES",
    "OptimizeOptions",
    "PassStats",
    "run_passes",
    "substitute_inputs",
    "rebuild_module",
]
