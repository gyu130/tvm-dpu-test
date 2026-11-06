"""图优化 pass 集合与流水线。"""

from .base import PassOutput, PassStats
from .fusion import fuse
from .graph_passes import (
    canonicalize,
    constant_fold,
    cse,
    dce,
    simplify,
    validate_layouts,
)
from .partition import PartitionEntry, PartitionReport, analyze_partition, require_fully_npu
from .pipeline import DEFAULT_PASSES, OptimizeOptions, run_passes

__all__ = [
    "PassOutput",
    "PassStats",
    "canonicalize",
    "constant_fold",
    "cse",
    "dce",
    "simplify",
    "validate_layouts",
    "fuse",
    "PartitionEntry",
    "PartitionReport",
    "analyze_partition",
    "require_fully_npu",
    "DEFAULT_PASSES",
    "OptimizeOptions",
    "run_passes",
]
