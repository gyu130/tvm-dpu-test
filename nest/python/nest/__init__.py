"""NestCompiler（NPU End-to-End Stack Compiler）M0 Python 包。"""

from .frontend import ImportOptions, ImportResult, from_onnx, from_torch
from .ir import Module
from .registry import (
    GLOBAL_REGISTRY,
    OpRegistry,
    OpSpec,
    register_custom,
    register_fallback_kernel,
)
from .runtime import run_graph, run_module

__version__ = "0.1.0"

__all__ = [
    "ImportOptions",
    "ImportResult",
    "from_onnx",
    "from_torch",
    "Module",
    "GLOBAL_REGISTRY",
    "OpRegistry",
    "OpSpec",
    "register_custom",
    "register_fallback_kernel",
    "run_graph",
    "run_module",
]
