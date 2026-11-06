"""算子注册表入口。

导入本包即加载标准算子表；客户自定义算子模块通过 ``register_custom`` 注册。
"""

from .op_registry import (
    FALLBACK_BACKENDS,
    GLOBAL_REGISTRY,
    OpRegistry,
    OpSpec,
    register_custom,
    register_fallback_kernel,
)
from . import standard_ops  # noqa: F401  （导入即注册标准算子）

__all__ = [
    "FALLBACK_BACKENDS",
    "GLOBAL_REGISTRY",
    "OpRegistry",
    "OpSpec",
    "register_custom",
    "register_fallback_kernel",
]
