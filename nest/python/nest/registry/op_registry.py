"""算子注册表与分发。

注册表是前端与 IR 的唯一事实来源。分发顺序：
1. 精确匹配 (domain, op_type) 的客户自定义算子；
2. 标准域（"" 或 "ai.onnx"）查标准算子表；
3. 未命中则生成 Fallback 节点（或按策略报错）。

自定义算子可在独立模块中通过 :func:`register_custom` 注册，无需修改核心代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..errors import RegistryError
from ..ir.types import TensorType

InferFn = Callable[[dict, list[TensorType]], list[TensorType]]
KernelFn = Callable[[list[np.ndarray], dict], list[np.ndarray]]
NormalizeAttrFn = Callable[[dict, int | None], dict]
NPUSupportFn = Callable[[dict, list[TensorType]], bool]

STANDARD_DOMAINS = ("", "ai.onnx")
RESERVED_CUSTOM_DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml"})
FALLBACK_BACKENDS = ("scalar_host", "scalar_riscv")


@dataclass(frozen=True)
class OpSpec:
    """一个算子的完整注册信息。"""

    domain: str
    op_type: str
    canonical_name: str
    category: str  # standard / custom
    infer_types: InferFn | None = None
    kernel: KernelFn | None = None
    normalize_attrs: NormalizeAttrFn | None = None
    npu_support: bool | NPUSupportFn = True
    fallback_backend: str = "scalar_host"
    quant_rule: str = "propagate"
    version: int = 1

    def __post_init__(self) -> None:
        if self.category not in ("standard", "custom"):
            raise RegistryError(f"非法算子类别: {self.category}")
        if not self.canonical_name:
            raise RegistryError(f"{self.domain}::{self.op_type} 缺少 canonical_name")
        if self.fallback_backend not in FALLBACK_BACKENDS:
            raise RegistryError(
                f"{self.domain}::{self.op_type} 的 fallback backend 非法: "
                f"{self.fallback_backend}"
            )

    def supports_npu(self, attrs: dict, input_types: list[TensorType]) -> bool:
        if callable(self.npu_support):
            return bool(self.npu_support(attrs, input_types))
        return bool(self.npu_support)

    @property
    def key(self) -> tuple[str, str]:
        return (self.domain, self.op_type)


@dataclass
class OpRegistry:
    """全局 / 局部算子注册表。"""

    _standard: dict[tuple[str, str], OpSpec] = field(default_factory=dict)
    _custom: dict[tuple[str, str], OpSpec] = field(default_factory=dict)
    _fallback_kernels: dict[tuple[str, str, str], KernelFn] = field(default_factory=dict)

    # ---- 注册 ----
    def register(self, spec: OpSpec, *, override: bool = False) -> OpSpec:
        table = self._standard if spec.category == "standard" else self._custom
        if spec.key in table and not override:
            raise RegistryError(
                f"算子重复注册: {spec.domain or 'ai.onnx'}::{spec.op_type}"
            )
        table[spec.key] = spec
        return spec

    def register_custom(
        self,
        domain: str,
        op_type: str,
        *,
        canonical_name: str | None = None,
        kernel: KernelFn | None = None,
        infer_types: InferFn | None = None,
        npu_support: bool | NPUSupportFn = False,
        fallback_backend: str = "scalar_host",
        attrs_normalizer: NormalizeAttrFn | None = None,
        quant_rule: str = "custom",
        override: bool = False,
    ) -> OpSpec:
        """注册客户自定义算子。

        默认 ``npu_support=False``：客户算子没有稳定 NPU 映射时，
        导入结果为 Fallback，并可通过注册的 ``kernel`` 在标量核对拍。
        ``infer_types`` 可以为空，此时输出 shape/dtype 必须来自模型 value_info。
        """
        if not domain or domain in RESERVED_CUSTOM_DOMAINS:
            raise RegistryError(
                "客户自定义算子必须使用独立的 domain，"
                f"不能使用保留域: {domain!r}"
            )
        # kernel / infer_types 都允许为空：前者表示没有标量实现，
        # 后者表示依赖模型 value_info 提供输出类型。
        spec = OpSpec(
            domain=domain,
            op_type=op_type,
            canonical_name=canonical_name or f"custom.{op_type}",
            category="custom",
            infer_types=infer_types,
            kernel=kernel,
            normalize_attrs=attrs_normalizer,
            npu_support=npu_support,
            fallback_backend=fallback_backend,
            quant_rule=quant_rule,
        )
        return self.register(spec, override=override)

    def register_fallback_kernel(
        self, domain: str, op_type: str, backend: str, kernel: KernelFn
    ) -> None:
        """为 fallback 节点注册标量实现。

        未注册的 fallback 算子在执行期必须明确报错，不能静默跳过。
        """
        if backend not in FALLBACK_BACKENDS:
            raise RegistryError(f"非法 fallback backend: {backend}")
        self._fallback_kernels[(domain, op_type, backend)] = kernel

    # ---- 查询 ----
    def lookup(self, domain: str, op_type: str) -> OpSpec | None:
        custom = self._custom.get((domain, op_type))
        if custom is not None:
            return custom
        # 精确 key 优先，支持编译器内部融合算子（如 nest.internal::conv2d_relu）。
        exact = self._standard.get((domain, op_type))
        if exact is not None:
            return exact
        if domain in STANDARD_DOMAINS:
            return self._standard.get(("", op_type))
        return None

    def is_custom(self, domain: str, op_type: str) -> bool:
        return (domain, op_type) in self._custom

    def get_fallback_kernel(
        self, domain: str, op_type: str, backend: str
    ) -> KernelFn | None:
        return self._fallback_kernels.get((domain, op_type, backend))

    def list_standard(self) -> list[OpSpec]:
        return sorted(self._standard.values(), key=lambda s: s.op_type)

    def list_custom(self) -> list[OpSpec]:
        return sorted(self._custom.values(), key=lambda s: (s.domain, s.op_type))

    def snapshot(self) -> tuple[dict, dict, dict]:
        return (dict(self._standard), dict(self._custom), dict(self._fallback_kernels))

    def restore(self, snapshot: tuple[dict, dict, dict]) -> None:
        self._standard, self._custom, self._fallback_kernels = snapshot


GLOBAL_REGISTRY = OpRegistry()


def register_custom(*args, **kwargs) -> OpSpec:
    return GLOBAL_REGISTRY.register_custom(*args, **kwargs)


def register_fallback_kernel(
    domain: str, op_type: str, backend: str, kernel: KernelFn
) -> None:
    GLOBAL_REGISTRY.register_fallback_kernel(domain, op_type, backend, kernel)
