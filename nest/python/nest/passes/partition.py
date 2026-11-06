"""NPU / fallback 分区分析。

M1 只支持单个全 NPU 子图：任何 fallback 节点或缺少 NPU kernel 的节点都会
阻止指令生成，并在报告中给出节点、source span 与原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import PartitionError
from ..ir import CallNode, FallbackNode, Module
from ..registry import GLOBAL_REGISTRY, OpRegistry


@dataclass
class PartitionEntry:
    value: str
    op: str
    domain: str
    canonical: str
    span: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "op": self.op,
            "domain": self.domain,
            "canonical": self.canonical,
            "span": self.span,
            "reason": self.reason,
        }


@dataclass
class PartitionReport:
    npu: list[PartitionEntry] = field(default_factory=list)
    fallback: list[PartitionEntry] = field(default_factory=list)

    @property
    def is_fully_npu(self) -> bool:
        return not self.fallback

    def to_dict(self) -> dict:
        return {
            "is_fully_npu": self.is_fully_npu,
            "npu": [entry.to_dict() for entry in self.npu],
            "fallback": [entry.to_dict() for entry in self.fallback],
        }


def analyze_partition(
    module: Module, registry: OpRegistry = GLOBAL_REGISTRY
) -> PartitionReport:
    tensors = module.graph.tensors
    report = PartitionReport()
    for node in module.graph.nodes:
        if isinstance(node, FallbackNode):
            report.fallback.append(
                PartitionEntry(
                    value=node.outputs[0].name,
                    op=node.op_type,
                    domain=node.domain,
                    canonical=node.op_type,
                    span=str(node.span),
                    reason=node.reason,
                )
            )
            continue
        if not isinstance(node, CallNode):
            continue
        spec = registry.lookup(node.domain, node.op_type)
        input_types = [tensors[name].type for name in node.inputs]
        reason = None
        if spec is None:
            reason = "算子未注册，无法生成 NPU 指令"
        elif not spec.supports_npu(node.attrs, input_types):
            reason = "算子不满足 NPU 支持条件"
        elif spec.kernel is None:
            reason = "缺少 compute kernel，无法生成可仿真的 NPU 指令"
        if reason is None:
            report.npu.append(
                PartitionEntry(
                    value=node.outputs[0].name,
                    op=node.op_type,
                    domain=node.domain,
                    canonical=node.canonical_name,
                    span=str(node.span),
                )
            )
        else:
            report.fallback.append(
                PartitionEntry(
                    value=node.outputs[0].name,
                    op=node.op_type,
                    domain=node.domain,
                    canonical=node.canonical_name,
                    span=str(node.span),
                    reason=reason,
                )
            )
    return report


def require_fully_npu(report: PartitionReport) -> None:
    if report.is_fully_npu:
        return
    lines = ["M1 只支持单个全 NPU 子图，当前存在无法进入 NPU 调度的节点："]
    for entry in report.fallback:
        lines.append(
            f"  - {entry.domain + '::' if entry.domain else ''}{entry.op} "
            f"({entry.value}) @ {entry.span}: {entry.reason}"
        )
    lines.append(
        "建议：移除/替换该算子，或等待多段执行（NPU 子图 + 标量 fallback 子图）支持"
    )
    raise PartitionError("\n".join(lines))
