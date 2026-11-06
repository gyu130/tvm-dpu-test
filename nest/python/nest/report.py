"""导入报告。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FallbackEntry:
    op: str
    domain: str
    backend: str
    reason: str
    span: str

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "domain": self.domain,
            "backend": self.backend,
            "reason": self.reason,
            "span": self.span,
        }


@dataclass
class ImportReport:
    frontend: str
    model_file: str
    module_name: str
    opset: dict[str, int]
    node_counts: dict[str, int] = field(default_factory=dict)
    op_counts: dict[str, int] = field(default_factory=dict)
    fallback: list[FallbackEntry] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frontend": self.frontend,
            "model_file": self.model_file,
            "module_name": self.module_name,
            "opset": self.opset,
            "node_counts": self.node_counts,
            "op_counts": self.op_counts,
            "fallback": [f.to_dict() for f in self.fallback],
            "inputs": self.inputs,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def summary_text(self) -> str:
        lines = [
            f"导入报告: {self.model_file}",
            f"前端: {self.frontend}，模块: {self.module_name}",
            "opset: " + ", ".join(f"{k or 'ai.onnx'}={v}" for k, v in self.opset.items()),
            "节点分类: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.node_counts.items())),
        ]
        if self.fallback:
            lines.append(f"fallback ({len(self.fallback)}):")
            for entry in self.fallback:
                lines.append(
                    f"  - {entry.domain + '::' if entry.domain else ''}{entry.op} "
                    f"[{entry.backend}] {entry.reason} @ {entry.span}"
                )
        else:
            lines.append("fallback: 无")
        if self.warnings:
            lines.append("警告:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)
