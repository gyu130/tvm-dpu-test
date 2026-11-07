"""模型前端。"""

from .onnx import ImportOptions, ImportResult, import_onnx_model
from .p4 import (
    Add,
    And,
    Control,
    Deparser,
    Entry,
    Field,
    FieldRef,
    Header,
    IncrementCounter,
    IntConst,
    Or,
    P4Program,
    Parser,
    SetField,
    Shl,
    Shr,
    State,
    Sub,
    Table,
    build_ipv4_ttl_decrement,
)
from .torch import from_torch


def from_onnx(model, *, fallback_policy: str = "mark", input_shapes=None,
              module_name=None, options: ImportOptions | None = None) -> ImportResult:
    """导入 ONNX 模型。"""
    if options is None:
        options = ImportOptions(
            fallback_policy=fallback_policy,
            input_shapes=dict(input_shapes or {}),
            module_name=module_name,
        )
    return import_onnx_model(model, options=options)


__all__ = [
    "ImportOptions",
    "ImportResult",
    "from_onnx",
    "from_torch",
    "import_onnx_model",
    # P4 前端
    "Header",
    "Field",
    "FieldRef",
    "Parser",
    "State",
    "Table",
    "Entry",
    "Control",
    "Deparser",
    "P4Program",
    "IntConst",
    "Add",
    "Sub",
    "And",
    "Or",
    "Shl",
    "Shr",
    "SetField",
    "IncrementCounter",
    "build_ipv4_ttl_decrement",
]
