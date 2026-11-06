"""模型前端。"""

from .onnx import ImportOptions, ImportResult, import_onnx_model
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


__all__ = ["ImportOptions", "ImportResult", "from_onnx", "from_torch", "import_onnx_model"]
