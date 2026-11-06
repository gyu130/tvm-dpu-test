"""PyTorch 前端（M0）。

M0 采用“torch.onnx.export -> ONNX importer”的桥接方式：
Torch 只作为可选依赖，便于快速打通端到端；M1 再评估原生 FX/ATen 图导入。
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

from ..errors import ModelImportError
from .onnx import ImportOptions, ImportResult, import_onnx_model


def from_torch(
    model,
    args: tuple,
    *,
    opset: int = 17,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    module_name: str | None = None,
    options: ImportOptions | None = None,
) -> ImportResult:
    """导入 PyTorch 模块并返回 NestGraph 模块。

    ``args`` 是 torch.onnx.export 所需的示例输入元组。
    动态 shape 由 ``options.input_shapes`` 显式固化。
    """
    try:
        import torch
    except ImportError as exc:
        raise ModelImportError(
            "Torch 前端需要可选依赖 torch；"
            "请先安装后重试（M0 仅通过 ONNX 桥接导入）"
        ) from exc

    if not isinstance(args, tuple):
        raise ModelImportError("from_torch 的 args 必须是 tuple")

    with tempfile.TemporaryDirectory(prefix="nest_torch_") as tmpdir:
        onnx_path = str(Path(tmpdir) / "model.onnx")
        export_kwargs = {
            "opset_version": opset,
            "input_names": input_names,
            "output_names": output_names,
        }
        # torch>=2.6 默认 dynamo=True，M0 需要稳定的 ONNX 导出，显式关闭。
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
        try:
            torch.onnx.export(model, args, onnx_path, **export_kwargs)
        except Exception as exc:
            raise ModelImportError(f"torch.onnx.export 失败: {exc}") from exc

        result = import_onnx_model(onnx_path, options=options)

    module = result.module
    if module_name:
        module.name = module_name
        result.report.module_name = module_name
    module.metadata["frontend"] = "torch"
    module.metadata["torch_export_opset"] = opset
    module.metadata.pop("model_file", None)
    result.report.frontend = "torch"
    result.report.model_file = "<torch model via onnx export>"
    return result
