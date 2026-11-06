"""NestCompiler 错误类型。

所有对外错误都继承 NestError，便于 CLI 统一格式化输出，
并保证错误信息中包含算子名、source span 等可定位信息。
"""

from __future__ import annotations


class NestError(Exception):
    """NestCompiler 基础错误。"""


class GraphValidationError(NestError):
    """NestGraph 不变式校验失败。"""


class RegistryError(NestError):
    """算子注册表使用错误（重复注册、非法注册内容等）。"""


class ModelImportError(NestError):
    """模型导入失败。"""


class ShapeInferenceError(ModelImportError):
    """shape / dtype 无法推导。"""


class UnsupportedOperatorError(ModelImportError):
    """关闭 fallback 时存在无法映射的算子。

    ``unsupported`` 为 (算子描述, source span 字符串) 列表，
    错误信息一次性输出完整清单，避免逐个报错。
    """

    def __init__(self, unsupported: list[tuple[str, str]]):
        self.unsupported = unsupported
        lines = [f"存在 {len(unsupported)} 个无法映射到 NestGraph 的算子："]
        lines += [f"  - {op} @ {span}" for op, span in unsupported]
        super().__init__("\n".join(lines))


class MissingKernelError(NestError):
    """参考执行器缺少算子实现。"""


class MissingScalarImplementationError(MissingKernelError):
    """fallback 节点没有可用的标量实现。"""


class PassError(NestError):
    """图优化 pass 使用或执行失败。"""


class PartitionError(NestError):
    """分区结果无法进入单核调度（存在 fallback 或不支持的节点）。"""


class QuantizationError(NestError):
    """PTQ 量化配置、校准或量化参数非法。"""


class CalibrationError(QuantizationError):
    """校准集或校准过程失败（空样本、NaN/Inf、范围非法）。"""


class ScheduleError(NestError):
    """调度模板无法生成合法的 tile 程序。"""


class TuneError(NestError):
    """AutoTune / Ansor 的配置空间、日志或调优流程使用错误。"""


class TileValidationError(NestError):
    """NestTile 不变式校验失败。"""


class LoweringError(NestError):
    """NestTile -> NestNPU 指令生成失败。"""


class SimulationError(NestError):
    """NestNPU 功能仿真失败。"""
