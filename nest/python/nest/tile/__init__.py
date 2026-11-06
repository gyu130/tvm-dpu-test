"""NestTile 调度 IR、调度模板、校验、内存规划、DPU 降阶与指令编码。"""

from .ir import (
    ArithOp,
    AtomicOp,
    BranchOp,
    Buffer,
    Compute,
    Copy,
    EndOp,
    LabelOp,
    Loop,
    MemManageOp,
    MemOp,
    Program,
    QueueOp,
    RegionSpec,
    Stage,
    Sync,
    TileRef,
)
from .lower import (
    LOWER_RULES,
    LowerContext,
    RegFile,
    list_supported_ops,
    lower_program,
    register_lower_rule,
)
from .encoder import (
    EncoderResult,
    OP_ALU_I, OP_ALU_R, OP_ALU_RB, OP_CTRL, OP_END, OP_ISM, OP_MOV, OP_NOP,
    ALU_I_SUB, ALU_R_SUB, ALU_RB_SUB, MOV_SUB, ISM_SUB, CTRL_SUB,
    encode_alu_i, encode_alu_r, encode_alu_rb, encode_mov,
    encode_ism, encode_ctrl, encode_nop, encode_end,
    encode_arith, encode_items, disassemble,
)
from .memory import Allocation, MemoryPlan, plan_memory
from .multicore import (
    CoreBarrier,
    CoreLoad,
    CoreSyncGraph,
    HaloSpec,
    MultiCoreOptions,
    MultiCoreProgram,
    SplitPlan,
    schedule_multicore,
)
from .pipeline import DoubleBufferEntry, DoubleBufferPlan, plan_double_buffer
from .printer import print_program
from .schedule import ScheduleOptions, schedule_module
from .validate import validate_program

__all__ = [
    # IR
    "ArithOp", "AtomicOp", "BranchOp", "Buffer", "Compute", "Copy",
    "EndOp", "LabelOp", "Loop", "MemManageOp", "MemOp", "Program", "QueueOp",
    "RegionSpec", "Stage", "Sync", "TileRef",
    # Lower
    "LOWER_RULES", "LowerContext", "RegFile",
    "list_supported_ops", "lower_program", "register_lower_rule",
    # Encoder
    "EncoderResult",
    "OP_ALU_I", "OP_ALU_R", "OP_ALU_RB", "OP_CTRL", "OP_END", "OP_ISM", "OP_MOV", "OP_NOP",
    "ALU_I_SUB", "ALU_R_SUB", "ALU_RB_SUB", "MOV_SUB", "ISM_SUB", "CTRL_SUB",
    "encode_alu_i", "encode_alu_r", "encode_alu_rb", "encode_mov",
    "encode_ism", "encode_ctrl", "encode_nop", "encode_end",
    "encode_arith", "encode_items", "disassemble",
    # 其他
    "Allocation", "MemoryPlan", "plan_memory",
    "print_program", "ScheduleOptions", "schedule_module", "validate_program",
    "MultiCoreOptions", "MultiCoreProgram", "SplitPlan", "HaloSpec",
    "CoreBarrier", "CoreSyncGraph", "CoreLoad", "schedule_multicore",
    "DoubleBufferEntry", "DoubleBufferPlan", "plan_double_buffer",
]
