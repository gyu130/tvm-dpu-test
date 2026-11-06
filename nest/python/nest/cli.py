"""NestCompiler 命令行（M0/M1/M2）。

用法：
    python -m nest import model.onnx --print-graph --report report.json
    python -m nest run model.onnx --inputs input.npz --outputs output.npz
    python -m nest quantize model.onnx --save-module qmodule.json
    python -m nest debug model.onnx --report accuracy.json --report-csv acc.csv
    python -m nest compile model.onnx --cores 2 --double-buffer --report tile.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from .errors import NestError
from .frontend import ImportOptions, import_onnx_model
from .ir import Module
from .passes import (
    OptimizeOptions,
    analyze_partition,
    require_fully_npu,
    run_passes,
)
from .quant import (
    AccuracyThresholds,
    CalibrationSample,
    CalibrationSet,
    QuantConfig,
    build_accuracy_report,
    calibrate,
    make_calibration_set,
    quantize_module,
    run_quantized,
    run_reference_tensors,
)
from .runtime import run_module
from .tile import (
    MultiCoreOptions,
    ScheduleOptions,
    print_program,
    schedule_multicore,
)


def _parse_shapes(items: list[str] | None) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--input-shape 需要 NAME=SHAPE 形式: {item}")
        name, raw = item.split("=", 1)
        dims = tuple(int(dim) for dim in raw.split(",") if dim != "")
        if not name or not dims:
            raise ValueError(f"非法 --input-shape: {item}")
        shapes[name] = dims
    return shapes


def _load_custom_ops(target: str) -> None:
    """加载客户自定义算子模块；模块内需提供 register() 或直接执行注册。"""
    path = Path(target)
    if path.exists():
        spec = importlib.util.spec_from_file_location("nest_custom_ops", str(path))
        if spec is None or spec.loader is None:
            raise ValueError(f"无法加载自定义算子模块: {target}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    hook = getattr(module, "register", None)
    if callable(hook):
        hook()


def _build_options(args) -> ImportOptions:
    return ImportOptions(
        fallback_policy=args.fallback,
        input_shapes=_parse_shapes(args.input_shape),
        module_name=args.module_name,
    )


def _import(args):
    if args.custom_ops:
        _load_custom_ops(args.custom_ops)
    result = import_onnx_model(args.model, options=_build_options(args))
    print(result.report.summary_text())
    if args.print_graph:
        print()
        print(result.module)
    if args.report:
        result.report.save(args.report)
        print(f"\n导入报告已写入: {args.report}")
    if args.save_module:
        result.module.save(args.save_module)
        print(f"NestGraph 模块已写入: {args.save_module}")
    return 0


def _load_inputs(path: str | None, module: Module) -> dict[str, np.ndarray]:
    if path is None:
        return {
            value.name: np.zeros(value.type.shape, dtype=value.type.dtype)
            for value in module.graph.inputs
        }
    payload = np.load(path)
    return {name: np.asarray(payload[name]) for name in payload.files}


def _run(args):
    if args.custom_ops:
        _load_custom_ops(args.custom_ops)
    result = import_onnx_model(args.model, options=_build_options(args))
    print(result.report.summary_text())
    inputs = _load_inputs(args.inputs, result.module)
    tensor_sink = {} if args.dump_tensors else None
    outputs = run_module(result.module, inputs, tensor_sink=tensor_sink)
    for name, value in outputs.items():
        print(f"输出 {name}: shape={value.shape}, dtype={value.dtype}")
    if args.outputs:
        np.savez(args.outputs, **outputs)
        print(f"输出已写入: {args.outputs}")
    if args.report:
        result.report.save(args.report)
        print(f"导入报告已写入: {args.report}")
    if args.dump_tensors and args.dump_dir:
        outdir = Path(args.dump_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, value in tensor_sink.items():
            np.save(outdir / f"{name}.npy", value)
        print(f"中间张量已写入: {outdir}")
    return 0


def _build_quant_config(args) -> QuantConfig:
    per_op = {}
    if getattr(args, "per_op", None):
        per_op = json.loads(args.per_op)
        if not isinstance(per_op, dict):
            raise ValueError("--per-op 必须是 JSON 对象")
    return QuantConfig(
        activation_dtype=args.activation_dtype,
        weight_dtype=args.weight_dtype,
        accum_bits=args.accum_bits,
        activation_granularity=args.activation_granularity,
        weight_granularity=args.weight_granularity,
        activation_scheme=args.activation_scheme,
        weight_scheme=args.weight_scheme,
        rounding=args.rounding,
        saturation=not args.no_saturation,
        calibration=args.calibration_algo,
        percentile=args.percentile,
        block_size=args.block_size,
        bias_correction=not args.no_bias_correction,
        calibration_samples=args.calibration_samples,
        calibration_strategy=args.calibration_strategy,
        seed=args.seed,
        per_op=per_op,
        float_ops=frozenset(args.float_ops or []),
    )


def _load_calibration(args, module: Module, config: QuantConfig) -> CalibrationSet:
    if not args.calibration_npz:
        return make_calibration_set(
            module,
            config,
            seed=config.seed,
            count=config.calibration_samples,
            strategy=config.calibration_strategy,
        )
    payload = np.load(args.calibration_npz)
    expected = [value.name for value in module.graph.inputs]
    missing = [name for name in expected if name not in payload.files]
    if missing:
        raise ValueError(f"校准 npz 缺少输入: {missing}")
    extra = [name for name in payload.files if name not in expected]
    if extra:
        raise ValueError(f"校准 npz 包含多余输入: {extra}")
    inputs = {name: np.asarray(payload[name]) for name in expected}
    return CalibrationSet(
        (CalibrationSample(inputs),), seed=config.seed, strategy="file"
    )


def _summary_of_quantization(result) -> str:
    lines = [
        f"量化配置哈希: {result.config_hash}",
        f"校准集哈希: {result.calibration_hash}",
        f"混合精度回退算子: {sorted(result.float_ops)}",
        f"量化张量数: {sum(1 for d in result.decisions if d.kind != 'float')}",
    ]
    return "\n".join(lines)


def _run_quantize(args):
    if args.custom_ops:
        _load_custom_ops(args.custom_ops)
    imported = import_onnx_model(args.model, options=_build_options(args))
    print(imported.report.summary_text())
    config = _build_quant_config(args)
    calibration_set = _load_calibration(args, imported.module, config)
    calibration = calibrate(imported.module, calibration_set, config)
    result = quantize_module(imported.module, calibration, config)
    print()
    print(_summary_of_quantization(result))
    if args.print_quant_params:
        print()
        for item in result.quant_params():
            if item["mode"] != "int":
                continue
            print(
                f"{item['tensor']}: {item['dtype']} axis={item['axis']} "
                f"scale={item['scale']} zero_point={item['zero_point']}"
            )
    if args.quant_report:
        result.save(args.quant_report)
        print(f"\n量化报告已写入: {args.quant_report}")
    if args.save_module:
        result.module.save(args.save_module)
        print(f"量化模块已写入: {args.save_module}")
    return 0


def _run_debug(args):
    if args.custom_ops:
        _load_custom_ops(args.custom_ops)
    imported = import_onnx_model(args.model, options=_build_options(args))
    config = _build_quant_config(args)
    calibration_set = _load_calibration(args, imported.module, config)
    calibration = calibrate(imported.module, calibration_set, config)
    quantized = quantize_module(imported.module, calibration, config)
    sample = calibration_set.samples[0].inputs
    reference = run_reference_tensors(imported.module, sample)
    simulated = run_quantized(quantized.module, sample)
    thresholds = AccuracyThresholds(
        min_cosine=args.min_cosine, min_sqnr_db=args.min_sqnr
    )
    reproduction = (
        f"python -m nest debug {args.model} "
        f"--calibration-samples {config.calibration_samples} "
        f"--seed {config.seed} --calibration-algo {config.calibration}"
    )
    report = build_accuracy_report(
        quantized.module,
        reference,
        simulated.tensors,
        config_hash=quantized.config_hash,
        calibration_hash=calibration.calibration_hash,
        seed=config.seed,
        thresholds=thresholds,
        reproduction=reproduction,
        source=args.model,
    )
    print(report.summary_text())
    if args.accuracy_report:
        report.save_json(args.accuracy_report)
        print(f"\n精度报告已写入: {args.accuracy_report}")
    if args.accuracy_csv:
        report.save_csv(args.accuracy_csv)
        print(f"精度 CSV 已写入: {args.accuracy_csv}")
    if args.dump_tensors and args.dump_dir:
        outdir = Path(args.dump_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, value in simulated.tensors.items():
            np.save(outdir / f"{name}.npy", value)
        print(f"QDQ 张量已写入: {outdir}")
    if args.fail_on_divergence and report.first_divergent is not None:
        print(f"错误: 首个发散层 {report.first_divergent}", file=sys.stderr)
        return 1
    return 0


def _run_compile(args):
    if args.custom_ops:
        _load_custom_ops(args.custom_ops)
    imported = import_onnx_model(args.model, options=_build_options(args))
    print(imported.report.summary_text())
    module, stats = run_passes(imported.module, OptimizeOptions())
    partition = analyze_partition(module)
    require_fully_npu(partition)
    if args.print_graph:
        print()
        print(module)
    schedule_options = ScheduleOptions(
        sram_bytes=args.sram_bytes,
        alignment=args.alignment,
        conv_oc_tile=args.conv_oc_tile,
        gemm_n_tile=args.gemm_n_tile,
    )
    result = schedule_multicore(
        module,
        MultiCoreOptions(
            cores=args.cores,
            split_dim=args.split_dim,
            schedule_options=schedule_options,
            double_buffer=args.double_buffer,
        ),
    )
    result.validate()
    print()
    print(result.summary_text())
    if args.print_program:
        for program in result.programs:
            print()
            print(print_program(program))
    if args.tile_report:
        payload = {
            "passes": [item.to_dict() for item in stats],
            "multicore": result.to_dict(),
        }
        with open(args.tile_report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nTile 报告已写入: {args.tile_report}")
    if args.save_module:
        module.save(args.save_module)
        print(f"优化后模块已写入: {args.save_module}")
    return 0


def _add_quant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--activation-dtype", default="uint8",
                        choices=("uint8", "int8", "int16", "uint16"))
    parser.add_argument("--weight-dtype", default="int8",
                        choices=("int8", "int16"))
    parser.add_argument("--accum-bits", type=int, default=32)
    parser.add_argument("--activation-granularity", default="per_tensor",
                        choices=("per_tensor", "per_channel", "per_block"))
    parser.add_argument("--weight-granularity", default="per_channel",
                        choices=("per_tensor", "per_channel", "per_block"))
    parser.add_argument("--activation-scheme", default="asymmetric",
                        choices=("symmetric", "asymmetric"))
    parser.add_argument("--weight-scheme", default="symmetric",
                        choices=("symmetric", "asymmetric"))
    parser.add_argument("--rounding", default="nearest_even",
                        choices=("nearest_even", "nearest_away", "truncate"))
    parser.add_argument("--no-saturation", action="store_true")
    parser.add_argument("--no-bias-correction", action="store_true")
    parser.add_argument("--calibration-algo", default="minmax",
                        choices=("minmax", "percentile", "kl", "mse"))
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--calibration-strategy", default="normal",
                        choices=("normal", "uniform"))
    parser.add_argument("--calibration-npz", default=None,
                        help="用单个 .npz 校准样本替代随机生成")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--float-ops", action="append", default=None,
                        help="显式保持浮点的 canonical 算子名，可重复")
    parser.add_argument("--per-op", default=None,
                        help="per-op 覆盖的 JSON 对象")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nest",
        description=(
            "NestCompiler M0/M1/M2：ONNX 导入、PTQ 量化与精度 Debug、"
            "多核 NestTile 调度"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("model", help="ONNX 模型路径")
    common.add_argument(
        "--fallback",
        choices=("mark", "error"),
        default="mark",
        help="无法映射算子的策略：mark 生成 Fallback 节点，error 直接报错",
    )
    common.add_argument(
        "--input-shape",
        action="append",
        metavar="NAME=N,C,H,W",
        help="静态化指定输入，可重复",
    )
    common.add_argument("--module-name", default=None, help="覆盖模块名")
    common.add_argument("--custom-ops", default=None,
                        help="客户自定义算子模块（.py 路径或可导入模块名）")
    common.add_argument("--report", default=None, help="导入报告输出 JSON 路径")
    common.add_argument("--save-module", default=None, help="NestGraph 模块序列化路径")

    p_import = sub.add_parser("import", parents=[common], help="导入模型并输出报告")
    p_import.add_argument("--print-graph", action="store_true", help="打印 NestGraph")
    p_import.set_defaults(func=_import)

    p_run = sub.add_parser("run", parents=[common], help="导入并执行参考实现")
    p_run.add_argument("--inputs", default=None, help="输入 .npz；缺省为全零")
    p_run.add_argument("--outputs", default=None, help="输出 .npz 路径")
    p_run.add_argument("--dump-tensors", action="store_true",
                       help="dump 所有中间张量")
    p_run.add_argument("--dump-dir", default=None, help="中间张量输出目录")
    p_run.set_defaults(func=_run)

    p_quant = sub.add_parser(
        "quantize",
        parents=[common],
        help="PTQ 量化：校准 + 标注 + QDQ 仿真参数",
    )
    _add_quant_args(p_quant)
    p_quant.add_argument("--print-quant-params", action="store_true",
                         help="打印整数张量的 scale/zero_point")
    p_quant.add_argument("--quant-report", default=None,
                         help="量化报告（含 scale/zero_point）JSON 路径")
    p_quant.set_defaults(func=_run_quantize)

    p_debug = sub.add_parser(
        "debug",
        parents=[common],
        help="逐层精度 Debug：QDQ 仿真、cosine/SQNR 与首个发散层",
    )
    _add_quant_args(p_debug)
    p_debug.add_argument("--min-cosine", type=float, default=0.99,
                         help="单层 cosine 合格阈值")
    p_debug.add_argument("--min-sqnr", type=float, default=20.0,
                         help="单层 SQNR(dB) 合格阈值")
    p_debug.add_argument("--accuracy-report", default=None,
                         help="精度报告 JSON 路径")
    p_debug.add_argument("--accuracy-csv", default=None,
                         help="逐层精度 CSV 路径")
    p_debug.add_argument("--fail-on-divergence", action="store_true",
                         help="存在首个发散层时以非零码退出")
    p_debug.add_argument("--dump-tensors", action="store_true",
                         help="dump QDQ 仿真所有张量")
    p_debug.add_argument("--dump-dir", default=None,
                         help="QDQ 张量输出目录")
    p_debug.set_defaults(func=_run_debug)

    p_compile = sub.add_parser(
        "compile",
        parents=[common],
        help="多核 NestTile 编译：切分、同步、内存规划与双缓冲",
    )
    p_compile.add_argument("--cores", type=int, default=2, help="核数")
    p_compile.add_argument("--split-dim", default="auto",
                           choices=("auto", "batch", "oc", "n", "h", "w"),
                           help="强制切分轴；auto 按模板自动选择")
    p_compile.add_argument("--double-buffer", action="store_true",
                           help="生成 DMA 双缓冲计划")
    p_compile.add_argument("--sram-bytes", type=int, default=16 * 1024 * 1024)
    p_compile.add_argument("--alignment", type=int, default=64)
    p_compile.add_argument("--conv-oc-tile", type=int, default=64)
    p_compile.add_argument("--gemm-n-tile", type=int, default=64)
    p_compile.add_argument("--print-graph", action="store_true",
                           help="打印优化后的 NestGraph")
    p_compile.add_argument("--print-program", action="store_true",
                           help="打印每核 NestTile 程序")
    p_compile.add_argument("--tile-report", default=None,
                           help="Tile/多核负载报告 JSON 路径")
    p_compile.set_defaults(func=_run_compile)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (NestError, ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
