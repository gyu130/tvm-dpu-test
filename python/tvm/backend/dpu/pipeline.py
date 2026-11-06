# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""DPU TIRx pipeline entrypoints."""

import tvm
from tvm import tirx
from tvm.tirx.compilation_pipeline import finalize_host_passes


def dpu_pipeline():
    """The DPU pipeline used in tvm.tirx.build."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """Lower TIRx for the DPU backend."""
        config = tvm.transform.PassContext.current().config
        passes = [
            tirx.transform.LowerTIRx(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tirx.transform.StmtSimplify(),
            tirx.transform.LowerTIRxOpaque(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            tirx.transform.VectorizeLoop(not bool(config.get("tir.disable_vectorize", False))),
            tirx.transform.UnrollLoop(),
            tirx.transform.StmtSimplify(),
        ]
        if not bool(config.get("tir.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                tirx.transform.FP8ComputeLegalize(),
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
                tirx.transform.SplitHostDevice(),
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
            ]
        )
        return tvm.ir.transform.Sequential(passes)(mod)

    return _pipeline, finalize_host_passes, finalize_device_passes_dpu


def finalize_device_passes_dpu():  # pylint: disable=unused-argument
    """The finalization passes for the DPU backend."""
    return tvm.ir.transform.Sequential(
        [tirx.transform.LowerIntrin(), tirx.transform.StmtSimplify()]
    )


__all__ = ["dpu_pipeline", "finalize_device_passes_dpu"]
