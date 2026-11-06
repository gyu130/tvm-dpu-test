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

import re

import tvm
import tvm.testing
from tvm.script import tirx as T


def _device_source(func, name):
    mod = tvm.IRModule({name: func.with_attr("global_symbol", name)})
    target = tvm.target.Target("dpu")
    compiled = tvm.compile(mod, target=target, tir_pipeline="dpu")
    assert len(compiled.mod.imports) == 1
    return compiled.mod.imports[0].inspect_source()


def _assert_labels_resolve(source):
    defined = set(re.findall(r"^L(\d+):", source, flags=re.MULTILINE))
    referenced = set(re.findall(r"L(\d+)", source)) - defined
    assert not referenced, f"undefined labels referenced: {sorted(referenced)}"


def test_dpu_target_kind():
    target = tvm.target.Target("dpu")
    assert target.kind.name == "dpu"
    assert target.attrs["num-cores"] == 2
    assert target.attrs["sram-size-bytes"] == 16 * 1024 * 1024


def test_dpu_split_host_device():
    @T.prim_func
    def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
        for i in range(16):
            B[i] = A[i] + 1

    mod = tvm.IRModule({"main": main.with_attr("global_symbol", "main")})
    target = tvm.target.Target("dpu")
    compiled = tvm.compile(mod, target=target, tir_pipeline="dpu")

    assert len(compiled.mod.imports) == 1
    device_source = compiled.mod.imports[0].inspect_source()
    assert ".dpu_func main_kernel" in device_source


def test_dpu_elementwise_codegen():
    @T.prim_func
    def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
        for i in range(16):
            B[i] = A[i] + 1

    source = _device_source(main, "main")
    assert ".dpu_func main_kernel" in source
    assert "; pattern = elementwise" in source
    assert "; args: A_ptr=r24, B_ptr=r25" in source

    for mnemonic in ["MOVI", "SHL", "ADD", "LOADM", "STOREM", "BLT", "END"]:
        assert mnemonic in source, f"missing {mnemonic} in generated DPU code"

    words = [line.split(";")[0].strip() for line in source.splitlines() if line.startswith("  0x")]
    assert words, "no encoded instruction words were emitted"
    assert words[-1] == "0xC0000000", "the stream must terminate with END"
    for word in words:
        assert re.fullmatch(r"0x[0-9A-F]{8}", word)

    _assert_labels_resolve(source)


def test_dpu_gemm_pattern_codegen():
    @T.prim_func
    def main(
        A: T.Buffer((4, 4), "int32"),
        B: T.Buffer((4, 4), "int32"),
        C: T.Buffer((4, 4), "int32"),
    ):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

    source = _device_source(main, "main")
    # The uniform ISA keeps the instruction stream generic, but the pattern
    # annotation records the recognized gemm loop nest.
    assert "; pattern = gemm" in source
    assert "MUL " in source
    # One compare-and-branch per loop nest level.
    assert source.count("BLT ") == 3
    _assert_labels_resolve(source)


if __name__ == "__main__":
    tvm.testing.main()
