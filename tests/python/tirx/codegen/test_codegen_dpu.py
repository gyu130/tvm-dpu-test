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

import tvm
import tvm.testing
from tvm.script import tirx as T


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


if __name__ == "__main__":
    tvm.testing.main()
