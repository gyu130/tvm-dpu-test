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
"""P4 前端：P4 程序 -> TIR -> DPU 指令 全链路测试。"""
import sys
from pathlib import Path

import pytest

# 确保 nest 包可导入
_NEST_PY = Path(__file__).resolve().parents[5] / "nest" / "python"
if str(_NEST_PY) not in sys.path:
    sys.path.insert(0, str(_NEST_PY))

import tvm
from nest.frontend import (
    Control,
    Deparser,
    Entry,
    Field,
    Header,
    IncrementCounter,
    P4Program,
    Parser,
    SetField,
    State,
    Table,
    build_ipv4_ttl_decrement,
)


def test_field_offset_layout():
    """Header 字段偏移按字节序计算。"""
    eth = Header("ethernet", [Field("dst", 48), Field("src", 48), Field("ethertype", 16)])
    assert eth.byte_width == 14
    assert eth.field_offset("ethertype") == 12


def test_to_tir_source_has_three_stages():
    """生成的 TVMScript 包含 parser / match-action / deparser 三阶段。"""
    prog = build_ipv4_ttl_decrement(pkt_len=64)
    src = prog.to_tir_source()
    # parser: 从 pkt 拷贝到 hdr
    assert "hdr[i] = pkt[i]" in src
    # match-action: ethertype == 0x0800
    assert "== 2048" in src
    # deparser: 从 hdr 写回 pkt
    assert "pkt[i] = hdr[i]" in src


def test_p4_to_tir_parses():
    """P4 程序可解析为合法的 TIR PrimFunc。"""
    prog = build_ipv4_ttl_decrement(pkt_len=64)
    func = prog.to_tir()
    assert func is not None
    # 函数有两个参数：pkt 和 counters
    assert len(func.params) == 2


def test_p4_to_dpu_generates_instructions():
    """P4 -> TIR -> DPU 全链路生成合法指令流。"""
    prog = build_ipv4_ttl_decrement(pkt_len=64)
    func = prog.to_tir()
    mod = tvm.IRModule({"main": func.with_attr("global_symbol", "main")})
    target = tvm.target.Target("dpu")
    compiled = tvm.compile(mod, target=target, tir_pipeline="dpu")

    sources = [imp.inspect_source() for imp in compiled.mod.imports]
    assert any(".dpu_func" in s for s in sources), "缺少 .dpu_func 声明"
    assert any("END" in s for s in sources), "缺少 END 指令"
    # deparser 必须产生 STOREM
    assert any("STOREM" in s for s in sources), "缺少 deparser 的 STOREM 指令"
    # match-action 必须产生条件分支
    assert any("BNE" in s or "BEQ" in s for s in sources), "缺少 match-action 分支"


def test_custom_vlan_passthrough():
    """自定义程序：识别 VLAN 帧（ethertype=0x8100）并累加计数，不改写。"""
    eth = Header("ethernet", [Field("dst", 48), Field("src", 48), Field("ethertype", 16)])
    parser = Parser([State("start", extract=eth, default="accept")])
    vlan_action = [IncrementCounter("vlan_counter", 1)]
    table = Table(
        name="vlan_table",
        key=eth.ethertype,
        entries=[Entry(0x8100, vlan_action)],
        default=None,
    )
    prog = P4Program(
        headers=[eth],
        parser=parser,
        control=Control([table]),
        deparser=Deparser([eth]),
        pkt_len=64,
        counters={"vlan_counter": 0},
    )
    func = prog.to_tir()
    src = prog.to_tir_source()
    assert "== 33024" in src  # 0x8100
    assert func is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
