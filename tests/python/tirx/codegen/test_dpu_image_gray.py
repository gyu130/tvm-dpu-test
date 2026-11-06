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
"""End-to-end DPU image test.

Loads a JPEG photo, compiles an RGB -> grayscale kernel to the DPU target,
and executes the generated 32-bit instruction stream with a small simulator
that follows the encoding rules in src/backend/dpu/codegen/dpu_isa.h.

The simulator plays the role of the DPU runtime (there is no hardware in
CI): argument registers r24/r25 point at the input/output buffers, and the
simulated memory is a sparse byte array.
"""

import os

import numpy as np
import pytest

import tvm
from tvm.script import tirx as T

PIL = pytest.importorskip("PIL.Image", reason="PIL is required to load the test image")

IMAGE_PATH = os.environ.get(
    "DPU_TEST_IMAGE",
    "/home/server/.trae-cn-server/data/User/workspaceStorage/b4ed2653d881b56a549dbef4d45c12f1/"
    "paste-files/bf4e8cc0-9209-4c04-be2b-56bad1f002fb_微信图片_20260515144730_558_5.jpg",
)
OUTPUT_PATH = os.environ.get("DPU_TEST_OUTPUT", "/tmp/dpu_gray_out.jpg")

# Working resolution for the test.  N is a literal in the TVMScript below.
W, H = 128, 96
N = W * H  # 12288 pixels
N3 = N * 3

A_BASE = 0x10000
B_BASE = 0x40000


# ---------------------------------------------------------------------------
# Minimal DPU ISA simulator (see dpu_isa.h for the field layouts).
# ---------------------------------------------------------------------------
def _sext(value, bits):
    sign = 1 << (bits - 1)
    return (value ^ sign) - sign


def _s32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


class DpuSim:
    def __init__(self, words):
        self.words = words
        self.reg = [0] * 32
        self.mem = {}
        self.steps = 0

    def load(self, addr, nbytes, signed):
        value = 0
        for i in range(nbytes):
            value |= self.mem.get(addr + i, 0) << (8 * i)
        if signed and nbytes < 4 and (value >> (8 * nbytes - 1)):
            value -= 1 << (8 * nbytes)
        return value & 0xFFFFFFFF

    def store(self, addr, value, nbytes):
        for i in range(nbytes):
            self.mem[addr + i] = (value >> (8 * i)) & 0xFF

    @staticmethod
    def _alu(op, lhs, rhs):
        lhs &= 0xFFFFFFFF
        rhs &= 0xFFFFFFFF
        if op == 0:
            return (lhs + rhs) & 0xFFFFFFFF  # ADD
        if op == 1:
            return (lhs - rhs) & 0xFFFFFFFF  # SUB
        if op == 2:
            return (lhs * rhs) & 0xFFFFFFFF  # MUL
        if op == 3:
            return lhs & rhs  # AND
        if op == 4:
            return lhs | rhs  # OR
        if op == 5:
            return lhs ^ rhs  # XOR
        if op == 6:
            return (lhs << (rhs & 31)) & 0xFFFFFFFF  # SHL
        if op == 7:
            return lhs >> (rhs & 31)  # SHR (logical)
        if op == 8:
            return _s32(lhs) >> (rhs & 31) & 0xFFFFFFFF  # SAR
        if op == 9:
            return max(_s32(lhs), _s32(rhs)) & 0xFFFFFFFF  # MAX
        if op == 10:
            return min(_s32(lhs), _s32(rhs)) & 0xFFFFFFFF  # MIN
        cmp_ops = {11: lambda a, b: a == b, 12: lambda a, b: a != b, 13: lambda a, b: a < b,
                   14: lambda a, b: a <= b, 15: lambda a, b: a > b, 16: lambda a, b: a >= b}
        return 1 if cmp_ops[op](_s32(lhs), _s32(rhs)) else 0

    def run(self, max_steps=10_000_000):
        pc = 0
        while pc < len(self.words):
            self.steps += 1
            assert self.steps <= max_steps, "DPU simulator exceeded the step budget"
            word = self.words[pc]
            cls = word >> 29
            if cls == 0:  # ALU_I
                op = (word >> 24) & 0x1F
                rd, rs1 = (word >> 19) & 0x1F, (word >> 14) & 0x1F
                self.reg[rd] = self._alu(op, self.reg[rs1], _sext(word & 0x3FFF, 14))
            elif cls in (1, 2):  # ALU_R / ALU_RB
                op = (word >> 24) & 0x1F
                rd, rs1, rs2 = (word >> 19) & 0x1F, (word >> 14) & 0x1F, (word >> 9) & 0x1F
                if cls == 2:  # DIV / MOD / MULH
                    lhs, rhs = _s32(self.reg[rs1]), _s32(self.reg[rs2])
                    assert rhs != 0, "DPU division by zero"
                    if op == 0:
                        self.reg[rd] = int(lhs / rhs) & 0xFFFFFFFF
                    elif op == 1:
                        self.reg[rd] = (lhs - int(lhs / rhs) * rhs) & 0xFFFFFFFF
                    else:
                        self.reg[rd] = ((lhs * rhs) >> 32) & 0xFFFFFFFF
                else:
                    self.reg[rd] = self._alu(op, self.reg[rs1], self.reg[rs2])
            elif cls == 3:  # MOV / MOVI
                rd = (word >> 18) & 0x1F
                if (word >> 28) & 1:
                    self.reg[rd] = _sext(word & 0x3FFFF, 18) & 0xFFFFFFFF
                else:
                    self.reg[rd] = self.reg[(word >> 13) & 0x1F]
            elif cls == 4:  # ISM: only LOADM/STOREM appear in this kernel
                subop = (word >> 24) & 0x1F
                nbytes = (1, 2, 4)[(word >> 5) & 0xF]
                addr = self.reg[(word >> 14) & 0x1F]
                if subop == 0:
                    self.reg[(word >> 19) & 0x1F] = self.load(addr, nbytes, bool((word >> 4) & 1))
                elif subop == 1:
                    self.store(addr, self.reg[(word >> 19) & 0x1F], nbytes)
                else:
                    raise AssertionError(f"unsupported ISM subop {subop} in the simulator")
            elif cls == 5:  # CTRL
                op = (word >> 24) & 0x1F
                lhs, rhs = _s32(self.reg[(word >> 19) & 0x1F]), _s32(self.reg[(word >> 14) & 0x1F])
                taken = {0: True, 1: lhs == rhs, 2: lhs != rhs, 3: lhs < rhs,
                         4: lhs >= rhs, 5: lhs <= rhs, 6: lhs > rhs}[op]
                if taken:
                    pc = pc + 1 + _sext(word & 0x3FFF, 14)
                    self.reg[0] = 0
                    continue
            elif cls == 6:  # END
                break
            self.reg[0] = 0
            pc += 1


# ---------------------------------------------------------------------------
# Kernel under test: fixed-point RGB -> grayscale on the DPU.
# ---------------------------------------------------------------------------
@T.prim_func
def gray_kernel(A: T.Buffer((N3,), "int32"), B: T.Buffer((N,), "int32")):
    for p in range(N):
        B[p] = T.shift_right(A[p * 3] * 76 + A[p * 3 + 1] * 150 + A[p * 3 + 2] * 30, 8)


def _compile_and_extract(source_holder):
    mod = tvm.IRModule({"main": gray_kernel.with_attr("global_symbol", "main")})
    target = tvm.target.Target("dpu")
    compiled = tvm.compile(mod, target=target, tir_pipeline="dpu")
    assert len(compiled.mod.imports) == 1
    source = compiled.mod.imports[0].inspect_source()
    source_holder.append(source)
    words = []
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("0x"):
            words.append(int(line.split()[0], 16))
    assert words, "no DPU instruction words were generated"
    return words


@pytest.mark.skipif(not os.path.exists(IMAGE_PATH), reason="test image is not available")
def test_dpu_image_grayscale():
    from PIL import Image

    rgb = np.asarray(
        Image.open(IMAGE_PATH).convert("RGB").resize((W, H)), dtype=np.uint8
    ).astype(np.int32)

    words = _compile_and_extract(source_holder := [])

    sim = DpuSim(words)
    sim.reg[24] = A_BASE  # A_ptr
    sim.reg[25] = B_BASE  # B_ptr
    flat_in = rgb.reshape(-1)
    for i, v in enumerate(flat_in):
        sim.store(A_BASE + 4 * i, int(v) & 0xFFFFFFFF, 4)
    sim.run()

    flat_out = np.array([_s32(sim.load(B_BASE + 4 * i, 4, True)) for i in range(N)])

    # Bit-exact match against the fixed-point reference.
    r, g, b = flat_in[0::3], flat_in[1::3], flat_in[2::3]
    expected = ((76 * r + 150 * g + 30 * b) >> 8).astype(np.int32)
    assert np.array_equal(flat_out, expected)

    gray_image = Image.fromarray(flat_out.reshape(H, W).astype(np.uint8), mode="L")
    gray_image.save(OUTPUT_PATH)

    print(f"\nDPU executed {sim.steps} instructions over {len(words)} encoded words")
    print(f"grayscale output written to {OUTPUT_PATH}")
    print("---- first 16 generated instructions ----")
    inst_lines = [l for l in source_holder[0].splitlines() if l.strip().startswith("0x")]
    for line in inst_lines[:16]:
        print(line)
    assert flat_out.min() >= 0 and flat_out.max() <= 255


if __name__ == "__main__":
    test_dpu_image_grayscale()
