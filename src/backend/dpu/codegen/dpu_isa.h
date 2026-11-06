/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file backend/dpu/codegen/dpu_isa.h
 * \brief DPU instruction encoding used by the TVM DPU backend.
 *
 * Every DPU instruction is a fixed 32-bit word.  Bits [31:29] hold the
 * instruction class; each class has the following field layout:
 *
 *   ALU_I   [31:29]=0  [28:24]=op  [23:19]=rd  [18:14]=rs1  [13:0]=imm14 (signed)
 *   ALU_R   [31:29]=1  [28:24]=op  [23:19]=rd  [18:14]=rs1  [13:9]=rs2
 *   ALU_RB  [31:29]=2  [28:24]=op  [23:19]=rd  [18:14]=rs1  [13:9]=rs2
 *   MOV     [31:29]=3  [28]=0     [22:18]=rd  [17:13]=rs
 *   MOVI    [31:29]=3  [28]=1     [22:18]=rd  [17:0]=imm18 (signed)
 *   ISM     [31:29]=4  [28:24]=subop [23:19]=rd [18:14]=rs1 [13:9]=rs2
 *                      [8:5]=width  [4]=signed
 *   CTRL    [31:29]=5  [28:24]=op  [23:19]=rs1  [18:14]=rs2  [13:0]=offset14
 *   END     [31:29]=6  all other bits are zero
 *
 * Register conventions:
 *   r0       hard-wired zero
 *   r1-r23   temporaries / local values
 *   r24-r31  kernel arguments (a0-a7); buffer pointers are pre-loaded here
 *            by the runtime before the kernel starts.
 *
 * ALU_R/ALU_RB operands must be registers.  Use ALU_I for immediate forms.
 * Branch offsets are signed multiples of four bytes, relative to the
 * instruction following the branch.
 */
#ifndef TVM_BACKEND_DPU_CODEGEN_DPU_ISA_H_
#define TVM_BACKEND_DPU_CODEGEN_DPU_ISA_H_

#include <tvm/runtime/logging.h>

#include <cstdint>
#include <string>

namespace tvm {
namespace codegen {
namespace dpu {

constexpr uint32_t kZeroReg = 0;
constexpr uint32_t kFirstTempReg = 1;
constexpr uint32_t kLastTempReg = 23;
constexpr uint32_t kFirstArgReg = 24;
constexpr uint32_t kLastArgReg = 31;

constexpr uint32_t kClassShift = 29;

enum class InstClass : uint32_t {
  kAluI = 0,
  kAluR = 1,
  kAluRB = 2,
  kMov = 3,
  kIsm = 4,
  kCtrl = 5,
  kEnd = 6,
};

/*! \brief ALU opcodes, shared by the ALU_I and ALU_R formats. */
enum AluOp : uint32_t {
  kAdd = 0,
  kSub = 1,
  kMul = 2,
  kAnd = 3,
  kOr = 4,
  kXor = 5,
  kShl = 6,
  kShr = 7,
  kSar = 8,
  kMax = 9,
  kMin = 10,
  kEq = 11,
  kNe = 12,
  kLt = 13,
  kLe = 14,
  kGt = 15,
  kGe = 16,
};

/*! \brief ALU_RB opcodes.  The second source must be a register. */
enum AluRbOp : uint32_t {
  kDiv = 0,
  kMod = 1,
  kMulh = 2,
};

/*! \brief ISM (memory and synchronization) sub-opcodes. */
enum IsmOp : uint32_t {
  kLoad = 0,
  kStore = 1,
  kIsb = 2,
  kIsbc = 3,
  kAtomAddTle = 4,
  kAtomAddTbe = 5,
  kPushQ = 6,
  kPopQ = 7,
  kMalcSmb = 8,
  kFreeSmb = 9,
};

/*! \brief CTRL (branch) opcodes. */
enum CtrlOp : uint32_t {
  kBranch = 0,
  kBeq = 1,
  kBne = 2,
  kBlt = 3,
  kBge = 4,
  kBle = 5,
  kBgt = 6,
};

/*! \brief Memory access width encoding used by LOADM/STOREM. */
enum MemWidth : uint32_t {
  kWidthByte = 0,
  kWidthHalf = 1,
  kWidthWord = 2,
};

inline uint32_t EncodeAluI(uint32_t op, uint32_t rd, uint32_t rs1, int64_t imm) {
  TVM_FFI_ICHECK(op <= 16) << "invalid ALU_I op " << op;
  TVM_FFI_ICHECK(rd <= 31 && rs1 <= 31) << "invalid register in ALU_I";
  TVM_FFI_ICHECK(imm >= -(1 << 13) && imm < (1 << 13)) << "ALU_I immediate out of range: " << imm;
  return (static_cast<uint32_t>(InstClass::kAluI) << kClassShift) | (op << 24) | (rd << 19) |
         (rs1 << 14) | (static_cast<uint32_t>(imm) & 0x3FFFu);
}

inline uint32_t EncodeAluR(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2) {
  TVM_FFI_ICHECK(op <= 16) << "invalid ALU_R op " << op;
  TVM_FFI_ICHECK(rd <= 31 && rs1 <= 31 && rs2 <= 31) << "invalid register in ALU_R";
  return (static_cast<uint32_t>(InstClass::kAluR) << kClassShift) | (op << 24) | (rd << 19) |
         (rs1 << 14) | (rs2 << 9);
}

inline uint32_t EncodeAluRB(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2) {
  TVM_FFI_ICHECK(op <= 2) << "invalid ALU_RB op " << op;
  TVM_FFI_ICHECK(rd <= 31 && rs1 <= 31 && rs2 <= 31) << "invalid register in ALU_RB";
  return (static_cast<uint32_t>(InstClass::kAluRB) << kClassShift) | (op << 24) | (rd << 19) |
         (rs1 << 14) | (rs2 << 9);
}

inline uint32_t EncodeMov(uint32_t rd, uint32_t rs) {
  TVM_FFI_ICHECK(rd <= 31 && rs <= 31) << "invalid register in MOV";
  return (static_cast<uint32_t>(InstClass::kMov) << kClassShift) | (rd << 18) | (rs << 13);
}

inline uint32_t EncodeMovi(uint32_t rd, int64_t imm) {
  TVM_FFI_ICHECK(rd <= 31) << "invalid register in MOVI";
  TVM_FFI_ICHECK(imm >= -(1 << 17) && imm < (1 << 17)) << "MOVI immediate out of range: " << imm;
  return (static_cast<uint32_t>(InstClass::kMov) << kClassShift) | (1u << 28) | (rd << 18) |
         (static_cast<uint32_t>(imm) & 0x3FFFFu);
}

inline uint32_t EncodeLoad(uint32_t rd, uint32_t addr, uint32_t width, bool is_signed) {
  TVM_FFI_ICHECK(rd <= 31 && addr <= 31) << "invalid register in LOADM";
  return (static_cast<uint32_t>(InstClass::kIsm) << kClassShift) | (kLoad << 24) | (rd << 19) |
         (addr << 14) | (width << 5) | (is_signed ? (1u << 4) : 0u);
}

inline uint32_t EncodeStore(uint32_t addr, uint32_t value, uint32_t width) {
  TVM_FFI_ICHECK(addr <= 31 && value <= 31) << "invalid register in STOREM";
  return (static_cast<uint32_t>(InstClass::kIsm) << kClassShift) | (kStore << 24) |
         (value << 19) | (addr << 14) | (width << 5);
}

inline uint32_t EncodeIsm(uint32_t subop) {
  TVM_FFI_ICHECK(subop <= 9) << "invalid ISM subop " << subop;
  return (static_cast<uint32_t>(InstClass::kIsm) << kClassShift) | (subop << 24);
}

inline uint32_t EncodeCtrl(uint32_t op, uint32_t rs1, uint32_t rs2, int32_t offset_words) {
  TVM_FFI_ICHECK(op <= 6) << "invalid CTRL op " << op;
  TVM_FFI_ICHECK(rs1 <= 31 && rs2 <= 31) << "invalid register in CTRL";
  TVM_FFI_ICHECK(offset_words >= -(1 << 13) && offset_words < (1 << 13))
      << "branch offset out of range: " << offset_words;
  return (static_cast<uint32_t>(InstClass::kCtrl) << kClassShift) | (op << 24) | (rs1 << 19) |
         (rs2 << 14) | (static_cast<uint32_t>(offset_words) & 0x3FFFu);
}

inline uint32_t EncodeEnd() { return static_cast<uint32_t>(InstClass::kEnd) << kClassShift; }

inline std::string RegName(uint32_t reg) { return "r" + std::to_string(reg); }

inline const char* AluName(uint32_t op) {
  static const char* kNames[] = {"ADD", "SUB", "MUL", "AND", "OR",  "XOR",
                                 "SHL", "SHR", "SAR", "MAX", "MIN", "EQ",
                                 "NE",  "LT",  "LE",  "GT",  "GE"};
  return op <= 16 ? kNames[op] : "?";
}

inline const char* AluRbName(uint32_t op) {
  static const char* kNames[] = {"DIV", "MOD", "MULH"};
  return op <= 2 ? kNames[op] : "?";
}

inline const char* CtrlName(uint32_t op) {
  static const char* kNames[] = {"B", "BEQ", "BNE", "BLT", "BGE", "BLE", "BGT"};
  return op <= 6 ? kNames[op] : "?";
}

}  // namespace dpu
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_BACKEND_DPU_CODEGEN_DPU_ISA_H_
