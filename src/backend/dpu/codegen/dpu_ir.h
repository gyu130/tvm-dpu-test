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
 * \file backend/dpu/codegen/dpu_ir.h
 * \brief DPU intermediate representation.
 *
 * The DPU compilation flow is split into two stages:
 *
 *   1. Lowering   TIR   ->  DpuProgram  (a list of DpuInst objects)
 *   2. Encoding   DpuProgram -> std::vector<uint32_t>  (32-bit instruction words)
 *
 * DpuInst holds the semantic operands of an instruction (opcode, registers,
 * immediate, branch target, ...).  The bit-level packing lives entirely in
 * dpu_isa.h and is applied by DpuProgram::Encode().  This separation mirrors
 * the project's "lower_program -> encode_items" pipeline and keeps the
 * lowering rules independent of the encoding layout.
 */
#ifndef TVM_BACKEND_DPU_CODEGEN_DPU_IR_H_
#define TVM_BACKEND_DPU_CODEGEN_DPU_IR_H_

#include <tvm/runtime/logging.h>

#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "dpu_isa.h"

namespace tvm {
namespace codegen {
namespace dpu {

/*! \brief DPU instruction kinds.  One per encoding class (plus sync ops). */
enum class DpuInstKind : uint8_t {
  kAluI,    // ALU with immediate:     op, rd, rs1, imm14
  kAluR,    // ALU register-register:  op, rd, rs1, rs2
  kAluRB,   // ALU_RB (DIV/MOD/MULH):  op, rd, rs1, rs2
  kMov,     // register move:          rd, rs1
  kMovi,    // immediate move:         rd, imm18
  kLoad,    // memory load:            rd, rs1(addr), width, signed_load
  kStore,   // memory store:           rs1(addr), rs2(value), width
  kBranch,  // control branch:         op, rs1, rs2, label
  kSync,    // synchronization:        op (ISM subop)
  kEnd,     // kernel terminator
};

/*!
 * \brief A single DPU instruction in its decoded (pre-encoding) form.
 *
 * Not every field is meaningful for every kind; the unused fields are left
 * at their default values.  The `text` field carries the human-readable
 * assembly mnemonic used when formatting the program listing.
 */
struct DpuInst {
  DpuInstKind kind = DpuInstKind::kEnd;
  uint32_t op = 0;       // ALU op / ISM subop / CTRL op
  uint32_t rd = 0;
  uint32_t rs1 = 0;
  uint32_t rs2 = 0;
  int32_t imm = 0;       // ALU_I imm14 / MOVI imm18
  uint32_t width = kWidthWord;
  bool signed_load = false;
  int label = -1;        // branch target label id (unresolved)
  std::string text;

  static DpuInst AluI(uint32_t op, uint32_t rd, uint32_t rs1, int32_t imm, std::string text) {
    return DpuInst{DpuInstKind::kAluI, op, rd, rs1, 0, imm, kWidthWord, false, -1, std::move(text)};
  }
  static DpuInst AluR(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2, std::string text) {
    return DpuInst{DpuInstKind::kAluR, op, rd, rs1, rs2, 0, kWidthWord, false, -1, std::move(text)};
  }
  static DpuInst AluRB(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2, std::string text) {
    return DpuInst{DpuInstKind::kAluRB, op, rd, rs1, rs2, 0, kWidthWord, false, -1,
                   std::move(text)};
  }
  static DpuInst Mov(uint32_t rd, uint32_t rs, std::string text) {
    return DpuInst{DpuInstKind::kMov, 0, rd, rs, 0, 0, kWidthWord, false, -1, std::move(text)};
  }
  static DpuInst Movi(uint32_t rd, int32_t imm, std::string text) {
    return DpuInst{DpuInstKind::kMovi, 0, rd, 0, 0, imm, kWidthWord, false, -1, std::move(text)};
  }
  static DpuInst Load(uint32_t rd, uint32_t addr, uint32_t width, bool is_signed, std::string text) {
    return DpuInst{DpuInstKind::kLoad, 0, rd, addr, 0, 0, width, is_signed, -1, std::move(text)};
  }
  static DpuInst Store(uint32_t addr, uint32_t value, uint32_t width, std::string text) {
    return DpuInst{DpuInstKind::kStore, 0, 0, addr, value, 0, width, false, -1, std::move(text)};
  }
  static DpuInst Branch(uint32_t op, uint32_t rs1, uint32_t rs2, int label, std::string text) {
    return DpuInst{DpuInstKind::kBranch, op, 0, rs1, rs2, 0, kWidthWord, false, label,
                   std::move(text)};
  }
  static DpuInst Sync(uint32_t subop, std::string text) {
    return DpuInst{DpuInstKind::kSync, subop, 0, 0, 0, 0, kWidthWord, false, -1, std::move(text)};
  }
  static DpuInst End(std::string text) {
    return DpuInst{DpuInstKind::kEnd, 0, 0, 0, 0, 0, kWidthWord, false, -1, std::move(text)};
  }
};

/*!
 * \brief A lowered DPU program: an ordered list of instructions plus the
 *        label-to-instruction-position mapping.
 */
class DpuProgram {
 public:
  void Append(DpuInst inst) { insts_.push_back(std::move(inst)); }

  /*! \brief Record that \p label is anchored at the current program position. */
  void MarkLabel(int label) { label_pos_[label] = insts_.size(); }

  size_t size() const { return insts_.size(); }
  const std::vector<DpuInst>& insts() const { return insts_; }
  const std::map<int, size_t>& label_pos() const { return label_pos_; }

  /*!
   * \brief Encode every instruction into a 32-bit word.
   *
   * Branch offsets are resolved against the label positions recorded during
   * lowering.  An unresolved label aborts via TVM_FFI_ICHECK.
   */
  std::vector<uint32_t> Encode() const {
    std::vector<uint32_t> words;
    words.reserve(insts_.size());
    for (size_t i = 0; i < insts_.size(); ++i) {
      const DpuInst& ins = insts_[i];
      uint32_t word = 0;
      switch (ins.kind) {
        case DpuInstKind::kAluI:
          word = EncodeAluI(ins.op, ins.rd, ins.rs1, ins.imm);
          break;
        case DpuInstKind::kAluR:
          word = EncodeAluR(ins.op, ins.rd, ins.rs1, ins.rs2);
          break;
        case DpuInstKind::kAluRB:
          word = EncodeAluRB(ins.op, ins.rd, ins.rs1, ins.rs2);
          break;
        case DpuInstKind::kMov:
          word = EncodeMov(ins.rd, ins.rs1);
          break;
        case DpuInstKind::kMovi:
          word = EncodeMovi(ins.rd, ins.imm);
          break;
        case DpuInstKind::kLoad:
          word = EncodeLoad(ins.rd, ins.rs1, ins.width, ins.signed_load);
          break;
        case DpuInstKind::kStore:
          word = EncodeStore(ins.rs1, ins.rs2, ins.width);
          break;
        case DpuInstKind::kBranch: {
          auto it = label_pos_.find(ins.label);
          TVM_FFI_ICHECK(it != label_pos_.end())
              << "unresolved DPU branch label L" << ins.label;
          int32_t offset =
              static_cast<int32_t>(it->second) - static_cast<int32_t>(i + 1);
          word = EncodeCtrl(ins.op, ins.rs1, ins.rs2, offset);
          break;
        }
        case DpuInstKind::kSync:
          word = EncodeIsm(ins.op);
          break;
        case DpuInstKind::kEnd:
          word = EncodeEnd();
          break;
      }
      words.push_back(word);
    }
    return words;
  }

  /*!
   * \brief Format the program as an assembly listing with hex words.
   *
   * Produces exactly the same output as the previous monolithic codegen so
   * that existing tests and the Python simulator keep working.
   */
  std::string Format() const {
    std::vector<uint32_t> words = Encode();
    std::ostringstream os;
    for (size_t i = 0; i < insts_.size(); ++i) {
      for (const auto& kv : label_pos_) {
        if (kv.second == i) {
          os << "L" << kv.first << ":\n";
        }
      }
      os << "  0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << words[i]
         << std::dec << std::nouppercase << std::setfill(' ') << "  ; " << insts_[i].text << "\n";
    }
    return os.str();
  }

 private:
  std::vector<DpuInst> insts_;
  std::map<int, size_t> label_pos_;
};

}  // namespace dpu
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_BACKEND_DPU_CODEGEN_DPU_IR_H_
