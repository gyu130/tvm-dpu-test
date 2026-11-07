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
 * \file backend/dpu/codegen/codegen_dpu.cc
 * \brief TIR -> DPU instruction selection for the DPU backend.
 *
 * The DPU has no dedicated conv/gemm/relu instructions.  Every operator is
 * lowered into a sequence of generic load/store, ALU, shift and branch
 * instructions.  This file implements that selection directly on the
 * device-side TIR produced by SplitHostDevice:
 *
 *  - Buffers are addressed through the kernel argument registers (r24-r31)
 *    with MOVI/SHL/ADD byte-offset computations.
 *  - `for` loops become a compare-and-branch loop whose induction variable
 *    lives in a register.
 *  - `if` and comparisons become CTRL branches.
 *  - Local `alloc_buffer` storage is placed in SRAM starting at a fixed base.
 *
 * The generated source contains both a readable assembly listing and the
 * encoded 32-bit instruction words (see dpu_isa.h for the field layouts).
 */
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../target/build_common.h"
#include "../../../target/source/codegen_source_base.h"
#include "dpu_ir.h"
#include "dpu_isa.h"

namespace tvm {
namespace codegen {
namespace dpu {

namespace {

constexpr int64_t kSramBase = 0x10000000;

int64_t GetTargetAttr(const Target& target, const char* key, int64_t default_value) {
  if (auto opt = target->GetAttr<int64_t>(key)) {
    return opt.value();
  }
  return default_value;
}

[[noreturn]] void Unsupported(const std::string& msg) {
  TVM_FFI_ICHECK(false) << "CodeGenDPU: " << msg;
  std::abort();
}

/*! \brief A value produced while evaluating a TIR expression. */
struct Value {
  uint32_t reg = 0;
  /*! \brief Whether the register is a temporary owned by the caller. */
  bool temp = false;
};

struct PatternInfo {
  int max_depth = 0;
  bool has_reduction = false;
  bool has_mul = false;
  bool has_const_offset = false;
};

/*!
 * \brief Classify the access pattern of a device kernel.
 *
 * The DPU instruction set is uniform, so the classification only annotates
 * the generated source.  It lets downstream tools recognize `gemm`/`conv2d`
 * loop nests while the instruction stream remains fully generic.
 */
class PatternAnalyzer {
 public:
  static PatternInfo Analyze(const tirx::PrimFunc& func) {
    PatternInfo info;
    AnalyzeStmt(func->body, 0, &info);
    return info;
  }

 private:
  static bool SameBuffer(const tirx::BufferVar& a, const tirx::BufferVar& b) {
    return a.get() == b.get();
  }

  static void AnalyzeExpr(const Expr& e, PatternInfo* info) {
    if (auto mul = e.as<tirx::MulNode>()) {
      info->has_mul = true;
      AnalyzeExpr(mul->a, info);
      AnalyzeExpr(mul->b, info);
      return;
    }
    if (auto add = e.as<tirx::AddNode>()) {
      AnalyzeExpr(add->a, info);
      AnalyzeExpr(add->b, info);
      return;
    }
    if (auto sub = e.as<tirx::SubNode>()) {
      AnalyzeExpr(sub->a, info);
      AnalyzeExpr(sub->b, info);
      return;
    }
    if (auto load = e.as<tirx::BufferLoadNode>()) {
      for (const PrimExpr& index : load->indices) {
        if (auto add = index.as<tirx::AddNode>()) {
          if (add->a.as<tirx::IntImmNode>() || add->b.as<tirx::IntImmNode>()) {
            info->has_const_offset = true;
          }
        }
      }
      for (const PrimExpr& index : load->indices) {
        AnalyzeExpr(index, info);
      }
    }
  }

  static bool IsReductionUpdate(const tirx::BufferStoreNode* store) {
    if (auto add = store->value.as<tirx::AddNode>()) {
      for (const PrimExpr& operand : {add->a, add->b}) {
        if (auto load = operand.as<tirx::BufferLoadNode>()) {
          if (SameBuffer(load->buffer, store->buffer)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  static void AnalyzeStmt(const tirx::Stmt& stmt, int depth, PatternInfo* info) {
    if (auto seq = stmt.as<tirx::SeqStmtNode>()) {
      for (const tirx::Stmt& s : seq->seq) {
        AnalyzeStmt(s, depth, info);
      }
    } else if (auto for_node = stmt.as<tirx::ForNode>()) {
      info->max_depth = std::max(info->max_depth, depth + 1);
      AnalyzeStmt(for_node->body, depth + 1, info);
    } else if (auto while_node = stmt.as<tirx::WhileNode>()) {
      info->max_depth = std::max(info->max_depth, depth + 1);
      AnalyzeStmt(while_node->body, depth + 1, info);
    } else if (auto if_node = stmt.as<tirx::IfThenElseNode>()) {
      AnalyzeStmt(if_node->then_case, depth, info);
      if (if_node->else_case.defined()) {
        AnalyzeStmt(if_node->else_case.value(), depth, info);
      }
    } else if (auto store = stmt.as<tirx::BufferStoreNode>()) {
      if (IsReductionUpdate(store)) {
        info->has_reduction = true;
      }
      AnalyzeExpr(store->value, info);
      for (const PrimExpr& index : store->indices) {
        AnalyzeExpr(index, info);
      }
    } else if (auto bind = stmt.as<tirx::BindNode>()) {
      AnalyzeExpr(bind->value, info);
    }
  }
};

class DpuCodeGen {
 public:
  DpuCodeGen(const Target& target, const tirx::PrimFunc& func)
      : target_(target), func_(func), alignment_(GetTargetAttr(target, "sram-alignment", 64)) {
    TVM_FFI_ICHECK(func->params.size() <= static_cast<size_t>(kLastArgReg - kFirstArgReg + 1))
        << "DPU kernel supports at most 8 arguments, got " << func->params.size();
    for (size_t i = 0; i < func->params.size(); ++i) {
      param_regs_[func->params[i].get()] = kFirstArgReg + static_cast<uint32_t>(i);
    }
    for (uint32_t r = 0; r <= 31; ++r) {
      reg_used_[r] = false;
    }
    reg_used_[kZeroReg] = true;
    for (uint32_t r = kFirstArgReg; r <= kLastArgReg; ++r) {
      reg_used_[r] = true;
    }
  }

  std::string Generate(const ffi::String& func_name) {
    pattern_ = PatternAnalyzer::Analyze(func_);

    std::ostringstream os;
    os << "; DPU kernel generated by TVM\n";
    os << "; num-cores = " << GetTargetAttr(target_, "num-cores", 2) << "\n";
    os << "; sram-size-bytes = " << GetTargetAttr(target_, "sram-size-bytes", 16 << 20) << "\n";
    os << "; shared-memory-size-bytes = "
       << GetTargetAttr(target_, "shared-memory-size-bytes", 64 << 10) << "\n";
    os << "; pattern = " << PatternName() << "\n";
    os << "; ABI: r0 = zero, r1-r23 = temporaries, r24-r31 = arguments\n";
    os << "; args: ";
    for (size_t i = 0; i < func_->params.size(); ++i) {
      if (i != 0) os << ", ";
      os << func_->params[i]->name << "=" << RegName(kFirstArgReg + static_cast<uint32_t>(i));
    }
    os << "\n";
    os << ".dpu_func " << func_name << "\n";

    end_label_ = NewLabel();
    PushScope();
    VisitStmt(func_->body);
    PopScope();
    MarkLabel(end_label_);
    EmitEnd("END");

    // The DPU IR (DpuInst list) is now complete.  Encode it to 32-bit words
    // and render the assembly listing; branch offsets are resolved here.
    os << program_.Format();
    return os.str();
  }

 private:
  // ------------------------------------------------------------------
  // Emission helpers
  //
  // Lowering emits DpuInst objects (the DPU IR).  The bit-level encoding
  // into 32-bit words happens later in DpuProgram::Encode().
  // ------------------------------------------------------------------
  void EmitAluI(uint32_t op, uint32_t rd, uint32_t rs1, int32_t imm, std::string text) {
    program_.Append(DpuInst::AluI(op, rd, rs1, imm, std::move(text)));
  }
  void EmitAluR(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2, std::string text) {
    program_.Append(DpuInst::AluR(op, rd, rs1, rs2, std::move(text)));
  }
  void EmitAluRB(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2, std::string text) {
    program_.Append(DpuInst::AluRB(op, rd, rs1, rs2, std::move(text)));
  }
  void EmitMov(uint32_t rd, uint32_t rs, std::string text) {
    program_.Append(DpuInst::Mov(rd, rs, std::move(text)));
  }
  void EmitMovi(uint32_t rd, int32_t imm, std::string text) {
    program_.Append(DpuInst::Movi(rd, imm, std::move(text)));
  }
  void EmitLoad(uint32_t rd, uint32_t addr, uint32_t width, bool is_signed, std::string text) {
    program_.Append(DpuInst::Load(rd, addr, width, is_signed, std::move(text)));
  }
  void EmitStore(uint32_t addr, uint32_t value, uint32_t width, std::string text) {
    program_.Append(DpuInst::Store(addr, value, width, std::move(text)));
  }
  void EmitSync(uint32_t subop, std::string text) {
    program_.Append(DpuInst::Sync(subop, std::move(text)));
  }
  void EmitEnd(std::string text) { program_.Append(DpuInst::End(std::move(text))); }

  int NewLabel() { return next_label_++; }

  void MarkLabel(int label) { program_.MarkLabel(label); }

  void EmitBranch(uint32_t op, uint32_t rs1, uint32_t rs2, int label, const std::string& text) {
    program_.Append(DpuInst::Branch(op, rs1, rs2, label, text));
  }

  /*! \brief Materialize a 32-bit constant into \p rd. */
  void EmitConst(uint32_t rd, int64_t value) {
    TVM_FFI_ICHECK(value >= std::numeric_limits<int32_t>::min() &&
                  value <= std::numeric_limits<int32_t>::max())
        << "DPU constant out of 32-bit range: " << value;
    if (value >= -(1 << 17) && value < (1 << 17)) {
      EmitMovi(rd, value, "MOVI " + RegName(rd) + ", " + std::to_string(value));
      return;
    }
    int64_t lo = value & 0xFFFF;
    int64_t hi = (value >> 16) & 0xFFFF;
    EmitMovi(rd, lo, "MOVI " + RegName(rd) + ", " + std::to_string(lo));
    if (hi != 0) {
      uint32_t tmp = AllocReg();
      EmitMovi(tmp, hi, "MOVI " + RegName(tmp) + ", " + std::to_string(hi));
      EmitAluI(kShl, tmp, tmp, 16, "SHL " + RegName(tmp) + ", " + RegName(tmp) + ", 16");
      EmitAluR(kOr, rd, rd, tmp,
           "OR " + RegName(rd) + ", " + RegName(rd) + ", " + RegName(tmp));
      FreeReg(tmp);
    }
  }

  uint32_t Materialize(int64_t value) {
    uint32_t reg = AllocReg();
    EmitConst(reg, value);
    return reg;
  }

  uint32_t AllocReg() {
    for (uint32_t r = kFirstTempReg; r <= kLastTempReg; ++r) {
      if (!reg_used_[r]) {
        reg_used_[r] = true;
        return r;
      }
    }
    Unsupported("out of DPU registers (r1-r23)");
  }

  void FreeReg(uint32_t reg) {
    if (reg >= kFirstTempReg && reg <= kLastTempReg) {
      TVM_FFI_ICHECK(reg_used_[reg]) << "double free of " << RegName(reg);
      reg_used_[reg] = false;
    }
  }

  void ReleaseIfTemp(Value& value) {
    if (value.temp) {
      FreeReg(value.reg);
      value.temp = false;
    }
  }

  // ------------------------------------------------------------------
  // Variable scopes
  // ------------------------------------------------------------------
  struct Binding {
    uint32_t reg;
    bool owned;
  };

  void PushScope() { scopes_.emplace_back(); }

  void PopScope() {
    for (const auto& kv : scopes_.back()) {
      if (kv.second.owned) {
        FreeReg(kv.second.reg);
      }
    }
    scopes_.pop_back();
  }

  void BindVar(const tvm::Var& var, uint32_t reg, bool owned) {
    TVM_FFI_ICHECK(!scopes_.empty());
    scopes_.back()[var.get()] = Binding{reg, owned};
  }

  uint32_t LookupVar(const tvm::VarNode* var) {
    auto pit = param_regs_.find(var);
    if (pit != param_regs_.end()) {
      return pit->second;
    }
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      auto vit = it->find(var);
      if (vit != it->end()) {
        return vit->second.reg;
      }
    }
    Unsupported("undefined variable " + std::string(var->name));
  }

  // ------------------------------------------------------------------
  // Expression lowering
  // ------------------------------------------------------------------
  Value EvalExpr(const Expr& e) {
    if (auto imm = e.as<tirx::IntImmNode>()) {
      return Value{Materialize(imm->value), true};
    }
    if (auto var = e.as<tvm::VarNode>()) {
      return Value{LookupVar(var), false};
    }
    if (auto load = e.as<tirx::BufferLoadNode>()) {
      return EvalBufferLoad(load);
    }
    if (auto cast = e.as<tirx::CastNode>()) {
      return EvalCast(cast);
    }
    if (auto add = e.as<tirx::AddNode>()) {
      return EvalAddSub(add->a, add->b, true);
    }
    if (auto sub = e.as<tirx::SubNode>()) {
      return EvalAddSub(sub->a, sub->b, false);
    }
    if (auto mul = e.as<tirx::MulNode>()) {
      return EvalMul(mul->a, mul->b);
    }
    if (auto div = e.as<tirx::DivNode>()) {
      return EvalDivMod(div->a, div->b, kDiv);
    }
    if (auto mod = e.as<tirx::ModNode>()) {
      return EvalDivMod(mod->a, mod->b, kMod);
    }
    if (auto fdiv = e.as<tirx::FloorDivNode>()) {
      return EvalDivMod(fdiv->a, fdiv->b, kDiv);
    }
    if (auto fmod = e.as<tirx::FloorModNode>()) {
      return EvalDivMod(fmod->a, fmod->b, kMod);
    }
    if (auto min = e.as<tirx::MinNode>()) {
      return EvalAluRR(min->a, min->b, kMin);
    }
    if (auto max = e.as<tirx::MaxNode>()) {
      return EvalAluRR(max->a, max->b, kMax);
    }
    if (auto eq = e.as<tirx::EQNode>()) {
      return EvalAluRR(eq->a, eq->b, kEq);
    }
    if (auto ne = e.as<tirx::NENode>()) {
      return EvalAluRR(ne->a, ne->b, kNe);
    }
    if (auto lt = e.as<tirx::LTNode>()) {
      return EvalAluRR(lt->a, lt->b, kLt);
    }
    if (auto le = e.as<tirx::LENode>()) {
      return EvalAluRR(le->a, le->b, kLe);
    }
    if (auto gt = e.as<tirx::GTNode>()) {
      return EvalAluRR(gt->a, gt->b, kGt);
    }
    if (auto ge = e.as<tirx::GENode>()) {
      return EvalAluRR(ge->a, ge->b, kGe);
    }
    if (auto select = e.as<tirx::SelectNode>()) {
      return EvalSelect(select);
    }
    if (auto not_node = e.as<tirx::NotNode>()) {
      Value value = EvalExpr(not_node->a);
      uint32_t dest = AllocReg();
      EmitAluI(kXor, dest, value.reg, 1,
               "XOR " + RegName(dest) + ", " + RegName(value.reg) + ", 1");
      ReleaseIfTemp(value);
      return Value{dest, true};
    }
    if (auto and_node = e.as<tirx::AndNode>()) {
      return EvalBitwise(and_node->a, and_node->b, kAnd);
    }
    if (auto or_node = e.as<tirx::OrNode>()) {
      return EvalBitwise(or_node->a, or_node->b, kOr);
    }
    if (auto call = e.as<tvm::CallNode>()) {
      return EvalCall(call);
    }
    if (e.as<tirx::FloatImmNode>()) {
      Unsupported("floating-point constant is not supported on DPU");
    }
    Unsupported("unsupported expression node in DPU codegen");
  }

  static bool IsIntImm(const Expr& e, int64_t* value) {
    if (auto imm = e.as<tirx::IntImmNode>()) {
      *value = imm->value;
      return true;
    }
    return false;
  }

  static bool FitsImm14(int64_t value) {
    return value >= -(1 << 13) && value < (1 << 13);
  }

  Value EvalAluIOrR(const Expr& a, const Expr& b, uint32_t op) {
    int64_t imm = 0;
    if (IsIntImm(b, &imm) && FitsImm14(imm)) {
      Value lhs = EvalExpr(a);
      uint32_t dest = AllocReg();
      EmitAluI(op, dest, lhs.reg, imm,
               std::string(AluName(op)) + " " + RegName(dest) + ", " + RegName(lhs.reg) + ", " +
                   std::to_string(imm));
      ReleaseIfTemp(lhs);
      return Value{dest, true};
    }
    if (IsIntImm(a, &imm) && FitsImm14(imm)) {
      Value rhs = EvalExpr(b);
      uint32_t dest = AllocReg();
      EmitAluI(op, dest, rhs.reg, imm,
               std::string(AluName(op)) + " " + RegName(dest) + ", " + RegName(rhs.reg) + ", " +
                   std::to_string(imm));
      ReleaseIfTemp(rhs);
      return Value{dest, true};
    }
    return EvalAluRR(a, b, op);
  }

  Value EvalAluRR(const Expr& a, const Expr& b, uint32_t op) {
    Value lhs = EvalExpr(a);
    int64_t imm = 0;
    // The zero register can stand in for a constant zero operand.
    if (IsIntImm(b, &imm) && imm == 0) {
      uint32_t dest = AllocReg();
      EmitAluR(op, dest, lhs.reg, kZeroReg,
               std::string(AluName(op)) + " " + RegName(dest) + ", " + RegName(lhs.reg) + ", r0");
      ReleaseIfTemp(lhs);
      return Value{dest, true};
    }
    Value rhs = EvalExpr(b);
    uint32_t dest = AllocReg();
    EmitAluR(op, dest, lhs.reg, rhs.reg,
             std::string(AluName(op)) + " " + RegName(dest) + ", " + RegName(lhs.reg) + ", " +
                 RegName(rhs.reg));
    ReleaseIfTemp(lhs);
    ReleaseIfTemp(rhs);
    return Value{dest, true};
  }

  Value EvalAddSub(const Expr& a, const Expr& b, bool is_add) {
    return EvalAluIOrR(a, b, is_add ? kAdd : kSub);
  }

  Value EvalBitwise(const Expr& a, const Expr& b, uint32_t op) {
    return EvalAluIOrR(a, b, op);
  }

  Value EvalMul(const Expr& a, const Expr& b) {
    int64_t lhs_imm = 0;
    int64_t rhs_imm = 0;
    const Expr* variable_side = nullptr;
    int64_t constant = 0;
    bool lhs_const = IsIntImm(a, &lhs_imm);
    bool rhs_const = IsIntImm(b, &rhs_imm);
    if (rhs_const && !lhs_const) {
      variable_side = &a;
      constant = rhs_imm;
    } else if (lhs_const && !rhs_const) {
      variable_side = &b;
      constant = lhs_imm;
    } else if (lhs_const && rhs_const) {
      // Constant folding should have handled this; materialize the result.
      uint32_t dest = AllocReg();
      EmitConst(dest, lhs_imm * rhs_imm);
      return Value{dest, true};
    } else {
      Value lhs = EvalExpr(a);
      Value rhs = EvalExpr(b);
      uint32_t dest = AllocReg();
      EmitAluR(kMul, dest, lhs.reg, rhs.reg,
               "MUL " + RegName(dest) + ", " + RegName(lhs.reg) + ", " + RegName(rhs.reg));
      ReleaseIfTemp(lhs);
      ReleaseIfTemp(rhs);
      return Value{dest, true};
    }

    Value src = EvalExpr(*variable_side);
    uint32_t dest = 0;
    if (constant == 0) {
      dest = AllocReg();
      EmitConst(dest, 0);
    } else if (constant > 0 && (constant & (constant - 1)) == 0) {
      int shift = 0;
      while ((int64_t(1) << shift) < constant) {
        ++shift;
      }
      dest = AllocReg();
      EmitAluI(kShl, dest, src.reg, shift,
               "SHL " + RegName(dest) + ", " + RegName(src.reg) + ", " + std::to_string(shift));
    } else if (constant > 1 && ((constant - 1) & (constant - 2)) == 0) {
      // constant == 2^k + 1
      int shift = 0;
      while ((int64_t(1) << shift) < constant - 1) {
        ++shift;
      }
      uint32_t shifted = AllocReg();
      EmitAluI(kShl, shifted, src.reg, shift,
               "SHL " + RegName(shifted) + ", " + RegName(src.reg) + ", " + std::to_string(shift));
      dest = AllocReg();
      EmitAluR(kAdd, dest, shifted, src.reg,
               "ADD " + RegName(dest) + ", " + RegName(shifted) + ", " + RegName(src.reg));
      FreeReg(shifted);
    } else if (constant > 1 && ((constant + 1) & constant) == 0) {
      // constant == 2^k - 1
      int shift = 0;
      while ((int64_t(1) << shift) < constant + 1) {
        ++shift;
      }
      uint32_t shifted = AllocReg();
      EmitAluI(kShl, shifted, src.reg, shift,
               "SHL " + RegName(shifted) + ", " + RegName(src.reg) + ", " + std::to_string(shift));
      dest = AllocReg();
      EmitAluR(kSub, dest, shifted, src.reg,
               "SUB " + RegName(dest) + ", " + RegName(shifted) + ", " + RegName(src.reg));
      FreeReg(shifted);
    } else {
      uint32_t cst = Materialize(constant);
      dest = AllocReg();
      EmitAluR(kMul, dest, src.reg, cst,
               "MUL " + RegName(dest) + ", " + RegName(src.reg) + ", " + RegName(cst));
      FreeReg(cst);
    }
    ReleaseIfTemp(src);
    return Value{dest, true};
  }

  Value EvalDivMod(const Expr& a, const Expr& b, uint32_t op) {
    Value lhs = EvalExpr(a);
    Value rhs = EvalExpr(b);
    uint32_t dest = AllocReg();
    EmitAluRB(op, dest, lhs.reg, rhs.reg,
              std::string(AluRbName(op)) + " " + RegName(dest) + ", " + RegName(lhs.reg) + ", " +
                  RegName(rhs.reg));
    ReleaseIfTemp(lhs);
    ReleaseIfTemp(rhs);
    return Value{dest, true};
  }

  Value EvalSelect(const tirx::SelectNode* select) {
    Value cond = EvalExpr(select->condition);
    Value true_value = EvalExpr(select->true_value);
    Value false_value = EvalExpr(select->false_value);
    uint32_t dest = AllocReg();
    int true_label = NewLabel();
    int end_label = NewLabel();
    EmitBranch(kBne, cond.reg, kZeroReg, true_label,
               "BNE " + RegName(cond.reg) + ", r0, L" + std::to_string(true_label));
    EmitMov(dest, false_value.reg,
            "MOV " + RegName(dest) + ", " + RegName(false_value.reg));
    EmitBranch(kBranch, kZeroReg, kZeroReg, end_label, "B L" + std::to_string(end_label));
    MarkLabel(true_label);
    EmitMov(dest, true_value.reg,
            "MOV " + RegName(dest) + ", " + RegName(true_value.reg));
    MarkLabel(end_label);
    ReleaseIfTemp(cond);
    ReleaseIfTemp(true_value);
    ReleaseIfTemp(false_value);
    return Value{dest, true};
  }

  Value EvalCall(const tvm::CallNode* call) {
    auto op = call->op;
    if (op.same_as(tirx::builtin::shift_left())) {
      return EvalAluIOrR(call->args[0], call->args[1],
                         kShl);
    }
    if (op.same_as(tirx::builtin::shift_right())) {
      return EvalAluIOrR(call->args[0], call->args[1],
                         kShr);
    }
    if (op.same_as(tirx::builtin::bitwise_and())) {
      return EvalBitwise(call->args[0], call->args[1],
                         kAnd);
    }
    if (op.same_as(tirx::builtin::bitwise_or())) {
      return EvalBitwise(call->args[0], call->args[1],
                         kOr);
    }
    if (op.same_as(tirx::builtin::bitwise_xor())) {
      return EvalBitwise(call->args[0], call->args[1],
                         kXor);
    }
    if (op.same_as(tirx::builtin::bitwise_not())) {
      Value value = EvalExpr(call->args[0]);
      uint32_t dest = AllocReg();
      uint32_t mask = Materialize(0xFFFFFFFF);
      EmitAluR(kXor, dest, value.reg, mask,
               "XOR " + RegName(dest) + ", " + RegName(value.reg) + ", " + RegName(mask));
      FreeReg(mask);
      ReleaseIfTemp(value);
      return Value{dest, true};
    }
    Unsupported("unsupported call `" + std::string(op->GetTypeKey()) + "` in DPU device code");
  }

  Value EvalCast(const tirx::CastNode* cast) {
    Value src = EvalExpr(cast->value);
    PrimType dst_ty = cast->ty.as_or_throw<PrimType>();
    PrimType src_ty = cast->value.ty();
    if (dst_ty.code() == kDLFloat || src_ty.code() == kDLFloat) {
      Unsupported("floating-point cast is not supported on DPU");
    }
    int dst_bits = dst_ty.bits();
    int src_bits = src_ty.bits();
    if (dst_bits == src_bits) {
      return src;
    }
    if (dst_bits == 32 && src_bits < 32) {
      int shift = 32 - src_bits;
      uint32_t dest = AllocReg();
      EmitAluI(kShl, dest, src.reg, shift,
               "SHL " + RegName(dest) + ", " + RegName(src.reg) + ", " + std::to_string(shift));
      bool is_signed = src_ty.code() == kDLInt;
      EmitAluI(is_signed ? kSar : kShr, dest, dest, shift,
               std::string(AluName(is_signed ? kSar : kShr)) + " " + RegName(dest) + ", " +
                   RegName(dest) + ", " + std::to_string(shift));
      ReleaseIfTemp(src);
      return Value{dest, true};
    }
    // Narrowing casts are implicit in the STOREM width.
    return src;
  }

  // ------------------------------------------------------------------
  // Buffer addressing
  // ------------------------------------------------------------------
  static int ElemBytes(const tirx::BufferVar& buffer) {
    PrimType dtype = buffer->dtype;
    TVM_FFI_ICHECK_EQ(dtype.lanes(), 1) << "DPU codegen does not support vector buffers";
    int bits = dtype.bits();
    TVM_FFI_ICHECK(bits == 8 || bits == 16 || bits == 32)
        << "DPU codegen only supports 8/16/32-bit buffers, got " << bits;
    return bits / 8;
  }

  static uint32_t WidthCode(int bytes) {
    switch (bytes) {
      case 1:
        return kWidthByte;
      case 2:
        return kWidthHalf;
      case 4:
        return kWidthWord;
      default:
        Unsupported("unsupported memory access width " + std::to_string(bytes));
    }
  }

  const tvm::VarNode* ResolveBufferData(const tvm::VarNode* buffer_var, int depth) {
    TVM_FFI_ICHECK(depth < 16) << "cyclic buffer data definition";
    if (param_regs_.count(buffer_var)) {
      return buffer_var;
    }
    if (alloc_offsets_.count(buffer_var)) {
      return nullptr;
    }
    auto it = decl_data_.find(buffer_var);
    if (it == decl_data_.end()) {
      Unsupported("cannot resolve the data pointer of buffer " + std::string(buffer_var->name));
    }
    const Expr& data = it->second;
    if (auto var = data.as<tvm::VarNode>()) {
      return ResolveBufferData(var, depth + 1);
    }
    if (auto call = data.as<tvm::CallNode>()) {
      if (call->op.same_as(tirx::builtin::buffer_data()) && call->args.size() == 1) {
        if (auto var = call->args[0].as<tvm::VarNode>()) {
          return ResolveBufferData(var, depth + 1);
        }
      }
    }
    Unsupported("unsupported buffer data expression for " + std::string(buffer_var->name));
  }

  /*! \brief Return the byte address of `buffer[indices]` in a temporary register. */
  uint32_t EvalAddress(const tirx::BufferVar& buffer, const ffi::Array<PrimExpr>& indices) {
    PrimExpr elem_offset = buffer->ElemOffset(indices)[0];
    int bytes = ElemBytes(buffer);
    PrimExpr byte_offset = elem_offset * tirx::MakeConst(elem_offset.ty(), bytes);
    Value offset = EvalExpr(byte_offset);

    uint32_t base = 0;
    bool base_temp = false;
    if (auto alloc_it = alloc_offsets_.find(buffer.get()); alloc_it != alloc_offsets_.end()) {
      base = Materialize(kSramBase + alloc_it->second);
      base_temp = true;
    } else {
      const tvm::VarNode* data = ResolveBufferData(buffer.get(), 0);
      TVM_FFI_ICHECK(data != nullptr);
      base = param_regs_.at(data);
    }

    uint32_t addr = AllocReg();
    EmitAluR(kAdd, addr, base, offset.reg,
             "ADD " + RegName(addr) + ", " + RegName(base) + ", " + RegName(offset.reg));
    if (base_temp) {
      FreeReg(base);
    }
    ReleaseIfTemp(offset);
    return addr;
  }

  Value EvalBufferLoad(const tirx::BufferLoadNode* load) {
    uint32_t addr = EvalAddress(load->buffer, load->indices);
    int bytes = ElemBytes(load->buffer);
    bool is_signed = load->buffer->dtype.code() == kDLInt;
    uint32_t dest = AllocReg();
    EmitLoad(dest, addr, WidthCode(bytes), is_signed,
             "LOADM " + RegName(dest) + ", [" + RegName(addr) + "], " + std::to_string(bytes) + "B" +
                 (is_signed ? " signed" : " unsigned"));
    FreeReg(addr);
    return Value{dest, true};
  }

  void VisitBufferStore(const tirx::BufferStoreNode* store) {
    uint32_t addr = EvalAddress(store->buffer, store->indices);
    Value value = EvalExpr(store->value);
    int bytes = ElemBytes(store->buffer);
    EmitStore(addr, value.reg, WidthCode(bytes),
              "STOREM [" + RegName(addr) + "], " + RegName(value.reg) + ", " + std::to_string(bytes) +
                  "B");
    FreeReg(addr);
    ReleaseIfTemp(value);
  }

  // ------------------------------------------------------------------
  // Statement lowering
  // ------------------------------------------------------------------
  void VisitStmt(const tirx::Stmt& stmt) {
    if (auto seq = stmt.as<tirx::SeqStmtNode>()) {
      for (const tirx::Stmt& s : seq->seq) {
        VisitStmt(s);
      }
    } else if (auto bind = stmt.as<tirx::BindNode>()) {
      int64_t imm = 0;
      uint32_t reg = 0;
      if (IsIntImm(bind->value, &imm)) {
        reg = AllocReg();
        EmitConst(reg, imm);
      } else {
        Value value = EvalExpr(bind->value);
        reg = AllocReg();
        EmitMov(reg, value.reg, "MOV " + RegName(reg) + ", " + RegName(value.reg));
        ReleaseIfTemp(value);
      }
      BindVar(bind->var, reg, true);
    } else if (auto decl = stmt.as<tirx::DeclBufferNode>()) {
      decl_data_[decl->buffer.get()] = decl->data;
    } else if (auto alloc = stmt.as<tirx::AllocBufferNode>()) {
      int64_t elems = 1;
      for (const PrimExpr& extent : alloc->buffer->shape) {
        if (auto imm = extent.as<tirx::IntImmNode>()) {
          elems *= imm->value;
        } else {
          Unsupported("DPU alloc_buffer requires a constant shape");
        }
      }
      int64_t allocation = elems * static_cast<int64_t>(ElemBytes(alloc->buffer));
      int64_t aligned = (next_sram_offset_ + alignment_ - 1) / alignment_ * alignment_;
      alloc_offsets_[alloc->buffer.get()] = aligned;
      next_sram_offset_ = aligned + allocation;
      TVM_FFI_ICHECK(next_sram_offset_ <= GetTargetAttr(target_, "sram-size-bytes", 16 << 20))
          << "DPU SRAM allocation exceeds the target sram-size-bytes";
    } else if (auto store = stmt.as<tirx::BufferStoreNode>()) {
      VisitBufferStore(store);
    } else if (auto if_node = stmt.as<tirx::IfThenElseNode>()) {
      VisitIfThenElse(if_node);
    } else if (auto for_node = stmt.as<tirx::ForNode>()) {
      VisitFor(for_node);
    } else if (auto while_node = stmt.as<tirx::WhileNode>()) {
      VisitWhile(while_node);
    } else if (auto attr = stmt.as<tirx::AttrStmtNode>()) {
      VisitStmt(attr->body);
    } else if (auto assert_stmt = stmt.as<tirx::AssertStmtNode>()) {
      // Asserts are host-side checks; skip them in the device kernel.
      (void)assert_stmt;
    } else if (auto eval = stmt.as<tirx::EvaluateNode>()) {
      int64_t imm = 0;
      if (!IsIntImm(eval->value, &imm)) {
        Value value = EvalExpr(eval->value);
        ReleaseIfTemp(value);
      }
    } else if (auto ret = stmt.as<tirx::ReturnNode>()) {
      if (ret->value.defined()) {
        Value value = EvalExpr(ret->value);
        ReleaseIfTemp(value);
      }
      EmitBranch(kBranch, kZeroReg, kZeroReg, end_label_,
                 "B L" + std::to_string(end_label_) + "  ; return");
    } else if (auto brk = stmt.as<tirx::BreakNode>()) {
      TVM_FFI_ICHECK(!loop_stack_.empty()) << "break outside of a loop";
      EmitBranch(kBranch, kZeroReg, kZeroReg, loop_stack_.back().break_label, "B break");
      (void)brk;
    } else if (auto cont = stmt.as<tirx::ContinueNode>()) {
      TVM_FFI_ICHECK(!loop_stack_.empty()) << "continue outside of a loop";
      EmitBranch(kBranch, kZeroReg, kZeroReg, loop_stack_.back().continue_label, "B continue");
      (void)cont;
    } else {
      Unsupported("unsupported statement in DPU device code");
    }
  }

  void VisitIfThenElse(const tirx::IfThenElseNode* if_node) {
    Value cond = EvalExpr(if_node->condition);
    int then_label = NewLabel();
    int end_label = NewLabel();
    EmitBranch(kBne, cond.reg, kZeroReg, then_label,
               "BNE " + RegName(cond.reg) + ", r0, L" + std::to_string(then_label));
    ReleaseIfTemp(cond);
    if (if_node->else_case.defined()) {
      VisitStmt(if_node->else_case.value());
    }
    EmitBranch(kBranch, kZeroReg, kZeroReg, end_label, "B L" + std::to_string(end_label));
    MarkLabel(then_label);
    VisitStmt(if_node->then_case);
    MarkLabel(end_label);
  }

  void VisitFor(const tirx::ForNode* for_node) {
    if (for_node->kind == tirx::ForKind::kThreadBinding) {
      Unsupported("thread-bound loops are not supported on DPU");
    }
    // counter = min
    uint32_t counter = AllocReg();
    int64_t min_imm = 0;
    if (IsIntImm(for_node->min, &min_imm) && min_imm == 0) {
      EmitConst(counter, 0);
    } else {
      Value min_value = EvalExpr(for_node->min);
      EmitMov(counter, min_value.reg,
              "MOV " + RegName(counter) + ", " + RegName(min_value.reg));
      ReleaseIfTemp(min_value);
    }
    // limit = min + extent
    PrimExpr limit_expr = for_node->min + for_node->extent;
    Value limit_value = EvalExpr(limit_expr);
    uint32_t limit = limit_value.reg;

    int loop_label = NewLabel();
    int continue_label = NewLabel();
    int break_label = NewLabel();
    MarkLabel(loop_label);

    PushScope();
    BindVar(for_node->loop_var, counter, false);
    loop_stack_.push_back(LoopLabels{break_label, continue_label});
    VisitStmt(for_node->body);
    loop_stack_.pop_back();
    PopScope();

    MarkLabel(continue_label);
    EmitAluI(kAdd, counter, counter, 1,
             "ADD " + RegName(counter) + ", " + RegName(counter) + ", 1");
    EmitBranch(kBlt, counter, limit, loop_label,
               "BLT " + RegName(counter) + ", " + RegName(limit) + ", L" +
                   std::to_string(loop_label));
    MarkLabel(break_label);

    ReleaseIfTemp(limit_value);
    FreeReg(counter);
  }

  void VisitWhile(const tirx::WhileNode* while_node) {
    int cond_label = NewLabel();
    int body_label = NewLabel();
    int continue_label = body_label;
    int break_label = NewLabel();
    MarkLabel(cond_label);
    Value cond = EvalExpr(while_node->condition);
    EmitBranch(kBne, cond.reg, kZeroReg, body_label,
               "BNE " + RegName(cond.reg) + ", r0, L" + std::to_string(body_label));
    ReleaseIfTemp(cond);
    EmitBranch(kBranch, kZeroReg, kZeroReg, break_label, "B break");
    MarkLabel(body_label);
    PushScope();
    loop_stack_.push_back(LoopLabels{break_label, continue_label});
    VisitStmt(while_node->body);
    loop_stack_.pop_back();
    PopScope();
    EmitBranch(kBranch, kZeroReg, kZeroReg, cond_label, "B L" + std::to_string(cond_label));
    MarkLabel(break_label);
  }

  std::string PatternName() const {
    if (pattern_.max_depth >= 4 && pattern_.has_reduction && pattern_.has_mul &&
        pattern_.has_const_offset) {
      return "conv2d";
    }
    if (pattern_.max_depth >= 3 && pattern_.has_reduction && pattern_.has_mul) {
      return "gemm";
    }
    if (pattern_.has_reduction) {
      return "reduction";
    }
    return pattern_.max_depth > 0 ? "elementwise" : "straightline";
  }

  struct LoopLabels {
    int break_label;
    int continue_label;
  };

  // State -----------------------------------------------------------------
  const Target& target_;
  const tirx::PrimFunc& func_;
  int64_t alignment_;
  int64_t next_sram_offset_ = 0;

  DpuProgram program_;
  std::vector<bool> reg_used_ = std::vector<bool>(32, false);
  std::vector<std::unordered_map<const tvm::VarNode*, Binding>> scopes_;
  std::unordered_map<const tvm::VarNode*, uint32_t> param_regs_;
  std::unordered_map<const tvm::VarNode*, Expr> decl_data_;
  std::unordered_map<const tvm::VarNode*, int64_t> alloc_offsets_;
  std::vector<LoopLabels> loop_stack_;

  int next_label_ = 0;
  int end_label_ = -1;
  PatternInfo pattern_;
};

}  // namespace

}  // namespace dpu

ffi::Module BuildDPU(IRModule mod, Target target) {
  std::ostringstream source_maker;
  source_maker << "; DPU source generated by TVM\n";

  for (const auto& kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "CodeGenDPU: Can only take PrimFunc";
    const tirx::PrimFunc& func = kv.second.as_or_throw<tirx::PrimFunc>();
    auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(global_symbol.has_value())
        << "CodeGenDPU: Expect PrimFunc to have the global_symbol attribute";
    dpu::DpuCodeGen codegen(target, func);
    source_maker << codegen.Generate(global_symbol.value()) << "\n";
  }

  return codegen::DeviceSourceModuleCreate(source_maker.str(), "dpu", ExtractFuncInfo(mod), "dpu");
}

void RegisterDPUCodegen() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.dpu", BuildDPU);
}

}  // namespace codegen
}  // namespace tvm
