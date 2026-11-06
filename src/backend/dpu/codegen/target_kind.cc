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
 * \file target_kind.cc
 * \brief DPU compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

namespace backend {
namespace dpu {

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  TVM_REGISTER_TARGET_KIND("dpu", kDLDPU)
      .add_attr_option<int64_t>("num-cores", refl::DefaultValue(2))
      .add_attr_option<int64_t>("sram-size-bytes", refl::DefaultValue(16 * 1024 * 1024))
      .add_attr_option<int64_t>("shared-memory-size-bytes", refl::DefaultValue(64 * 1024))
      .add_attr_option<int64_t>("sram-alignment", refl::DefaultValue(64))
      .set_default_keys({"dpu"});
}

}  // namespace dpu
}  // namespace backend

namespace codegen {
void RegisterDPUCodegen();
}  // namespace codegen

}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::dpu::RegisterTargetKind();
  tvm::codegen::RegisterDPUCodegen();
}
