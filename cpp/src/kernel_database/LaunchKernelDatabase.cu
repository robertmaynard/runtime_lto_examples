/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LaunchKernelDatabase.h"
#include "LaunchKernelEntry.h"

#include "NVRTCLTOFragmentCompiler.h"

#include <iostream>

LaunchKernelDatabase::LaunchKernelDatabase() {}

LaunchKernelEntry* LaunchKernelDatabase::get_kernel(
    std::vector<std::string> const& params) const {

  auto it = this->entries.find(params);
  if(it != this->entries.end()){
    return it->get();
  }
  return nullptr;
}

bool LaunchKernelDatabase::has_kernel(std::vector<std::string> const& params) const {
  return this->entries.contains(params);
}

namespace {
std::string make_template_arg_syntax(std::vector<std::string> const& params) {
  std::string k{};
  for (auto& p : params) {
    k += p + ", ";
  }
  return k;
}
}  // namespace

bool LaunchKernelDatabase::add_nvrtc_kernel(
    std::vector<std::string> const& params) {
  if (this->has_kernel(params)) {
    return false;
  }

  if(this->nvrtc_compiler == nullptr) {
    this->nvrtc_compiler = std::make_unique<NRTCLTOFragmentCompiler>();
  }

  //Todo make this configurable
  std::string code;
  code += "#include \"grid_stride.h\"\n";
  code += "template __global__ void grid_stride(";
  code += make_template_arg_syntax(params);
  code += " size_t);\n\n";

  auto result =
      this->nvrtc_compiler->compile(code, this->nvrtc_includes);
  auto entry = std::make_unique<NVRTCLaunchKernelEntry>(
      params, result.mangled_name, result.size, std::move(result.program));
  this->entries.insert(std::move(entry));

  return true;
}

bool LaunchKernelDatabase::add_fatbin_kernel(
    std::vector<std::string> const& params, unsigned char const* blob) {
  if (this->has_kernel(params)) {
    return false;
  }
  std::cout << "add a kernel" << std::endl;

  auto entry = std::make_unique<FatbinLaunchKernelEntry>(params, blob);
  this->entries.insert(std::move(entry));
  return true;
}

bool LaunchKernelDatabase::add_nvrtc_include(std::string const& include_name,
                                             char const* blob)
{
  this->nvrtc_includes[include_name]=blob;
  return true;
}

LaunchKernelDatabase& build_launch_kernel_database() {
  // Left to the reader to make this thread safe
  static LaunchKernelDatabase database;
  return database;
}

void registerFatbinLaunchKernel(std::vector<std::string> const& params,
                                unsigned char const* blob) {
  auto& db = build_launch_kernel_database();
  db.add_fatbin_kernel(params, blob);
}

void registerNVRTCKernelInclude(std::string const& name,
                                char const* blob) {
  auto& db = build_launch_kernel_database();
  db.add_nvrtc_include(name, blob);
}

#define NVRTC_GET_TYPE_NAME 1
#include <nvrtc.h>
namespace detail {
std::string nvrtc_name(std::type_info const& info) {
  std::string type_name;
  nvrtcGetTypeName(info, &type_name);
  return type_name;
}
}  // namespace detail
