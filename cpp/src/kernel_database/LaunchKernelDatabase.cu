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

#include <iostream>

LaunchKernelDatabase::LaunchKernelDatabase() {}

LaunchKernelEntry LaunchKernelDatabase::get_kernel(
    std::vector<std::string> const& params) const {
  return LaunchKernelEntry{};
}

bool LaunchKernelDatabase::has_kernel(std::vector<std::string> const& params) const {
  return true;
}

bool LaunchKernelDatabase::add_nvrtc_kernel(
    std::vector<std::string> const& params) {
  return true;
}

bool LaunchKernelDatabase::add_fatbin_kernel(
    std::vector<std::string> const& params, unsigned char const* blob) {
  std::cout << "adding lto kernel" << std::endl;
  for(auto a : params) { std::cout << a; }
  std::cout << "\n";
  return true;
}

LaunchKernelDatabase& build_launch_kernel_database() {
  // Left to the reader to make this thread safe
  static LaunchKernelDatabase database;

  // We need to init the nvrtc source data here

  return database;
}

void registerFatbinLaunchKernel(std::vector<std::string> const& params,
                                unsigned char const* blob) {
  auto& db = build_launch_kernel_database();
  db.add_fatbin_kernel(params, blob);
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
