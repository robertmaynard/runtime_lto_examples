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

#pragma once

#include "LaunchKernelEntry.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct NRTCLTOFragmentCompiler;

namespace detail {
  std::string nvrtc_name(std::type_info const& info);
}


template <typename... Ts>
std::vector<std::string> make_launch_key() {
  // Create an array of type names using std::vector

  std::vector<std::string> result;
  (result.push_back(detail::nvrtc_name(typeid(Ts))), ...);
  return result;
}

// Holds all the LTO fragments for `__global__` entry points
//
// Can construct new `__global__` entries based on the requested
// argument types if it doesn't exist via NVRTC
//
// Used by the KernelPlanner's to construct the full program
// to launch
//
class LaunchKernelDatabase {

public:
  LaunchKernelDatabase(LaunchKernelDatabase const&) = delete;
  LaunchKernelDatabase(LaunchKernelDatabase&&) = delete;

  LaunchKernelDatabase& operator=(LaunchKernelDatabase&&) = delete;
  LaunchKernelDatabase& operator=(LaunchKernelDatabase const&) = delete;

  template <typename... Args>
  LaunchKernelEntry* get() const {
    auto launch_key = make_launch_key<Args...>();
    return this->get_kernel(launch_key);
 }

 template <typename... Args>
 bool has() const {
   auto launch_key = make_launch_key<Args...>();
   return this->has_kernel(launch_key);
 }

 template <typename... Args>
 bool add() {
   auto launch_key = make_launch_key<Args...>();
   return this->add_nvrtc_kernel(launch_key);
 }

private:
  friend LaunchKernelDatabase& build_launch_kernel_database();
  friend void registerFatbinLaunchKernel(std::vector<std::string> const& params,
                                         unsigned char const* blob);
  friend void registerNVRTCKernelInclude(std::string const& include_name,
                                         char const* blob);

  LaunchKernelDatabase();

  LaunchKernelEntry* get_kernel(std::vector<std::string> const& params) const;
  bool has_kernel(std::vector<std::string> const& params) const;
  bool add_nvrtc_kernel(std::vector<std::string> const& params);
  bool add_fatbin_kernel(std::vector<std::string> const& params, unsigned char const* blob);
  bool add_nvrtc_include(std::string const& include_name, char const* blob);

  std::unordered_set<std::unique_ptr<LaunchKernelEntry>,
                     LaunchKernelEntryHash,
                     LaunchKernelEntryEqual> entries;

  //needs to be lazily created to not call cuda functions at library load time
  std::unique_ptr<NRTCLTOFragmentCompiler> nvrtc_compiler;
  std::unordered_map<std::string, char const*> nvrtc_includes;
};


// Returns a reference to the static singelton
//
// A very basic factory pattern
LaunchKernelDatabase& build_launch_kernel_database();

void registerFatbinLaunchKernel(std::vector<std::string> const& params,
                                unsigned char const* blob);
void registerNVRTCKernelInclude(std::string const& include_name,
                                char const* blob);
