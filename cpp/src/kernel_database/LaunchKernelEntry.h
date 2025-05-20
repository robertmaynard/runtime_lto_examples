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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvJitLink.h>


struct LaunchKernelEntry {
  CUkernel get_kernel(CUlibrary lib) const;

  bool operator==(const LaunchKernelEntry& rhs) const {
    return launch_key == rhs.launch_key;
  }

  //This needs to hold if the entry is
  //         nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoIR.data.get(),
  //                         ltoIR.size, file_name.c_str());
  // or whatever the nvrtc version is
  //
  bool add_to(nvJitLinkHandle& handle) const;

  std::vector<std::string> params{};
  std::string launch_key{};
  std::string compute_key{};
};

template <>
struct std::hash<LaunchKernelEntry> {
  std::size_t operator()(LaunchKernelEntry const& ke) const {
    return std::hash<std::string>{}(ke.launch_key);
  }
};
