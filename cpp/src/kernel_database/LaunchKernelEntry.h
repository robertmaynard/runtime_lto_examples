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

#include <nvJitLink.h>


struct LaunchKernelEntry {
  LaunchKernelEntry(std::vector<std::string> const& params);

  bool operator==(const LaunchKernelEntry& rhs) const {
    return launch_key == rhs.launch_key;
  }

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  std::size_t launch_arg_count = 0; //optimization for equality checks
  std::string launch_key{};
};

struct LaunchKernelEntryHash {
  using is_transparent = void;

  std::size_t operator()(std::unique_ptr<LaunchKernelEntry> const& entry) const noexcept {
    return std::hash<std::string>{}(entry->launch_key);
  }
    std::size_t operator()(LaunchKernelEntry const* entry) const noexcept {
    return std::hash<std::string>{}(entry->launch_key);
  }
  std::size_t operator()(std::vector<std::string> const& params) const noexcept;
};

struct LaunchKernelEntryEqual {
  using is_transparent = void;

  template <typename T, typename U>
  bool operator()(T const& t, U const& u) const {
    return std::to_address(t) == std::to_address(u);
  }

  bool operator()(std::unique_ptr<LaunchKernelEntry> const& entry,
                  std::vector<std::string> const& params) const noexcept
  {
    return this->operator()(params, entry);
  }

  bool operator()(std::vector<std::string> const& params,
                  std::unique_ptr<LaunchKernelEntry> const& entry) const noexcept;
};

struct FatbinLaunchKernelEntry final : LaunchKernelEntry {
  FatbinLaunchKernelEntry(std::vector<std::string> const& params,
                          unsigned char const* view);

  // This needs to hold if the entry is
  //          nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoIR.data.get(),
  //                          ltoIR.size, file_name.c_str());
  //  or whatever the nvrtc version is
  //
  virtual bool add_to(nvJitLinkHandle& handle) const;

  std::size_t data_size = 0;
  unsigned char const* data_view = nullptr;
};

struct NVRTCLaunchKernelEntry final : LaunchKernelEntry {

  NVRTCLaunchKernelEntry(std::vector<std::string> const& params,
                         std::string const& mname,
                         std::size_t size,
                         std::unique_ptr<char[]>&& p);

  virtual bool add_to(nvJitLinkHandle& handle) const;

  std::string mangled_name{};
  std::size_t data_size = 0;
  std::unique_ptr<char[]> program{};
};
