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


struct ComputeFragmentEntry {
  ComputeFragmentEntry(std::vector<std::string> const& params);

  bool operator==(const ComputeFragmentEntry& rhs) const {
    return compute_key == rhs.compute_key;
  }

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  std::size_t compute_arg_count = 0; //optimization for equality checks
  std::string compute_key{};
};

struct ComputeFragmentEntryHash {
  using is_transparent = void;

  std::size_t operator()(std::unique_ptr<ComputeFragmentEntry> const& entry) const noexcept {
    return std::hash<std::string>{}(entry->compute_key);
  }
    std::size_t operator()(ComputeFragmentEntry const* entry) const noexcept {
    return std::hash<std::string>{}(entry->compute_key);
  }
  std::size_t operator()(std::vector<std::string> const& params) const noexcept;
};

struct ComputeFragmentEntryEqual {
  using is_transparent = void;

  template <typename T, typename U>
  bool operator()(T const& t, U const& u) const {
    return std::to_address(t) == std::to_address(u);
  }

  bool operator()(std::unique_ptr<ComputeFragmentEntry> const& entry,
                  std::vector<std::string> const& params) const noexcept
  {
    return this->operator()(params, entry);
  }

  bool operator()(std::vector<std::string> const& params,
                  std::unique_ptr<ComputeFragmentEntry> const& entry) const noexcept;
};

struct FatbinComputeFragmentEntry final : ComputeFragmentEntry {
  FatbinComputeFragmentEntry(std::vector<std::string> const& params,
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

struct NVRTCComputeFragmentEntry final : ComputeFragmentEntry {

  NVRTCComputeFragmentEntry(std::vector<std::string> const& params,
                         std::size_t size,
                         std::unique_ptr<char[]>&& p);

  virtual bool add_to(nvJitLinkHandle& handle) const;

  std::size_t data_size = 0;
  std::unique_ptr<char[]> program{};
};
