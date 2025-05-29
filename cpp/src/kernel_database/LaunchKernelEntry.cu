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

#include "LaunchKernelEntry.h"

#include <cstring>
#include <iostream>


namespace {
  std::string make_launch_key(std::vector<std::string> const& params) {
    std::string k{};
    for(auto& p : params) {
      k += p + "_";
    }
    return k;
  }

  // We can make a better RAII wrapper around nvjitlinkhandle
  void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result) {
    if (result != NVJITLINK_SUCCESS) {
      std::cerr << "\n nvJITLink failed with error " << result << '\n';
      size_t log_size = 0;
      result = nvJitLinkGetErrorLogSize(handle, &log_size);
      if (result == NVJITLINK_SUCCESS && log_size > 0) {
        std::unique_ptr<char[]> log{new char[log_size]};
        result = nvJitLinkGetErrorLog(handle, log.get());
        if (result == NVJITLINK_SUCCESS) {
          std::cerr << "nvJITLink error log: " << log.get() << '\n';
        }
      }
      exit(1);
    }
  }
}

LaunchKernelEntry::LaunchKernelEntry(std::vector<std::string> const& params)
    : launch_arg_count(params.size()),
    launch_key(make_launch_key(params)){}

std::size_t LaunchKernelEntryHash::operator()(
    std::vector<std::string> const& params) const noexcept {
    return std::hash<std::string>{}(make_launch_key(params));
}

bool LaunchKernelEntryEqual::operator()(
    std::vector<std::string> const& params,
    std::unique_ptr<LaunchKernelEntry> const& entry) const noexcept {

    if( params.size() == entry->launch_arg_count ) {
      auto key = make_launch_key(params);
      return entry->launch_key == key;
    }
    return false;
}

FatbinLaunchKernelEntry::FatbinLaunchKernelEntry(
    std::vector<std::string> const& params, unsigned char const* view)
    : LaunchKernelEntry(params),
      data_size(std::strlen((char*)view)),
      data_view(view) {}

bool FatbinLaunchKernelEntry::add_to(nvJitLinkHandle& handle) const {
    auto result =
        nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, this->data_view,
                         this->data_size, this->launch_key.c_str());
    check_nvjitlink_result(handle, result);
    return true;
}

NVRTCLaunchKernelEntry::NVRTCLaunchKernelEntry(
    std::vector<std::string> const& params, std::string const& mname,
    std::size_t size,
    std::unique_ptr<char[]>&& p)
    : LaunchKernelEntry(params),
      mangled_name(mname),
      data_size(size),
      program(std::move(p)) {}

bool NVRTCLaunchKernelEntry::add_to(nvJitLinkHandle& handle) const {
    auto result =
        nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, this->program.get(),
                         this->data_size, this->launch_key.c_str());
    check_nvjitlink_result(handle, result);

    return true;
}
