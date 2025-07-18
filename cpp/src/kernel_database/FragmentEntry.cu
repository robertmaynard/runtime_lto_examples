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

#include <cstring>
#include <iostream>

#include "FragmentEntry.h"

namespace {
std::string make_compute_key(std::vector<std::string> const& params) {
  std::string k{};
  for (auto& p : params) {
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
}  // namespace

FragmentEntry::FragmentEntry(std::vector<std::string> const& params)
    : compute_arg_count(params.size()), compute_key(make_compute_key(params)) {}

std::size_t FragmentEntryHash::operator()(
    std::vector<std::string> const& params) const noexcept {
  return std::hash<std::string>{}(make_compute_key(params));
}

bool FragmentEntryEqual::operator()(
    std::vector<std::string> const& params,
    std::unique_ptr<FragmentEntry> const& entry) const noexcept {
  if (params.size() == entry->compute_arg_count) {
    auto key = make_compute_key(params);
    return entry->compute_key == key;
  }
  return false;
}

FatbinFragmentEntry::FatbinFragmentEntry(std::vector<std::string> const& params,
                                         unsigned char const* view)
    : FragmentEntry(params),
      data_size(std::strlen((char*)view)),
      data_view(view) {}

bool FatbinFragmentEntry::add_to(nvJitLinkHandle& handle) const {
  auto result =
      nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, this->data_view,
                       this->data_size, this->compute_key.c_str());
  check_nvjitlink_result(handle, result);
  return true;
}

NVRTCFragmentEntry::NVRTCFragmentEntry(std::vector<std::string> const& params,
                                       std::string const& mname,
                                       std::size_t size,
                                       std::unique_ptr<char[]>&& p)
    : FragmentEntry(params),
      mangled_name(mname),
      data_size(size),
      program(std::move(p)) {}

bool NVRTCFragmentEntry::add_to(nvJitLinkHandle& handle) const {
  auto result =
      nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, this->program.get(),
                       this->data_size, this->compute_key.c_str());
  check_nvjitlink_result(handle, result);

  return true;
}
