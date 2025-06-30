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
#include <unordered_map>
#include <unordered_set>

#include "FragmentEntry.h"
#include "MakeFragmentKey.h"
#include "iteration_spaces/LaunchTypes.h"

struct NRTCLTOFragmentCompiler;


struct PerEntryCachedInfo {
  std::unordered_set<std::unique_ptr<FragmentEntry>, FragmentEntryHash,
                     FragmentEntryEqual>
      entries;
};

class FragmentDatabase {
 public:
  FragmentDatabase(FragmentDatabase const&) = delete;
  FragmentDatabase(FragmentDatabase&&) = delete;

  FragmentDatabase& operator=(FragmentDatabase&&) = delete;
  FragmentDatabase& operator=(FragmentDatabase const&) = delete;

  FragmentEntry* nvrtc_fragment(LaunchType type,
                                std::vector<std::string> const& params) {
    return this->nvrtc_fragment(true, to_string(type), params);
  }

  FragmentEntry* nvrtc_fragment(std::string const& name,
                                std::vector<std::string> const& params) {
    return this->nvrtc_fragment(false, name, params);
  }

 private:
  FragmentDatabase();

  void make_cache_entry(std::string const& name);

  FragmentEntry* nvrtc_fragment(bool is_entry_point,
                                std::string const& name,
                                std::vector<std::string> const& params);
  // Assumptions:
  //   make_cache_entry already called
  //   No existing fragment for `params` exists
  bool add_nvrtc_fragment(bool is_entry_point,
                          std::string const& name,
                          std::vector<std::string> const& params);

  bool has_fragment(std::string const& name,
                    std::vector<std::string> const& params) const;

  // Assumptions:
  //   make_cache_entry already called
  bool add_nvrtc_include(std::string const& name,
                         std::string const& include_name, char const* blob);

  // Assumptions:
  //   make_cache_entry already called
  //   No existing fragment for `params` exists
  bool add_fatbin_fragment(std::string const& name,
                           std::vector<std::string> const& params,
                           unsigned char const* blob);

  friend FragmentDatabase& fragment_database();

  friend void registerFatbinFragment(std::string const& algo,
                                     std::vector<std::string> const& params,
                                     unsigned char const* blob);

  friend void registerNVRTCFragmentInclude(std::string const& algo,
                                           std::string const& name,
                                           char const* blob);

  std::unordered_map<std::string, PerEntryCachedInfo> cache;

  std::unique_ptr<NRTCLTOFragmentCompiler> nvrtc_compiler;
  std::unordered_map<std::string, char const*> nvrtc_includes;
};

FragmentDatabase& fragment_database();

void registerFatbinFragment(std::string const& algo,
                        std::vector<std::string> const& params,
                        unsigned char const* blob);

void registerNVRTCFragmentInclude(std::string const& algo,
                                  std::string const& name, char const* blob);
