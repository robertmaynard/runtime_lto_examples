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


#include "ComputeFragmentEntry.h"

struct NRTCLTOFragmentCompiler;

struct PerAlgoCachedInfo {

  std::unordered_set<std::unique_ptr<ComputeFragmentEntry>,
                     ComputeFragmentEntryHash,
                     ComputeFragmentEntryEqual> entries;
  std::unique_ptr<NRTCLTOFragmentCompiler> nvrtc_compiler;
  std::unordered_map<std::string, char const*> nvrtc_includes;
};

class AlgorithmFragmentDatabase {
public:
  AlgorithmFragmentDatabase(AlgorithmFragmentDatabase const&) = delete;
  AlgorithmFragmentDatabase(AlgorithmFragmentDatabase&&) = delete;

  AlgorithmFragmentDatabase& operator=(AlgorithmFragmentDatabase&&) = delete;
  AlgorithmFragmentDatabase& operator=(AlgorithmFragmentDatabase const&) = delete;


 ComputeFragmentEntry* get_algo(std::string const& name,
                                std::vector<std::string> const& params) const;

 bool has_fragment(std::string const& name,
                 std::vector<std::string> const& params) const;
 bool add_nvrtc_fragment(std::string const& name,
                       std::vector<std::string> const& params);
 bool add_nvrtc_include(std::string const& name,
                        std::string const& include_name, char const* blob);

private:
  AlgorithmFragmentDatabase();

  void make_cache_entry(std::string const& name);
  bool add_fatbin_fragment(std::string const& name,
                        std::vector<std::string> const& params,
                        unsigned char const* blob);

  friend AlgorithmFragmentDatabase& algo_fragment_database();

  friend void registerFatbinAlgo(std::string const& algo,
                                 std::vector<std::string> const& params,
                                 unsigned char const* blob);

  friend void registerNVRTCAlgoInclude(std::string algo,
                                       std::string const& name,
                                       char const* blob);

  std::unordered_map<std::string, PerAlgoCachedInfo> cache;
};

AlgorithmFragmentDatabase& algo_fragment_database();

void registerFatbinAlgo(std::string const& algo,
                        std::vector<std::string> const& params,
                        unsigned char const* blob);

void registerNVRTCAlgoInclude(std::string const& algo,
                              std::string const& name,
                              char const* blob);
