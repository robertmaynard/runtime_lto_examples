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

#include "AlgorithmFragmentDatabase.h"
#include "ComputeFragmentEntry.h"

#include "NVRTCLTOFragmentCompiler.h"

#include <iostream>

AlgorithmFragmentDatabase::AlgorithmFragmentDatabase() {}

ComputeFragmentEntry* AlgorithmFragmentDatabase::get_algo(
    std::string const& name, std::vector<std::string> const& params) const {
  auto cache_it = this->cache.find(name);
  if (cache_it != this->cache.end()) {
    auto it = cache_it->second.entries.find(params);
    if(it != cache_it->second.entries.end()){
      return it->get();
    }
  }
  return nullptr;
}

bool AlgorithmFragmentDatabase::has_fragment(
    std::string const& name, std::vector<std::string> const& params) const {
  auto cache_it = this->cache.find(name);
  if (cache_it != this->cache.end()) {
    cache_it->second.entries.contains(params);
  }
  return false;
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

bool AlgorithmFragmentDatabase::add_nvrtc_fragment(
    std::string const& name, std::vector<std::string> const& params) {
  if (this->has_fragment(name, params)) {
    return false;
  }

  std::cout << name << " compile a add_nvrtc_fragment " << std::endl;

  this->make_cache_entry(name);

  auto& cache_entry = this->cache[name];
  if (cache_entry.nvrtc_compiler == nullptr) {
    cache_entry.nvrtc_compiler = std::make_unique<NRTCLTOFragmentCompiler>();
  }

  // Todo make this configurable
  std::string code;
  code += "#include \"" + name + ".hpp\"\n";
  code += "template __global__ void grid_stride(";
  code += make_template_arg_syntax(params);
  code += ");\n\n";

  auto result =
      cache_entry.nvrtc_compiler->compile(code, cache_entry.nvrtc_includes);
  auto entry = std::make_unique<NVRTCComputeFragmentEntry>(
      params, result.size, std::move(result.program));
  cache_entry.entries.insert(std::move(entry));

  return true;
}

bool AlgorithmFragmentDatabase::add_fatbin_fragment(
    std::string const& name, std::vector<std::string> const& params,
    unsigned char const* blob) {
  if (this->has_fragment(name, params)) {
    return false;
  }
  std::cout << name << " add a fatbin algo fragment " << std::endl;

  this->make_cache_entry(name);

  auto entry = std::make_unique<FatbinComputeFragmentEntry>(params, blob);
  this->cache[name].entries.insert(std::move(entry));
  return true;
}

bool AlgorithmFragmentDatabase::add_nvrtc_include(
    std::string const& name, std::string const& include_name,
    char const* blob) {

  std::cout << name << " add a nvrtc algo include for " << include_name << std::endl;
  //need to factor in `name`
  this->make_cache_entry(name);

  this->cache[name].nvrtc_includes[include_name]=blob;
  return true;
}

void AlgorithmFragmentDatabase::make_cache_entry(std::string const& name)
{
  if (this->cache.count(name) == 0) {
    this->cache[name] = PerAlgoCachedInfo{};
  }
  return;
}

AlgorithmFragmentDatabase& algo_fragment_database()
{
  // Left to the reader to make this thread safe
  static AlgorithmFragmentDatabase database;
  return database;
}

void registerFatbinAlgo(std::string const& algo,
                        std::vector<std::string> const& params,
                        unsigned char const* blob) {
  auto& planner = algo_fragment_database();
  planner.add_fatbin_fragment(algo, params, blob);
}

void registerNVRTCAlgoInclude(std::string const& algo,
                              std::string const& name,
                              char const* blob) {
  auto& planner = algo_fragment_database();
  planner.add_nvrtc_include(algo, name, blob);
}
