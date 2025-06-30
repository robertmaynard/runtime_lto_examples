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

#include <iostream>

#include "FragmentDatabase.h"
#include "FragmentEntry.h"
#include "NVRTCLTOFragmentCompiler.h"

FragmentDatabase::FragmentDatabase() {}


void FragmentDatabase::make_cache_entry(std::string const& name) {
  if (this->cache.count(name) == 0) {
    this->cache[name] = PerEntryCachedInfo{};
  }
  return;
}

FragmentEntry* FragmentDatabase::nvrtc_fragment(
    bool is_entry_point, std::string const& name,
    std::vector<std::string> const& params) {
  this->make_cache_entry(name);
  auto cache_it = this->cache.find(name);
  auto it = cache_it->second.entries.find(params);
  if (it == cache_it->second.entries.end()) {
    bool r = this->add_nvrtc_fragment(is_entry_point, name, params);
    if (!r) {
      return nullptr;
    }
    it = cache_it->second.entries.find(params);
  }
  return it->get();
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

bool FragmentDatabase::add_nvrtc_fragment(
    bool is_entry_point, std::string const& name,
    std::vector<std::string> const& params) {
  std::cout << name << " compiled as nvrtc_fragment " << std::endl;

  auto& cache_entry = this->cache[name];
  if (this->nvrtc_compiler == nullptr) {
    this->nvrtc_compiler = std::make_unique<NRTCLTOFragmentCompiler>();
  }

  std::string code;
  if(is_entry_point) {
    code += "#include \"" + name + ".hpp\"\n";
    code += "template __global__ void grid_stride(";
    code += make_template_arg_syntax(params);
    code += ");\n\n";
  } else {
    code += "#include \"" + name + ".hpp\"\n";
    code += "template __device__ void compute(";
    code += make_template_arg_syntax(params);
    code += ");\n\n";
  }

  auto result =
      this->nvrtc_compiler->compile(code, this->nvrtc_includes);
  auto entry = std::make_unique<NVRTCFragmentEntry>(
      params, result.mangled_name, result.size, std::move(result.program));
  cache_entry.entries.insert(std::move(entry));
  return true;
}

bool FragmentDatabase::has_fragment(
    std::string const& name, std::vector<std::string> const& params) const {
  auto cache_it = this->cache.find(name);
  if (cache_it != this->cache.end()) {
    cache_it->second.entries.contains(params);
  }
  return false;
}


bool FragmentDatabase::add_fatbin_fragment(
    std::string const& name, std::vector<std::string> const& params,
    unsigned char const* blob) {

  std::cout << name << " add a fatbin algo fragment " << std::endl;

  auto entry = std::make_unique<FatbinFragmentEntry>(params, blob);
  this->cache[name].entries.insert(std::move(entry));
  return true;
}

bool FragmentDatabase::add_nvrtc_include(
    std::string const& name, std::string const& include_name,
    char const* blob) {
  std::cout << name << " add a nvrtc algo include for " << include_name
            << std::endl;
  this->nvrtc_includes[include_name] = blob;
  return true;
}

FragmentDatabase& fragment_database() {
  // Left to the reader to make this thread safe
  static FragmentDatabase database;
  return database;
}

void registerFatbinFragment(std::string const& algo,
                        std::vector<std::string> const& params,
                        unsigned char const* blob) {
  auto& planner = fragment_database();
  planner.make_cache_entry(algo);
  if(!planner.has_fragment(algo, params)) {
    planner.add_fatbin_fragment(algo, params, blob);
  }
}

void registerNVRTCFragmentInclude(std::string const& algo,
                                  std::string const& name, char const* blob) {
  auto& planner = fragment_database();
  planner.make_cache_entry(algo);
  planner.add_nvrtc_include(algo, name, blob);
}

