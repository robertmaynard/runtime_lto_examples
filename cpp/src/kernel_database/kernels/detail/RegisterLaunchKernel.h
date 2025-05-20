
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

#define NVRTC_GET_TYPE_NAME 1
#include <nvrtc.h>

#include <vector>

void registerFatbinLaunchKernel(std::vector<std::string> const& params,
                                unsigned char const* blob);

namespace {

template <typename T>
std::string nvrtc_name() {
  std::string type_name;
  nvrtcGetTypeName<T>(&type_name);
  return type_name;
}

template <typename... Ts>
std::vector<std::string> make_launch_key() {
  // Create an array of type names using std::vector

  std::vector<std::string> result;
  (result.push_back(nvrtc_name<Ts>()), ...);
  return result;
}

template <typename... Ts>
void registerLaunchKernel(unsigned char const* blob) {
  auto key = make_launch_key<Ts...>();
  registerFatbinLaunchKernel(key, blob);
}
}  // namespace
