
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

#include <string>
#include <vector>
#include <typeinfo>

namespace detail {
  std::string nvrtc_name(std::type_info const& info);

  template<typename T>
  std::string type_as_string() {
    if constexpr (std::is_reference_v<T>) {
        return detail::nvrtc_name(typeid(T))+"&";
    } else {
        return detail::nvrtc_name(typeid(T));
    }
  }
}

template <typename... Ts>
std::vector<std::string> make_fragment_key() {
  // Create an array of type names using std::vector
  //
  std::vector<std::string> result;
  (result.push_back(detail::type_as_string<Ts>()), ...);
  return result;
}
