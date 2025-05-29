# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================


function(generate_nvrtc_include_cpp_file target_name)
  # build up a c++ .cpp file that has all the includes

  get_target_property(libcudacxx_raw_includes CCCL::libcudacxx INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(libcub_raw_includes CCCL::CUB INTERFACE_INCLUDE_DIRECTORIES)
  set(includes ${libcub_raw_includes} ${libcudacxx_raw_includes} ${CUDAToolkit_INCLUDE_DIRS})

  set(textual_string)
  foreach(inc IN LISTS includes)
    string(APPEND textual_string "\"-I${inc}\", ")
  endforeach()

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/nvrtc_includes.cpp
    "
    #include <string>
    #include <vector>
    std::vector<std::string> get_include_dirs_from_cmake() {
      return std::vector<std::string>{${textual_string} " "};
    }
    ")

  target_sources(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/nvrtc_includes.cpp)
endfunction()

