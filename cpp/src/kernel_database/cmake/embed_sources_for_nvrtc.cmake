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


function(embed_sources_for_nvrtc nvrtc_target_to_embed_in )
  # We need to embed the source file as C++ string that
  # has an identifier equal to the name of the file.ext
  #
  # This is done to allow nvrtc to find the file when
  # it encounters `#include "<filename.ext>"
  #
  #
  set(input_files ${ARGN})
  list(TRANSFORM input_files PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
  set(output_file ${CMAKE_CURRENT_BINARY_DIR}/embedded_sources_for_nvrtc.cpp)

  # Generates the header(s) with the inline fatbins
  add_custom_command(
    OUTPUT "${output_file}"
    COMMAND ${CMAKE_COMMAND}
      "-DINPUTS=${input_files}"
      "-DOUTPUT=${output_file}"
      -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_nvrtc_source.cmake
    VERBATIM
    DEPENDS ${ARGN}
    COMMENT "Converting C++ sources to embeded strings in C++ source"
    )

  # add those c++ sources to `nvrtc_target_to_embed_in`
  target_sources(${nvrtc_target_to_embed_in} PRIVATE ${output_file})
  target_compile_features(${nvrtc_target_to_embed_in} PRIVATE cxx_std_20)
  target_include_directories(${nvrtc_target_to_embed_in} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

