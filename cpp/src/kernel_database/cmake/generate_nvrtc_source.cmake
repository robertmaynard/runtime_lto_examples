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


set(file_contents
[=[
  #include "detail/RegisterNVRTCHeader.h"
]=])
set(template_contents
[=[
__attribute__((__constructor__)) static void register_nvrtc_@header_name@()
  {
  static std::string file_contents = R"(
    @contents@
  )";
  std::string name{"@header_name@"};
  std::string include_name{"@header_name@@header_ext@"};
  registerNVRTCFragmentInclude(name, include_name, file_contents.c_str());
  }
]=])
foreach(f ${INPUTS})
  get_filename_component(header_name ${f} NAME_WE)
  get_filename_component(header_ext ${f} EXT)
  get_filename_component(file_dir ${f} DIRECTORY)

  # We need to read the file and embed it as a textual string
  file(READ ${f} contents)

  string(CONFIGURE "${template_contents}" generated_cxx @ONLY)
  string(APPEND file_contents "\n${generated_cxx}\n")

endforeach()
file(WRITE "${OUTPUT}" "${file_contents}")
