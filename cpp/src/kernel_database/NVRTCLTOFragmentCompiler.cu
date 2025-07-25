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

#include "NVRTCLTOFragmentCompiler.h"

#include <iostream>

#include "cuda.h"
#include "nvrtc.h"


std::vector<std::string> get_include_dirs_from_cmake(); //generated by generate_nvrtc_include_cpp_file

#define NVRTC_SAFE_CALL(_call)                              \
  do {                                                      \
    nvrtcResult result = _call;                             \
    if (result != NVRTC_SUCCESS) {                          \
      std::cerr << "\nerror: " #_call " failed with error " \
                << nvrtcGetErrorString(result) << '\n';     \
      exit(1);                                              \
    }                                                       \
  } while (0)



NRTCLTOFragmentCompiler::NRTCLTOFragmentCompiler() {

  int device=0;
  int major = 0;
  int minor = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(
      &minor, cudaDevAttrComputeCapabilityMinor, device);

  auto include_directories = get_include_dirs_from_cmake();
  this->standard_compile_opts.resize(4 + include_directories.size());

  std::size_t i=0;
  this->standard_compile_opts[i++]=std::string{"-arch=sm_" + std::to_string((major * 10 + minor))};
  this->standard_compile_opts[i++]=std::string{"-dlto"};
  this->standard_compile_opts[i++]=std::string{"-rdc=true"};
  this->standard_compile_opts[i++]=std::string{"-default-device"};
  for(const auto& o : include_directories) {
    this->standard_compile_opts[i++] = o.c_str();
  }
}

LTOIRFromNVRTC NRTCLTOFragmentCompiler::compile(
  std::string const& code,
  std::unordered_map<std::string, char const*> const& nvrtc_includes) const
{
  nvrtcProgram prog;

  int numHeaders = static_cast<int>(nvrtc_includes.size());
  std::vector<const char*> header_contents(numHeaders);
  std::vector<const char*> header_names(numHeaders);
  {
    std::size_t i = 0;
    for (const auto& o : nvrtc_includes) {
      header_contents[i] = o.second;
      header_names[i] = o.first.c_str();
      ++i;
    }
  }

  NVRTC_SAFE_CALL(nvrtcCreateProgram(
      &prog, code.c_str(), "nvrtc_lto_fragment",
      numHeaders,
      header_contents.data(),  // array of pointers to header contents
      header_names.data()));   // array of pointers to header names

  std::vector<const char*> compile_opts(this->standard_compile_opts.size());
  {
    std::size_t i = 0;
    for (const auto& o : this->standard_compile_opts) {
      compile_opts[i++] = o.c_str();
    }
  }

  nvrtcResult compileResult =
      nvrtcCompileProgram(prog,                  // prog
                          compile_opts.size(),   // numOptions
                          compile_opts.data());  // options

  if (compileResult != NVRTC_SUCCESS) {
    // Obtain compilation log from the program.
    size_t log_size;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::unique_ptr<char[]> log{new char[log_size]};
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
    std::cerr << "nvrtrc compile error log: \n";
    std::cerr << log.get() << '\n';
    exit(1);
  }

  // Obtain generated LTO IR from the program.
  std::size_t ltoIRSize;
  NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoIRSize));

  LTOIRFromNVRTC result{ltoIRSize};
  nvrtcGetLTOIR(prog, result.program.get());

  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  return std::move(result);
}
