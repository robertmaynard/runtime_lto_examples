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
#include <iterator>
#include <memory>
#include <string>
#include <fstream>
#include <vector>

#include "cuda_wrapper.hpp"

#include "cuda.h"
#include <nvJitLink.h>
#include <nvrtc.h>

#define NVRTC_SAFE_CALL(_call)                              \
  do {                                                      \
    nvrtcResult result = _call;                             \
    if (result != NVRTC_SUCCESS) {                          \
      std::cerr << "\nerror: " #_call " failed with error " \
                << nvrtcGetErrorString(result) << '\n';     \
      exit(1);                                              \
    }                                                       \
  } while (0)

namespace {
// We can make a better RAII wrapper around nvjitlinkhandle
void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result) {
  if (result != NVJITLINK_SUCCESS) {
    std::cerr << "\n nvJITLink failed with error " << result << '\n';
    size_t log_size = 0;
    result = nvJitLinkGetErrorLogSize(handle, &log_size);
    if (result == NVJITLINK_SUCCESS && log_size > 0) {
      std::unique_ptr<char[]> log{new char[log_size]};
      result = nvJitLinkGetErrorLog(handle, log.get());
      if (result == NVJITLINK_SUCCESS) {
        std::cerr << "nvJITLink error log: " << log.get() << '\n';
      }
    }
    exit(1);
  }
}

struct LTOIRFromNVRTC {
  std::size_t size;
  std::unique_ptr<char[]> data;

  LTOIRFromNVRTC(std::size_t s)
      : size(s), data{std::make_unique<char[]>(s)} {}
};

LTOIRFromNVRTC compile_nvrtc_to_ltoir(CUdevice device,
                                      std::string nvrtc_file_path,
                                      std::string program_name,
                                      std::string compile_target) {
  nvrtcProgram prog;

  std::ifstream file{nvrtc_file_path};
  if(!file.is_open())
  {
    std::cerr << "Unable to open file " << nvrtc_file_path << "\n";
    exit(1);
  }

  std::string code{std::istreambuf_iterator<char>(file),
                   std::istreambuf_iterator<char>()};
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, code.c_str(), program_name.c_str(), 0,
                                     nullptr, nullptr));

  const char *compile_opts[] = {"-dlto", "-rdc=true", compile_target.c_str()};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,           // prog
                                                  3,              // numOptions
                                                  compile_opts);  // options
  if (compileResult != NVRTC_SUCCESS) {
    // Obtain compilation log from the program.
    size_t log_size;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::unique_ptr<char[]> log{new char[log_size]};
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
    std::cerr <<  "nvrtrc compile error log: \n";
    std::cerr << log.get() << '\n';
    exit(1);
  }

  // Obtain generated LTO IR from the program.
  std::size_t ltoIRSize;
  NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoIRSize));

  LTOIRFromNVRTC result{ltoIRSize};
  nvrtcGetLTOIR(prog, result.data.get());

  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  return std::move(result);
}

}  // namespace


CUlibrary load_nvrtc_and_fatbins(CUdevice device,
                                 std::vector<std::string> nvrtc_file_paths,
                                 std::vector<std::string> fatbin_names) {
  int major = 0;
  int minor = 0;
  DEMO_CUDA_TRY(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  DEMO_CUDA_TRY(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

  // Load the generated LTO IR and link them together
  nvJitLinkHandle handle;
  const char *lopts[] = {"-lto", archs.c_str()};
  auto result = nvJitLinkCreate(&handle, 2, lopts);
  check_nvjitlink_result(handle, result);

  for (auto file_path : nvrtc_file_paths) {
    auto file_name = file_path.substr(file_path.find_last_of("\\") + 1);
    auto ltoIR = compile_nvrtc_to_ltoir(device, file_path, file_name, archs);
    result =
        nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoIR.data.get(),
                         ltoIR.size, file_name.c_str());
    check_nvjitlink_result(handle, result);
    std::cout << "\t\tadding " << file_name << " to the nvJITLink module \n";
  }

  for (auto name : fatbin_names) {
    // need to compute the path to `name`
    result = nvJitLinkAddFile(handle, NVJITLINK_INPUT_FATBIN, name.c_str());
    check_nvjitlink_result(handle, result);
    std::cout << "\t\tadding " << name << " to the nvJITLink module \n";
  }

  // Call to nvJitLinkComplete causes linker to link together all the LTO-IR
  // modules perform any optimizations and generate cubin from it.
  std::cout << "\tStarted LTO runtime linking \n";
  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);
  std::cout << "\tCompleted LTO runtime linking \n";

  // get cubin from nvJitLink
  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  check_nvjitlink_result(handle, result);

  // cubin is linked, so now load it
  CUlibrary library;
  DEMO_CUDA_TRY(cuLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0,
                                  nullptr, nullptr, 0));
  return library;
}

CUlibrary load_fatbins(CUdevice device, std::vector<std::string> fatbin_names) {
  // load the requested FATBINs into a CUDA driver module
  std::vector<std::string> nvrtc_files;
  return load_nvrtc_and_fatbins(device, nvrtc_files, fatbin_names);
}
