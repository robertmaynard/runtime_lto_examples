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

#include <memory>
#include <vector>

#include "cuda_wrapper.hpp"

#include "cuda.h"

std::unique_ptr<CUmodule> load_fatbins(CUdevice, std::vector<std::string>);


// NOTICES:
// When converting this to production code we need to use a
// dlopen wrapper around cuda driver so that we can gracefully fail
// at runtime

int main() {

  CUdevice cuda_device;
  CUcontext cuda_context;
  cuInit(0);
  DEMO_CUDA_TRY(cuDeviceGet(&cuda_device, 0));
  DEMO_CUDA_TRY(cuCtxCreate(&cuda_context, 0, cuda_device));

  std::cout << "Started Loading LTO FATBINS \n";
  auto cuda_module = load_fatbins(cuda_device, std::vector<std::string>{"kernels.fatbin"});
  std::cout << "Finished Loading LTO FATBINS \n";

  return 0;
}
