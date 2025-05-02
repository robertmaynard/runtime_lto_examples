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

#include <cub/detail/launcher/cuda_driver.cuh>

CUlibrary load_fatbins(CUdevice, std::vector<std::string>);

// NOTICES:
// When converting this to production code we need to use a
// dlopen wrapper around cuda driver so that we can gracefully fail
// at runtime
void load_and_call_helloworld(CUdevice cuda_device) {
  std::cout << "Start loading Hello World LTO FATBINS \n";
  auto cuda_lib = load_fatbins(
    cuda_device,
    std::vector<std::string>{"hello_world_kernel.fatbin"});
  std::cout << "Finished loading \n";

  //Build up a launcher for kernels with the same grid, block, etc
  constexpr dim3 grid = {1, 1, 1};
  constexpr dim3 block = {1, 1, 1};
  constexpr size_t shared_mem = 0;
  CUstream stream;
  DEMO_CUDA_TRY(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  cub::detail::CudaDriverLauncher launcher{grid, block, shared_mem, stream};

  // Get kernel pointer out of the library
  CUkernel kernel;
  std::cout << "Launch Hello World \n";
  DEMO_CUDA_TRY(cuLibraryGetKernel(&kernel, cuda_lib, "hello_world"));
  launcher.doit(kernel);

  DEMO_CUDA_TRY(cuStreamSynchronize(stream));
  DEMO_CUDA_TRY(cuStreamDestroy(stream));
  DEMO_CUDA_TRY(cuLibraryUnload(cuda_lib));
}
