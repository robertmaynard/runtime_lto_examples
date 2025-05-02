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

#include <random>
#include <vector>

#include "cuda.h"
#include "cuda_wrapper.hpp"
#include "grid_1d.hpp"
#include "saxpy_setup.h"

#include <cub/detail/launcher/cuda_driver.cuh>


CUlibrary load_fatbins(CUdevice, std::vector<std::string>);


int main(int, char**) {

  CUdevice cuda_device;
  CUcontext cuda_context;
  cuInit(0);
  DEMO_CUDA_TRY(cuDeviceGet(&cuda_device, 0));
  DEMO_CUDA_TRY(cuCtxCreate(&cuda_context, 0, cuda_device));

  rmm::cuda_stream stream{};
  saxpy_memory saxpy{stream};

  std::cout << "Start loading SAXPY LTO FATBINS \n";
  auto cuda_lib = load_fatbins(
      cuda_device,
      std::vector<std::string>{"saxpy_compute.fatbin", "saxpy_grid_stride.fatbin"});
  std::cout << "Finished loading \n";

  // Build up a launcher for kernels with the same grid, block, etc
  const auto n = static_cast<std::int64_t>(saxpy.x->size());

  common::grid_1d grid_config{n, common::block_size};
  const auto grid_size = static_cast<std::uint32_t>(grid_config.num_blocks);
  constexpr size_t shared_mem = 0;
  cub::detail::CudaDriverLauncher launcher{
    dim3{grid_size, 1, 1},
    dim3{common::block_size, 1, 1},
    shared_mem,
    stream.value()};

  // Get kernel pointer out of the library
  CUkernel kernel;
  std::cout << "Launch SAXPY with " << saxpy.x->size() << " elements\n";
  DEMO_CUDA_TRY(cuLibraryGetKernel(&kernel, cuda_lib, "saxpy"));
  launcher.doit(kernel, saxpy.x->begin(), saxpy.y->begin(), n);

  std::vector<float> host_y;
  host_y.resize(n);

  cudaMemcpyAsync(host_y.data(), saxpy.y->begin(), n * sizeof(float),
                  cudaMemcpyDefault, stream.value());

  cudaStreamSynchronize(stream.value());

  // DEMO_CUDA_TRY(cuCtxDestroy(cuda_context));
  return 0;
}
