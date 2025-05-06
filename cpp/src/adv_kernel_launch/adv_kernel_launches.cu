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


#include "kernel_lookup.h"
#include "saxpy_setup.h"


#include <cub/detail/launcher/cuda_driver.cuh>

CUlibrary load_nvrtc_and_fatbins(CUdevice, std::vector<std::string>, std::vector<std::string>);

template<typename T>
void run_saxpy(CUdevice cuda_device,
               cub::detail::CudaDriverLauncher& launcher,
               std::string const& algorithm_text,
               std::string const& nvrtc_file_path,
               std::size_t array_size,
               rmm::cuda_stream_view stream) {

  saxpy_memory<T> saxpy{array_size, stream};

  const auto n = static_cast<std::int64_t>(array_size);

  auto entry_points = build_grid_database();
  auto saxpy_impls = build_saxpy_database();

  std::vector<std::string> fatbins;
  std::vector<std::string> nvrtc_files;

  auto entry_point = find_entry(entry_points, saxpy.x->begin(), saxpy.y->begin(), saxpy.y->begin());
  fatbins.push_back(entry_point.file);

  if (nvrtc_file_path.empty()) {
    fatbins.push_back(saxpy_impls[entry_point.compute_key]);
  } else {
    nvrtc_files.push_back(nvrtc_file_path);
  }

  std::cout << "Start loading " << algorithm_text << " LTO-IR \n";
  auto cuda_lib = load_nvrtc_and_fatbins(cuda_device, nvrtc_files, fatbins);
  std::cout << "Finished loading \n";

  auto kernel = entry_point.get_kernel(cuda_lib);
  launcher.doit(kernel, saxpy.x->begin(), saxpy.y->begin(), saxpy.y->begin(), n);

  // return the results to the host
  std::vector<T> host_y;
  host_y.resize(n);
  cudaMemcpyAsync(host_y.data(), saxpy.y->begin(), n * sizeof(T),
                  cudaMemcpyDefault, stream.value());
}

// program command line syntax
// adv_kernel_launches -> float no nvrtc
// adv_kernel_launches 'f' -> float no nvrtc
// adv_kernel_launches 'd' -> double no nvrtc
// adv_kernel_launches 'f' /path/to/nvrtc -> float with nvrtc
// adv_kernel_launches 'd' /path/to/nvrtc -> float with nvrtc
int main(int argc, char** argv) {

  CUdevice cuda_device;
  CUcontext cuda_context;
  cuInit(0);
  DEMO_CUDA_TRY(cuDeviceGet(&cuda_device, 0));
  DEMO_CUDA_TRY(cuCtxCreate(&cuda_context, 0, cuda_device));

  rmm::cuda_stream stream{};

  constexpr std::size_t array_size = 1 << 27;

  // Build up a launcher for kernels with the same grid, block, etc
  const auto n = static_cast<std::int64_t>(array_size);
  common::grid_1d grid_config{n, common::block_size};
  const auto grid_size = static_cast<std::uint32_t>(grid_config.num_blocks);
  constexpr size_t shared_mem = 0;
  cub::detail::CudaDriverLauncher launcher{
    dim3{grid_size, 1, 1},
    dim3{common::block_size, 1, 1},
    shared_mem,
    stream.value()};

  std::string nvrtc_file_path;
  bool nvrtc = (argc == 3);
  bool do_float = true;
  if (argc >= 2) {
    if (argv[1][0] == 'd') {
      do_float = false;
    }
  }
  if (nvrtc) {
    nvrtc_file_path = std::string(argv[2]);
  }

  if (do_float) {
    run_saxpy<float>(cuda_device, launcher, "float saxpy", nvrtc_file_path,
                     array_size, stream);
  } else if (!do_float) {
    run_saxpy<double>(cuda_device, launcher, "double saxpy", nvrtc_file_path,
                      array_size, stream);
  }

  cudaStreamSynchronize(stream.value());
  return 0;
}
