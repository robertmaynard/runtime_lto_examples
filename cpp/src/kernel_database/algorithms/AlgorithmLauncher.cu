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


#include "AlgorithmLauncher.h"

#include <iostream>

AlgorithmLauncher::AlgorithmLauncher(CUlibrary l, CUkernel k, LaunchType t)
    : library{l}, kernel{k}, launch_type{t}, stream{}, shared_mem{0} {}

void AlgorithmLauncher::exec_info(cudaStream_t stream, std::size_t shared_mem)
{
  this->stream = stream;
  this->shared_mem = shared_mem;
}

void AlgorithmLauncher::call(std::int64_t length, void** kernel_args) {

  static constexpr int block_size{256};
  dim3 grid;
  dim3 block;

  if(this->launch_type == GRID_1D) {
    // Build up a launcher for kernels with the same grid, block, etc
    common::grid_1d grid_config{length, block_size};
    const auto grid_size = static_cast<std::uint32_t>(grid_config.num_blocks);
    grid = dim3{grid_size, 1, 1};
    block = dim3{block_size, 1, 1};
  } else {
    std::cerr<< "non grid1d launch types implemented" << std::endl;
    std::exit(1);
  }

  CUlaunchAttribute attribute[1];
  attribute[0].id                                           = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attribute[0].value.programmaticStreamSerializationAllowed = 1;

  CUlaunchConfig config{};
  config.gridDimX       = grid.x;
  config.gridDimY       = grid.y;
  config.gridDimZ       = grid.z;
  config.blockDimX      = block.x;
  config.blockDimY      = block.y;
  config.blockDimZ      = block.z;
  config.sharedMemBytes = shared_mem;
  config.hStream        = stream;
  config.attrs          = attribute;
  config.numAttrs       = 1;

  cuLaunchKernelEx(&config, (CUfunction)kernel, kernel_args, 0);
}
