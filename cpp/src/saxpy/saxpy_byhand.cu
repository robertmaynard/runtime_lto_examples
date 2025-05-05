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

#include "saxpy_setup.h"
#include "grid_1d.hpp"


__global__ void saxpy_fast(float* x, float* y, size_t n) {
  auto tidx = common::grid_1d::global_thread_id();
  auto const stride = common::grid_1d::grid_stride();
  while (tidx < n) {
    y[tidx] = 2.0f * x[tidx] + y[tidx];
    tidx += stride;
  }
}

__global__ void saxpy_pass_1(float* x, size_t n) {
  auto tidx = common::grid_1d::global_thread_id();
  auto const stride = common::grid_1d::grid_stride();
  while (tidx < n) {
    x[tidx] = 2.0f * x[tidx];
    tidx += stride;
  }
}

__global__ void saxpy_pass_2(float* x, float* y, size_t n) {
  auto tidx = common::grid_1d::global_thread_id();
  auto const stride = common::grid_1d::grid_stride();
  while (tidx < n) {
    y[tidx] = x[tidx] + y[tidx];
    tidx += stride;
  }
}


int main(int, char**) {
  rmm::cuda_stream stream{};
  saxpy_memory saxpy{stream};

  const auto n = static_cast<std::int64_t>(saxpy.x->size());
  common::grid_1d grid_config{n, common::block_size};
  const auto grid_size = static_cast<std::uint32_t>(grid_config.num_blocks);

  std::cout << "Launch <<< >>> fast saxpy with " << n << " elements\n";
  saxpy_fast<<<grid_size, common::block_size, 0, stream.value()>>>(
      saxpy.x->begin(), saxpy.y->begin(), n);

  std::cout << "Launch <<< >>> slow saxpy pass 1 with " << n << " elements\n";
  saxpy_pass_1<<<grid_size, common::block_size, 0, stream.value()>>>(
      saxpy.x->begin(), n);

  std::cout << "Launch <<< >>> slow saxpy pass 1 with " << n << " elements\n";
  saxpy_pass_2<<<grid_size, common::block_size, 0, stream.value()>>>(
      saxpy.x->begin(), saxpy.y->begin(), n);

  std::vector<float> host_y;
  host_y.resize(saxpy.y->size());

  cudaMemcpyAsync(host_y.data(), saxpy.y->begin(),
                  saxpy.y->size() * sizeof(float), cudaMemcpyDefault,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  return 0;
}
