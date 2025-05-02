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

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <thrust/transform.h>

void saxypy_thrust() {

  using cuda_async_mr = rmm::mr::cuda_async_memory_resource;
  constexpr std::size_t array_size = 1<<24;
  constexpr std::size_t bytes = array_size * sizeof(float);
  constexpr auto pool_init_size{bytes * 3 + 1<<9};
  cuda_async_mr mr{pool_init_size};

  std::default_random_engine generator;
  auto constexpr range_min{10.f};
  auto constexpr range_max{100000.f};
  std::uniform_real_distribution<float> distribution(range_min, range_max);

  std::vector<float> host_x{array_size};
  std::vector<float> host_y{array_size};
  std::fill(host_x.begin(), host_x.end(), [&]() { return distribution(generator); });
  std::fill(host_y.begin(), host_y.end(), [&]() { return distribution(generator); });


  rmm::cuda_stream stream{};
  rmm::device_uvector<float> d_x{array_size, stream, mr};
  rmm::device_uvector<float> d_y{array_size, stream, mr};

  cudaMemcpyAsync(d_x.begin(), host_x.data(), bytes, cudaMemcpyDefault, stream.value());
  cudaMemcpyAsync(d_y.begin(), host_y.data(), bytes, cudaMemcpyDefault, stream.value());

  thrust::transform(rmm::exec_policy(stream),
    d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),
    [] __device__(float x, float y) { return 2.0f * x + y; });

  cudaMemcpyAsync(host_y.data(), d_y.begin(), bytes, cudaMemcpyDefault, stream.value());

  cudaStreamSynchronize(stream.value());
  return;
}
