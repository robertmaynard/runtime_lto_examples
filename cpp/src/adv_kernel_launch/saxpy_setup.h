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

#pragma once

#include <memory>
#include <random>
#include <vector>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

template<typename T>
struct saxpy_memory {
  std::unique_ptr<rmm::mr::cuda_async_memory_resource> mr;
  std::unique_ptr<rmm::device_uvector<T>> x;
  std::unique_ptr<rmm::device_uvector<T>> y;

  saxpy_memory(std::size_t array_size, rmm::cuda_stream_view stream) {
    const std::size_t bytes = array_size * sizeof(T);
    const auto pool_init_size{bytes * 2};
    this->mr =
        std::make_unique<rmm::mr::cuda_async_memory_resource>(pool_init_size);

    std::default_random_engine generator;
    auto constexpr range_min{-1000.f};
    auto constexpr range_max{1000.f};
    std::uniform_real_distribution<T> distribution(range_min, range_max);

    this->x =
        std::make_unique<rmm::device_uvector<T>>(array_size, stream, mr.get());
    this->y =
        std::make_unique<rmm::device_uvector<T>>(array_size, stream, mr.get());

    std::vector<T> host_x;
    host_x.resize(array_size);
    std::generate(host_x.begin(), host_x.end(),
                  [&]() { return distribution(generator); });
    cudaMemcpyAsync(this->x->begin(), host_x.data(), bytes, cudaMemcpyDefault,
                    stream.value());

    std::vector<T> host_y;
    host_y.resize(array_size);
    std::generate(host_y.begin(), host_y.end(),
                  [&]() { return distribution(generator); });
    cudaMemcpyAsync(this->y->begin(), host_y.data(), bytes, cudaMemcpyDefault,
                    stream.value());
  }
};
