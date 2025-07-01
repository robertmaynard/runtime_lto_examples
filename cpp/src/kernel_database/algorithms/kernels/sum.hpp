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

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>

template<typename T>
__device__ void compute(T& input, T& result)
{
  constexpr int block_size = 256;
  using BlockReduce = cub::BlockReduce<T, block_size>;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  T sum = BlockReduce(temp_storage).Sum(input);

  if (threadIdx.x == 0)
  {
    cuda::atomic_ref<T, cuda::thread_scope_device> atomic_result(result);
    atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
  }
}
