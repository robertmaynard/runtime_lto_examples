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

#include <cuda/std/type_traits>

namespace common {

namespace detail {
template <typename I>
__host__ __device__ constexpr I div_rounding_up_safe(cuda::std::false_type,
                                                     I dividend,
                                                     I divisor) noexcept {
  // TODO: This could probably be implemented faster
  return (dividend > divisor)
             ? 1 + div_rounding_up_unsafe(dividend - divisor, divisor)
             : (dividend > 0);
}

template <typename I>
__host__ __device__ constexpr I div_rounding_up_safe(cuda::std::true_type,
                                                     I dividend,
                                                     I divisor) noexcept {
  auto quotient = dividend / divisor;
  auto remainder = dividend % divisor;
  return quotient + (remainder != 0);
}
}  // namespace detail

template <typename I>
__host__ __device__ constexpr I div_rounding_up_safe(I dividend,
                                                     I divisor) noexcept {
  using i_is_a_signed_type =
      cuda::std::bool_constant<cuda::std::is_signed_v<I>>;
  return detail::div_rounding_up_safe(i_is_a_signed_type{}, dividend, divisor);
}


static constexpr int warp_size{32};
static constexpr int block_size{256};
using thread_index_type = std::int64_t;
class grid_1d {
 public:
  thread_index_type const num_threads_per_block;
  thread_index_type const num_blocks;

  grid_1d(thread_index_type overall_num_elements,
          thread_index_type num_threads_per_block,
          thread_index_type elements_per_thread = 1)
      : num_threads_per_block(num_threads_per_block),
        num_blocks(div_rounding_up_safe(
            overall_num_elements,
            elements_per_thread * num_threads_per_block)) {}

  __device__ static constexpr thread_index_type global_thread_id(
      thread_index_type thread_id, thread_index_type block_id,
      thread_index_type num_threads_per_block) {
    return thread_id + block_id * num_threads_per_block;
  }

  static __device__ thread_index_type global_thread_id() {
    return global_thread_id(threadIdx.x, blockIdx.x, blockDim.x);
  }

  template <thread_index_type num_threads_per_block>
  static __device__ thread_index_type global_thread_id() {
    return global_thread_id(threadIdx.x, blockIdx.x, num_threads_per_block);
  }

  __device__ static constexpr thread_index_type grid_stride(
      thread_index_type num_threads_per_block,
      thread_index_type num_blocks_per_grid) {
    return num_threads_per_block * num_blocks_per_grid;
  }

  static __device__ thread_index_type grid_stride() {
    return grid_stride(blockDim.x, gridDim.x);
  }

  template <thread_index_type num_threads_per_block>
  static __device__ thread_index_type grid_stride() {
    return grid_stride(num_threads_per_block, gridDim.x);
  }
};
}  // namespace common
