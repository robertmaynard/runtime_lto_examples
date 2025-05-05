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
#include "grid_1d.hpp"


template<typename T, typename U, typename R>
extern void __device__ compute(T x, U y, R& r);

// We are using a templated kernel now to
// show how to compute C++ names at runtime
template<typename T, typename U, typename R>
__global__ void grid_stride(T* x, U* y, R* z, size_t n) {
  auto tidx = common::grid_1d::global_thread_id();
  auto const stride = common::grid_1d::grid_stride();
  while (tidx < n) {
    compute(x[tidx], y[tidx], z[tidx]);
    tidx += stride;
  }
}
