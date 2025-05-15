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


template<typename... Args, typename B>
extern void __device__ compute(Args... args);

template<class T>
T load(T t, std::size_t) {
  return t;
}
template<class T>
T& load(T* t, std::size_t i) {
  return t[i];
}

template<typename A>
__global__ void grid_stride(A a,size_t n) {
  auto tidx = common::grid_1d::global_thread_id();
  auto const stride = common::grid_1d::grid_stride();
  while (tidx < n) {
    compute( load(a,tidx) );
    tidx += stride;
  }
}
template<typename A, typename B>
__global__ void grid_stride(A a, B b, size_t n) {
  auto tidx = common::grid_1d::global_thread_id();
  auto const stride = common::grid_1d::grid_stride();
  while (tidx < n) {
    compute( load(a,tidx), load(b,tidx) );
    tidx += stride;
  }
}
template<typename A, typename B, typename C>
__global__ void grid_stride(A a, B b, C c, size_t n) {
  auto tidx = common::grid_1d::global_thread_id();
  auto const stride = common::grid_1d::grid_stride();
  while (tidx < n) {
    compute( load(a,tidx), load(b,tidx), load(c,tidx) );
    tidx += stride;
  }
}
