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

#include <cuda/std/__algorithm/max.h>

template<typename T, typename U, typename R>
__device__ void compute(T x, U y, R& r) {
  r = ::cuda::std::max(x, y); //compile-error on purpose
}

template __device__ void compute<float, float, float>(float, float, float&);
template __device__ void compute<double, double, double>(double, double, double&);
