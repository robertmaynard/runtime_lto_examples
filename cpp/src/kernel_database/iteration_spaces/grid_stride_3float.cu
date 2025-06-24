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

#ifdef BUILD_KERNELS

#include "grid_stride.h"

template __global__ void grid_stride(float*, float*,float*, size_t);

#else

#include "embedded_fatbins.h"
#include "../detail/RegisterLaunchKernel.h"

__attribute__((__constructor__)) static void register_3float() {
  registerLaunchKernel<float*, float*, float*>(embedded_grid_stride_3float);
}

#endif
