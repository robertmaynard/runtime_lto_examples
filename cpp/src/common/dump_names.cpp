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

#include <memory>
#include "cuda_wrapper.hpp"

void dump_all_kernel_names(CUlibrary cuda_lib) {
  unsigned int count = 9000;
  cuLibraryGetKernelCount(&count, cuda_lib);
  std::cout << "count " << count << std::endl;

  std::unique_ptr<CUkernel[]> kernels_{new CUkernel[count]};
  cuLibraryEnumerateKernels(kernels_.get(), count, cuda_lib);
  for (unsigned int i = 0; i < count; ++i) {
    const char* result;
    cuKernelGetName(&result, kernels_[i]);
    std::cout << result << std::endl;
  }
}

