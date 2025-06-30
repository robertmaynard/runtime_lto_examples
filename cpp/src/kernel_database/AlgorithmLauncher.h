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

#include "iteration_spaces/LaunchTypes.h"

#include <cstdint>

#include "cuda.h"

struct AlgorithmLauncher {

  AlgorithmLauncher(CUlibrary l, CUkernel k, LaunchType t);

  void exec_info(cudaStream_t stream, std::size_t shared_mem);

  template <typename... Args>
  void operator()(std::int64_t length, Args&&... args) {
    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call(length, kernel_args);
  }

private:
  void call(std::int64_t length, void** args);
  CUlibrary library;
  CUkernel kernel;
  LaunchType launch_type;
  cudaStream_t stream;
  std::size_t shared_mem;
};
