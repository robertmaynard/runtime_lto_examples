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

#include "kernel_lookup.h"
#include "cuda_wrapper.hpp"


CUkernel KernelEntry::get_kernel(CUlibrary lib) const {
  //The trick way is to use cuLibraryEnumerateKernel and cuKernelGetName
  //Since our modules only have 1 kernel we can fetch the kernel without
  //ever knowing the mangled name
  //
  // There should be a better way, but I need to do some research
  unsigned int count = 1;
  std::unique_ptr<CUkernel[]> kernels_{new CUkernel[count]};
  cuLibraryEnumerateKernels(kernels_.get(), count, lib);
  return kernels_[0];
}


// This simulates a database of grid launch fatbins
//
// projects use a python code generator via `add_custom_target` to build this logic
// from some set of input files.
//
// Note: while we use `.fatbin` we can also load from C strings
KernelDatabase build_grid_database() {
  KernelDatabase db;
  {
  auto key = make_db_key<float*, float*, float*>();
  auto compute_key = make_db_key<float, float, float>(); //std::remove_pointer ?
  db.insert(  KernelEntry{ key,
                           compute_key,
                           "grid_stride_" + key + ".fatbin"});
  }

  {
  auto key = make_db_key<double*, double*, double*>();
  auto compute_key = make_db_key<double, double, double>();
  db.insert(  KernelEntry{ key,
                           compute_key,
                           "grid_stride_" + key + ".fatbin"});
  }

  return db;
}

// This simulates a database of grid launch fatbins
ComputeDatabase build_saxpy_database() {
  ComputeDatabase saxpy;

  {
  auto key = make_db_key<float, float, float>();
  saxpy[key] = "saxpy_compute_" + key + ".fatbin";
  }

  {
  auto key = make_db_key<double, double, double>();
  saxpy[key] = "saxpy_compute_" + key + ".fatbin";
  }

  return saxpy;
}
