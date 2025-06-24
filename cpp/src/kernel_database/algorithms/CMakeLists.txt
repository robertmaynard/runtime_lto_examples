# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# ##################################################################################################
# * kernels  ---------------------------------------------------------------------------------------
add_library(algo_kernels OBJECT
  kernels/saxpy_float.cu
  kernels/saxpy_double.cu
  kernels/sum_int32.cu
  kernels/sum_int64.cu
  )
target_compile_definitions(algo_kernels PRIVATE BUILD_KERNELS)
target_compile_options(algo_kernels PRIVATE -Xfatbin=--compress-all --compress-mode=size)
set_target_properties(algo_kernels PROPERTIES
  CUDA_ARCHITECTURES "60-real"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_FATBIN_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_link_libraries(algo_kernels PRIVATE CCCL::CCCL)

add_library(algo_kernels_as_cpp_sources STATIC
  AlgorithmLauncher.cu
  AlgorithmFragmentDatabase.cu
  AlgorithmPlanner.cu
  ComputeFragmentEntry.cu
)
target_include_directories(algo_kernels_as_cpp_sources PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(algo_kernels_as_cpp_sources PRIVATE CUDA::cuda_driver)

# ##################################################################################################
# * convert kernels to cpp sources for embedding  --------------------------------------------------
embed_fatbins(algo_kernels_as_cpp_sources algo_kernels)

# We need to leverage gcc -c23 #embed here!
embed_sources_for_nvrtc(algo_kernels_as_cpp_sources
  kernels/saxpy.hpp
  kernels/sum.hpp)
