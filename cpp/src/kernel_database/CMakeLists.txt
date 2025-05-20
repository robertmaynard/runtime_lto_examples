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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/kernel_database)

include(cmake/embed_fatbins.cmake)

# ##################################################################################################
# * kernels  ---------------------------------------------------------------------------------------
add_library(kernels OBJECT
  kernels/grid_stride_3double.cu
  kernels/grid_stride_3float.cu
  )
target_compile_definitions(kernels PRIVATE BUILD_KERNELS)
target_compile_options(kernels PRIVATE -Xfatbin=--compress-all --compress-mode=size)
set_target_properties(kernels PROPERTIES
  CUDA_ARCHITECTURES "60-real"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_FATBIN_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_link_libraries(kernels PRIVATE CCCL::CCCL demo::common)

# ##################################################################################################
# * convert kernels to cpp sources for embedding  --------------------------------------------------
add_library(kernels_as_cpp_sources STATIC)
embed_fatbins(kernels_as_cpp_sources kernels)


# # ##################################################################################################
# # * kernel launch program  -------------------------------------------------------------------------
add_executable(kernel_database
  main.cu
  LaunchKernelDatabase.cu
  KernelPlanner.cu
)

target_compile_features(kernel_database PRIVATE cuda_std_20)
target_link_libraries(kernel_database PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,kernels_as_cpp_sources>)
# target_link_libraries(kernel_database PRIVATE kernels_as_cpp_sources)
target_link_libraries(kernel_database PRIVATE CUDA::nvJitLink CCCL::CCCL)
# set_target_properties(kernel_database PROPERTIES
#   CUDA_ARCHITECTURES "60-real"
#   CUDA_SEPARABLE_COMPILATION OFF)

# This shouldn't exist in production code, and instead be a dlopen wrapper
target_link_libraries(kernel_database PRIVATE CUDA::cuda_driver)

# Background reading
# https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/
# https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/
# https://github.com/NVIDIA/cccl/blob/3f5016148f269513e3e9a252e65c8dd3f81e0f1b/docs/cuda_parallel/developer_overview.rst
