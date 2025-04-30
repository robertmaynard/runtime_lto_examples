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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(cmake/rapids_config.cmake)

include(rapids-cmake)
include(rapids-cuda)
include(rapids-cpm)
include(rapids-find)

set(CMAKE_CUDA_ARCHITECTURES "60-real")
rapids_cuda_init_architectures(runtime_lto_demo)

project(
  runtime_lto_demo
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA)
rapids_cmake_build_type(Release)

# ##################################################################################################
# * dependencies -----------------------------------------------------------------------------------

rapids_cpm_init()
rapids_find_package(CUDAToolkit REQUIRED)

include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl()

# ##################################################################################################
# * kernels  ---------------------------------------------------------------------------------------
add_library(kernels OBJECT src/kernels.cu)
set_target_properties(kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_FATBIN_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_link_libraries(kernels PRIVATE CCCL::CCCL)


# ##################################################################################################
# * kernel launch program  -------------------------------------------------------------------------
add_executable(demo src/main.cu src/launch.cpp src/load.cpp)
target_link_libraries(demo PRIVATE CUDA::nvJitLink)
target_link_libraries(demo PRIVATE CCCL::CCCL)
target_include_directories(demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# This shouldn't exist in production code, and instead be a dlopen wrapper
target_link_libraries(demo PRIVATE CUDA::cuda_driver)

# Background reading
# https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/
# https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/
# https://github.com/NVIDIA/cccl/blob/3f5016148f269513e3e9a252e65c8dd3f81e0f1b/docs/cuda_parallel/developer_overview.rst
