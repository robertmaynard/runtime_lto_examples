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

include(../../cmake/rapids_config.cmake)

include(rapids-cmake)
include(rapids-cuda)
include(rapids-cpm)
include(rapids-find)

rapids_cuda_init_architectures(runtime_lto_demo)

project(
  kernel_database
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA)

rapids_cmake_build_type(Release)

include(cmake/embed_fatbins.cmake)
include(cmake/embed_sources_for_nvrtc.cmake)
include(cmake/nvrtc_includes_as_cpp_file.cmake)


# ##################################################################################################
# * dependencies -----------------------------------------------------------------------------------
rapids_cpm_init()
rapids_find_package(CUDAToolkit REQUIRED)

include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl()

add_subdirectory(iteration_spaces)
add_subdirectory(algorithms)

# # ##################################################################################################
# # * kernel launch program  -------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/kernel_database)
add_executable(kernel_database
  main.cu
  LaunchKernelDatabase.cu
  LaunchKernelEntry.cu
  NVRTCLTOFragmentCompiler.cu
)
target_include_directories(kernel_database PRIVATE "${PROJECT_SOURCE_DIR}")
generate_nvrtc_include_cpp_file(kernel_database)

target_compile_features(kernel_database PRIVATE cuda_std_20)
target_link_libraries(kernel_database PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,iteration_kernels_as_cpp_sources>)
target_link_libraries(kernel_database PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,algo_kernels_as_cpp_sources>)
target_link_libraries(kernel_database PRIVATE CUDA::nvJitLink CUDA::nvrtc CCCL::CCCL)
set_target_properties(kernel_database PROPERTIES
  CUDA_ARCHITECTURES "native"
  CUDA_SEPARABLE_COMPILATION OFF)


# Background reading
# https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/
# https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/
# https://github.com/NVIDIA/cccl/blob/3f5016148f269513e3e9a252e65c8dd3f81e0f1b/docs/cuda_parallel/developer_overview.rst
