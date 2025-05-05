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

include(${rapids-cmake-dir}/cpm/rmm.cmake)
set(CUDA_STATIC_RUNTIME ON)
rapids_cpm_rmm()


# ##################################################################################################
# * kernels  ---------------------------------------------------------------------------------------
add_library(adv_kernels OBJECT
  kernels/grid_stride_PfPfPf.cu
  kernels/grid_stride_PdPdPd.cu
  kernels/saxpy_compute_fff.cu
  kernels/saxpy_compute_ddd.cu

  )
target_compile_options(adv_kernels PRIVATE -Xfatbin=--compress-all --compress-mode=size)
set_target_properties(adv_kernels PROPERTIES
  CUDA_ARCHITECTURES "60-real"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_FATBIN_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_link_libraries(adv_kernels PRIVATE CCCL::CCCL demo::common)


# ##################################################################################################
# * kernel launch program  -------------------------------------------------------------------------
add_executable(adv_kernel_launches
  adv_kernel_launches.cu
  kernel_lookup.cpp
)
target_compile_features(adv_kernel_launches PRIVATE cuda_std_20)
target_link_libraries(adv_kernel_launches PRIVATE CUDA::nvJitLink CCCL::CCCL rmm::rmm demo::common)
set_target_properties(adv_kernel_launches PROPERTIES
  CUDA_ARCHITECTURES "60-real"
  CUDA_SEPARABLE_COMPILATION OFF)

# This shouldn't exist in production code, and instead be a dlopen wrapper
target_link_libraries(adv_kernel_launches PRIVATE CUDA::cuda_driver)

# Background reading
# https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/
# https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/
# https://github.com/NVIDIA/cccl/blob/3f5016148f269513e3e9a252e65c8dd3f81e0f1b/docs/cuda_parallel/developer_overview.rst
