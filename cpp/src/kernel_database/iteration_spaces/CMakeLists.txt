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
add_library(iteration_kernels OBJECT
  grid_stride_3double.cu
  grid_stride_3float.cu
  )
target_compile_definitions(iteration_kernels PRIVATE BUILD_KERNELS)
target_compile_options(iteration_kernels PRIVATE -Xfatbin=--compress-all --compress-mode=size)
set_target_properties(iteration_kernels PROPERTIES
  CUDA_ARCHITECTURES "60-real"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_FATBIN_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_link_libraries(iteration_kernels PRIVATE CCCL::CCCL)

# ##################################################################################################
# * convert kernels to cpp sources for embedding  --------------------------------------------------
add_library(iteration_kernels_as_cpp_sources STATIC)
embed_fatbins(iteration_kernels_as_cpp_sources iteration_kernels)

# We need to leverage gcc -c23 #embed here!
embed_sources_for_nvrtc(iteration_kernels_as_cpp_sources
  grid_1d.hpp
  grid_stride.hpp)
