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
set(CUDA_STATIC_RUNTIME OFF)
rapids_cpm_rmm()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/mandlebrot/)

find_package(OpenGL)
find_package(GLUT)

if(NOT (TARGET GLUT::GLUT AND TARGET OpenGL::GL))
  message(FATAL_ERROR "you need to install opengl+glut (for example `sudo apt install libglut-dev`)")
endif()

# ##################################################################################################
# * kernels  ---------------------------------------------------------------------------------------
add_library(mandlebrot_kernels OBJECT
  kernels/mandlebrot_grid_stride.cu
  kernels/mandlebrot_compute.cu
  )
target_compile_options(mandlebrot_kernels PRIVATE -Xfatbin=--compress-all --compress-mode=size)
set_target_properties(mandlebrot_kernels PROPERTIES
  CUDA_ARCHITECTURES "60-real"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_FATBIN_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_link_libraries(mandlebrot_kernels PRIVATE CCCL::CCCL demo::common)
add_copy_fatbins_command(mandlebrot_kernels)


# ##################################################################################################
# * kernel launch program  -------------------------------------------------------------------------
add_executable(mandlebrot mandlebrot_lto.cu)
target_compile_features(mandlebrot PRIVATE cuda_std_20)
target_include_directories(mandlebrot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mandlebrot PRIVATE CUDA::nvJitLink CCCL::CCCL rmm::rmm demo::common)
set_target_properties(mandlebrot PROPERTIES
  CUDA_ARCHITECTURES "60-real"
  CUDA_SEPARABLE_COMPILATION OFF)

# This shouldn't exist in production code, and instead be a dlopen wrapper
target_link_libraries(mandlebrot PRIVATE CUDA::cuda_driver)
target_link_libraries(mandlebrot PRIVATE rmm::rmm demo::common OpenGL::OpenGL OpenGL::GLU GLUT::GLUT)

# ##################################################################################################
# * baseline impl to compare against --------------------------------------------------------
add_executable(mandlebrot_byhand mandlebrot_byhand.cu)
target_compile_features(mandlebrot_byhand PRIVATE cuda_std_20)
target_compile_options(mandlebrot_byhand PRIVATE -extended-lambda --compress-mode=size)
target_include_directories(mandlebrot_byhand PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(mandlebrot_byhand PROPERTIES
  CUDA_ARCHITECTURES "70-real;75-real;80-real;86-real;90-real;100-real;120"
  CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(mandlebrot_byhand PRIVATE CUDA::cuda_driver)
target_link_libraries(mandlebrot_byhand PRIVATE rmm::rmm demo::common OpenGL::OpenGL OpenGL::GLU GLUT::GLUT)


target_compile_options(mandlebrot PRIVATE -g)
