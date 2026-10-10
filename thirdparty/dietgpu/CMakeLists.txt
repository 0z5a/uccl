# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(DIETGPU_EXPLICIT_CUDA_ARCHITECTURES OFF)
if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
    set(DIETGPU_EXPLICIT_CUDA_ARCHITECTURES ON)
endif()
project(dietgpu LANGUAGES CUDA CXX VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")

if(NOT DIETGPU_EXPLICIT_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

# Set default build type.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE
  )
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CTest)
if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
endif()

add_subdirectory(dietgpu/utils)
add_subdirectory(dietgpu/ans)
add_subdirectory(dietgpu/float)
add_subdirectory(dietgpu)
