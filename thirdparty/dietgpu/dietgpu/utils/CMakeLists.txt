add_library(dietgpu_utils SHARED
  DeviceUtils.cpp
  StackDeviceMemory.cpp
)

target_include_directories(dietgpu_utils PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}/../../include>
)
target_link_libraries(dietgpu_utils PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
)
target_compile_options(dietgpu_utils PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --generate-line-info
  #--device-debug
>)
