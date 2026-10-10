add_library(gpu_ans SHARED
  GpuANSDecode.cu
  GpuANSEncode.cu
  GpuANSInfo.cu
)
add_dependencies(gpu_ans
  dietgpu_utils
)
target_include_directories(gpu_ans PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
)
target_link_libraries(gpu_ans PUBLIC
  dietgpu_utils
)
target_compile_options(gpu_ans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --generate-line-info
  #--device-debug
>)


if(BUILD_TESTING)
  add_executable(ans_test ../test/ANSTest.cu)
  add_executable(ans_statistics_test ../test/ANSStatisticsTest.cu)
  add_executable(batch_prefix_sum_test ../test/BatchPrefixSumTest.cu)
  foreach(test ans_test ans_statistics_test batch_prefix_sum_test)
    target_link_libraries(${test} gpu_ans GTest::gtest_main)
    gtest_discover_tests(${test} DISCOVERY_MODE PRE_TEST)
  endforeach()
endif()
