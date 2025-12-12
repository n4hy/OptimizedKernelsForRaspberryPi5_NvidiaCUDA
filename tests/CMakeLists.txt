add_executable(test_basic test_basic.cpp)
target_link_libraries(test_basic PRIVATE OptMathKernels::OptMathKernels)
add_test(NAME test_basic COMMAND test_basic)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_vulkan_vector test_vulkan_vector.cpp)
target_link_libraries(test_vulkan_vector PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_vulkan_vector COMMAND test_vulkan_vector)

add_executable(test_vulkan_matrix test_vulkan_matrix.cpp)
target_link_libraries(test_vulkan_matrix PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_vulkan_matrix COMMAND test_vulkan_matrix)

add_executable(test_vulkan_dsp test_vulkan_dsp.cpp)
target_link_libraries(test_vulkan_dsp PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_vulkan_dsp COMMAND test_vulkan_dsp)

add_executable(test_vulkan_advanced test_vulkan_advanced.cpp)
target_link_libraries(test_vulkan_advanced PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_vulkan_advanced COMMAND test_vulkan_advanced)

add_executable(test_neon_kernels test_neon_kernels.cpp)
target_link_libraries(test_neon_kernels PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_neon_kernels COMMAND test_neon_kernels)
