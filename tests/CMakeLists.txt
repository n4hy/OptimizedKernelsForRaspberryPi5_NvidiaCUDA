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

add_executable(test_neon_complex test_neon_complex.cpp)
target_link_libraries(test_neon_complex PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_neon_complex COMMAND test_neon_complex)

add_executable(test_neon_transcendentals test_neon_transcendentals.cpp)
target_link_libraries(test_neon_transcendentals PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_neon_transcendentals COMMAND test_neon_transcendentals)

add_executable(test_radar_caf test_radar_caf.cpp)
target_link_libraries(test_radar_caf PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_radar_caf COMMAND test_radar_caf)

add_executable(test_radar_cfar test_radar_cfar.cpp)
target_link_libraries(test_radar_cfar PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_radar_cfar COMMAND test_radar_cfar)

add_executable(test_neon_resample test_neon_resample.cpp)
target_link_libraries(test_neon_resample PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_neon_resample COMMAND test_neon_resample)

add_executable(test_neon_iir test_neon_iir.cpp)
target_link_libraries(test_neon_iir PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_neon_iir COMMAND test_neon_iir)

add_executable(test_neon_conv2d test_neon_conv2d.cpp)
target_link_libraries(test_neon_conv2d PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_neon_conv2d COMMAND test_neon_conv2d)

add_executable(test_neon_linalg test_neon_linalg.cpp)
target_link_libraries(test_neon_linalg PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_neon_linalg COMMAND test_neon_linalg)

# SVE2 tests (only if SVE2 is enabled)
if(ENABLE_SVE2 AND ON_ARM AND HAS_SVE2)
    add_executable(test_sve2_kernels test_sve2_kernels.cpp)
    target_link_libraries(test_sve2_kernels PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
    add_test(NAME test_sve2_kernels COMMAND test_sve2_kernels)
endif()

# Platform detection tests
add_executable(test_platform test_platform.cpp)
target_link_libraries(test_platform PRIVATE OptMathKernels::OptMathKernels gtest gtest_main)
add_test(NAME test_platform COMMAND test_platform)

# CUDA tests (only if CUDA is enabled)
if(ENABLE_CUDA)
    add_executable(test_cuda_kernels test_cuda_kernels.cpp)
    target_link_libraries(test_cuda_kernels PRIVATE
        OptMathKernels::OptMathKernels
        gtest
        gtest_main
        CUDA::cudart
    )
    add_test(NAME test_cuda_kernels COMMAND test_cuda_kernels)

    # TODO: Enable when shorthand radar functions are implemented
    # add_executable(test_cuda_radar test_cuda_radar.cpp)
    # target_link_libraries(test_cuda_radar PRIVATE
    #     OptMathKernels::OptMathKernels
    #     gtest
    #     gtest_main
    #     CUDA::cudart
    # )
    # add_test(NAME test_cuda_radar COMMAND test_cuda_radar)
endif()
