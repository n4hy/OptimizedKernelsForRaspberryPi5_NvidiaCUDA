cmake_minimum_required(VERSION 3.18)

project(OptMathExamples)

add_executable(demo demo.cpp)

target_link_libraries(demo PRIVATE OptMathKernels::OptMathKernels)

# If you want to enable specific compiler flags for the example
target_compile_features(demo PRIVATE cxx_std_20)

# CUDA demo (only if CUDA is enabled)
# TODO: Enable when shorthand radar functions are implemented
# if(ENABLE_CUDA)
#     add_executable(cuda_demo cuda_demo.cpp)
#     target_link_libraries(cuda_demo PRIVATE
#         OptMathKernels::OptMathKernels
#         CUDA::cudart
#         CUDA::cublas
#         CUDA::cufft
#     )
#     target_compile_features(cuda_demo PRIVATE cxx_std_20)
# endif()
