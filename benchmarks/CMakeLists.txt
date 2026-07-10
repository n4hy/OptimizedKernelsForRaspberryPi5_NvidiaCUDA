# Benchmark suite using Google Benchmark
include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)

# Disable benchmark tests
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

# Common benchmark header
# bench_common.hpp is in this directory

# NEON GEMM Benchmark
add_executable(bench_neon_gemm bench_neon_gemm.cpp)
target_link_libraries(bench_neon_gemm PRIVATE OptMathKernels::OptMathKernels benchmark::benchmark)
target_include_directories(bench_neon_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# NEON FFT Benchmark
add_executable(bench_neon_fft bench_neon_fft.cpp)
target_link_libraries(bench_neon_fft PRIVATE OptMathKernels::OptMathKernels benchmark::benchmark)
target_include_directories(bench_neon_fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Vulkan Matrix Multiplication Benchmark
add_executable(bench_vulkan_matmul bench_vulkan_matmul.cpp)
target_link_libraries(bench_vulkan_matmul PRIVATE OptMathKernels::OptMathKernels benchmark::benchmark)
target_include_directories(bench_vulkan_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Vulkan Reduction A/B Benchmark (barrier tree vs subgroup shuffle)
add_executable(bench_vulkan_reduce bench_vulkan_reduce.cpp)
target_link_libraries(bench_vulkan_reduce PRIVATE OptMathKernels::OptMathKernels benchmark::benchmark)
target_include_directories(bench_vulkan_reduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Radar CAF Benchmark
add_executable(bench_radar_caf bench_radar_caf.cpp)
target_link_libraries(bench_radar_caf PRIVATE OptMathKernels::OptMathKernels benchmark::benchmark)
target_include_directories(bench_radar_caf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# NEON Transcendentals Benchmark
add_executable(bench_neon_transcendentals bench_neon_transcendentals.cpp)
target_link_libraries(bench_neon_transcendentals PRIVATE OptMathKernels::OptMathKernels benchmark::benchmark)
target_include_directories(bench_neon_transcendentals PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
