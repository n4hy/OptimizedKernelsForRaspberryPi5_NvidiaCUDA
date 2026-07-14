#include "bench_common.hpp"
#include <optmath/vulkan_backend.hpp>

using namespace optmath::vulkan;

// Benchmark: Vulkan matrix multiplication (original)
static void BM_Vulkan_MatMul(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::MatrixXf A = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf B = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf C;

    // Warm up
    C = vulkan_mat_mul(A, B);

    for (auto _ : state) {
        C = vulkan_mat_mul(A, B);
        benchmark::DoNotOptimize(C.data());
    }

    bench::set_flops(state, bench::gemm_flops(N, N, N));
}
BENCHMARK(BM_Vulkan_MatMul)->RangeMultiplier(2)->Range(64, 1024)->UseRealTime();

// Benchmark: Vulkan vector operations
static void BM_Vulkan_VecAdd(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::VectorXf a = bench::random_vector_f32(N);
    Eigen::VectorXf b = bench::random_vector_f32(N);
    Eigen::VectorXf c;

    for (auto _ : state) {
        c = vulkan_vec_add(a, b);
        benchmark::DoNotOptimize(c.data());
    }

    bench::set_bytes_processed(state, 3 * N * sizeof(float));  // Read 2, write 1
}
BENCHMARK(BM_Vulkan_VecAdd)->RangeMultiplier(4)->Range(1024, 4194304)->UseRealTime();

// Benchmark: Vulkan vector multiplication
static void BM_Vulkan_VecMul(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::VectorXf a = bench::random_vector_f32(N);
    Eigen::VectorXf b = bench::random_vector_f32(N);
    Eigen::VectorXf c;

    for (auto _ : state) {
        c = vulkan_vec_mul(a, b);
        benchmark::DoNotOptimize(c.data());
    }

    bench::set_bytes_processed(state, 3 * N * sizeof(float));
}
BENCHMARK(BM_Vulkan_VecMul)->RangeMultiplier(4)->Range(1024, 4194304)->UseRealTime();

// Benchmark: Vulkan dot product
static void BM_Vulkan_VecDot(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::VectorXf a = bench::random_vector_f32(N);
    Eigen::VectorXf b = bench::random_vector_f32(N);
    float result;

    for (auto _ : state) {
        result = vulkan_vec_dot(a, b);
        benchmark::DoNotOptimize(result);
    }

    bench::set_flops(state, 2.0 * N);  // N muls + N-1 adds
}
BENCHMARK(BM_Vulkan_VecDot)->RangeMultiplier(4)->Range(1024, 4194304)->UseRealTime();

// Benchmark: Vulkan reduction
static void BM_Vulkan_ReduceSum(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::VectorXf a = bench::random_vector_f32(N);
    float result;

    for (auto _ : state) {
        result = vulkan_reduce_sum(a);
        benchmark::DoNotOptimize(result);
    }

    bench::set_flops(state, static_cast<double>(N));
}
BENCHMARK(BM_Vulkan_ReduceSum)->RangeMultiplier(4)->Range(1024, 4194304)->UseRealTime();

// Benchmark: Vulkan prefix sum
static void BM_Vulkan_PrefixSum(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::VectorXf a = bench::random_vector_f32(N);
    Eigen::VectorXf result;

    for (auto _ : state) {
        result = vulkan_scan_prefix_sum(a);
        benchmark::DoNotOptimize(result.data());
    }

    bench::set_flops(state, static_cast<double>(N));
}
BENCHMARK(BM_Vulkan_PrefixSum)->RangeMultiplier(2)->Range(256, 4096)->UseRealTime();

// Benchmark: Vulkan 1D convolution
static void BM_Vulkan_Conv1D(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t signal_len = state.range(0);
    size_t kernel_len = state.range(1);

    Eigen::VectorXf x = bench::random_vector_f32(signal_len);
    Eigen::VectorXf k = bench::random_vector_f32(kernel_len);
    Eigen::VectorXf result;

    for (auto _ : state) {
        result = vulkan_convolution_1d(x, k);
        benchmark::DoNotOptimize(result.data());
    }

    bench::set_flops(state, 2.0 * signal_len * kernel_len);
}
BENCHMARK(BM_Vulkan_Conv1D)
    ->Args({4096, 16})
    ->Args({16384, 32})
    ->Args({65536, 64})
    ->Args({262144, 128})
    ->UseRealTime();

// Benchmark: Vulkan 2D convolution
static void BM_Vulkan_Conv2D(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t img_size = state.range(0);
    size_t kernel_size = state.range(1);

    Eigen::MatrixXf x = bench::random_matrix_f32(img_size, img_size);
    Eigen::MatrixXf k = bench::random_matrix_f32(kernel_size, kernel_size);
    Eigen::MatrixXf result;

    for (auto _ : state) {
        result = vulkan_convolution_2d(x, k);
        benchmark::DoNotOptimize(result.data());
    }

    bench::set_flops(state, 2.0 * img_size * img_size * kernel_size * kernel_size);
}
BENCHMARK(BM_Vulkan_Conv2D)
    ->Args({128, 3})
    ->Args({256, 5})
    ->Args({512, 7})
    ->Args({1024, 3})
    ->UseRealTime();

// Benchmark: Matrix transpose
static void BM_Vulkan_MatTranspose(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("Vulkan not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::MatrixXf A = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf result;

    for (auto _ : state) {
        result = vulkan_mat_transpose(A);
        benchmark::DoNotOptimize(result.data());
    }

    bench::set_bytes_processed(state, 2 * N * N * sizeof(float));
}
BENCHMARK(BM_Vulkan_MatTranspose)->RangeMultiplier(2)->Range(64, 2048)->UseRealTime();

BENCHMARK_MAIN();
