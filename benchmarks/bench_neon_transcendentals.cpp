#include "bench_common.hpp"
#include <optmath/neon_kernels.hpp>
#include <cmath>

using namespace optmath::neon;

// Benchmark: Vectorized exp approximation
static void BM_NEON_Exp_Approx(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    std::vector<float> input(N), output(N);

    // Initialize with values in reasonable range
    for (size_t i = 0; i < N; ++i) {
        input[i] = -10.0f + 20.0f * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        neon_fast_exp_f32(output.data(), input.data(), N);
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, 15.0 * N);  // ~15 ops per exp
}
BENCHMARK(BM_NEON_Exp_Approx)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: std::exp for comparison
static void BM_Std_Exp(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> input(N), output(N);

    for (size_t i = 0; i < N; ++i) {
        input[i] = -10.0f + 20.0f * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) {
            output[i] = std::exp(input[i]);
        }
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, static_cast<double>(N));
}
BENCHMARK(BM_Std_Exp)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: Vectorized sin approximation
static void BM_NEON_Sin_Approx(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    std::vector<float> input(N), output(N);

    const float pi = 3.14159265f;
    for (size_t i = 0; i < N; ++i) {
        input[i] = -4.0f * pi + 8.0f * pi * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        neon_fast_sin_f32(output.data(), input.data(), N);
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, 12.0 * N);
}
BENCHMARK(BM_NEON_Sin_Approx)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: std::sin for comparison
static void BM_Std_Sin(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> input(N), output(N);

    const float pi = 3.14159265f;
    for (size_t i = 0; i < N; ++i) {
        input[i] = -4.0f * pi + 8.0f * pi * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) {
            output[i] = std::sin(input[i]);
        }
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, static_cast<double>(N));
}
BENCHMARK(BM_Std_Sin)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: Vectorized cos approximation
static void BM_NEON_Cos_Approx(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    std::vector<float> input(N), output(N);

    const float pi = 3.14159265f;
    for (size_t i = 0; i < N; ++i) {
        input[i] = -4.0f * pi + 8.0f * pi * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        neon_fast_cos_f32(output.data(), input.data(), N);
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, 12.0 * N);
}
BENCHMARK(BM_NEON_Cos_Approx)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: Fast sigmoid
static void BM_NEON_Sigmoid_Fast(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    std::vector<float> input(N), output(N);

    for (size_t i = 0; i < N; ++i) {
        input[i] = -10.0f + 20.0f * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        neon_fast_sigmoid_f32(output.data(), input.data(), N);
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, 20.0 * N);  // exp + division
}
BENCHMARK(BM_NEON_Sigmoid_Fast)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: Scalar sigmoid for comparison
static void BM_Scalar_Sigmoid(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> data(N);

    for (size_t i = 0; i < N; ++i) {
        data[i] = -10.0f + 20.0f * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
        benchmark::DoNotOptimize(data.data());
    }

    bench::set_flops(state, 2.0 * N);
}
BENCHMARK(BM_Scalar_Sigmoid)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: Fast tanh
static void BM_NEON_Tanh_Fast(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    std::vector<float> input(N), output(N);

    for (size_t i = 0; i < N; ++i) {
        input[i] = -5.0f + 10.0f * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        neon_fast_tanh_f32(output.data(), input.data(), N);
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, 25.0 * N);
}
BENCHMARK(BM_NEON_Tanh_Fast)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: std::tanh for comparison
static void BM_Std_Tanh(benchmark::State& state) {
    size_t N = state.range(0);
    std::vector<float> data(N);

    for (size_t i = 0; i < N; ++i) {
        data[i] = -5.0f + 10.0f * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = std::tanh(data[i]);
        }
        benchmark::DoNotOptimize(data.data());
    }

    bench::set_flops(state, static_cast<double>(N));
}
BENCHMARK(BM_Std_Tanh)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: ReLU (should be very fast)
static void BM_NEON_ReLU(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    std::vector<float> data(N);

    for (size_t i = 0; i < N; ++i) {
        data[i] = -5.0f + 10.0f * static_cast<float>(i) / N;
    }

    for (auto _ : state) {
        neon_relu_f32(data.data(), N);
        benchmark::DoNotOptimize(data.data());
    }

    bench::set_bytes_processed(state, N * sizeof(float));
}
BENCHMARK(BM_NEON_ReLU)->RangeMultiplier(4)->Range(256, 4194304);

// Benchmark: Vector dot product
static void BM_NEON_Dot(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::VectorXf a = bench::random_vector_f32(N);
    Eigen::VectorXf b = bench::random_vector_f32(N);
    float result;

    for (auto _ : state) {
        result = neon_dot(a, b);
        benchmark::DoNotOptimize(result);
    }

    bench::set_flops(state, 2.0 * N);  // N muls + N-1 adds
}
BENCHMARK(BM_NEON_Dot)->RangeMultiplier(4)->Range(256, 4194304);

// Benchmark: Eigen dot product for comparison
static void BM_Eigen_Dot(benchmark::State& state) {
    size_t N = state.range(0);
    Eigen::VectorXf a = bench::random_vector_f32(N);
    Eigen::VectorXf b = bench::random_vector_f32(N);
    float result;

    for (auto _ : state) {
        result = a.dot(b);
        benchmark::DoNotOptimize(result);
    }

    bench::set_flops(state, 2.0 * N);
}
BENCHMARK(BM_Eigen_Dot)->RangeMultiplier(4)->Range(256, 4194304);

BENCHMARK_MAIN();
