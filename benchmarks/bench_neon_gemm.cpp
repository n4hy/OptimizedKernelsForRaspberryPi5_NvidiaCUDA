#include "bench_common.hpp"
#include <optmath/neon_kernels.hpp>

using namespace optmath::neon;

// Benchmark: Basic 4x4 GEMM microkernel
static void BM_NEON_GEMM_4x4(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t K = state.range(0);
    Eigen::MatrixXf A = bench::random_matrix_f32(4, K);
    Eigen::MatrixXf B = bench::random_matrix_f32(K, 4);
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(4, 4);

    for (auto _ : state) {
        C.setZero();
        // Iterate over K in blocks of 4
        for (size_t k = 0; k + 3 < K; k += 4) {
            neon_gemm_4x4_f32(C.data(), A.data() + k * 4, 4,
                              B.data() + k, K, 4);
        }
        benchmark::DoNotOptimize(C.data());
    }

    bench::set_flops(state, bench::gemm_flops(4, 4, K));
}
BENCHMARK(BM_NEON_GEMM_4x4)->RangeMultiplier(2)->Range(16, 1024)->UseRealTime();

// Benchmark: Simple tiled GEMM (existing implementation)
static void BM_NEON_GEMM_Simple(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::MatrixXf A = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf B = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf C;

    for (auto _ : state) {
        C = neon_gemm(A, B);
        benchmark::DoNotOptimize(C.data());
    }

    bench::set_flops(state, bench::gemm_flops(N, N, N));
}
BENCHMARK(BM_NEON_GEMM_Simple)->RangeMultiplier(2)->Range(32, 512)->UseRealTime();

// Benchmark: Optimized blocked GEMM
static void BM_NEON_GEMM_Blocked(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::MatrixXf A = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf B = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf C;

    for (auto _ : state) {
        C = neon_gemm_blocked(A, B);
        benchmark::DoNotOptimize(C.data());
    }

    bench::set_flops(state, bench::gemm_flops(N, N, N));
}
BENCHMARK(BM_NEON_GEMM_Blocked)->RangeMultiplier(2)->Range(32, 512)->UseRealTime();

// Benchmark: Compare with Eigen
static void BM_Eigen_GEMM(benchmark::State& state) {
    size_t N = state.range(0);
    Eigen::MatrixXf A = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf B = bench::random_matrix_f32(N, N);
    Eigen::MatrixXf C;

    for (auto _ : state) {
        C = A * B;
        benchmark::DoNotOptimize(C.data());
    }

    bench::set_flops(state, bench::gemm_flops(N, N, N));
}
BENCHMARK(BM_Eigen_GEMM)->RangeMultiplier(2)->Range(32, 512)->UseRealTime();

// Benchmark: Non-square matrices
static void BM_NEON_GEMM_Blocked_MNK(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t M = state.range(0);
    size_t N = state.range(1);
    size_t K = state.range(2);

    Eigen::MatrixXf A = bench::random_matrix_f32(M, K);
    Eigen::MatrixXf B = bench::random_matrix_f32(K, N);
    Eigen::MatrixXf C;

    for (auto _ : state) {
        C = neon_gemm_blocked(A, B);
        benchmark::DoNotOptimize(C.data());
    }

    bench::set_flops(state, bench::gemm_flops(M, N, K));
}
BENCHMARK(BM_NEON_GEMM_Blocked_MNK)
    ->Args({128, 256, 64})
    ->Args({256, 128, 512})
    ->Args({512, 64, 256})
    ->UseRealTime();

// Benchmark: Matrix-vector multiplication
static void BM_NEON_MatVec(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);
    Eigen::MatrixXf A = bench::random_matrix_f32(N, N);
    Eigen::VectorXf v = bench::random_vector_f32(N);
    Eigen::VectorXf result;

    for (auto _ : state) {
        result = neon_mat_vec_mul(A, v);
        benchmark::DoNotOptimize(result.data());
    }

    // 2*N*N flops for mat-vec
    bench::set_flops(state, 2.0 * N * N);
}
BENCHMARK(BM_NEON_MatVec)->RangeMultiplier(2)->Range(64, 2048)->UseRealTime();

// Benchmark: the PUBLIC neon_gemm wrapper on shapes with one small dimension.
//
// This exists because of a v0.6.3 bug the rest of this file structurally could not
// see. neon_gemm guarded its fast path with `M >= 64 && N >= 64 && K >= 64`, so any
// ONE small dimension dropped an arbitrarily large GEMM onto a naive 4x4 loop --
// 4096x32x4096 ran at 0.9 GFLOPS instead of 20.6, a 22.9x penalty, through the API
// the README advertises. Every other GEMM benchmark here sweeps SQUARE sizes >= 64,
// which always satisfy the guard, so none of them could ever observe the slow path.
// Correctness tests could not catch it either: the naive path was right, just slow.
//
// Keep at least one non-square shape per axis here. A guard regression shows up as
// these dropping toward single-digit GFLOPS while the square benchmarks stay flat.
static void BM_NEON_GEMM_Public_SmallDim(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }
    size_t M = state.range(0), N = state.range(1), K = state.range(2);
    Eigen::MatrixXf A = bench::random_matrix_f32(M, K);
    Eigen::MatrixXf B = bench::random_matrix_f32(K, N);
    Eigen::MatrixXf C;

    for (auto _ : state) {
        C = neon_gemm(A, B);
        benchmark::DoNotOptimize(C.data());
    }

    bench::set_flops(state, bench::gemm_flops(M, N, K));
}
BENCHMARK(BM_NEON_GEMM_Public_SmallDim)
    ->Args({4096, 32, 4096})   // N < 64  -- was 22.9x slower
    ->Args({32, 4096, 4096})   // M < 64
    ->Args({4096, 4096, 32})   // K < 64
    ->Args({63, 1024, 1024})   // just under the old gate
    ->Args({256, 256, 256})    // square control: must match BM_NEON_GEMM_Blocked/256
    ->UseRealTime();

BENCHMARK_MAIN();
