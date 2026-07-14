#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cstring>
#include <vector>

// GEMM had no dedicated suite before v0.6.2, which is how a fully-serial
// `#pragma omp parallel for` over ceil(N/NC) blocks (one iteration at N<=256,
// so three of four A76 cores idled) survived unnoticed: every correctness test
// passed the whole time, because the bug only cost speed.
//
// These lock in the parts that are easy to break while tuning: the ragged edge
// tiles, the Eigen/NEON dispatch boundary, and freedom from races in the
// parallel (jr, ir) tile sweep.

using optmath::neon::neon_gemm_blocked;

static void expect_matches_eigen(int M, int N, int K, float tol = 1e-4f) {
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);
    Eigen::MatrixXf ref = A * B;
    Eigen::MatrixXf got = neon_gemm_blocked(A, B);

    ASSERT_EQ(got.rows(), M) << M << "x" << N << "x" << K;
    ASSERT_EQ(got.cols(), N) << M << "x" << N << "x" << K;
    // Scale-relative: random GEMM output grows with K.
    const float scale = std::max(1.0f, ref.cwiseAbs().maxCoeff());
    const float err = (got - ref).cwiseAbs().maxCoeff() / scale;
    EXPECT_LT(err, tol) << "shape " << M << "x" << N << "x" << K;
}

// Sizes straddling MR=8/NR=8 (microkernel) and MC=128/KC=256/NC=256 (blocking),
// so both the full microkernel and the ragged scalar edge path get exercised.
TEST(NeonGemm, MatchesEigenAcrossShapes) {
    const std::vector<int> dims = {1, 2, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65,
                                   127, 128, 129, 255, 256, 257};
    for (int M : dims)
        for (int N : dims)
            for (int K : dims) {
                if ((long)M * N * K > 600000) continue;  // keep the suite quick
                expect_matches_eigen(M, N, K);
                if (HasFatalFailure()) return;
            }
}

// Non-square / skinny shapes: M, N, K independently on either side of the
// blocking parameters.
TEST(NeonGemm, MatchesEigenNonSquare) {
    expect_matches_eigen(128, 256, 64);
    expect_matches_eigen(256, 128, 129);
    expect_matches_eigen(1, 512, 256);    // single row
    expect_matches_eigen(512, 1, 256);    // single column
    expect_matches_eigen(300, 7, 300);    // skinny N, past the dispatch boundary
    expect_matches_eigen(7, 300, 300);    // skinny M
}

// OPTMATH_GEMM_EIGEN_MAX = 80^3 routes small GEMMs to Eigen and larger ones to
// the NEON microkernel. Both sides of that branch must be correct -- and the
// strided Eigen::Map on the small side is easy to get wrong (a bad OuterStride
// silently corrupts non-square results).
TEST(NeonGemm, DispatchBoundaryBothSidesCorrect) {
    expect_matches_eigen(79, 79, 79);   // below  80^3 -> Eigen path
    expect_matches_eigen(80, 80, 80);   // at     80^3 -> NEON path
    expect_matches_eigen(81, 81, 81);   // above  80^3 -> NEON path
    // Same total work, very different shapes, straddling the boundary.
    expect_matches_eigen(512, 512, 1);
    expect_matches_eigen(8, 8, 8192);   // long-k, one tile
}

// The (jr, ir) tile sweep writes disjoint blocks of C from multiple threads.
// The kernel is deterministic by construction (fixed k order per tile), so any
// run-to-run difference means a race.
TEST(NeonGemm, ParallelTileSweepIsDeterministic) {
    for (int N : {96, 129, 256}) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf first = neon_gemm_blocked(A, B);
        for (int rep = 0; rep < 10; ++rep) {
            Eigen::MatrixXf again = neon_gemm_blocked(A, B);
            ASSERT_EQ(0, std::memcmp(first.data(), again.data(),
                                     (size_t)N * N * sizeof(float)))
                << "nondeterministic at N=" << N << " rep=" << rep
                << " -- race in the parallel tile sweep";
        }
    }
}

// C is fully overwritten, never accumulated into: a dirty buffer must not leak
// through (the kernel zeroes C, and the Eigen path assigns with noalias()).
TEST(NeonGemm, OverwritesDestination) {
    for (int N : {32, 128}) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf once = neon_gemm_blocked(A, B);
        Eigen::MatrixXf twice = neon_gemm_blocked(A, B);
        EXPECT_TRUE(once.isApprox(twice)) << "N=" << N;
        EXPECT_TRUE(once.isApprox(A * B, 1e-4f)) << "N=" << N;
    }
}

// neon_gemm is the public API the README advertises. Until v0.6.3 it guarded the
// blocked path with `Md >= 64 && Nd >= 64 && Kd >= 64`, so any ONE small
// dimension dropped an arbitrarily large GEMM onto a naive 4x4 loop -- measured
// 22.9x slower at 4096x32x4096. Nothing caught it: the only test that called
// neon_gemm used 64x64x64, the exact size that takes the other branch.
// These shapes all took the naive path before the fix.
TEST(NeonGemm, PublicWrapperMatchesEigenIncludingSmallDimensions) {
    using optmath::neon::neon_gemm;
    struct { int M, N, K; } shapes[] = {
        {64, 64, 64},      // the only size the old suite tested
        {63, 63, 63},      // just under the old gate on every axis
        {4096 / 32, 32, 128},
        {128, 32, 128},    // N < 64  -> was the naive path
        {32, 128, 128},    // M < 64
        {128, 128, 32},    // K < 64
        {1, 128, 128},     // degenerate M
        {128, 1, 128},     // degenerate N
        {128, 128, 1},     // degenerate K
        {8, 8, 8}, {4, 4, 4}, {3, 5, 7},  // tiny / ragged
    };
    for (const auto& s : shapes) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(s.M, s.K);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(s.K, s.N);
        Eigen::MatrixXf got = neon_gemm(A, B);
        Eigen::MatrixXf ref = A * B;
        ASSERT_EQ(got.rows(), s.M);
        ASSERT_EQ(got.cols(), s.N);
        const float scale = std::max(1.0f, ref.cwiseAbs().maxCoeff());
        EXPECT_LT((got - ref).cwiseAbs().maxCoeff() / scale, 1e-4f)
            << "neon_gemm wrong at " << s.M << "x" << s.N << "x" << s.K;
    }
}

// neon_gemm_4x4_f32 stays exported and benchmarked but is no longer reachable
// through neon_gemm, so it needs its own test. It ACCUMULATES into C.
TEST(NeonGemm, Microkernel4x4AccumulatesWithStrides) {
    using optmath::neon::neon_gemm_4x4_f32;
    // Column-major 4x4 blocks with non-trivial leading dimensions.
    const std::size_t lda = 8, ldb = 8, ldc = 8;
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(lda, 4);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(ldb, 4);
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(ldc, 4);
    Eigen::MatrixXf C0 = C;

    neon_gemm_4x4_f32(C.data(), A.data(), lda, B.data(), ldb, ldc);

    Eigen::MatrixXf expected = C0.topLeftCorner(4, 4)
                             + A.topLeftCorner(4, 4) * B.topLeftCorner(4, 4);
    EXPECT_TRUE(C.topLeftCorner(4, 4).isApprox(expected, 1e-4f))
        << "4x4 microkernel result:\n" << C.topLeftCorner(4, 4)
        << "\nexpected:\n" << expected;
}
