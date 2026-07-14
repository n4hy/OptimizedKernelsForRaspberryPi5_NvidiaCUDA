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
