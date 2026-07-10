/**
 * Tests for the quantized/half-precision GEMM kernels added for the Pi 5
 * Cortex-A76: int8 SDOT GEMM and fp16 GEMM/GEMV.
 */
#include "optmath/neon_int8.hpp"
#include "optmath/platform.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>
#include <vector>

using I8  = Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic>;
using I32 = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>;

// ---- int8 SDOT GEMM (exact vs int32 reference) ----
static void check_int8(int M, int N, int K, std::mt19937& g) {
    std::uniform_int_distribution<int> d(-32, 31);
    I8 A(M, K), B(K, N);
    for (int i = 0; i < M; ++i) for (int k = 0; k < K; ++k) A(i, k) = (std::int8_t)d(g);
    for (int k = 0; k < K; ++k) for (int j = 0; j < N; ++j) B(k, j) = (std::int8_t)d(g);

    I32 C = optmath::neon::neon_gemm_int8(A, B);
    ASSERT_EQ(C.rows(), M);
    ASSERT_EQ(C.cols(), N);

    I32 R = I32::Zero(M, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            std::int32_t s = 0;
            for (int k = 0; k < K; ++k) s += (std::int32_t)A(i, k) * (std::int32_t)B(k, j);
            R(i, j) = s;
        }
    EXPECT_EQ((C - R).cwiseAbs().maxCoeff(), 0) << "int8 " << M << "x" << N << "x" << K;
}

TEST(QuantGemm, Int8ExactVariousShapes) {
    std::mt19937 g(12345);
    check_int8(4, 4, 16, g);      // aligned
    check_int8(7, 5, 19, g);      // M,N not mult of 4; K not mult of 16
    check_int8(13, 11, 3, g);     // K < 16 (all scalar tail)
    check_int8(1, 1, 1, g);       // degenerate
    check_int8(64, 48, 96, g);    // larger, aligned
    check_int8(65, 33, 100, g);   // larger, ragged
    check_int8(300, 260, 130, g); // multi cache-block + threads + ragged edges
    check_int8(256, 32, 512, g);  // exactly one NC panel wide
}

// ---- int8 conv2d (exact vs int32 reference) ----
static void check_conv8(int H, int W, int Kh, int Kw, std::mt19937& g) {
    std::uniform_int_distribution<int> d(-40, 40);
    int OH = H - Kh + 1, OW = W - Kw + 1;
    std::vector<std::int8_t> in(H * W), ker(Kh * Kw);
    std::vector<std::int32_t> out(OH * OW, 0), ref(OH * OW, 0);
    for (auto& x : in) x = (std::int8_t)d(g);
    for (auto& x : ker) x = (std::int8_t)d(g);
    optmath::neon::neon_conv2d_s8s8s32(out.data(), in.data(), H, W, ker.data(), Kh, Kw);
    for (int r = 0; r < OH; ++r)
        for (int c = 0; c < OW; ++c) {
            std::int32_t s = 0;
            for (int kr = 0; kr < Kh; ++kr)
                for (int kc = 0; kc < Kw; ++kc)
                    s += (std::int32_t)in[(r + kr) * W + c + kc] * (std::int32_t)ker[kr * Kw + kc];
            ref[r * OW + c] = s;
        }
    for (int i = 0; i < OH * OW; ++i)
        ASSERT_EQ(out[i], ref[i]) << "conv " << H << "x" << W << " k" << Kh << "x" << Kw << " idx " << i;
}

TEST(QuantGemm, Int8Conv2dExact) {
    std::mt19937 g(2468);
    check_conv8(16, 16, 3, 3, g);    // small
    check_conv8(64, 50, 5, 5, g);    // vectorized cols + tail
    check_conv8(33, 27, 3, 7, g);    // asymmetric kernel
    check_conv8(9, 9, 9, 9, g);      // kernel == image (1 output)
    check_conv8(128, 96, 3, 3, g);   // threaded (out_rows >= 64)
}

TEST(QuantGemm, Int8DimMismatchReturnsEmpty) {
    I8 A(4, 5), B(6, 4);  // 5 != 6
    I32 C = optmath::neon::neon_gemm_int8(A, B);
    EXPECT_EQ(C.size(), 0);
}

TEST(QuantGemm, DotprodDetectedOnA76) {
    // On the Pi 5 the runtime probe must report dot-product support.
    const auto& info = optmath::platform::detect_cpu_info();
    if (info.model_name.find("A76") != std::string::npos) {
        EXPECT_TRUE(info.has_dotprod);
        EXPECT_TRUE(info.has_fp16);
        EXPECT_TRUE(info.has_neon);
    }
    SUCCEED();
}

// NOTE: fp16 GEMM/GEMV were removed (slower than fp32 on the A76, which lacks
// FEAT_FHM). Quantized matmul speedups are provided by the int8 SDOT path above.
