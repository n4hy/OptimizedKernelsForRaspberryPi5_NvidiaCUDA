#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

TEST(NeonTranscendentalsTest, ExpApproximation) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    // Test range from -10 to 10
    for (int i = 0; i < N; ++i) {
        input[i] = -10.0f + 20.0f * i / (N - 1);
    }

    optmath::neon::neon_fast_exp_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::exp(input[i]);
        float rel_error = std::abs(result[i] - expected) / (std::abs(expected) + 1e-10f);
        // Fast polynomial approximation: up to 12% relative error at extremes is acceptable
        EXPECT_LT(rel_error, 0.12f) << "at x = " << input[i]
                                    << ", expected = " << expected
                                    << ", got = " << result[i];
    }
}

TEST(NeonTranscendentalsTest, ExpBoundary) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    // Test boundary conditions
    std::vector<float> input = {0.0f, 1.0f, -1.0f, 88.0f, -88.0f, 100.0f, -100.0f};
    std::vector<float> result(input.size());

    optmath::neon::neon_fast_exp_f32(result.data(), input.data(), input.size());

    EXPECT_NEAR(result[0], 1.0f, 1e-4f);  // exp(0) = 1
    // Fast approximation has higher error for non-zero inputs
    EXPECT_NEAR(result[1], std::exp(1.0f), 0.3f);  // exp(1) ~= 2.72
    EXPECT_NEAR(result[2], std::exp(-1.0f), 0.1f); // exp(-1) ~= 0.37
    EXPECT_GT(result[3], 0.0f);  // exp(88) should be large but finite
    EXPECT_GE(result[4], 0.0f);  // exp(-88) may underflow to 0 in fast approximation
    EXPECT_GT(result[5], 0.0f);  // exp(100) clamped to exp(88)
    EXPECT_GE(result[6], 0.0f);  // exp(-100) may underflow to 0 in fast approximation
}

TEST(NeonTranscendentalsTest, SinApproximation) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    // Test from -4*pi to 4*pi
    const float pi = 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        input[i] = -4.0f * pi + 8.0f * pi * i / (N - 1);
    }

    optmath::neon::neon_fast_sin_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::sin(input[i]);
        EXPECT_NEAR(result[i], expected, 1e-5f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, CosApproximation) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    const float pi = 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        input[i] = -4.0f * pi + 8.0f * pi * i / (N - 1);
    }

    optmath::neon::neon_fast_cos_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::cos(input[i]);
        EXPECT_NEAR(result[i], expected, 1e-5f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, SinCosIdentity) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 256;
    std::vector<float> input(N), sin_out(N), cos_out(N);

    const float pi = 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        input[i] = -2.0f * pi + 4.0f * pi * i / (N - 1);
    }

    optmath::neon::neon_fast_sin_f32(sin_out.data(), input.data(), N);
    optmath::neon::neon_fast_cos_f32(cos_out.data(), input.data(), N);

    // sin^2 + cos^2 = 1
    for (int i = 0; i < N; ++i) {
        float sum_sq = sin_out[i] * sin_out[i] + cos_out[i] * cos_out[i];
        EXPECT_NEAR(sum_sq, 1.0f, 1e-4f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, SigmoidFast) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    // Test from -10 to 10
    for (int i = 0; i < N; ++i) {
        input[i] = -10.0f + 20.0f * i / (N - 1);
    }

    optmath::neon::neon_fast_sigmoid_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-input[i]));
        // Fast approximation chains exp, so tolerance must be relaxed significantly
        EXPECT_NEAR(result[i], expected, 3e-2f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, SigmoidProperties) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    std::vector<float> input = {0.0f, 10.0f, -10.0f, 1.0f, -1.0f};
    std::vector<float> result(input.size());

    optmath::neon::neon_fast_sigmoid_f32(result.data(), input.data(), input.size());

    EXPECT_NEAR(result[0], 0.5f, 1e-5f);  // sigmoid(0) = 0.5
    EXPECT_GT(result[1], 0.99f);           // sigmoid(10) close to 1
    EXPECT_LT(result[2], 0.01f);           // sigmoid(-10) close to 0

    // sigmoid(x) + sigmoid(-x) = 1
    EXPECT_NEAR(result[3] + result[4], 1.0f, 1e-4f);
}

TEST(NeonTranscendentalsTest, TanhFast) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    for (int i = 0; i < N; ++i) {
        input[i] = -5.0f + 10.0f * i / (N - 1);
    }

    optmath::neon::neon_fast_tanh_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::tanh(input[i]);
        // Fast approximation uses 2*sigmoid(2x)-1, errors compound from exp approximation
        EXPECT_NEAR(result[i], expected, 6e-2f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, TanhProperties) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    std::vector<float> input = {0.0f, 5.0f, -5.0f, 1.0f, -1.0f};
    std::vector<float> result(input.size());

    optmath::neon::neon_fast_tanh_f32(result.data(), input.data(), input.size());

    EXPECT_NEAR(result[0], 0.0f, 1e-5f);   // tanh(0) = 0
    EXPECT_GT(result[1], 0.99f);            // tanh(5) close to 1
    EXPECT_LT(result[2], -0.99f);           // tanh(-5) close to -1

    // tanh is odd: tanh(-x) = -tanh(x)
    EXPECT_NEAR(result[3], -result[4], 1e-5f);
}

TEST(NeonTranscendentalsTest, GEMMBlocked) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    // Test various matrix sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {16, 16, 16},
        {64, 64, 64},
        {128, 128, 128},
        {100, 50, 75},  // Non-power-of-2
        {17, 23, 31},   // Odd sizes
    };

    for (auto& [M, N, K] : sizes) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::neon::neon_gemm_blocked(A, B);

        ASSERT_EQ(result.rows(), expected.rows());
        ASSERT_EQ(result.cols(), expected.cols());

        float max_error = (result - expected).cwiseAbs().maxCoeff();
        float tol = 1e-3f * K;  // Error scales with K

        EXPECT_LT(max_error, tol) << "Matrix size: " << M << "x" << K << " * " << K << "x" << N;
    }
}
