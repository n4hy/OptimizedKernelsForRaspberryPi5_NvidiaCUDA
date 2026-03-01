#include <gtest/gtest.h>
#include <optmath/sve2_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <vector>

// Helper to check approximate equality
static void expect_approx_equal(const Eigen::VectorXf& a, const Eigen::VectorXf& b, float tol = 1e-4) {
    ASSERT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "at index " << i;
    }
}

static void expect_approx_equal_mat(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, float tol = 1e-4) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a(i), b(i), tol) << "at index " << i;
    }
}

// =========================================================================
// Correctness Tests
// =========================================================================

TEST(SVE2KernelTest, DotProduct_ExactMatch) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    std::vector<int> sizes = {1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33,
                              63, 64, 65, 127, 128, 1000, 10000, 100000};

    for (int N : sizes) {
        Eigen::VectorXf a = Eigen::VectorXf::Random(N);
        Eigen::VectorXf b = Eigen::VectorXf::Random(N);

        float expected = a.dot(b);
        float result = optmath::sve2::sve2_dot(a, b);

        EXPECT_NEAR(result, expected, 1e-2f * N)
            << "Dot product mismatch for size " << N;
    }
}

TEST(SVE2KernelTest, VectorOps_AllSizes) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    std::vector<int> sizes = {1, 3, 7, 15, 16, 17, 31, 32, 33, 64, 128, 1000};

    for (int N : sizes) {
        Eigen::VectorXf a = Eigen::VectorXf::Random(N);
        Eigen::VectorXf b = Eigen::VectorXf::Random(N);
        // Ensure b has no zeros for div
        b = b.cwiseAbs().array() + 0.1f;

        // Add
        {
            Eigen::VectorXf expected = a + b;
            Eigen::VectorXf result = optmath::sve2::sve2_add(a, b);
            expect_approx_equal(result, expected);
        }

        // Sub
        {
            Eigen::VectorXf expected = a - b;
            Eigen::VectorXf result = optmath::sve2::sve2_sub(a, b);
            expect_approx_equal(result, expected);
        }

        // Mul
        {
            Eigen::VectorXf expected = a.array() * b.array();
            Eigen::VectorXf result = optmath::sve2::sve2_mul(a, b);
            expect_approx_equal(result, expected);
        }

        // Div
        {
            Eigen::VectorXf expected = a.array() / b.array();
            Eigen::VectorXf result = optmath::sve2::sve2_div(a, b);
            expect_approx_equal(result, expected);
        }
    }
}

TEST(SVE2KernelTest, Reductions_AllSizes) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    std::vector<int> sizes = {1, 3, 7, 15, 16, 17, 32, 64, 128, 256, 1000};

    for (int N : sizes) {
        Eigen::VectorXf a = Eigen::VectorXf::Random(N);

        EXPECT_NEAR(optmath::sve2::sve2_reduce_sum(a), a.sum(), 1e-3f)
            << "reduce_sum mismatch for size " << N;
        EXPECT_EQ(optmath::sve2::sve2_reduce_max(a), a.maxCoeff())
            << "reduce_max mismatch for size " << N;
        EXPECT_EQ(optmath::sve2::sve2_reduce_min(a), a.minCoeff())
            << "reduce_min mismatch for size " << N;
    }
}

TEST(SVE2KernelTest, ComplexMul_FCMAvsReference) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    int N = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(N);

    Eigen::VectorXcf result = optmath::sve2::sve2_complex_mul(a, b);
    Eigen::VectorXcf expected = a.array() * b.array();

    ASSERT_EQ(result.size(), expected.size());
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(result[i].real(), expected[i].real(), 1e-4f) << "real part at index " << i;
        EXPECT_NEAR(result[i].imag(), expected[i].imag(), 1e-4f) << "imag part at index " << i;
    }
}

TEST(SVE2KernelTest, ComplexConjMul_FCMAvsReference) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    int N = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(N);

    Eigen::VectorXcf result = optmath::sve2::sve2_complex_conj_mul(a, b);
    Eigen::VectorXcf expected = a.array() * b.conjugate().array();

    ASSERT_EQ(result.size(), expected.size());
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(result[i].real(), expected[i].real(), 1e-4f) << "real part at index " << i;
        EXPECT_NEAR(result[i].imag(), expected[i].imag(), 1e-4f) << "imag part at index " << i;
    }
}

TEST(SVE2KernelTest, ComplexDot_Accumulation) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    std::vector<int> sizes = {16, 64, 256, 1024};

    for (int N : sizes) {
        Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);
        Eigen::VectorXcf b = Eigen::VectorXcf::Random(N);

        std::complex<float> result = optmath::sve2::sve2_complex_dot(a, b);

        // Manual reference: sum(a[i] * conj(b[i]))
        std::complex<float> expected(0.0f, 0.0f);
        for (int i = 0; i < N; ++i) {
            expected += a[i] * std::conj(b[i]);
        }

        float tol = 1e-2f * std::sqrt(static_cast<float>(N));
        EXPECT_NEAR(result.real(), expected.real(), tol)
            << "real part mismatch for size " << N;
        EXPECT_NEAR(result.imag(), expected.imag(), tol)
            << "imag part mismatch for size " << N;
    }
}

TEST(SVE2KernelTest, GEMM_SmallKnown) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    // 2x2
    {
        Eigen::MatrixXf A(2, 2);
        A << 1, 2, 3, 4;
        Eigen::MatrixXf B(2, 2);
        B << 5, 6, 7, 8;
        Eigen::MatrixXf expected(2, 2);
        expected << 19, 22, 43, 50;

        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);
        expect_approx_equal_mat(result, expected, 1e-4f);
    }

    // 3x3
    {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 3);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(3, 3);
        Eigen::MatrixXf expected = A * B;

        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);
        expect_approx_equal_mat(result, expected, 1e-4f);
    }

    // 4x4
    {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(4, 4);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(4, 4);
        Eigen::MatrixXf expected = A * B;

        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);
        expect_approx_equal_mat(result, expected, 1e-4f);
    }
}

TEST(SVE2KernelTest, GEMM_MediumRandom) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    std::vector<int> sizes = {64, 128, 256};

    for (int N : sizes) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);

        expect_approx_equal_mat(result, expected, 1e-2f);
    }
}

TEST(SVE2KernelTest, GEMM_LargeStress) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    // 512x512
    {
        int N = 512;
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);

        expect_approx_equal_mat(result, expected, 5e-2f);
    }

    // 1024x1024
    {
        int N = 1024;
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);

        expect_approx_equal_mat(result, expected, 1e-1f);
    }
}

TEST(SVE2KernelTest, GEMM_NonSquare) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    // 32x64 * 64x16
    {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(32, 64);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(64, 16);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);

        expect_approx_equal_mat(result, expected, 1e-2f);
    }

    // 128x32 * 32x256
    {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(128, 32);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(32, 256);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);

        expect_approx_equal_mat(result, expected, 1e-2f);
    }

    // 17x33 * 33x9
    {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(17, 33);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(33, 9);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);

        expect_approx_equal_mat(result, expected, 1e-2f);
    }
}

TEST(SVE2KernelTest, Transcendentals_Accuracy) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    int N = 10000;
    std::vector<float> input(N);
    std::vector<float> output(N);

    // exp: range [-5, 5], relative tolerance 0.15
    {
        for (int i = 0; i < N; ++i) {
            input[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N - 1);
        }

        optmath::sve2::sve2_fast_exp_f32(output.data(), input.data(), N);

        for (int i = 0; i < N; ++i) {
            float ref = std::exp(input[i]);
            float rel_err = std::abs(output[i] - ref) / (std::abs(ref) + 1e-10f);
            EXPECT_LT(rel_err, 0.15f) << "exp relative error at index " << i
                << " input=" << input[i] << " output=" << output[i] << " ref=" << ref;
        }
    }

    // sin: range [-pi, pi], absolute tolerance 1e-4
    {
        for (int i = 0; i < N; ++i) {
            input[i] = -static_cast<float>(M_PI) + 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(N - 1);
        }

        optmath::sve2::sve2_fast_sin_f32(output.data(), input.data(), N);

        for (int i = 0; i < N; ++i) {
            float ref = std::sin(input[i]);
            EXPECT_NEAR(output[i], ref, 1e-4f) << "sin at index " << i
                << " input=" << input[i];
        }
    }

    // cos: range [-pi, pi], absolute tolerance 1e-4
    {
        for (int i = 0; i < N; ++i) {
            input[i] = -static_cast<float>(M_PI) + 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(N - 1);
        }

        optmath::sve2::sve2_fast_cos_f32(output.data(), input.data(), N);

        for (int i = 0; i < N; ++i) {
            float ref = std::cos(input[i]);
            EXPECT_NEAR(output[i], ref, 1e-4f) << "cos at index " << i
                << " input=" << input[i];
        }
    }

    // sigmoid: range [-10, 10], absolute tolerance 0.03
    {
        for (int i = 0; i < N; ++i) {
            input[i] = -10.0f + 20.0f * static_cast<float>(i) / static_cast<float>(N - 1);
        }

        optmath::sve2::sve2_fast_sigmoid_f32(output.data(), input.data(), N);

        for (int i = 0; i < N; ++i) {
            float ref = 1.0f / (1.0f + std::exp(-input[i]));
            EXPECT_NEAR(output[i], ref, 0.03f) << "sigmoid at index " << i
                << " input=" << input[i];
        }
    }

    // tanh: range [-5, 5], absolute tolerance 0.06
    {
        for (int i = 0; i < N; ++i) {
            input[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N - 1);
        }

        optmath::sve2::sve2_fast_tanh_f32(output.data(), input.data(), N);

        for (int i = 0; i < N; ++i) {
            float ref = std::tanh(input[i]);
            EXPECT_NEAR(output[i], ref, 0.06f) << "tanh at index " << i
                << " input=" << input[i];
        }
    }
}

TEST(SVE2KernelTest, FIR_Filter) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    int nx = 100;
    int nh = 5;
    Eigen::VectorXf x = Eigen::VectorXf::Random(nx);
    Eigen::VectorXf h = Eigen::VectorXf::Random(nh);

    // Manual reference: y[i] = sum(x[i+k] * h[k])
    Eigen::VectorXf expected(nx - nh + 1);
    for (int i = 0; i < expected.size(); ++i) {
        float sum = 0.0f;
        for (int k = 0; k < nh; ++k) {
            sum += x[i + k] * h[k];
        }
        expected[i] = sum;
    }

    Eigen::VectorXf result = optmath::sve2::sve2_fir(x, h);
    expect_approx_equal(result, expected, 1e-4f);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

TEST(SVE2KernelTest, ZeroLength) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    Eigen::VectorXf a(0);
    Eigen::VectorXf b(0);

    // sve2_add with zero-length vectors should not crash
    Eigen::VectorXf add_result = optmath::sve2::sve2_add(a, b);
    EXPECT_EQ(add_result.size(), 0);

    // sve2_dot with zero-length vectors should return 0
    float dot_result = optmath::sve2::sve2_dot(a, b);
    EXPECT_EQ(dot_result, 0.0f);

    // sve2_reduce_sum with zero-length vector should return 0
    float sum_result = optmath::sve2::sve2_reduce_sum(a);
    EXPECT_EQ(sum_result, 0.0f);
}

TEST(SVE2KernelTest, SingleElement) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    Eigen::VectorXf a(1);
    a[0] = 3.0f;
    Eigen::VectorXf b(1);
    b[0] = 7.0f;

    // sve2_add
    {
        Eigen::VectorXf result = optmath::sve2::sve2_add(a, b);
        ASSERT_EQ(result.size(), 1);
        EXPECT_NEAR(result[0], 10.0f, 1e-6f);
    }

    // sve2_dot
    {
        float result = optmath::sve2::sve2_dot(a, b);
        EXPECT_NEAR(result, 21.0f, 1e-6f);
    }

    // sve2_reduce_sum
    {
        float result = optmath::sve2::sve2_reduce_sum(a);
        EXPECT_NEAR(result, 3.0f, 1e-6f);
    }
}

TEST(SVE2KernelTest, OddSizes) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    std::vector<int> sizes = {3, 5, 7, 9, 11, 13};

    for (int N : sizes) {
        Eigen::VectorXf a = Eigen::VectorXf::Random(N);
        Eigen::VectorXf b = Eigen::VectorXf::Random(N);

        // sve2_add
        {
            Eigen::VectorXf expected = a + b;
            Eigen::VectorXf result = optmath::sve2::sve2_add(a, b);
            expect_approx_equal(result, expected);
        }

        // sve2_dot
        {
            float expected = a.dot(b);
            float result = optmath::sve2::sve2_dot(a, b);
            EXPECT_NEAR(result, expected, 1e-4f)
                << "dot mismatch for odd size " << N;
        }
    }
}

TEST(SVE2KernelTest, NaN_Inf_Handling) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    // Test with NaN
    {
        Eigen::VectorXf a(4);
        a << 1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f;

        float result = optmath::sve2::sve2_reduce_sum(a);
        EXPECT_TRUE(std::isnan(result)) << "Expected NaN result when input contains NaN, got " << result;
    }

    // Test with Inf
    {
        Eigen::VectorXf a(4);
        a << 1.0f, std::numeric_limits<float>::infinity(), 3.0f, 4.0f;

        float result = optmath::sve2::sve2_reduce_sum(a);
        EXPECT_TRUE(std::isinf(result)) << "Expected Inf result when input contains Inf, got " << result;
    }
}

// =========================================================================
// Stress Tests
// =========================================================================

TEST(SVE2KernelTest, LargeVector_1M) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    int N = 1000000;
    Eigen::VectorXf a = Eigen::VectorXf::Random(N);
    Eigen::VectorXf b = Eigen::VectorXf::Random(N);

    // Add
    {
        Eigen::VectorXf expected = a + b;
        Eigen::VectorXf result = optmath::sve2::sve2_add(a, b);
        expect_approx_equal(result, expected);
    }

    // Dot product
    {
        float expected = a.dot(b);
        float result = optmath::sve2::sve2_dot(a, b);
        EXPECT_NEAR(result, expected, 1e-2f * N);
    }
}

TEST(SVE2KernelTest, RepeatedGEMM_Stability) {
    if (!optmath::sve2::is_available()) {
        GTEST_SKIP() << "SVE2 not available, skipping test.";
    }

    int N = 64;
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);

    Eigen::MatrixXf first_result = optmath::sve2::sve2_gemm_blocked(A, B);

    for (int iter = 1; iter < 100; ++iter) {
        Eigen::MatrixXf result = optmath::sve2::sve2_gemm_blocked(A, B);
        expect_approx_equal_mat(result, first_result, 1e-4f);
    }
}
