#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>

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

TEST(NeonKernelTest, VectorOperations) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1000;
    Eigen::VectorXf a = Eigen::VectorXf::Random(N);
    Eigen::VectorXf b = Eigen::VectorXf::Random(N);
    // Ensure no division by zero for div test
    b = b.cwiseAbs().array() + 0.1f;

    // Add
    {
        Eigen::VectorXf expected = a + b;
        Eigen::VectorXf result = optmath::neon::neon_add(a, b);
        expect_approx_equal(result, expected);
    }

    // Sub
    {
        Eigen::VectorXf expected = a - b;
        Eigen::VectorXf result = optmath::neon::neon_sub(a, b);
        expect_approx_equal(result, expected);
    }

    // Mul
    {
        Eigen::VectorXf expected = a.array() * b.array();
        Eigen::VectorXf result = optmath::neon::neon_mul(a, b);
        expect_approx_equal(result, expected);
    }

    // Div
    {
        Eigen::VectorXf expected = a.array() / b.array();
        Eigen::VectorXf result = optmath::neon::neon_div(a, b);
        expect_approx_equal(result, expected);
    }

    // Dot
    {
        float expected = a.dot(b);
        float result = optmath::neon::neon_dot(a, b);
        EXPECT_NEAR(result, expected, 1e-2 * N);
    }

    // Norm
    {
        float expected = a.norm();
        float result = optmath::neon::neon_norm(a);
        EXPECT_NEAR(result, expected, 1e-2 * N);
    }
}

TEST(NeonKernelTest, MatrixOperations) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int M = 64;
    int N = 64;
    int K = 64;

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);

    // Gemm (Mul)
    {
        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::neon::neon_gemm(A, B);
        expect_approx_equal_mat(result, expected, 1e-2); // Matrix mul accumulates error
    }

    // Scale
    {
        float s = 2.5f;
        Eigen::MatrixXf expected = A * s;
        Eigen::MatrixXf result = optmath::neon::neon_mat_scale(A, s);
        expect_approx_equal_mat(result, expected);
    }

    // Transpose
    {
        Eigen::MatrixXf expected = A.transpose();
        Eigen::MatrixXf result = optmath::neon::neon_mat_transpose(A);
        expect_approx_equal_mat(result, expected);
    }

    // Mat-Vec Mul
    {
        Eigen::VectorXf v = Eigen::VectorXf::Random(K);
        Eigen::VectorXf expected = A * v;
        Eigen::VectorXf result = optmath::neon::neon_mat_vec_mul(A, v);
        expect_approx_equal(result, expected, 1e-2);
    }
}

TEST(NeonKernelTest, DSPOperations) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    // FIR / Convolution 1D
    {
        int nx = 100;
        int nh = 5;
        Eigen::VectorXf x = Eigen::VectorXf::Random(nx);
        Eigen::VectorXf h = Eigen::VectorXf::Random(nh);

        // Eigen/Manual Reference
        // Our neon_fir implements correlation/dot-product style sliding window.
        // y[i] = sum(x[i+k] * h[k])

        Eigen::VectorXf expected(nx - nh + 1);
        for(int i=0; i < expected.size(); ++i) {
            float sum = 0.0f;
            for(int k=0; k < nh; ++k) sum += x[i+k] * h[k];
            expected[i] = sum;
        }

        Eigen::VectorXf result = optmath::neon::neon_fir(x, h);
        expect_approx_equal(result, expected);
    }
}

TEST(NeonKernelTest, Reductions) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 256;
    Eigen::VectorXf a = Eigen::VectorXf::Random(N);

    EXPECT_NEAR(optmath::neon::neon_reduce_sum(a), a.sum(), 1e-3);
    EXPECT_EQ(optmath::neon::neon_reduce_max(a), a.maxCoeff());
    EXPECT_EQ(optmath::neon::neon_reduce_min(a), a.minCoeff());
}
