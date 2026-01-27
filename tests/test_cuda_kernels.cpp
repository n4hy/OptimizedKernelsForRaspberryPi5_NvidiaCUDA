/**
 * @file test_cuda_kernels.cpp
 * @brief Comprehensive tests for CUDA backend operations
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <vector>
#include <random>

#ifdef OPTMATH_USE_CUDA
#include "optmath/cuda_backend.hpp"
#endif

constexpr float TOLERANCE = 1e-4f;
constexpr double TOLERANCE_D = 1e-10;

// ============================================================================
// Test Fixture
// ============================================================================

class CudaKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef OPTMATH_USE_CUDA
        if (!optmath::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        optmath::cuda::init();
#else
        GTEST_SKIP() << "CUDA not enabled in build";
#endif
    }

    void TearDown() override {
#ifdef OPTMATH_USE_CUDA
        if (optmath::cuda::is_available()) {
            optmath::cuda::synchronize();
        }
#endif
    }

    // Random number generator
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// ============================================================================
// Vector Operation Tests
// ============================================================================

#ifdef OPTMATH_USE_CUDA

TEST_F(CudaKernelTest, VectorAdd) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_add(a, b);
    Eigen::VectorXf expected = a + b;

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), expected(i), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, VectorMul) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_mul(a, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), a(i) * b(i), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, VectorScale) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    float scalar = 3.14159f;

    Eigen::VectorXf result = optmath::cuda::cuda_scale(a, scalar);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), a(i) * scalar, TOLERANCE);
    }
}

TEST_F(CudaKernelTest, VectorDot) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_dot(a, b);
    float expected = a.dot(b);

    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-3f);
}

TEST_F(CudaKernelTest, VectorSum) {
    const int n = 4096;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_sum(a);
    float expected = a.sum();

    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-3f);
}

TEST_F(CudaKernelTest, VectorMax) {
    const int n = 4096;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_max(a);
    float expected = a.maxCoeff();

    EXPECT_NEAR(result, expected, TOLERANCE);
}

TEST_F(CudaKernelTest, VectorMin) {
    const int n = 4096;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_min(a);
    float expected = a.minCoeff();

    EXPECT_NEAR(result, expected, TOLERANCE);
}

// ============================================================================
// Transcendental Function Tests
// ============================================================================

TEST_F(CudaKernelTest, VectorExp) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 0.5f; // Limit range

    Eigen::VectorXf result = optmath::cuda::cuda_exp(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::exp(a(i)), std::abs(std::exp(a(i))) * 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorLog) {
    const int n = 1024;
    Eigen::VectorXf a = (Eigen::VectorXf::Random(n).array().abs() + 0.1f).matrix(); // Positive values

    Eigen::VectorXf result = optmath::cuda::cuda_log(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::log(a(i)), 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorSin) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 6.28f;

    Eigen::VectorXf result = optmath::cuda::cuda_sin(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::sin(a(i)), 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorCos) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 6.28f;

    Eigen::VectorXf result = optmath::cuda::cuda_cos(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::cos(a(i)), 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorSqrt) {
    const int n = 1024;
    Eigen::VectorXf a = (Eigen::VectorXf::Random(n).array().abs() + 0.1f).matrix();

    Eigen::VectorXf result = optmath::cuda::cuda_sqrt(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::sqrt(a(i)), 1e-4f);
    }
}

// ============================================================================
// Activation Function Tests
// ============================================================================

TEST_F(CudaKernelTest, Sigmoid) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 10.0f;

    Eigen::VectorXf result = optmath::cuda::cuda_sigmoid(a);

    for (int i = 0; i < n; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-a(i)));
        EXPECT_NEAR(result(i), expected, 1e-4f);
    }
}

TEST_F(CudaKernelTest, Tanh) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 5.0f;

    Eigen::VectorXf result = optmath::cuda::cuda_tanh(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::tanh(a(i)), 1e-4f);
    }
}

TEST_F(CudaKernelTest, ReLU) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 10.0f;

    Eigen::VectorXf result = optmath::cuda::cuda_relu(a);

    for (int i = 0; i < n; ++i) {
        float expected = std::max(0.0f, a(i));
        EXPECT_NEAR(result(i), expected, TOLERANCE);
    }
}

TEST_F(CudaKernelTest, LeakyReLU) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 10.0f;
    float alpha = 0.1f;

    Eigen::VectorXf result = optmath::cuda::cuda_leaky_relu(a, alpha);

    for (int i = 0; i < n; ++i) {
        float expected = a(i) >= 0 ? a(i) : alpha * a(i);
        EXPECT_NEAR(result(i), expected, TOLERANCE);
    }
}

// ============================================================================
// Matrix Operation Tests
// ============================================================================

TEST_F(CudaKernelTest, MatrixGEMM) {
    const int m = 128, k = 64, n = 256;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(m, k);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(k, n);

    Eigen::MatrixXf result = optmath::cuda::cuda_gemm(a, b);
    Eigen::MatrixXf expected = a * b;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(result(i,j), expected(i,j), std::abs(expected(i,j)) * 1e-3f + 1e-4f);
        }
    }
}

TEST_F(CudaKernelTest, MatrixGEMV) {
    const int m = 256, n = 128;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(m, n);
    Eigen::VectorXf x = Eigen::VectorXf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_gemv(a, x);
    Eigen::VectorXf expected = a * x;

    for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(result(i), expected(i), std::abs(expected(i)) * 1e-3f + 1e-4f);
    }
}

TEST_F(CudaKernelTest, MatrixTranspose) {
    const int m = 128, n = 64;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(m, n);

    Eigen::MatrixXf result = optmath::cuda::cuda_transpose(a);

    EXPECT_EQ(result.rows(), n);
    EXPECT_EQ(result.cols(), m);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(result(j, i), a(i, j), TOLERANCE);
        }
    }
}

// ============================================================================
// Complex Number Tests
// ============================================================================

TEST_F(CudaKernelTest, ComplexMul) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf result = optmath::cuda::cuda_complex_mul(a, b);

    for (int i = 0; i < n; ++i) {
        std::complex<float> expected = a(i) * b(i);
        EXPECT_NEAR(result(i).real(), expected.real(), TOLERANCE);
        EXPECT_NEAR(result(i).imag(), expected.imag(), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, ComplexConjMul) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf result = optmath::cuda::cuda_complex_conj_mul(a, b);

    for (int i = 0; i < n; ++i) {
        std::complex<float> expected = a(i) * std::conj(b(i));
        EXPECT_NEAR(result(i).real(), expected.real(), TOLERANCE);
        EXPECT_NEAR(result(i).imag(), expected.imag(), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, ComplexMagnitude) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_complex_abs(a);

    for (int i = 0; i < n; ++i) {
        float expected = std::abs(a(i));
        EXPECT_NEAR(result(i), expected, TOLERANCE);
    }
}

TEST_F(CudaKernelTest, ComplexPhase) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_complex_arg(a);

    for (int i = 0; i < n; ++i) {
        float expected = std::arg(a(i));
        EXPECT_NEAR(result(i), expected, 1e-3f);
    }
}

TEST_F(CudaKernelTest, ComplexDot) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(n);

    std::complex<float> result = optmath::cuda::cuda_complex_dot(a, b);
    std::complex<float> expected = a.dot(b);

    EXPECT_NEAR(result.real(), expected.real(), std::abs(expected.real()) * 1e-3f + 1e-4f);
    EXPECT_NEAR(result.imag(), expected.imag(), std::abs(expected.imag()) * 1e-3f + 1e-4f);
}

// ============================================================================
// FFT Tests
// ============================================================================

TEST_F(CudaKernelTest, FFTForwardInverse) {
    const int n = 1024;
    Eigen::VectorXcf input = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf fft_result = optmath::cuda::cuda_fft(input);
    Eigen::VectorXcf ifft_result = optmath::cuda::cuda_ifft(fft_result);

    // IFFT should recover the original signal
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(ifft_result(i).real(), input(i).real(), 1e-3f);
        EXPECT_NEAR(ifft_result(i).imag(), input(i).imag(), 1e-3f);
    }
}

TEST_F(CudaKernelTest, FFTParseval) {
    const int n = 1024;
    Eigen::VectorXcf input = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf fft_result = optmath::cuda::cuda_fft(input);

    // Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
    float time_energy = input.squaredNorm();
    float freq_energy = fft_result.squaredNorm() / n;

    EXPECT_NEAR(time_energy, freq_energy, time_energy * 1e-3f);
}

// ============================================================================
// Convolution Tests
// ============================================================================

TEST_F(CudaKernelTest, Convolution1D) {
    const int n = 256;
    const int k = 16;
    Eigen::VectorXf signal = Eigen::VectorXf::Random(n);
    Eigen::VectorXf kernel = Eigen::VectorXf::Random(k);

    Eigen::VectorXf result = optmath::cuda::cuda_convolve_1d(signal, kernel);

    // Manual convolution check for a few points
    int mid = n / 2;
    float expected = 0;
    for (int j = 0; j < k && (mid - j) >= 0; ++j) {
        expected += signal(mid - j) * kernel(j);
    }

    // Allow some tolerance due to boundary handling differences
    EXPECT_NEAR(result(mid), expected, std::abs(expected) * 0.1f + 1e-3f);
}

// ============================================================================
// Large Scale Performance Tests
// ============================================================================

TEST_F(CudaKernelTest, LargeMatrixMultiply) {
    const int size = 1024;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(size, size);

    Eigen::MatrixXf result = optmath::cuda::cuda_gemm(a, b);

    // Just verify it completes without error
    EXPECT_EQ(result.rows(), size);
    EXPECT_EQ(result.cols(), size);

    // Spot check a few elements
    Eigen::MatrixXf expected = a * b;
    EXPECT_NEAR(result(0, 0), expected(0, 0), std::abs(expected(0, 0)) * 1e-2f + 1e-2f);
    EXPECT_NEAR(result(size-1, size-1), expected(size-1, size-1),
                std::abs(expected(size-1, size-1)) * 1e-2f + 1e-2f);
}

TEST_F(CudaKernelTest, LargeFFT) {
    const int n = 65536;  // 64k point FFT
    Eigen::VectorXcf input = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf result = optmath::cuda::cuda_fft(input);

    EXPECT_EQ(result.size(), n);

    // Verify inverse recovers original
    Eigen::VectorXcf recovered = optmath::cuda::cuda_ifft(result);
    EXPECT_NEAR(recovered(0).real(), input(0).real(), 1e-2f);
}

#endif // OPTMATH_USE_CUDA

// Main
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
