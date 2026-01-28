#pragma once

#include <vector>
#include <cstddef>
#include <complex>
#include <Eigen/Dense>

namespace optmath {
namespace neon {

    /**
     * @brief Checks if NEON acceleration was compiled in.
     */
    bool is_available();

    // =========================================================================
    // Core Intrinsics Wrappers
    // =========================================================================

    float neon_dot_f32(const float* a, const float* b, std::size_t n);
    double neon_dot_f64(const double* a, const double* b, std::size_t n);

    void neon_add_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_sub_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_mul_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_div_f32(float* out, const float* a, const float* b, std::size_t n);

    // Reductions
    float neon_norm_f32(const float* a, std::size_t n);
    float neon_reduce_sum_f32(const float* a, std::size_t n);
    float neon_reduce_max_f32(const float* a, std::size_t n);
    float neon_reduce_min_f32(const float* a, std::size_t n);

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    // Basic 4x4 microkernel: C += A * B
    void neon_gemm_4x4_f32(float* C, const float* A, std::size_t lda, const float* B, std::size_t ldb, std::size_t ldc);

    // Optimized blocked GEMM with cache blocking (MC=128, KC=256, NC=512)
    void neon_gemm_blocked_f32(float* C, const float* A, const float* B,
                               std::size_t M, std::size_t N, std::size_t K,
                               std::size_t lda, std::size_t ldb, std::size_t ldc);

    // =========================================================================
    // DSP / Filter Operations
    // =========================================================================

    void neon_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y);

    // =========================================================================
    // Activation Functions
    // =========================================================================

    void neon_relu_f32(float* data, std::size_t n);
    void neon_sigmoid_f32(float* data, std::size_t n);  // Scalar version
    void neon_tanh_f32(float* data, std::size_t n);     // Scalar version

    // =========================================================================
    // Vectorized Transcendental Functions (Fast Approximations)
    // =========================================================================
    // These are SIMD-optimized approximations trading accuracy for speed.
    // Typical accuracy: exp ~12%, sigmoid ~3%, tanh ~6% relative error.
    // Suitable for ML inference and non-precision-critical DSP.

    /**
     * @brief Fast vectorized exp using range reduction and 6th-order polynomial
     * ~12% relative error at extremes, better near zero
     */
    void neon_fast_exp_f32(float* out, const float* in, std::size_t n);

    /**
     * @brief Fast vectorized sin using Chebyshev polynomial
     * ~1e-5 accuracy (uses range reduction to [-pi, pi])
     */
    void neon_fast_sin_f32(float* out, const float* in, std::size_t n);

    /**
     * @brief Fast vectorized cos using sin(x + pi/2)
     */
    void neon_fast_cos_f32(float* out, const float* in, std::size_t n);

    /**
     * @brief Fast vectorized sigmoid: 1/(1+exp(-x))
     * ~3% error (inherits from fast exp)
     */
    void neon_fast_sigmoid_f32(float* out, const float* in, std::size_t n);

    /**
     * @brief Fast vectorized tanh: 2*sigmoid(2x)-1
     * ~6% error (compounds from sigmoid)
     */
    void neon_fast_tanh_f32(float* out, const float* in, std::size_t n);

    // =========================================================================
    // Complex Number Operations
    // =========================================================================

    // Separate real/imaginary format
    void neon_complex_mul_f32(float* out_re, float* out_im,
                              const float* a_re, const float* a_im,
                              const float* b_re, const float* b_im,
                              std::size_t n);

    void neon_complex_conj_mul_f32(float* out_re, float* out_im,
                                   const float* a_re, const float* a_im,
                                   const float* b_re, const float* b_im,
                                   std::size_t n);

    // Interleaved format (IQ: real, imag, real, imag, ...)
    void neon_complex_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_complex_conj_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n);

    // Complex dot product: sum(a * conj(b))
    void neon_complex_dot_f32(float* out_re, float* out_im,
                              const float* a_re, const float* a_im,
                              const float* b_re, const float* b_im,
                              std::size_t n);

    // Magnitude and phase
    void neon_complex_magnitude_f32(float* out, const float* re, const float* im, std::size_t n);
    void neon_complex_magnitude_squared_f32(float* out, const float* re, const float* im, std::size_t n);
    void neon_complex_phase_f32(float* out, const float* re, const float* im, std::size_t n);

    // Complex arithmetic
    void neon_complex_add_f32(float* out_re, float* out_im,
                              const float* a_re, const float* a_im,
                              const float* b_re, const float* b_im,
                              std::size_t n);

    void neon_complex_scale_f32(float* out_re, float* out_im,
                                const float* in_re, const float* in_im,
                                float scale_re, float scale_im,
                                std::size_t n);

    void neon_complex_exp_f32(float* out_re, float* out_im, const float* phase, std::size_t n);

    // --- Eigen Wrappers ---

    float neon_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    double neon_dot(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

    Eigen::VectorXf neon_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf neon_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b);

    // Computes y = x * h
    Eigen::VectorXf neon_fir(const Eigen::VectorXf& x, const Eigen::VectorXf& h);

    // In-place activations
    void neon_relu(Eigen::VectorXf& x);
    void neon_sigmoid(Eigen::VectorXf& x);
    void neon_tanh(Eigen::VectorXf& x);

    // Simple Matrix Multiplication wrapper (A * B)
    Eigen::MatrixXf neon_gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);

    // Optimized blocked GEMM
    Eigen::MatrixXf neon_gemm_blocked(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);

    Eigen::VectorXf neon_sub(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf neon_div(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    float neon_norm(const Eigen::VectorXf& a);

    float neon_reduce_sum(const Eigen::VectorXf& a);
    float neon_reduce_max(const Eigen::VectorXf& a);
    float neon_reduce_min(const Eigen::VectorXf& a);

    Eigen::MatrixXf neon_mat_scale(const Eigen::MatrixXf& A, float s);
    Eigen::MatrixXf neon_mat_transpose(const Eigen::MatrixXf& A);
    Eigen::VectorXf neon_mat_vec_mul(const Eigen::MatrixXf& A, const Eigen::VectorXf& v);

    // =========================================================================
    // Eigen Wrappers for Complex Operations
    // =========================================================================

    Eigen::VectorXcf neon_complex_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
    Eigen::VectorXcf neon_complex_conj_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
    std::complex<float> neon_complex_dot(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
    Eigen::VectorXf neon_complex_magnitude(const Eigen::VectorXcf& a);
    Eigen::VectorXf neon_complex_phase(const Eigen::VectorXcf& a);

}
}
