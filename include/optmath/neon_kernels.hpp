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
    // Polyphase Resampler
    // =========================================================================

    struct PolyphaseResamplerState {
        std::vector<std::vector<float>> phases;  // Polyphase decomposition [L][n_taps]
        std::size_t L;              // Interpolation factor
        std::size_t M;              // Decimation factor
        std::size_t n_taps;         // Taps per phase
        std::vector<float> delay;   // Delay line for streaming
        std::size_t delay_pos;      // Write position in circular delay line
        std::size_t phase_acc;      // Phase accumulator
    };

    /**
     * @brief Initialize polyphase resampler state.
     * Decomposes prototype filter into L polyphase phases.
     * Filter should be a lowpass with cutoff at min(pi/L, pi/M).
     * Scale filter by L to preserve unity gain after interpolation.
     */
    void neon_resample_init(PolyphaseResamplerState& state,
                            const float* filter, std::size_t filter_len,
                            std::size_t L, std::size_t M);

    /**
     * @brief Process a block through the polyphase resampler (streaming).
     * Returns the number of output samples produced.
     * Caller must allocate out with at least ceil(input_len * L / M) + n_taps floats.
     */
    std::size_t neon_resample_f32(float* out, const float* in, std::size_t input_len,
                                   PolyphaseResamplerState& state);

    /**
     * @brief One-shot polyphase resampler (non-streaming, zero-padded edges).
     * Output length is written to *output_len.
     * Caller must allocate out with at least ceil(input_len * L / M) + 1 floats.
     */
    void neon_resample_oneshot_f32(float* out, std::size_t* output_len,
                                    const float* in, std::size_t input_len,
                                    const float* filter, std::size_t filter_len,
                                    std::size_t L, std::size_t M);

    // =========================================================================
    // Biquad IIR Filter (Direct Form II Transposed)
    // =========================================================================

    struct BiquadCoeffs {
        float b0, b1, b2;  // Numerator (feedforward)
        float a1, a2;       // Denominator (feedback), a0 normalized to 1
    };

    struct BiquadState {
        float s1 = 0.0f;   // DF2T state variable 1
        float s2 = 0.0f;   // DF2T state variable 2
    };

    /**
     * @brief Process samples through a single biquad section (DF2T).
     * out and in may alias (in-place processing supported).
     */
    void neon_biquad_f32(float* out, const float* in, std::size_t n,
                         const BiquadCoeffs& coeffs, BiquadState& state);

    /**
     * @brief Process samples through a cascade of biquad sections.
     * Each section is applied sequentially. out and in may alias.
     */
    void neon_biquad_cascade_f32(float* out, const float* in, std::size_t n,
                                  const BiquadCoeffs* coeffs, BiquadState* states,
                                  std::size_t n_sections);

    // Biquad design helpers (Audio EQ Cookbook formulas)
    BiquadCoeffs neon_biquad_lowpass(float fc, float fs, float Q = 0.7071067811865476f);
    BiquadCoeffs neon_biquad_highpass(float fc, float fs, float Q = 0.7071067811865476f);
    BiquadCoeffs neon_biquad_bandpass(float fc, float fs, float Q = 1.0f);
    BiquadCoeffs neon_biquad_notch(float fc, float fs, float Q = 1.0f);

    // =========================================================================
    // 2D Convolution (row-major layout)
    // =========================================================================

    /**
     * @brief General 2D convolution (valid mode, no padding).
     * Input and kernel in row-major order.
     * Output size: (in_rows - kernel_rows + 1) x (in_cols - kernel_cols + 1).
     */
    void neon_conv2d_f32(float* out, const float* in,
                         std::size_t in_rows, std::size_t in_cols,
                         const float* kernel, std::size_t kernel_rows, std::size_t kernel_cols);

    /**
     * @brief Separable 2D convolution (valid mode).
     * Applies row_kernel along columns first, then col_kernel along rows.
     * Output size: (in_rows - col_len + 1) x (in_cols - row_len + 1).
     */
    void neon_conv2d_separable_f32(float* out, const float* in,
                                    std::size_t in_rows, std::size_t in_cols,
                                    const float* row_kernel, std::size_t row_kernel_len,
                                    const float* col_kernel, std::size_t col_kernel_len);

    /**
     * @brief Optimized 3x3 convolution with fully unrolled kernel.
     * Output size: (in_rows - 2) x (in_cols - 2).
     */
    void neon_conv2d_3x3_f32(float* out, const float* in,
                              std::size_t in_rows, std::size_t in_cols,
                              const float kernel[9]);

    /**
     * @brief Optimized 5x5 convolution with unrolled kernel.
     * Output size: (in_rows - 4) x (in_cols - 4).
     */
    void neon_conv2d_5x5_f32(float* out, const float* in,
                              std::size_t in_rows, std::size_t in_cols,
                              const float kernel[25]);

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

    // Eigen wrappers for resampler
    Eigen::VectorXf neon_resample(const Eigen::VectorXf& in,
                                   const Eigen::VectorXf& filter,
                                   std::size_t L, std::size_t M);

    // Eigen wrappers for biquad
    Eigen::VectorXf neon_biquad(const Eigen::VectorXf& in,
                                 const BiquadCoeffs& coeffs);

    // Eigen wrappers for 2D convolution
    Eigen::MatrixXf neon_conv2d(const Eigen::MatrixXf& in, const Eigen::MatrixXf& kernel);

    // =========================================================================
    // Dense Linear Algebra (column-major layout)
    // =========================================================================

    // --- Triangular Solve (raw pointer, column-major) ---

    /** @brief Forward substitution: solve L*x = b (L lower triangular). b overwritten with x. */
    void neon_trsv_lower_f32(float* b, const float* L, std::size_t n, std::size_t ldl);

    /** @brief Backward substitution: solve U*x = b (U upper triangular). b overwritten with x. */
    void neon_trsv_upper_f32(float* b, const float* U, std::size_t n, std::size_t ldu);

    /** @brief Forward substitution with unit diagonal: solve L*x = b where diag(L) = 1. */
    void neon_trsv_lower_unit_f32(float* b, const float* L, std::size_t n, std::size_t ldl);

    /** @brief Solve L^T*x = b using lower triangular L. b overwritten with x. */
    void neon_trsv_lower_trans_f32(float* b, const float* L, std::size_t n, std::size_t ldl);

    /** @brief Multi-RHS lower triangular solve: solve L*X = B. B overwritten with X. */
    void neon_trsm_lower_f32(float* B, const float* L, std::size_t n, std::size_t nrhs,
                              std::size_t ldl, std::size_t ldb);

    /** @brief Multi-RHS upper triangular solve: solve U*X = B. B overwritten with X. */
    void neon_trsm_upper_f32(float* B, const float* U, std::size_t n, std::size_t nrhs,
                              std::size_t ldu, std::size_t ldb);

    // --- Decompositions (in-place, column-major) ---

    /** @brief Cholesky decomposition A = L*L^T. A overwritten with L (lower).
     *  @return 0 on success, or 1-based index of failing pivot if not SPD. */
    int neon_cholesky_f32(float* A, std::size_t n, std::size_t lda);

    /** @brief LU decomposition with partial pivoting. A overwritten with L\U.
     *  @param piv Output pivot indices (size m).
     *  @return 0 on success, or 1-based index of zero pivot if singular. */
    int neon_lu_f32(float* A, int* piv, std::size_t m, std::size_t n, std::size_t lda);

    /** @brief QR decomposition via Householder reflections.
     *  A overwritten with R (upper part) and Householder vectors (lower part).
     *  @param tau Householder scalars (size min(m,n)). */
    void neon_qr_f32(float* A, float* tau, std::size_t m, std::size_t n, std::size_t lda);

    /** @brief Extract explicit Q from stored Householder reflectors. */
    void neon_qr_extract_q_f32(float* Q, const float* A, const float* tau,
                                 std::size_t m, std::size_t n, std::size_t lda, std::size_t ldq);

    // --- Solvers ---

    /** @brief General solve A*x = b via LU. A and b overwritten. @return 0 or error. */
    int neon_solve_f32(float* A, float* b, std::size_t n, std::size_t lda);

    /** @brief SPD solve A*x = b via Cholesky. A and b overwritten. @return 0 or error. */
    int neon_solve_spd_f32(float* A, float* b, std::size_t n, std::size_t lda);

    /** @brief Matrix inverse via LU: Ainv = A^{-1}. @return 0 or error. */
    int neon_inverse_f32(float* Ainv, const float* A, std::size_t n,
                          std::size_t lda, std::size_t ldinv);

    // --- Eigen Wrappers for Dense Linear Algebra ---

    /** @brief Cholesky: returns lower L such that A = L*L^T. Empty matrix on failure. */
    Eigen::MatrixXf neon_cholesky(const Eigen::MatrixXf& A);

    /** @brief LU with partial pivoting: returns (LU combined, pivot vector). */
    std::pair<Eigen::MatrixXf, Eigen::VectorXi> neon_lu(const Eigen::MatrixXf& A);

    /** @brief QR: returns (Q, R). */
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> neon_qr(const Eigen::MatrixXf& A);

    /** @brief Triangular solve L*x = b (L lower). */
    Eigen::VectorXf neon_trsv_lower(const Eigen::MatrixXf& L, const Eigen::VectorXf& b);

    /** @brief Triangular solve U*x = b (U upper). */
    Eigen::VectorXf neon_trsv_upper(const Eigen::MatrixXf& U, const Eigen::VectorXf& b);

    /** @brief General solve A*x = b. */
    Eigen::VectorXf neon_solve(const Eigen::MatrixXf& A, const Eigen::VectorXf& b);

    /** @brief SPD solve A*x = b. */
    Eigen::VectorXf neon_solve_spd(const Eigen::MatrixXf& A, const Eigen::VectorXf& b);

    /** @brief Matrix inverse. Empty matrix on failure. */
    Eigen::MatrixXf neon_inverse(const Eigen::MatrixXf& A);

}
}
