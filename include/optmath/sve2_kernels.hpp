#pragma once

#include <vector>
#include <cstddef>
#include <complex>
#include <Eigen/Dense>

namespace optmath {
namespace sve2 {

    /**
     * @brief Checks if SVE2 acceleration was compiled in and hardware supports it.
     */
    bool is_available();

    // =========================================================================
    // Core Vector Operations (predicated - no tail loops)
    // =========================================================================

    float sve2_dot_f32(const float* a, const float* b, std::size_t n);

    void sve2_add_f32(float* out, const float* a, const float* b, std::size_t n);
    void sve2_sub_f32(float* out, const float* a, const float* b, std::size_t n);
    void sve2_mul_f32(float* out, const float* a, const float* b, std::size_t n);
    void sve2_div_f32(float* out, const float* a, const float* b, std::size_t n);

    // Reductions
    float sve2_norm_f32(const float* a, std::size_t n);
    float sve2_reduce_sum_f32(const float* a, std::size_t n);
    float sve2_reduce_max_f32(const float* a, std::size_t n);
    float sve2_reduce_min_f32(const float* a, std::size_t n);

    // =========================================================================
    // Vectorized Transcendental Functions
    // =========================================================================

    void sve2_fast_exp_f32(float* out, const float* in, std::size_t n);
    void sve2_fast_sin_f32(float* out, const float* in, std::size_t n);
    void sve2_fast_cos_f32(float* out, const float* in, std::size_t n);
    void sve2_fast_sigmoid_f32(float* out, const float* in, std::size_t n);
    void sve2_fast_tanh_f32(float* out, const float* in, std::size_t n);

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    /**
     * @brief SVE2 cache-blocked GEMM with tuning for A720 (12MB L3).
     * MC=256, KC=512, NC=1024 for large L3 caches.
     */
    void sve2_gemm_blocked_f32(float* C, const float* A, const float* B,
                                std::size_t M, std::size_t N, std::size_t K,
                                std::size_t lda, std::size_t ldb, std::size_t ldc);

    /**
     * @brief Int8 matrix multiply using SVE2 I8MM instructions.
     * Computes C_f32 = dequant(A_i8 * B_i8) with per-tensor quantization.
     * @param scale_a Quantization scale for A
     * @param scale_b Quantization scale for B
     * @param zero_a Zero point for A
     * @param zero_b Zero point for B
     */
    void sve2_gemm_i8mm(float* C, const int8_t* A, const int8_t* B,
                         std::size_t M, std::size_t N, std::size_t K,
                         std::size_t lda, std::size_t ldb, std::size_t ldc,
                         float scale_a, float scale_b,
                         int32_t zero_a, int32_t zero_b);

    // =========================================================================
    // DSP / Filter Operations
    // =========================================================================

    void sve2_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y);

    // =========================================================================
    // Complex Number Operations (Separate real/imaginary format)
    // =========================================================================

    void sve2_complex_mul_f32(float* out_re, float* out_im,
                               const float* a_re, const float* a_im,
                               const float* b_re, const float* b_im,
                               std::size_t n);

    void sve2_complex_conj_mul_f32(float* out_re, float* out_im,
                                    const float* a_re, const float* a_im,
                                    const float* b_re, const float* b_im,
                                    std::size_t n);

    // =========================================================================
    // Complex Operations with FCMA (Interleaved format)
    // =========================================================================

    /**
     * @brief Complex multiply using FCMA instructions (2 ops instead of 4).
     * Interleaved format: [re0, im0, re1, im1, ...]
     * @param n Number of complex samples (array size is 2*n)
     */
    void sve2_complex_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n);

    void sve2_complex_conj_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n);

    // Complex dot product
    void sve2_complex_dot_f32(float* out_re, float* out_im,
                               const float* a_re, const float* a_im,
                               const float* b_re, const float* b_im,
                               std::size_t n);

    // Magnitude and phase
    void sve2_complex_magnitude_f32(float* out, const float* re, const float* im, std::size_t n);
    void sve2_complex_phase_f32(float* out, const float* re, const float* im, std::size_t n);

    // Complex arithmetic
    void sve2_complex_add_f32(float* out_re, float* out_im,
                               const float* a_re, const float* a_im,
                               const float* b_re, const float* b_im,
                               std::size_t n);

    void sve2_complex_scale_f32(float* out_re, float* out_im,
                                 const float* in_re, const float* in_im,
                                 float scale_re, float scale_im,
                                 std::size_t n);

    void sve2_complex_exp_f32(float* out_re, float* out_im, const float* phase, std::size_t n);

    // =========================================================================
    // Radar DSP Operations
    // =========================================================================

    /**
     * @brief Cross-Ambiguity Function with FCMA inner loop.
     * Uses svcmla for 2-instruction complex multiply instead of 4.
     */
    void sve2_caf_f32(float* out_mag,
                       const float* ref_re, const float* ref_im,
                       const float* surv_re, const float* surv_im,
                       std::size_t n_samples,
                       std::size_t n_doppler_bins,
                       float doppler_start, float doppler_step,
                       float sample_rate,
                       std::size_t n_range_bins);

    void sve2_xcorr_f32(float* out, const float* x, std::size_t nx,
                         const float* y, std::size_t ny);

    void sve2_xcorr_complex_f32(float* out_re, float* out_im,
                                 const float* x_re, const float* x_im, std::size_t nx,
                                 const float* y_re, const float* y_im, std::size_t ny);

    /**
     * @brief Phase-shift beamformer using FCMA for complex rotation.
     */
    void sve2_beamform_phase_f32(float* output_re, float* output_im,
                                  const float* inputs_re, const float* inputs_im,
                                  const float* phases,
                                  const float* weights,
                                  std::size_t n_channels,
                                  std::size_t n_samples);

    void sve2_apply_window_f32(float* data, const float* window, std::size_t n);
    void sve2_apply_window_complex_f32(float* data_re, float* data_im,
                                        const float* window, std::size_t n);

    // =========================================================================
    // Eigen Wrappers
    // =========================================================================

    float sve2_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf sve2_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf sve2_sub(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf sve2_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf sve2_div(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    float sve2_norm(const Eigen::VectorXf& a);
    float sve2_reduce_sum(const Eigen::VectorXf& a);
    float sve2_reduce_max(const Eigen::VectorXf& a);
    float sve2_reduce_min(const Eigen::VectorXf& a);

    Eigen::VectorXf sve2_fir(const Eigen::VectorXf& x, const Eigen::VectorXf& h);

    Eigen::MatrixXf sve2_gemm_blocked(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);

    Eigen::VectorXcf sve2_complex_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
    Eigen::VectorXcf sve2_complex_conj_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
    std::complex<float> sve2_complex_dot(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
    Eigen::VectorXf sve2_complex_magnitude(const Eigen::VectorXcf& a);
    Eigen::VectorXf sve2_complex_phase(const Eigen::VectorXcf& a);

    Eigen::MatrixXf sve2_caf(const Eigen::VectorXcf& ref,
                              const Eigen::VectorXcf& surv,
                              std::size_t n_doppler_bins,
                              float doppler_start, float doppler_step,
                              float sample_rate,
                              std::size_t n_range_bins);

} // namespace sve2
} // namespace optmath
