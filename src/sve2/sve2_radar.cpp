#include "optmath/sve2_kernels.hpp"
#include "optmath/radar_kernels.hpp"
#include "optmath/neon_kernels.hpp"
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#ifdef OPTMATH_USE_SVE2
#include <arm_sve.h>
#endif

namespace optmath {
namespace sve2 {

#ifdef OPTMATH_USE_SVE2

// =========================================================================
// Cross-Ambiguity Function (CAF) with SVE2 FCMA inner loop
// =========================================================================

void sve2_caf_f32(float* out_mag,
                   const float* ref_re, const float* ref_im,
                   const float* surv_re, const float* surv_im,
                   std::size_t n_samples,
                   std::size_t n_doppler_bins,
                   float doppler_start, float doppler_step,
                   float sample_rate,
                   std::size_t n_range_bins) {

    const float two_pi = 6.28318530717958647693f;

    // Temporary buffers for Doppler-shifted reference
    std::vector<float> shifted_re(n_samples);
    std::vector<float> shifted_im(n_samples);

    for (std::size_t d = 0; d < n_doppler_bins; ++d) {
        float doppler_freq = doppler_start + d * doppler_step;
        float phase_step = two_pi * doppler_freq / sample_rate;

        // Apply Doppler shift to reference signal
        // shifted = ref * exp(j * 2 * pi * fd * t)
        for (std::size_t i = 0; i < n_samples; ++i) {
            float phase = phase_step * i;
            float cos_p = std::cos(phase);
            float sin_p = std::sin(phase);

            // Complex multiply: ref * exp(j*phase)
            shifted_re[i] = ref_re[i] * cos_p - ref_im[i] * sin_p;
            shifted_im[i] = ref_re[i] * sin_p + ref_im[i] * cos_p;
        }

        // Cross-correlate shifted reference with surveillance for each range bin
        for (std::size_t r = 0; r < n_range_bins; ++r) {
            // CRITICAL: Bounds check to prevent out-of-bounds memory access
            // When r >= n_samples, max_i would underflow (wrap around for unsigned)
            // causing memory access beyond array bounds
            if (r >= n_samples) {
                out_mag[d * n_range_bins + r] = 0.0f;
                continue;
            }

            // Compute correlation at this range delay
            std::size_t max_i = n_samples - r;
            uint64_t max_i_u64 = static_cast<uint64_t>(max_i);

            // SVE2 predicated inner correlation loop
            svfloat32_t vsumr = svdup_f32(0.0f);
            svfloat32_t vsumi = svdup_f32(0.0f);

            uint64_t ii = 0;
            do {
                svbool_t pg = svwhilelt_b32(ii, max_i_u64);

                svfloat32_t sr = svld1_f32(pg, shifted_re.data() + ii);
                svfloat32_t si = svld1_f32(pg, shifted_im.data() + ii);
                svfloat32_t vr = svld1_f32(pg, surv_re + ii + r);
                svfloat32_t vi = svld1_f32(pg, surv_im + ii + r);

                // shifted * conj(surv): (sr + j*si) * (vr - j*vi)
                // Real part: sr*vr + si*vi
                // Imag part: si*vr - sr*vi
                vsumr = svmla_f32_z(pg, vsumr, sr, vr);    // sr*vr
                vsumr = svmla_f32_z(pg, vsumr, si, vi);    // + si*vi
                vsumi = svmla_f32_z(pg, vsumi, si, vr);    // si*vr
                vsumi = svmls_f32_z(pg, vsumi, sr, vi);    // - sr*vi

                ii += svcntw();
            } while (svptest_first(svptrue_b32(), svwhilelt_b32(ii, max_i_u64)));

            float corr_re = svaddv_f32(svptrue_b32(), vsumr);
            float corr_im = svaddv_f32(svptrue_b32(), vsumi);
            out_mag[d * n_range_bins + r] = std::sqrt(corr_re * corr_re + corr_im * corr_im);
        }
    }
}

// =========================================================================
// Real Cross-Correlation with SVE2 inner loop
// =========================================================================

void sve2_xcorr_f32(float* out, const float* x, std::size_t nx,
                     const float* y, std::size_t ny) {
    // Full cross-correlation: output size = nx + ny - 1
    std::size_t out_len = nx + ny - 1;

    for (std::size_t k = 0; k < out_len; ++k) {
        // Determine overlap range
        std::size_t x_start = (k >= ny - 1) ? 0 : (ny - 1 - k);
        std::size_t x_end = std::min(nx, out_len - k);
        std::size_t y_offset = (k >= ny - 1) ? (k - ny + 1) : 0;

        std::size_t len = x_end - x_start;
        const float* xp = x + x_start;
        const float* yp = y + y_offset;
        uint64_t len_u64 = static_cast<uint64_t>(len);

        svfloat32_t vsum = svdup_f32(0.0f);

        uint64_t i = 0;
        if (len > 0) {
            do {
                svbool_t pg = svwhilelt_b32(i, len_u64);

                svfloat32_t vx = svld1_f32(pg, xp + i);
                svfloat32_t vy = svld1_f32(pg, yp + i);
                vsum = svmla_f32_z(pg, vsum, vx, vy);

                i += svcntw();
            } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, len_u64)));
        }

        out[k] = svaddv_f32(svptrue_b32(), vsum);
    }
}

// =========================================================================
// Complex Cross-Correlation with SVE2 inner loop
// =========================================================================

void sve2_xcorr_complex_f32(float* out_re, float* out_im,
                             const float* x_re, const float* x_im, std::size_t nx,
                             const float* y_re, const float* y_im, std::size_t ny) {
    // Complex cross-correlation: x * conj(y)
    std::size_t out_len = nx + ny - 1;

    for (std::size_t k = 0; k < out_len; ++k) {
        std::size_t x_start = (k >= ny - 1) ? 0 : (ny - 1 - k);
        std::size_t x_end = std::min(nx, out_len - k);
        std::size_t y_offset = (k >= ny - 1) ? (k - ny + 1) : 0;

        std::size_t len = x_end - x_start;
        const float* xrp = x_re + x_start;
        const float* xip = x_im + x_start;
        const float* yrp = y_re + y_offset;
        const float* yip = y_im + y_offset;
        uint64_t len_u64 = static_cast<uint64_t>(len);

        svfloat32_t vsumr = svdup_f32(0.0f);
        svfloat32_t vsumi = svdup_f32(0.0f);

        uint64_t i = 0;
        if (len > 0) {
            do {
                svbool_t pg = svwhilelt_b32(i, len_u64);

                svfloat32_t xr = svld1_f32(pg, xrp + i);
                svfloat32_t xi = svld1_f32(pg, xip + i);
                svfloat32_t yr = svld1_f32(pg, yrp + i);
                svfloat32_t yi = svld1_f32(pg, yip + i);

                // x * conj(y) = (xr + j*xi) * (yr - j*yi)
                // Real: xr*yr + xi*yi
                // Imag: xi*yr - xr*yi
                vsumr = svmla_f32_z(pg, vsumr, xr, yr);
                vsumr = svmla_f32_z(pg, vsumr, xi, yi);
                vsumi = svmla_f32_z(pg, vsumi, xi, yr);
                vsumi = svmls_f32_z(pg, vsumi, xr, yi);

                i += svcntw();
            } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, len_u64)));
        }

        out_re[k] = svaddv_f32(svptrue_b32(), vsumr);
        out_im[k] = svaddv_f32(svptrue_b32(), vsumi);
    }
}

// =========================================================================
// Phase-Shift Beamformer with SVE2 inner loop
// =========================================================================

void sve2_beamform_phase_f32(float* output_re, float* output_im,
                              const float* inputs_re, const float* inputs_im,
                              const float* phases,
                              const float* weights,
                              std::size_t n_channels,
                              std::size_t n_samples) {

    std::memset(output_re, 0, n_samples * sizeof(float));
    std::memset(output_im, 0, n_samples * sizeof(float));

    uint64_t n_u64 = static_cast<uint64_t>(n_samples);

    for (std::size_t ch = 0; ch < n_channels; ++ch) {
        float phase = phases[ch];
        float weight = (weights != nullptr) ? weights[ch] : 1.0f;
        float cos_p = weight * std::cos(phase);
        float sin_p = weight * std::sin(phase);

        const float* ch_re = inputs_re + ch * n_samples;
        const float* ch_im = inputs_im + ch * n_samples;

        svfloat32_t vcos = svdup_f32(cos_p);
        svfloat32_t vsin = svdup_f32(sin_p);

        uint64_t i = 0;
        if (n_samples > 0) {
            do {
                svbool_t pg = svwhilelt_b32(i, n_u64);

                svfloat32_t ir = svld1_f32(pg, ch_re + i);
                svfloat32_t ii = svld1_f32(pg, ch_im + i);
                svfloat32_t otr = svld1_f32(pg, output_re + i);
                svfloat32_t oti = svld1_f32(pg, output_im + i);

                // Apply phase shift: (ir + j*ii) * (cos + j*sin)
                // Real: ir*cos - ii*sin
                // Imag: ir*sin + ii*cos
                otr = svmla_f32_z(pg, otr, ir, vcos);
                otr = svmls_f32_z(pg, otr, ii, vsin);
                oti = svmla_f32_z(pg, oti, ir, vsin);
                oti = svmla_f32_z(pg, oti, ii, vcos);

                svst1_f32(pg, output_re + i, otr);
                svst1_f32(pg, output_im + i, oti);

                i += svcntw();
            } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, n_u64)));
        }
    }
}

// =========================================================================
// Apply Window (real) with SVE2 predicated multiply
// =========================================================================

void sve2_apply_window_f32(float* data, const float* window, std::size_t n) {
    uint64_t n_u64 = static_cast<uint64_t>(n);

    uint64_t i = 0;
    if (n > 0) {
        do {
            svbool_t pg = svwhilelt_b32(i, n_u64);

            svfloat32_t d = svld1_f32(pg, data + i);
            svfloat32_t w = svld1_f32(pg, window + i);
            svst1_f32(pg, data + i, svmul_f32_z(pg, d, w));

            i += svcntw();
        } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, n_u64)));
    }
}

// =========================================================================
// Apply Window to Complex Data (separate re/im) with SVE2
// =========================================================================

void sve2_apply_window_complex_f32(float* data_re, float* data_im,
                                    const float* window, std::size_t n) {
    uint64_t n_u64 = static_cast<uint64_t>(n);

    uint64_t i = 0;
    if (n > 0) {
        do {
            svbool_t pg = svwhilelt_b32(i, n_u64);

            svfloat32_t w = svld1_f32(pg, window + i);
            svfloat32_t dr = svld1_f32(pg, data_re + i);
            svfloat32_t di = svld1_f32(pg, data_im + i);

            svst1_f32(pg, data_re + i, svmul_f32_z(pg, dr, w));
            svst1_f32(pg, data_im + i, svmul_f32_z(pg, di, w));

            i += svcntw();
        } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, n_u64)));
    }
}

// =========================================================================
// Eigen Wrapper: CAF
// =========================================================================

Eigen::MatrixXf sve2_caf(const Eigen::VectorXcf& ref,
                          const Eigen::VectorXcf& surv,
                          std::size_t n_doppler_bins,
                          float doppler_start, float doppler_step,
                          float sample_rate,
                          std::size_t n_range_bins) {

    Eigen::VectorXf ref_re = ref.real();
    Eigen::VectorXf ref_im = ref.imag();
    Eigen::VectorXf surv_re = surv.real();
    Eigen::VectorXf surv_im = surv.imag();

    Eigen::MatrixXf result(n_doppler_bins, n_range_bins);

    std::size_t n_samples = std::min(static_cast<std::size_t>(ref.size()),
                                     static_cast<std::size_t>(surv.size()));

    sve2_caf_f32(result.data(),
                 ref_re.data(), ref_im.data(),
                 surv_re.data(), surv_im.data(),
                 n_samples,
                 n_doppler_bins,
                 doppler_start, doppler_step,
                 sample_rate,
                 n_range_bins);

    return result;
}

#else // !OPTMATH_USE_SVE2

// =========================================================================
// Fallback: delegate to optmath::radar:: reference implementations
// =========================================================================

void sve2_caf_f32(float* out_mag,
                   const float* ref_re, const float* ref_im,
                   const float* surv_re, const float* surv_im,
                   std::size_t n_samples,
                   std::size_t n_doppler_bins,
                   float doppler_start, float doppler_step,
                   float sample_rate,
                   std::size_t n_range_bins) {
    optmath::radar::caf_f32(out_mag, ref_re, ref_im, surv_re, surv_im,
                            n_samples, n_doppler_bins,
                            doppler_start, doppler_step,
                            sample_rate, n_range_bins);
}

void sve2_xcorr_f32(float* out, const float* x, std::size_t nx,
                     const float* y, std::size_t ny) {
    optmath::radar::xcorr_f32(out, x, nx, y, ny);
}

void sve2_xcorr_complex_f32(float* out_re, float* out_im,
                             const float* x_re, const float* x_im, std::size_t nx,
                             const float* y_re, const float* y_im, std::size_t ny) {
    optmath::radar::xcorr_complex_f32(out_re, out_im,
                                      x_re, x_im, nx,
                                      y_re, y_im, ny);
}

void sve2_beamform_phase_f32(float* output_re, float* output_im,
                              const float* inputs_re, const float* inputs_im,
                              const float* phases,
                              const float* weights,
                              std::size_t n_channels,
                              std::size_t n_samples) {
    optmath::radar::beamform_phase_f32(output_re, output_im,
                                       inputs_re, inputs_im,
                                       phases, weights,
                                       n_channels, n_samples);
}

void sve2_apply_window_f32(float* data, const float* window, std::size_t n) {
    optmath::radar::apply_window_f32(data, window, n);
}

void sve2_apply_window_complex_f32(float* data_re, float* data_im,
                                    const float* window, std::size_t n) {
    optmath::radar::apply_window_complex_f32(data_re, data_im, window, n);
}

Eigen::MatrixXf sve2_caf(const Eigen::VectorXcf& ref,
                          const Eigen::VectorXcf& surv,
                          std::size_t n_doppler_bins,
                          float doppler_start, float doppler_step,
                          float sample_rate,
                          std::size_t n_range_bins) {
    return optmath::radar::caf(ref, surv, n_doppler_bins,
                               doppler_start, doppler_step,
                               sample_rate, n_range_bins);
}

#endif // OPTMATH_USE_SVE2

} // namespace sve2
} // namespace optmath
