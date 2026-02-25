#include "optmath/radar_kernels.hpp"
#include "optmath/neon_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace radar {

// =========================================================================
// Window Functions
// =========================================================================

void generate_window_f32(float* window, std::size_t n, WindowType type, float beta) {
    if (n == 0) return;

    const float pi = 3.14159265358979323846f;

    switch (type) {
        case WindowType::RECTANGULAR:
            for (std::size_t i = 0; i < n; ++i) {
                window[i] = 1.0f;
            }
            break;

        case WindowType::HAMMING:
            for (std::size_t i = 0; i < n; ++i) {
                window[i] = 0.54f - 0.46f * std::cos(2.0f * pi * i / (n - 1));
            }
            break;

        case WindowType::HANNING:
            for (std::size_t i = 0; i < n; ++i) {
                window[i] = 0.5f * (1.0f - std::cos(2.0f * pi * i / (n - 1)));
            }
            break;

        case WindowType::BLACKMAN:
            for (std::size_t i = 0; i < n; ++i) {
                float x = 2.0f * pi * i / (n - 1);
                window[i] = 0.42f - 0.5f * std::cos(x) + 0.08f * std::cos(2.0f * x);
            }
            break;

        case WindowType::BLACKMAN_HARRIS:
            for (std::size_t i = 0; i < n; ++i) {
                float x = 2.0f * pi * i / (n - 1);
                window[i] = 0.35875f - 0.48829f * std::cos(x)
                          + 0.14128f * std::cos(2.0f * x)
                          - 0.01168f * std::cos(3.0f * x);
            }
            break;

        case WindowType::KAISER: {
            // Kaiser window: w[n] = I0(beta * sqrt(1 - ((n - N/2) / (N/2))^2)) / I0(beta)
            // Using series approximation for I0 (modified Bessel function)
            auto bessel_i0 = [](float x) -> float {
                float sum = 1.0f;
                float term = 1.0f;
                for (int k = 1; k < 20; ++k) {
                    term *= (x / (2.0f * k)) * (x / (2.0f * k));
                    sum += term;
                    if (term < 1e-10f * sum) break;
                }
                return sum;
            };

            float i0_beta = bessel_i0(beta);
            float half_n = (n - 1) / 2.0f;

            for (std::size_t i = 0; i < n; ++i) {
                float t = (i - half_n) / half_n;
                float arg = beta * std::sqrt(std::max(0.0f, 1.0f - t * t));
                window[i] = bessel_i0(arg) / i0_beta;
            }
            break;
        }
    }
}

void apply_window_f32(float* data, const float* window, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vld1q_f32(data + i);
        float32x4_t w = vld1q_f32(window + i);
        vst1q_f32(data + i, vmulq_f32(d, w));
    }
    for (; i < n; ++i) {
        data[i] *= window[i];
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        data[i] *= window[i];
    }
#endif
}

void apply_window_complex_f32(float* data_re, float* data_im,
                               const float* window, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t w = vld1q_f32(window + i);
        float32x4_t dr = vld1q_f32(data_re + i);
        float32x4_t di = vld1q_f32(data_im + i);
        vst1q_f32(data_re + i, vmulq_f32(dr, w));
        vst1q_f32(data_im + i, vmulq_f32(di, w));
    }
    for (; i < n; ++i) {
        data_re[i] *= window[i];
        data_im[i] *= window[i];
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        data_re[i] *= window[i];
        data_im[i] *= window[i];
    }
#endif
}

Eigen::VectorXf generate_window(std::size_t n, WindowType type, float beta) {
    Eigen::VectorXf window(n);
    generate_window_f32(window.data(), n, type, beta);
    return window;
}

void apply_window(Eigen::VectorXf& data, const Eigen::VectorXf& window) {
    if (data.size() != window.size()) return;
    apply_window_f32(data.data(), window.data(), data.size());
}

void apply_window(Eigen::VectorXcf& data, const Eigen::VectorXf& window) {
    if (data.size() != window.size()) return;
    Eigen::VectorXf re = data.real();
    Eigen::VectorXf im = data.imag();
    apply_window_complex_f32(re.data(), im.data(), window.data(), data.size());
    data.real() = re;
    data.imag() = im;
}

// =========================================================================
// Cross-Correlation
// =========================================================================

void xcorr_f32(float* out, const float* x, std::size_t nx,
               const float* y, std::size_t ny) {
    // Full cross-correlation: output size = nx + ny - 1
    // out[k] = sum_n x[n] * y[n - k + (ny-1)]
    // For k = 0: y is shifted all the way right, minimal overlap
    // For k = ny-1: y is aligned with x starting at 0

    std::size_t out_len = nx + ny - 1;

    for (std::size_t k = 0; k < out_len; ++k) {
        float sum = 0.0f;

        // Determine overlap range
        std::size_t x_start = (k >= ny - 1) ? 0 : (ny - 1 - k);
        std::size_t x_end = std::min(nx, out_len - k);

        // Map to y indices
        std::size_t y_offset = (k >= ny - 1) ? (k - ny + 1) : 0;

#ifdef OPTMATH_USE_NEON
        std::size_t len = x_end - x_start;
        const float* xp = x + x_start;
        const float* yp = y + y_offset;

        std::size_t i = 0;
        float32x4_t vsum = vdupq_n_f32(0.0f);

        for (; i + 3 < len; i += 4) {
            float32x4_t vx = vld1q_f32(xp + i);
            float32x4_t vy = vld1q_f32(yp + i);
            vsum = vmlaq_f32(vsum, vx, vy);
        }
        sum = vaddvq_f32(vsum);
        for (; i < len; ++i) {
            sum += xp[i] * yp[i];
        }
#else
        for (std::size_t i = x_start; i < x_end; ++i) {
            sum += x[i] * y[i - x_start + y_offset];
        }
#endif
        out[k] = sum;
    }
}

void xcorr_complex_f32(float* out_re, float* out_im,
                       const float* x_re, const float* x_im, std::size_t nx,
                       const float* y_re, const float* y_im, std::size_t ny) {
    // Complex cross-correlation: x * conj(y)
    std::size_t out_len = nx + ny - 1;

    for (std::size_t k = 0; k < out_len; ++k) {
        float sum_re = 0.0f, sum_im = 0.0f;

        std::size_t x_start = (k >= ny - 1) ? 0 : (ny - 1 - k);
        std::size_t x_end = std::min(nx, out_len - k);
        std::size_t y_offset = (k >= ny - 1) ? (k - ny + 1) : 0;

#ifdef OPTMATH_USE_NEON
        std::size_t len = x_end - x_start;
        const float* xrp = x_re + x_start;
        const float* xip = x_im + x_start;
        const float* yrp = y_re + y_offset;
        const float* yip = y_im + y_offset;

        float32x4_t vsumr = vdupq_n_f32(0.0f);
        float32x4_t vsumi = vdupq_n_f32(0.0f);

        std::size_t i = 0;
        for (; i + 3 < len; i += 4) {
            float32x4_t xr = vld1q_f32(xrp + i);
            float32x4_t xi = vld1q_f32(xip + i);
            float32x4_t yr = vld1q_f32(yrp + i);
            float32x4_t yi = vld1q_f32(yip + i);

            // x * conj(y) = (xr + j*xi) * (yr - j*yi)
            // = (xr*yr + xi*yi) + j*(xi*yr - xr*yi)
            vsumr = vmlaq_f32(vsumr, xr, yr);
            vsumr = vmlaq_f32(vsumr, xi, yi);
            vsumi = vmlaq_f32(vsumi, xi, yr);
            vsumi = vmlsq_f32(vsumi, xr, yi);
        }
        sum_re = vaddvq_f32(vsumr);
        sum_im = vaddvq_f32(vsumi);

        for (; i < len; ++i) {
            sum_re += xrp[i] * yrp[i] + xip[i] * yip[i];
            sum_im += xip[i] * yrp[i] - xrp[i] * yip[i];
        }
#else
        for (std::size_t i = x_start; i < x_end; ++i) {
            std::size_t yi = i - x_start + y_offset;
            sum_re += x_re[i] * y_re[yi] + x_im[i] * y_im[yi];
            sum_im += x_im[i] * y_re[yi] - x_re[i] * y_im[yi];
        }
#endif
        out_re[k] = sum_re;
        out_im[k] = sum_im;
    }
}

Eigen::VectorXf xcorr(const Eigen::VectorXf& x, const Eigen::VectorXf& y) {
    Eigen::VectorXf result(x.size() + y.size() - 1);
    xcorr_f32(result.data(), x.data(), x.size(), y.data(), y.size());
    return result;
}

Eigen::VectorXcf xcorr(const Eigen::VectorXcf& x, const Eigen::VectorXcf& y) {
    std::size_t out_len = x.size() + y.size() - 1;
    Eigen::VectorXf x_re = x.real(), x_im = x.imag();
    Eigen::VectorXf y_re = y.real(), y_im = y.imag();
    Eigen::VectorXf out_re(out_len), out_im(out_len);

    xcorr_complex_f32(out_re.data(), out_im.data(),
                      x_re.data(), x_im.data(), x.size(),
                      y_re.data(), y_im.data(), y.size());

    Eigen::VectorXcf result(out_len);
    result.real() = out_re;
    result.imag() = out_im;
    return result;
}

// =========================================================================
// Cross-Ambiguity Function (CAF)
// =========================================================================

void caf_f32(float* out_mag,
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
                // No valid samples for this range bin, set magnitude to zero
                out_mag[d * n_range_bins + r] = 0.0f;
                continue;
            }

            // Compute correlation at this range delay
            float corr_re = 0.0f, corr_im = 0.0f;
            std::size_t max_i = n_samples - r;

#ifdef OPTMATH_USE_NEON
            float32x4_t vsumr = vdupq_n_f32(0.0f);
            float32x4_t vsumi = vdupq_n_f32(0.0f);
            std::size_t i = 0;

            for (; i + 3 < max_i; i += 4) {
                float32x4_t sr = vld1q_f32(shifted_re.data() + i);
                float32x4_t si = vld1q_f32(shifted_im.data() + i);
                float32x4_t vr = vld1q_f32(surv_re + i + r);
                float32x4_t vi = vld1q_f32(surv_im + i + r);

                // shifted * conj(surv)
                vsumr = vmlaq_f32(vsumr, sr, vr);
                vsumr = vmlaq_f32(vsumr, si, vi);
                vsumi = vmlaq_f32(vsumi, si, vr);
                vsumi = vmlsq_f32(vsumi, sr, vi);
            }
            corr_re = vaddvq_f32(vsumr);
            corr_im = vaddvq_f32(vsumi);

            for (; i < max_i; ++i) {
                corr_re += shifted_re[i] * surv_re[i + r] + shifted_im[i] * surv_im[i + r];
                corr_im += shifted_im[i] * surv_re[i + r] - shifted_re[i] * surv_im[i + r];
            }
#else
            for (std::size_t i = 0; i < max_i; ++i) {
                corr_re += shifted_re[i] * surv_re[i + r] + shifted_im[i] * surv_im[i + r];
                corr_im += shifted_im[i] * surv_re[i + r] - shifted_re[i] * surv_im[i + r];
            }
#endif
            // Store magnitude
            out_mag[d * n_range_bins + r] = std::sqrt(corr_re * corr_re + corr_im * corr_im);
        }
    }
}

Eigen::MatrixXf caf(const Eigen::VectorXcf& ref,
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

    caf_f32(result.data(),
            ref_re.data(), ref_im.data(),
            surv_re.data(), surv_im.data(),
            ref.size(),
            n_doppler_bins,
            doppler_start, doppler_step,
            sample_rate,
            n_range_bins);

    return result;
}

// =========================================================================
// CFAR Detection
// =========================================================================

void cfar_ca_f32(std::uint8_t* detections, float* threshold,
                 const float* input, std::size_t n,
                 std::size_t guard_cells, std::size_t reference_cells,
                 float pfa_factor) {

    for (std::size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        std::size_t count = 0;

        // Left reference cells
        for (std::size_t j = 1; j <= guard_cells + reference_cells; ++j) {
            if (i >= j && j > guard_cells) {
                sum += input[i - j];
                count++;
            }
        }

        // Right reference cells
        for (std::size_t j = 1; j <= guard_cells + reference_cells; ++j) {
            if (i + j < n && j > guard_cells) {
                sum += input[i + j];
                count++;
            }
        }

        // Compute threshold
        float avg = (count > 0) ? (sum / count) : 0.0f;
        float thresh = pfa_factor * avg;

        if (threshold) threshold[i] = thresh;
        detections[i] = (input[i] > thresh) ? 1 : 0;
    }
}

void cfar_2d_f32(std::uint8_t* detections,
                 const float* input,
                 std::size_t n_doppler, std::size_t n_range,
                 std::size_t guard_range, std::size_t guard_doppler,
                 std::size_t ref_range, std::size_t ref_doppler,
                 float pfa_factor) {

    for (std::size_t d = 0; d < n_doppler; ++d) {
        for (std::size_t r = 0; r < n_range; ++r) {
            float sum = 0.0f;
            std::size_t count = 0;

            // Scan the reference window (excluding guard cells)
            for (int dd = -(int)(guard_doppler + ref_doppler); dd <= (int)(guard_doppler + ref_doppler); ++dd) {
                for (int dr = -(int)(guard_range + ref_range); dr <= (int)(guard_range + ref_range); ++dr) {
                    // Skip cell under test
                    if (dd == 0 && dr == 0) continue;

                    // Skip guard cells
                    if (std::abs(dd) <= (int)guard_doppler && std::abs(dr) <= (int)guard_range) continue;

                    int nd = (int)d + dd;
                    int nr = (int)r + dr;

                    if (nd >= 0 && nd < (int)n_doppler && nr >= 0 && nr < (int)n_range) {
                        sum += input[nd * n_range + nr];
                        count++;
                    }
                }
            }

            float avg = (count > 0) ? (sum / count) : 0.0f;
            float thresh = pfa_factor * avg;

            detections[d * n_range + r] = (input[d * n_range + r] > thresh) ? 1 : 0;
        }
    }
}

void cfar_os_f32(std::uint8_t* detections, float* threshold,
                 const float* input, std::size_t n,
                 std::size_t guard_cells, std::size_t reference_cells,
                 std::size_t k_select,
                 float pfa_factor) {

    std::vector<float> ref_samples(2 * reference_cells);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t count = 0;

        // Collect left reference cells
        for (std::size_t j = guard_cells + 1; j <= guard_cells + reference_cells; ++j) {
            if (i >= j) {
                ref_samples[count++] = input[i - j];
            }
        }

        // Collect right reference cells
        for (std::size_t j = guard_cells + 1; j <= guard_cells + reference_cells; ++j) {
            if (i + j < n) {
                ref_samples[count++] = input[i + j];
            }
        }

        if (count == 0) {
            if (threshold) threshold[i] = 0.0f;
            detections[i] = 1; // Default to detection if no reference
            continue;
        }

        // Sort and select k-th order statistic
        std::sort(ref_samples.begin(), ref_samples.begin() + count);
        std::size_t k = std::min(k_select, count - 1);
        float selected = ref_samples[k];

        float thresh = pfa_factor * selected;
        if (threshold) threshold[i] = thresh;
        detections[i] = (input[i] > thresh) ? 1 : 0;
    }
}

Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 1> cfar_ca(
    const Eigen::VectorXf& input,
    std::size_t guard_cells, std::size_t reference_cells,
    float pfa_factor) {

    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 1> detections(input.size());
    cfar_ca_f32(detections.data(), nullptr, input.data(), input.size(),
                guard_cells, reference_cells, pfa_factor);
    return detections;
}

Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> cfar_2d(
    const Eigen::MatrixXf& input,
    std::size_t guard_range, std::size_t guard_doppler,
    std::size_t ref_range, std::size_t ref_doppler,
    float pfa_factor) {

    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> detections(input.rows(), input.cols());
    cfar_2d_f32(detections.data(), input.data(),
                input.rows(), input.cols(),
                guard_range, guard_doppler,
                ref_range, ref_doppler,
                pfa_factor);
    return detections;
}

// =========================================================================
// Clutter Filtering
// =========================================================================

void nlms_filter_f32(float* output, float* weights,
                     const float* input, const float* reference,
                     std::size_t n, std::size_t filter_length,
                     float mu, float eps) {

    // Initialize weights to zero if not pre-initialized
    // Caller should manage this

    for (std::size_t i = filter_length - 1; i < n; ++i) {
        // Compute filter output (estimate of clutter)
        float y = 0.0f;
        float power = eps;

        for (std::size_t j = 0; j < filter_length; ++j) {
            float ref_val = reference[i - j];
            y += weights[j] * ref_val;
            power += ref_val * ref_val;
        }

        // Error = input - filter output
        float error = input[i] - y;
        output[i] = error;

        // Update weights (NLMS)
        float step = mu / power;
        for (std::size_t j = 0; j < filter_length; ++j) {
            weights[j] += step * error * reference[i - j];
        }
    }

    // Fill initial samples where filter can't operate
    for (std::size_t i = 0; i < filter_length - 1; ++i) {
        output[i] = input[i]; // Pass through
    }
}

void projection_clutter_f32(float* output,
                            const float* input,
                            const float* clutter_subspace,
                            std::size_t n, std::size_t subspace_dim) {

    // Compute projection: P = I - U*U^H
    // output = (I - U*U^H) * input = input - U * (U^H * input)

    // First compute U^H * input (subspace_dim x 1)
    std::vector<float> coeff(subspace_dim, 0.0f);

    for (std::size_t d = 0; d < subspace_dim; ++d) {
        float dot = 0.0f;
        const float* u = clutter_subspace + d * n;

#ifdef OPTMATH_USE_NEON
        float32x4_t vsum = vdupq_n_f32(0.0f);
        std::size_t i = 0;
        for (; i + 3 < n; i += 4) {
            vsum = vmlaq_f32(vsum, vld1q_f32(u + i), vld1q_f32(input + i));
        }
        dot = vaddvq_f32(vsum);
        for (; i < n; ++i) {
            dot += u[i] * input[i];
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            dot += u[i] * input[i];
        }
#endif
        coeff[d] = dot;
    }

    // output = input - sum(coeff[d] * U[d])
    std::memcpy(output, input, n * sizeof(float));

    for (std::size_t d = 0; d < subspace_dim; ++d) {
        const float* u = clutter_subspace + d * n;
        float c = coeff[d];

#ifdef OPTMATH_USE_NEON
        float32x4_t vc = vdupq_n_f32(c);
        std::size_t i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t out_val = vld1q_f32(output + i);
            out_val = vmlsq_f32(out_val, vc, vld1q_f32(u + i));
            vst1q_f32(output + i, out_val);
        }
        for (; i < n; ++i) {
            output[i] -= c * u[i];
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            output[i] -= c * u[i];
        }
#endif
    }
}

Eigen::VectorXf nlms_filter(const Eigen::VectorXf& input,
                            const Eigen::VectorXf& reference,
                            std::size_t filter_length,
                            float mu, float eps) {

    if (input.size() != reference.size()) return Eigen::VectorXf();

    Eigen::VectorXf output(input.size());
    Eigen::VectorXf weights = Eigen::VectorXf::Zero(filter_length);

    nlms_filter_f32(output.data(), weights.data(),
                    input.data(), reference.data(),
                    input.size(), filter_length, mu, eps);

    return output;
}

Eigen::VectorXf projection_clutter(const Eigen::VectorXf& input,
                                   const Eigen::MatrixXf& clutter_subspace) {

    Eigen::VectorXf output(input.size());
    projection_clutter_f32(output.data(), input.data(),
                           clutter_subspace.data(),
                           input.size(), clutter_subspace.cols());
    return output;
}

// =========================================================================
// Doppler Processing
// =========================================================================

void doppler_fft_f32(float* output_re, float* output_im,
                     const float* input_re, const float* input_im,
                     std::size_t n_pulses, std::size_t n_range,
                     std::size_t fft_size) {

    // Simple DFT implementation for each range bin
    // In production, this would use an optimized FFT library

    const float two_pi = 6.28318530717958647693f;

    for (std::size_t r = 0; r < n_range; ++r) {
        for (std::size_t k = 0; k < fft_size; ++k) {
            float sum_re = 0.0f, sum_im = 0.0f;

            for (std::size_t p = 0; p < n_pulses; ++p) {
                float angle = -two_pi * k * p / fft_size;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float in_re = input_re[p * n_range + r];
                float in_im = input_im[p * n_range + r];

                // Complex multiply: input * exp(-j*angle)
                sum_re += in_re * cos_a - in_im * sin_a;
                sum_im += in_re * sin_a + in_im * cos_a;
            }

            output_re[k * n_range + r] = sum_re;
            output_im[k * n_range + r] = sum_im;
        }
    }
}

void mti_filter_f32(float* output, const float* input,
                    std::size_t n_pulses, std::size_t n_range,
                    const float* coeffs, std::size_t n_coeffs) {

    std::size_t out_pulses = n_pulses - n_coeffs + 1;

    for (std::size_t p = 0; p < out_pulses; ++p) {
        for (std::size_t r = 0; r < n_range; ++r) {
            float sum = 0.0f;
            for (std::size_t c = 0; c < n_coeffs; ++c) {
                sum += coeffs[c] * input[(p + c) * n_range + r];
            }
            output[p * n_range + r] = sum;
        }
    }
}

Eigen::MatrixXcf doppler_fft(const Eigen::MatrixXcf& input, std::size_t fft_size) {
    std::size_t n_pulses = input.rows();
    std::size_t n_range = input.cols();

    Eigen::VectorXf in_re(n_pulses * n_range), in_im(n_pulses * n_range);
    Eigen::VectorXf out_re(fft_size * n_range), out_im(fft_size * n_range);

    // Flatten input (row-major)
    for (std::size_t p = 0; p < n_pulses; ++p) {
        for (std::size_t r = 0; r < n_range; ++r) {
            in_re[p * n_range + r] = input(p, r).real();
            in_im[p * n_range + r] = input(p, r).imag();
        }
    }

    doppler_fft_f32(out_re.data(), out_im.data(),
                    in_re.data(), in_im.data(),
                    n_pulses, n_range, fft_size);

    Eigen::MatrixXcf output(fft_size, n_range);
    for (std::size_t k = 0; k < fft_size; ++k) {
        for (std::size_t r = 0; r < n_range; ++r) {
            output(k, r) = std::complex<float>(out_re[k * n_range + r], out_im[k * n_range + r]);
        }
    }

    return output;
}

Eigen::MatrixXf mti_filter(const Eigen::MatrixXf& input, const Eigen::VectorXf& coeffs) {
    std::size_t n_pulses = input.rows();
    std::size_t n_range = input.cols();
    std::size_t n_coeffs = coeffs.size();
    std::size_t out_pulses = n_pulses - n_coeffs + 1;

    Eigen::MatrixXf output(out_pulses, n_range);

    // Eigen is column-major, so we do the filtering directly here
    // to handle the memory layout correctly
    for (std::size_t p = 0; p < out_pulses; ++p) {
        for (std::size_t r = 0; r < n_range; ++r) {
            float sum = 0.0f;
            for (std::size_t c = 0; c < n_coeffs; ++c) {
                sum += coeffs[c] * input(p + c, r);
            }
            output(p, r) = sum;
        }
    }

    return output;
}

// =========================================================================
// Beamforming Primitives
// =========================================================================

void beamform_delay_sum_f32(float* output,
                            const float* inputs,
                            const int* delays,
                            const float* weights,
                            std::size_t n_channels,
                            std::size_t n_samples) {

    std::memset(output, 0, n_samples * sizeof(float));

    for (std::size_t ch = 0; ch < n_channels; ++ch) {
        int delay = delays[ch];
        float weight = (weights != nullptr) ? weights[ch] : 1.0f;
        const float* ch_data = inputs + ch * n_samples;

        for (std::size_t i = 0; i < n_samples; ++i) {
            int src_idx = (int)i - delay;
            if (src_idx >= 0 && src_idx < (int)n_samples) {
                output[i] += weight * ch_data[src_idx];
            }
        }
    }
}

void beamform_phase_f32(float* output_re, float* output_im,
                        const float* inputs_re, const float* inputs_im,
                        const float* phases,
                        const float* weights,
                        std::size_t n_channels,
                        std::size_t n_samples) {

    std::memset(output_re, 0, n_samples * sizeof(float));
    std::memset(output_im, 0, n_samples * sizeof(float));

    for (std::size_t ch = 0; ch < n_channels; ++ch) {
        float phase = phases[ch];
        float weight = (weights != nullptr) ? weights[ch] : 1.0f;
        float cos_p = weight * std::cos(phase);
        float sin_p = weight * std::sin(phase);

        const float* ch_re = inputs_re + ch * n_samples;
        const float* ch_im = inputs_im + ch * n_samples;

#ifdef OPTMATH_USE_NEON
        float32x4_t vcos = vdupq_n_f32(cos_p);
        float32x4_t vsin = vdupq_n_f32(sin_p);

        std::size_t i = 0;
        for (; i + 3 < n_samples; i += 4) {
            float32x4_t ir = vld1q_f32(ch_re + i);
            float32x4_t ii = vld1q_f32(ch_im + i);
            float32x4_t otr = vld1q_f32(output_re + i);
            float32x4_t oti = vld1q_f32(output_im + i);

            // Apply phase shift: (ir + j*ii) * (cos + j*sin)
            // = (ir*cos - ii*sin) + j*(ir*sin + ii*cos)
            otr = vmlaq_f32(otr, ir, vcos);
            otr = vmlsq_f32(otr, ii, vsin);
            oti = vmlaq_f32(oti, ir, vsin);
            oti = vmlaq_f32(oti, ii, vcos);

            vst1q_f32(output_re + i, otr);
            vst1q_f32(output_im + i, oti);
        }
        for (; i < n_samples; ++i) {
            output_re[i] += ch_re[i] * cos_p - ch_im[i] * sin_p;
            output_im[i] += ch_re[i] * sin_p + ch_im[i] * cos_p;
        }
#else
        for (std::size_t i = 0; i < n_samples; ++i) {
            output_re[i] += ch_re[i] * cos_p - ch_im[i] * sin_p;
            output_im[i] += ch_re[i] * sin_p + ch_im[i] * cos_p;
        }
#endif
    }
}

void steering_vector_ula_f32(float* steering_re, float* steering_im,
                             std::size_t n_elements,
                             float d_lambda,
                             float theta_rad) {

    const float two_pi = 6.28318530717958647693f;
    float phase_step = two_pi * d_lambda * std::sin(theta_rad);

    for (std::size_t i = 0; i < n_elements; ++i) {
        float phase = phase_step * i;
        steering_re[i] = std::cos(phase);
        steering_im[i] = std::sin(phase);
    }
}

Eigen::VectorXf beamform_delay_sum(const Eigen::MatrixXf& inputs,
                                   const Eigen::VectorXi& delays,
                                   const Eigen::VectorXf& weights) {

    Eigen::VectorXf output(inputs.cols());
    beamform_delay_sum_f32(output.data(), inputs.data(),
                           delays.data(),
                           weights.size() > 0 ? weights.data() : nullptr,
                           inputs.rows(), inputs.cols());
    return output;
}

Eigen::VectorXcf beamform_phase(const Eigen::MatrixXcf& inputs,
                                const Eigen::VectorXf& phases,
                                const Eigen::VectorXf& weights) {

    std::size_t n_channels = inputs.rows();
    std::size_t n_samples = inputs.cols();

    Eigen::VectorXf in_re(n_channels * n_samples), in_im(n_channels * n_samples);
    Eigen::VectorXf out_re(n_samples), out_im(n_samples);

    for (std::size_t ch = 0; ch < n_channels; ++ch) {
        for (std::size_t i = 0; i < n_samples; ++i) {
            in_re[ch * n_samples + i] = inputs(ch, i).real();
            in_im[ch * n_samples + i] = inputs(ch, i).imag();
        }
    }

    beamform_phase_f32(out_re.data(), out_im.data(),
                       in_re.data(), in_im.data(),
                       phases.data(),
                       weights.size() > 0 ? weights.data() : nullptr,
                       n_channels, n_samples);

    Eigen::VectorXcf output(n_samples);
    output.real() = out_re;
    output.imag() = out_im;
    return output;
}

Eigen::VectorXcf steering_vector_ula(std::size_t n_elements,
                                     float d_lambda,
                                     float theta_rad) {

    Eigen::VectorXf re(n_elements), im(n_elements);
    steering_vector_ula_f32(re.data(), im.data(), n_elements, d_lambda, theta_rad);

    Eigen::VectorXcf result(n_elements);
    result.real() = re;
    result.imag() = im;
    return result;
}

} // namespace radar
} // namespace optmath
