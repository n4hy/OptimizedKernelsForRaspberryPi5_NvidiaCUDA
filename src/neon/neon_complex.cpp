#include "optmath/neon_kernels.hpp"
#include <cmath>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

// =========================================================================
// Complex Number Operations (Critical for Radar Signal Processing)
// =========================================================================
// Complex numbers are represented as separate real/imaginary arrays
// or as interleaved format (IQ: real, imag, real, imag, ...)

void neon_complex_mul_f32(float* out_re, float* out_im,
                          const float* a_re, const float* a_im,
                          const float* b_re, const float* b_im,
                          std::size_t n) {
    // (a_re + j*a_im) * (b_re + j*b_im) =
    // (a_re*b_re - a_im*b_im) + j*(a_re*b_im + a_im*b_re)
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t ar = vld1q_f32(a_re + i);
        float32x4_t ai = vld1q_f32(a_im + i);
        float32x4_t br = vld1q_f32(b_re + i);
        float32x4_t bi = vld1q_f32(b_im + i);

        // out_re = a_re*b_re - a_im*b_im
        float32x4_t or_val = vmulq_f32(ar, br);
        or_val = vmlsq_f32(or_val, ai, bi);

        // out_im = a_re*b_im + a_im*b_re
        float32x4_t oi_val = vmulq_f32(ar, bi);
        oi_val = vmlaq_f32(oi_val, ai, br);

        vst1q_f32(out_re + i, or_val);
        vst1q_f32(out_im + i, oi_val);
    }
    for (; i < n; ++i) {
        out_re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        out_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        out_re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        out_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
    }
#endif
}

void neon_complex_conj_mul_f32(float* out_re, float* out_im,
                               const float* a_re, const float* a_im,
                               const float* b_re, const float* b_im,
                               std::size_t n) {
    // a * conj(b) = (a_re + j*a_im) * (b_re - j*b_im)
    // = (a_re*b_re + a_im*b_im) + j*(a_im*b_re - a_re*b_im)
    // Critical for cross-correlation in frequency domain
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t ar = vld1q_f32(a_re + i);
        float32x4_t ai = vld1q_f32(a_im + i);
        float32x4_t br = vld1q_f32(b_re + i);
        float32x4_t bi = vld1q_f32(b_im + i);

        // out_re = a_re*b_re + a_im*b_im
        float32x4_t or_val = vmulq_f32(ar, br);
        or_val = vmlaq_f32(or_val, ai, bi);

        // out_im = a_im*b_re - a_re*b_im
        float32x4_t oi_val = vmulq_f32(ai, br);
        oi_val = vmlsq_f32(oi_val, ar, bi);

        vst1q_f32(out_re + i, or_val);
        vst1q_f32(out_im + i, oi_val);
    }
    for (; i < n; ++i) {
        out_re[i] = a_re[i] * b_re[i] + a_im[i] * b_im[i];
        out_im[i] = a_im[i] * b_re[i] - a_re[i] * b_im[i];
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        out_re[i] = a_re[i] * b_re[i] + a_im[i] * b_im[i];
        out_im[i] = a_im[i] * b_re[i] - a_re[i] * b_im[i];
    }
#endif
}

void neon_complex_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n) {
    // Interleaved format: [re0, im0, re1, im1, ...]
    // n is the number of complex samples (array size is 2*n)
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    // Process 4 complex numbers at a time (8 floats)
    for (; i + 3 < n; i += 4) {
        // Load interleaved: vld2q_f32 deinterleaves into separate re/im vectors
        float32x4x2_t av = vld2q_f32(a + 2*i);
        float32x4x2_t bv = vld2q_f32(b + 2*i);

        float32x4_t ar = av.val[0];
        float32x4_t ai = av.val[1];
        float32x4_t br = bv.val[0];
        float32x4_t bi = bv.val[1];

        // out_re = a_re*b_re - a_im*b_im
        float32x4_t or_val = vmulq_f32(ar, br);
        or_val = vmlsq_f32(or_val, ai, bi);

        // out_im = a_re*b_im + a_im*b_re
        float32x4_t oi_val = vmulq_f32(ar, bi);
        oi_val = vmlaq_f32(oi_val, ai, br);

        float32x4x2_t result;
        result.val[0] = or_val;
        result.val[1] = oi_val;
        vst2q_f32(out + 2*i, result);
    }
    // Scalar tail
    for (; i < n; ++i) {
        float ar = a[2*i];
        float ai = a[2*i + 1];
        float br = b[2*i];
        float bi = b[2*i + 1];
        out[2*i] = ar * br - ai * bi;
        out[2*i + 1] = ar * bi + ai * br;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        float ar = a[2*i];
        float ai = a[2*i + 1];
        float br = b[2*i];
        float bi = b[2*i + 1];
        out[2*i] = ar * br - ai * bi;
        out[2*i + 1] = ar * bi + ai * br;
    }
#endif
}

void neon_complex_conj_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n) {
    // a * conj(b) for interleaved format
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4x2_t av = vld2q_f32(a + 2*i);
        float32x4x2_t bv = vld2q_f32(b + 2*i);

        float32x4_t ar = av.val[0];
        float32x4_t ai = av.val[1];
        float32x4_t br = bv.val[0];
        float32x4_t bi = bv.val[1];

        // out_re = a_re*b_re + a_im*b_im
        float32x4_t or_val = vmulq_f32(ar, br);
        or_val = vmlaq_f32(or_val, ai, bi);

        // out_im = a_im*b_re - a_re*b_im
        float32x4_t oi_val = vmulq_f32(ai, br);
        oi_val = vmlsq_f32(oi_val, ar, bi);

        float32x4x2_t result;
        result.val[0] = or_val;
        result.val[1] = oi_val;
        vst2q_f32(out + 2*i, result);
    }
    for (; i < n; ++i) {
        float ar = a[2*i];
        float ai = a[2*i + 1];
        float br = b[2*i];
        float bi = b[2*i + 1];
        out[2*i] = ar * br + ai * bi;
        out[2*i + 1] = ai * br - ar * bi;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        float ar = a[2*i];
        float ai = a[2*i + 1];
        float br = b[2*i];
        float bi = b[2*i + 1];
        out[2*i] = ar * br + ai * bi;
        out[2*i + 1] = ai * br - ar * bi;
    }
#endif
}

void neon_complex_dot_f32(float* out_re, float* out_im,
                          const float* a_re, const float* a_im,
                          const float* b_re, const float* b_im,
                          std::size_t n) {
    // Complex dot product: sum(conj(a) * b) - standard mathematical inner product
    // This matches Eigen's VectorXcf::dot() behavior
#ifdef OPTMATH_USE_NEON
    float32x4_t sum_re = vdupq_n_f32(0.0f);
    float32x4_t sum_im = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 3 < n; i += 4) {
        float32x4_t ar = vld1q_f32(a_re + i);
        float32x4_t ai = vld1q_f32(a_im + i);
        float32x4_t br = vld1q_f32(b_re + i);
        float32x4_t bi = vld1q_f32(b_im + i);

        // conj(a) * b = (ar - j*ai) * (br + j*bi)
        //             = (ar*br + ai*bi) + j*(ar*bi - ai*br)
        sum_re = vmlaq_f32(sum_re, ar, br);
        sum_re = vmlaq_f32(sum_re, ai, bi);
        sum_im = vmlaq_f32(sum_im, ar, bi);
        sum_im = vmlsq_f32(sum_im, ai, br);
    }

    float re = vaddvq_f32(sum_re);
    float im = vaddvq_f32(sum_im);

    for (; i < n; ++i) {
        re += a_re[i] * b_re[i] + a_im[i] * b_im[i];
        im += a_re[i] * b_im[i] - a_im[i] * b_re[i];
    }

    *out_re = re;
    *out_im = im;
#else
    float re = 0.0f, im = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        re += a_re[i] * b_re[i] + a_im[i] * b_im[i];
        im += a_re[i] * b_im[i] - a_im[i] * b_re[i];
    }
    *out_re = re;
    *out_im = im;
#endif
}

void neon_complex_magnitude_f32(float* out, const float* re, const float* im, std::size_t n) {
    // |z| = sqrt(re^2 + im^2)
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vr = vld1q_f32(re + i);
        float32x4_t vi = vld1q_f32(im + i);

        float32x4_t mag_sq = vmulq_f32(vr, vr);
        mag_sq = vmlaq_f32(mag_sq, vi, vi);

        // Use NEON fast reciprocal sqrt approximation followed by multiplication
        // sqrt(x) = x * rsqrt(x)
        float32x4_t rsqrt = vrsqrteq_f32(mag_sq);
        // Newton-Raphson refinement for better accuracy
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(mag_sq, rsqrt), rsqrt));
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(mag_sq, rsqrt), rsqrt));

        float32x4_t mag = vmulq_f32(mag_sq, rsqrt);

        // Handle zero case (mag_sq = 0 -> rsqrt = inf, result should be 0)
        uint32x4_t zero_mask = vceqzq_f32(mag_sq);
        mag = vbslq_f32(zero_mask, vdupq_n_f32(0.0f), mag);

        vst1q_f32(out + i, mag);
    }
    for (; i < n; ++i) {
        out[i] = std::sqrt(re[i] * re[i] + im[i] * im[i]);
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::sqrt(re[i] * re[i] + im[i] * im[i]);
    }
#endif
}

void neon_complex_magnitude_squared_f32(float* out, const float* re, const float* im, std::size_t n) {
    // |z|^2 = re^2 + im^2 (avoids sqrt, useful for power calculations)
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vr = vld1q_f32(re + i);
        float32x4_t vi = vld1q_f32(im + i);

        float32x4_t mag_sq = vmulq_f32(vr, vr);
        mag_sq = vmlaq_f32(mag_sq, vi, vi);

        vst1q_f32(out + i, mag_sq);
    }
    for (; i < n; ++i) {
        out[i] = re[i] * re[i] + im[i] * im[i];
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = re[i] * re[i] + im[i] * im[i];
    }
#endif
}

void neon_complex_phase_f32(float* out, const float* re, const float* im, std::size_t n) {
    // Phase = atan2(im, re)
    // Note: atan2 is not easily vectorizable without approximation
    // Using scalar implementation for accuracy
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::atan2(im[i], re[i]);
    }
}

void neon_complex_add_f32(float* out_re, float* out_im,
                          const float* a_re, const float* a_im,
                          const float* b_re, const float* b_im,
                          std::size_t n) {
#ifdef OPTMATH_USE_NEON
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t ar = vld1q_f32(a_re + i);
        float32x4_t ai = vld1q_f32(a_im + i);
        float32x4_t br = vld1q_f32(b_re + i);
        float32x4_t bi = vld1q_f32(b_im + i);

        vst1q_f32(out_re + i, vaddq_f32(ar, br));
        vst1q_f32(out_im + i, vaddq_f32(ai, bi));
    }
    for (; i < n; ++i) {
        out_re[i] = a_re[i] + b_re[i];
        out_im[i] = a_im[i] + b_im[i];
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        out_re[i] = a_re[i] + b_re[i];
        out_im[i] = a_im[i] + b_im[i];
    }
#endif
}

void neon_complex_scale_f32(float* out_re, float* out_im,
                            const float* in_re, const float* in_im,
                            float scale_re, float scale_im,
                            std::size_t n) {
    // (in_re + j*in_im) * (scale_re + j*scale_im)
#ifdef OPTMATH_USE_NEON
    float32x4_t sr = vdupq_n_f32(scale_re);
    float32x4_t si = vdupq_n_f32(scale_im);

    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t ir = vld1q_f32(in_re + i);
        float32x4_t ii = vld1q_f32(in_im + i);

        // out_re = in_re*scale_re - in_im*scale_im
        float32x4_t or_val = vmulq_f32(ir, sr);
        or_val = vmlsq_f32(or_val, ii, si);

        // out_im = in_re*scale_im + in_im*scale_re
        float32x4_t oi_val = vmulq_f32(ir, si);
        oi_val = vmlaq_f32(oi_val, ii, sr);

        vst1q_f32(out_re + i, or_val);
        vst1q_f32(out_im + i, oi_val);
    }
    for (; i < n; ++i) {
        out_re[i] = in_re[i] * scale_re - in_im[i] * scale_im;
        out_im[i] = in_re[i] * scale_im + in_im[i] * scale_re;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        out_re[i] = in_re[i] * scale_re - in_im[i] * scale_im;
        out_im[i] = in_re[i] * scale_im + in_im[i] * scale_re;
    }
#endif
}

void neon_complex_exp_f32(float* out_re, float* out_im, const float* phase, std::size_t n) {
    // exp(j*phase) = cos(phase) + j*sin(phase)
    // Using scalar for now; vectorized sin/cos can be added later
    for (std::size_t i = 0; i < n; ++i) {
        out_re[i] = std::cos(phase[i]);
        out_im[i] = std::sin(phase[i]);
    }
}

// =========================================================================
// Eigen Wrappers for Complex Operations
// =========================================================================

Eigen::VectorXcf neon_complex_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return Eigen::VectorXcf();

    Eigen::VectorXcf result(a.size());

    // Extract real and imaginary parts
    Eigen::VectorXf a_re = a.real();
    Eigen::VectorXf a_im = a.imag();
    Eigen::VectorXf b_re = b.real();
    Eigen::VectorXf b_im = b.imag();
    Eigen::VectorXf out_re(a.size());
    Eigen::VectorXf out_im(a.size());

    neon_complex_mul_f32(out_re.data(), out_im.data(),
                         a_re.data(), a_im.data(),
                         b_re.data(), b_im.data(),
                         a.size());

    result.real() = out_re;
    result.imag() = out_im;
    return result;
}

Eigen::VectorXcf neon_complex_conj_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return Eigen::VectorXcf();

    Eigen::VectorXcf result(a.size());

    Eigen::VectorXf a_re = a.real();
    Eigen::VectorXf a_im = a.imag();
    Eigen::VectorXf b_re = b.real();
    Eigen::VectorXf b_im = b.imag();
    Eigen::VectorXf out_re(a.size());
    Eigen::VectorXf out_im(a.size());

    neon_complex_conj_mul_f32(out_re.data(), out_im.data(),
                              a_re.data(), a_im.data(),
                              b_re.data(), b_im.data(),
                              a.size());

    result.real() = out_re;
    result.imag() = out_im;
    return result;
}

std::complex<float> neon_complex_dot(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return std::complex<float>(0, 0);

    Eigen::VectorXf a_re = a.real();
    Eigen::VectorXf a_im = a.imag();
    Eigen::VectorXf b_re = b.real();
    Eigen::VectorXf b_im = b.imag();

    float re, im;
    neon_complex_dot_f32(&re, &im,
                         a_re.data(), a_im.data(),
                         b_re.data(), b_im.data(),
                         a.size());

    return std::complex<float>(re, im);
}

Eigen::VectorXf neon_complex_magnitude(const Eigen::VectorXcf& a) {
    Eigen::VectorXf result(a.size());
    Eigen::VectorXf re = a.real();
    Eigen::VectorXf im = a.imag();

    neon_complex_magnitude_f32(result.data(), re.data(), im.data(), a.size());
    return result;
}

Eigen::VectorXf neon_complex_phase(const Eigen::VectorXcf& a) {
    Eigen::VectorXf result(a.size());
    Eigen::VectorXf re = a.real();
    Eigen::VectorXf im = a.imag();

    neon_complex_phase_f32(result.data(), re.data(), im.data(), a.size());
    return result;
}

} // namespace neon
} // namespace optmath
