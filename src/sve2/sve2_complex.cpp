#include "optmath/sve2_kernels.hpp"
#include "optmath/neon_kernels.hpp"
#include <cmath>

#ifdef OPTMATH_USE_SVE2
#include <arm_sve.h>
#endif

namespace optmath {
namespace sve2 {

// =========================================================================
// Complex Number Operations (SVE2/FCMA accelerated)
// =========================================================================
// Split format: separate real and imaginary arrays
// Interleaved format: [re0, im0, re1, im1, ...]
// SVE2 predicated loops eliminate scalar tail handling entirely.

// -------------------------------------------------------------------------
// 1. Complex Multiply (split format)
// -------------------------------------------------------------------------
void sve2_complex_mul_f32(float* out_re, float* out_im,
                           const float* a_re, const float* a_im,
                           const float* b_re, const float* b_im,
                           std::size_t n) {
    // (ar + j*ai) * (br + j*bi) = (ar*br - ai*bi) + j*(ar*bi + ai*br)
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        // out_re = ar*br - ai*bi
        svfloat32_t or_val = svmul_f32_z(pg, ar, br);
        or_val = svmls_f32_z(pg, or_val, ai, bi);

        // out_im = ar*bi + ai*br
        svfloat32_t oi_val = svmul_f32_z(pg, ar, bi);
        oi_val = svmla_f32_z(pg, oi_val, ai, br);

        svst1_f32(pg, out_re + i, or_val);
        svst1_f32(pg, out_im + i, oi_val);

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, (uint64_t)n)));
#else
    neon::neon_complex_mul_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

// -------------------------------------------------------------------------
// 2. Complex Conjugate Multiply (split format)
// -------------------------------------------------------------------------
void sve2_complex_conj_mul_f32(float* out_re, float* out_im,
                                const float* a_re, const float* a_im,
                                const float* b_re, const float* b_im,
                                std::size_t n) {
    // a * conj(b) = (ar + j*ai) * (br - j*bi)
    //            = (ar*br + ai*bi) + j*(ai*br - ar*bi)
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        // out_re = ar*br + ai*bi
        svfloat32_t or_val = svmul_f32_z(pg, ar, br);
        or_val = svmla_f32_z(pg, or_val, ai, bi);

        // out_im = ai*br - ar*bi
        svfloat32_t oi_val = svmul_f32_z(pg, ai, br);
        oi_val = svmls_f32_z(pg, oi_val, ar, bi);

        svst1_f32(pg, out_re + i, or_val);
        svst1_f32(pg, out_im + i, oi_val);

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, (uint64_t)n)));
#else
    neon::neon_complex_conj_mul_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

// -------------------------------------------------------------------------
// 3. Complex Multiply Interleaved (FCMA: 2 instructions!)
// -------------------------------------------------------------------------
void sve2_complex_mul_interleaved_f32(float* out, const float* a,
                                       const float* b, std::size_t n) {
    // Interleaved format: [re0, im0, re1, im1, ...]
    // n = number of complex samples; array size is 2*n floats
#ifdef OPTMATH_USE_SVE2
    uint64_t total = (uint64_t)(2 * n);
    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, total);

        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);

#ifdef OPTMATH_USE_FCMA
        // With FCMA, interleaved complex multiply is just 2 instructions:
        //   svcmla rotation=0   -> acc += a * b (real contribution)
        //   svcmla rotation=90  -> acc += a * b (imag contribution)
        svfloat32_t acc = svdup_f32(0.0f);
        acc = svcmla_f32_z(pg, acc, va, vb, 0);    // real contribution
        acc = svcmla_f32_z(pg, acc, va, vb, 90);   // imag contribution
#else
        // Manual deinterleave and multiply without FCMA
        // Extract even (real) and odd (imag) elements
        svfloat32_t ar = svuzp1_f32(va, va);  // real parts of a
        svfloat32_t ai = svuzp2_f32(va, va);  // imag parts of a
        svfloat32_t br = svuzp1_f32(vb, vb);  // real parts of b
        svfloat32_t bi = svuzp2_f32(vb, vb);  // imag parts of b

        // out_re = ar*br - ai*bi
        svfloat32_t or_val = svmul_f32_z(pg, ar, br);
        or_val = svmls_f32_z(pg, or_val, ai, bi);

        // out_im = ar*bi + ai*br
        svfloat32_t oi_val = svmul_f32_z(pg, ar, bi);
        oi_val = svmla_f32_z(pg, oi_val, ai, br);

        // Re-interleave results
        svfloat32_t acc = svzip1_f32(or_val, oi_val);
#endif

        svst1_f32(pg, out + i, acc);

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, total)));
#else
    neon::neon_complex_mul_interleaved_f32(out, a, b, n);
#endif
}

// -------------------------------------------------------------------------
// 4. Complex Conjugate Multiply Interleaved (FCMA rotations 0, 270)
// -------------------------------------------------------------------------
void sve2_complex_conj_mul_interleaved_f32(float* out, const float* a,
                                            const float* b, std::size_t n) {
    // a * conj(b) interleaved using FCMA
    // Rotation 270 gives conjugate multiply for imaginary part
#ifdef OPTMATH_USE_SVE2
    uint64_t total = (uint64_t)(2 * n);
    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, total);

        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);

#ifdef OPTMATH_USE_FCMA
        // FCMA conjugate multiply: rotations 0 and 270
        svfloat32_t acc = svdup_f32(0.0f);
        acc = svcmla_f32_z(pg, acc, va, vb, 0);    // real contribution
        acc = svcmla_f32_z(pg, acc, va, vb, 270);  // conjugate imag contribution
#else
        // Manual deinterleave and conjugate multiply without FCMA
        svfloat32_t ar = svuzp1_f32(va, va);
        svfloat32_t ai = svuzp2_f32(va, va);
        svfloat32_t br = svuzp1_f32(vb, vb);
        svfloat32_t bi = svuzp2_f32(vb, vb);

        // out_re = ar*br + ai*bi
        svfloat32_t or_val = svmul_f32_z(pg, ar, br);
        or_val = svmla_f32_z(pg, or_val, ai, bi);

        // out_im = ai*br - ar*bi
        svfloat32_t oi_val = svmul_f32_z(pg, ai, br);
        oi_val = svmls_f32_z(pg, oi_val, ar, bi);

        // Re-interleave results
        svfloat32_t acc = svzip1_f32(or_val, oi_val);
#endif

        svst1_f32(pg, out + i, acc);

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, total)));
#else
    neon::neon_complex_conj_mul_interleaved_f32(out, a, b, n);
#endif
}

// -------------------------------------------------------------------------
// 5. Complex Dot Product (split format)
// -------------------------------------------------------------------------
void sve2_complex_dot_f32(float* out_re, float* out_im,
                           const float* a_re, const float* a_im,
                           const float* b_re, const float* b_im,
                           std::size_t n) {
    // Complex dot product: sum(conj(a) * b)
    // conj(a) * b = (ar - j*ai) * (br + j*bi)
    //             = (ar*br + ai*bi) + j*(ar*bi - ai*br)
#ifdef OPTMATH_USE_SVE2
    svfloat32_t sum_re = svdup_f32(0.0f);
    svfloat32_t sum_im = svdup_f32(0.0f);

    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        // Accumulate real: ar*br + ai*bi
        sum_re = svmla_f32_m(pg, sum_re, ar, br);
        sum_re = svmla_f32_m(pg, sum_re, ai, bi);

        // Accumulate imag: ar*bi - ai*br
        sum_im = svmla_f32_m(pg, sum_im, ar, bi);
        sum_im = svmls_f32_m(pg, sum_im, ai, br);

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, (uint64_t)n)));

    // Horizontal reduction across all lanes
    *out_re = svaddv_f32(svptrue_b32(), sum_re);
    *out_im = svaddv_f32(svptrue_b32(), sum_im);
#else
    neon::neon_complex_dot_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

// -------------------------------------------------------------------------
// 6. Complex Magnitude (split format)
// -------------------------------------------------------------------------
void sve2_complex_magnitude_f32(float* out, const float* re,
                                 const float* im, std::size_t n) {
    // |z| = sqrt(re^2 + im^2)
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

        svfloat32_t vr = svld1_f32(pg, re + i);
        svfloat32_t vi = svld1_f32(pg, im + i);

        // mag_sq = re^2 + im^2
        svfloat32_t mag_sq = svmul_f32_z(pg, vr, vr);
        mag_sq = svmla_f32_z(pg, mag_sq, vi, vi);

        // SVE2 has a proper sqrt instruction
        svfloat32_t mag = svsqrt_f32_z(pg, mag_sq);

        svst1_f32(pg, out + i, mag);

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, (uint64_t)n)));
#else
    neon::neon_complex_magnitude_f32(out, re, im, n);
#endif
}

// -------------------------------------------------------------------------
// 7. Complex Phase (split format) - scalar atan2
// -------------------------------------------------------------------------
void sve2_complex_phase_f32(float* out, const float* re,
                             const float* im, std::size_t n) {
    // Phase = atan2(im, re)
    // atan2 is not easily vectorizable without approximation tables.
    // Using scalar implementation for accuracy.
#ifdef OPTMATH_USE_SVE2
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::atan2(im[i], re[i]);
    }
#else
    neon::neon_complex_phase_f32(out, re, im, n);
#endif
}

// -------------------------------------------------------------------------
// 8. Complex Add (split format)
// -------------------------------------------------------------------------
void sve2_complex_add_f32(float* out_re, float* out_im,
                           const float* a_re, const float* a_im,
                           const float* b_re, const float* b_im,
                           std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        svst1_f32(pg, out_re + i, svadd_f32_z(pg, ar, br));
        svst1_f32(pg, out_im + i, svadd_f32_z(pg, ai, bi));

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, (uint64_t)n)));
#else
    neon::neon_complex_add_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

// -------------------------------------------------------------------------
// 9. Complex Scale (split format) - complex scalar multiply
// -------------------------------------------------------------------------
void sve2_complex_scale_f32(float* out_re, float* out_im,
                              const float* in_re, const float* in_im,
                              float scale_re, float scale_im,
                              std::size_t n) {
    // (in_re + j*in_im) * (scale_re + j*scale_im)
    // = (in_re*scale_re - in_im*scale_im) + j*(in_re*scale_im + in_im*scale_re)
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

        svfloat32_t ir = svld1_f32(pg, in_re + i);
        svfloat32_t ii = svld1_f32(pg, in_im + i);

        svfloat32_t sr = svdup_f32(scale_re);
        svfloat32_t si = svdup_f32(scale_im);

        // out_re = in_re*scale_re - in_im*scale_im
        svfloat32_t or_val = svmul_f32_z(pg, ir, sr);
        or_val = svmls_f32_z(pg, or_val, ii, si);

        // out_im = in_re*scale_im + in_im*scale_re
        svfloat32_t oi_val = svmul_f32_z(pg, ir, si);
        oi_val = svmla_f32_z(pg, oi_val, ii, sr);

        svst1_f32(pg, out_re + i, or_val);
        svst1_f32(pg, out_im + i, oi_val);

        i += svcntw();
    } while (svptest_first(svptrue_b32(), svwhilelt_b32(i, (uint64_t)n)));
#else
    neon::neon_complex_scale_f32(out_re, out_im, in_re, in_im, scale_re, scale_im, n);
#endif
}

// -------------------------------------------------------------------------
// 10. Complex Exponential - exp(j*phase) = cos(phase) + j*sin(phase)
// -------------------------------------------------------------------------
void sve2_complex_exp_f32(float* out_re, float* out_im,
                           const float* phase, std::size_t n) {
    // exp(j*phase) = cos(phase) + j*sin(phase)
    // Using scalar cos/sin for now; vectorized sin/cos approximations
    // can be substituted once accuracy requirements are defined.
#ifdef OPTMATH_USE_SVE2
    for (std::size_t i = 0; i < n; ++i) {
        out_re[i] = std::cos(phase[i]);
        out_im[i] = std::sin(phase[i]);
    }
#else
    neon::neon_complex_exp_f32(out_re, out_im, phase, n);
#endif
}

// =========================================================================
// Eigen Wrappers for Complex Operations
// =========================================================================

Eigen::VectorXcf sve2_complex_mul(const Eigen::VectorXcf& a,
                                   const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return Eigen::VectorXcf();

    Eigen::VectorXcf result(a.size());

    Eigen::VectorXf a_re = a.real();
    Eigen::VectorXf a_im = a.imag();
    Eigen::VectorXf b_re = b.real();
    Eigen::VectorXf b_im = b.imag();
    Eigen::VectorXf out_re(a.size());
    Eigen::VectorXf out_im(a.size());

    sve2_complex_mul_f32(out_re.data(), out_im.data(),
                          a_re.data(), a_im.data(),
                          b_re.data(), b_im.data(),
                          a.size());

    result.real() = out_re;
    result.imag() = out_im;
    return result;
}

Eigen::VectorXcf sve2_complex_conj_mul(const Eigen::VectorXcf& a,
                                        const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return Eigen::VectorXcf();

    Eigen::VectorXcf result(a.size());

    Eigen::VectorXf a_re = a.real();
    Eigen::VectorXf a_im = a.imag();
    Eigen::VectorXf b_re = b.real();
    Eigen::VectorXf b_im = b.imag();
    Eigen::VectorXf out_re(a.size());
    Eigen::VectorXf out_im(a.size());

    sve2_complex_conj_mul_f32(out_re.data(), out_im.data(),
                               a_re.data(), a_im.data(),
                               b_re.data(), b_im.data(),
                               a.size());

    result.real() = out_re;
    result.imag() = out_im;
    return result;
}

std::complex<float> sve2_complex_dot(const Eigen::VectorXcf& a,
                                      const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return std::complex<float>(0, 0);

    Eigen::VectorXf a_re = a.real();
    Eigen::VectorXf a_im = a.imag();
    Eigen::VectorXf b_re = b.real();
    Eigen::VectorXf b_im = b.imag();

    float re, im;
    sve2_complex_dot_f32(&re, &im,
                          a_re.data(), a_im.data(),
                          b_re.data(), b_im.data(),
                          a.size());

    return std::complex<float>(re, im);
}

Eigen::VectorXf sve2_complex_magnitude(const Eigen::VectorXcf& a) {
    Eigen::VectorXf result(a.size());
    Eigen::VectorXf re = a.real();
    Eigen::VectorXf im = a.imag();

    sve2_complex_magnitude_f32(result.data(), re.data(), im.data(), a.size());
    return result;
}

Eigen::VectorXf sve2_complex_phase(const Eigen::VectorXcf& a) {
    Eigen::VectorXf result(a.size());
    Eigen::VectorXf re = a.real();
    Eigen::VectorXf im = a.imag();

    sve2_complex_phase_f32(result.data(), re.data(), im.data(), a.size());
    return result;
}

} // namespace sve2
} // namespace optmath
