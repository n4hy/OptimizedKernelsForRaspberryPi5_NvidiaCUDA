/**
 * OptMathKernels SVE2 Core Kernels
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Core SVE2 vectorized math kernels targeting AArch64 SVE2-capable processors
 * (Cortex-A720 / Orange Pi 5 Plus and similar).
 *
 * Core Vector Operations (Predicated):
 *   All loops use svwhilelt_b32/svptest_any count-up predication, eliminating
 *   scalar tail code entirely. Includes sve2_dot_f32 (predicated FMA with
 *   svaddv_f32 horizontal reduction), sve2_add/sub/mul/div_f32 (svadd/svsub/
 *   svmul/svdiv_f32_z zeroing predication), sve2_norm_f32 (dot self + sqrtf),
 *   and sve2_reduce_sum/max/min_f32 (svaddv/svmaxv/svminv_f32). Division uses
 *   a sign-preserving epsilon guard.
 *
 * Vectorized Transcendentals (Predicated Polynomial Approximation):
 *   sve2_fast_exp_f32 uses range reduction x=k*ln2+f with svrintn_f32_z
 *   rounding, integer exponent reconstruction via svcvt_s32_f32_z/
 *   svlsl_n_s32_z/svreinterpret_f32_s32, and a 6-coefficient Horner
 *   polynomial with svmla_n_f32_z. sve2_fast_sin_f32 uses pi-based range
 *   reduction with a 5-coefficient Chebyshev polynomial and predicated sign
 *   correction via svneg_f32_m. sve2_fast_cos_f32 = sin(x+pi/2).
 *   sve2_fast_sigmoid_f32 = 1/(1+exp(-x)) with clamping via svmin/svmax_f32_m.
 *   sve2_fast_tanh_f32 = 2*sigmoid(2x)-1.
 *
 * Cache-Blocked GEMM with 8x8 Microkernel:
 *   micro_kernel_8x8_sve2 uses column-oriented accumulators (8 columns x 2
 *   vector halves for VL-agnostic design). pack_A_panel_sve2 packs MR=8 row
 *   strips with predicated loads. pack_B_panel_sve2 packs NR=8 column strips.
 *   sve2_gemm_blocked_f32 implements 3-level Goto-style blocking: MC=256,
 *   KC=512, NC=1024 tuned for Cortex-A720 12MB L3 cache on Orange Pi 5 Plus.
 *
 * I8MM Integer Matrix Multiply:
 *   sve2_gemm_i8mm uses svmmla_s32 for 2x2 output tiles processing K in
 *   chunks of 8. Supports quantized int8 inference with scale/zero-point
 *   dequantization.
 *
 * FIR Filter:
 *   sve2_fir_f32 implements valid convolution as a series of predicated dot
 *   products.
 *
 * Eigen Wrappers:
 *   Complete set for VectorXf/MatrixXf with data pointer extraction.
 */
#include "optmath/sve2_kernels.hpp"
#include "optmath/neon_kernels.hpp"
#include "optmath/platform.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef OPTMATH_USE_SVE2
#include <arm_sve.h>
#endif

namespace optmath {
namespace sve2 {

// is_available() is in sve2_detect.cpp (compiled without SVE2 flags)

// =========================================================================
// Core Vector Operations (predicated - no scalar tail loops)
// =========================================================================

float sve2_dot_f32(const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    svfloat32_t vsum = svdup_f32(0.0f);
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        vsum = svmla_f32_z(svptrue_b32(), vsum, va, vb);
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), vsum);
#else
    return neon::neon_dot_f32(a, b, n);
#endif
}

void sve2_add_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, out + i, svadd_f32_z(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_add_f32(out, a, b, n);
#endif
}

void sve2_sub_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, out + i, svsub_f32_z(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_sub_f32(out, a, b, n);
#endif
}

void sve2_mul_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, out + i, svmul_f32_z(pg, va, vb));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_mul_f32(out, a, b, n);
#endif
}

void sve2_div_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    const float epsilon = 1e-10f;
    svfloat32_t veps = svdup_f32(epsilon);
    svfloat32_t vzero = svdup_f32(0.0f);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);

        // Sign-preserving epsilon: add epsilon if non-negative, subtract if negative
        svbool_t neg_mask = svcmplt_f32(pg, vb, vzero);
        svfloat32_t eps_signed = svneg_f32_m(veps, neg_mask, veps);
        svfloat32_t vb_safe = svadd_f32_z(pg, vb, eps_signed);

        svst1_f32(pg, out + i, svdiv_f32_z(pg, va, vb_safe));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_div_f32(out, a, b, n);
#endif
}

float sve2_norm_f32(const float* a, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    float dot = sve2_dot_f32(a, a, n);
    return std::sqrt(dot);
#else
    return neon::neon_norm_f32(a, n);
#endif
}

float sve2_reduce_sum_f32(const float* a, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    svfloat32_t vsum = svdup_f32(0.0f);
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        vsum = svadd_f32_m(svptrue_b32(), vsum, va);
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), vsum);
#else
    return neon::neon_reduce_sum_f32(a, n);
#endif
}

float sve2_reduce_max_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_SVE2
    svfloat32_t vmax = svdup_f32(-3.402823466e+38f);
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        vmax = svmax_f32_m(pg, vmax, va);
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    return svmaxv_f32(svptrue_b32(), vmax);
#else
    return neon::neon_reduce_max_f32(a, n);
#endif
}

float sve2_reduce_min_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_SVE2
    svfloat32_t vmin = svdup_f32(3.402823466e+38f);
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        vmin = svmin_f32_m(pg, vmin, va);
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    return svminv_f32(svptrue_b32(), vmin);
#else
    return neon::neon_reduce_min_f32(a, n);
#endif
}

// =========================================================================
// Vectorized Transcendental Functions
// =========================================================================
// Same polynomial coefficients as NEON implementations

void sve2_fast_exp_f32(float* out, const float* in, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // Polynomial coefficients for 2^x on [-0.5, 0.5] (same as NEON)
    const float log2e = 1.44269504088896341f;
    const float ln2   = 0.693147180559945309f;
    const float c0 = 1.0f;
    const float c1 = 0.693147182464599609f;
    const float c2 = 0.240226507186889648f;
    const float c3 = 0.055504187941551208f;
    const float c4 = 0.009618341922760010f;
    const float c5 = 0.001333355903625488f;
    const float c6 = 0.000154034309089184f;

    svfloat32_t vlog2e = svdup_f32(log2e);
    svfloat32_t vln2   = svdup_f32(ln2);
    svfloat32_t vc0 = svdup_f32(c0);
    svfloat32_t vc1 = svdup_f32(c1);
    svfloat32_t vc2 = svdup_f32(c2);
    svfloat32_t vc3 = svdup_f32(c3);
    svfloat32_t vc4 = svdup_f32(c4);
    svfloat32_t vc5 = svdup_f32(c5);
    svfloat32_t vc6 = svdup_f32(c6);

    svfloat32_t vmax = svdup_f32(88.0f);
    svfloat32_t vmin = svdup_f32(-88.0f);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svld1_f32(pg, in + i);

        // Clamp input to avoid overflow/underflow
        x = svmin_f32_z(pg, svmax_f32_z(pg, x, vmin), vmax);

        // Range reduction: x = k * ln2 + f, where k = round(x * log2e)
        svfloat32_t t = svmul_f32_z(pg, x, vlog2e);
        svfloat32_t k = svrintn_f32_z(pg, t);
        svfloat32_t f = svmls_f32_z(pg, x, k, vln2);  // f = x - k * ln2

        // Horner's method: p = c6*f + c5, p = p*f + c4, ...
        svfloat32_t p = svmad_f32_z(pg, vc6, f, vc5);   // p = c6*f + c5
        p = svmad_f32_z(pg, p, f, vc4);                  // p = p*f + c4
        p = svmad_f32_z(pg, p, f, vc3);                  // p = p*f + c3
        p = svmad_f32_z(pg, p, f, vc2);                  // p = p*f + c2
        p = svmad_f32_z(pg, p, f, vc1);                  // p = p*f + c1
        p = svmad_f32_z(pg, p, f, vc0);                  // p = p*f + c0

        // Reconstruct: exp(x) = 2^k * p via integer bit manipulation
        svint32_t ki = svcvt_s32_f32_z(pg, k);
        ki = svadd_s32_z(pg, ki, svdup_s32(127));        // Add IEEE754 bias
        ki = svlsl_n_s32_z(pg, ki, 23);                  // Shift to exponent position
        svfloat32_t scale = svreinterpret_f32_s32(ki);

        svfloat32_t result = svmul_f32_z(pg, p, scale);
        svst1_f32(pg, out + i, result);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_fast_exp_f32(out, in, n);
#endif
}

void sve2_fast_sin_f32(float* out, const float* in, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // Chebyshev polynomial coefficients for sin(x) (same as NEON)
    const float pi     = 3.14159265358979323846f;
    const float inv_pi = 0.31830988618379067154f;
    const float c1 =  1.0f;
    const float c3 = -0.16666667163372039795f;
    const float c5 =  0.00833333376795053482f;
    const float c7 = -0.00019841269776225090f;
    const float c9 =  0.00000275573189712526f;

    svfloat32_t vpi     = svdup_f32(pi);
    svfloat32_t vinv_pi = svdup_f32(inv_pi);
    svfloat32_t vc1 = svdup_f32(c1);
    svfloat32_t vc3 = svdup_f32(c3);
    svfloat32_t vc5 = svdup_f32(c5);
    svfloat32_t vc7 = svdup_f32(c7);
    svfloat32_t vc9 = svdup_f32(c9);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svld1_f32(pg, in + i);

        // Range reduction: x = x - round(x / pi) * pi
        svfloat32_t k = svrintn_f32_z(pg, svmul_f32_z(pg, x, vinv_pi));
        x = svmls_f32_z(pg, x, k, vpi);  // x = x - k * pi

        // sin(x) = x * (c1 + x^2*(c3 + x^2*(c5 + x^2*(c7 + x^2*c9))))
        svfloat32_t x2 = svmul_f32_z(pg, x, x);

        // Horner's method on x^2
        svfloat32_t p = svmad_f32_z(pg, vc9, x2, vc7);  // p = c9*x2 + c7
        p = svmad_f32_z(pg, p, x2, vc5);                 // p = p*x2 + c5
        p = svmad_f32_z(pg, p, x2, vc3);                 // p = p*x2 + c3
        p = svmad_f32_z(pg, p, x2, vc1);                 // p = p*x2 + c1
        p = svmul_f32_z(pg, p, x);                       // p = p * x

        // Handle sign flip for odd k: if k is odd, negate result
        svint32_t ki = svcvt_s32_f32_z(pg, k);
        svbool_t odd = svcmpne_s32(pg, svand_s32_z(pg, ki, svdup_s32(1)), svdup_s32(0));
        p = svneg_f32_m(p, odd, p);

        svst1_f32(pg, out + i, p);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_fast_sin_f32(out, in, n);
#endif
}

void sve2_fast_cos_f32(float* out, const float* in, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // cos(x) = sin(x + pi/2) — inline the sin polynomial to avoid heap allocation
    const float half_pi = 1.57079632679489661923f;
    const float pi      = 3.14159265358979323846f;
    const float inv_pi  = 0.31830988618379067154f;
    const float c1 =  1.0f;
    const float c3 = -0.16666667163372039795f;
    const float c5 =  0.00833333376795053482f;
    const float c7 = -0.00019841269776225090f;
    const float c9 =  0.00000275573189712526f;

    svfloat32_t vhalf_pi = svdup_f32(half_pi);
    svfloat32_t vpi      = svdup_f32(pi);
    svfloat32_t vinv_pi  = svdup_f32(inv_pi);
    svfloat32_t vc1 = svdup_f32(c1);
    svfloat32_t vc3 = svdup_f32(c3);
    svfloat32_t vc5 = svdup_f32(c5);
    svfloat32_t vc7 = svdup_f32(c7);
    svfloat32_t vc9 = svdup_f32(c9);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svadd_f32_z(pg, svld1_f32(pg, in + i), vhalf_pi);

        // Range reduction: x = x - round(x / pi) * pi
        svfloat32_t k = svrintn_f32_z(pg, svmul_f32_z(pg, x, vinv_pi));
        x = svmls_f32_z(pg, x, k, vpi);

        svfloat32_t x2 = svmul_f32_z(pg, x, x);

        svfloat32_t p = svmad_f32_z(pg, vc9, x2, vc7);
        p = svmad_f32_z(pg, p, x2, vc5);
        p = svmad_f32_z(pg, p, x2, vc3);
        p = svmad_f32_z(pg, p, x2, vc1);
        p = svmul_f32_z(pg, p, x);

        // Sign flip for odd k
        svint32_t ki = svcvt_s32_f32_z(pg, k);
        svbool_t odd = svcmpne_s32(pg, svand_s32_z(pg, ki, svdup_s32(1)), svdup_s32(0));
        p = svneg_f32_m(p, odd, p);

        svst1_f32(pg, out + i, p);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_fast_cos_f32(out, in, n);
#endif
}

void sve2_fast_sigmoid_f32(float* out, const float* in, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // Fast sigmoid: 1 / (1 + exp(-x)) — single-pass, no heap allocation
    const float log2e = 1.44269504088896341f;
    const float ln2   = 0.693147180559945309f;
    const float c0 = 1.0f;
    const float c1 = 0.693147182464599609f;
    const float c2 = 0.240226507186889648f;
    const float c3 = 0.055504187941551208f;
    const float c4 = 0.009618341922760010f;
    const float c5 = 0.001333355903625488f;
    const float c6 = 0.000154034309089184f;

    svfloat32_t vone   = svdup_f32(1.0f);
    svfloat32_t vlog2e = svdup_f32(log2e);
    svfloat32_t vln2   = svdup_f32(ln2);
    svfloat32_t vc0 = svdup_f32(c0), vc1 = svdup_f32(c1);
    svfloat32_t vc2 = svdup_f32(c2), vc3 = svdup_f32(c3);
    svfloat32_t vc4 = svdup_f32(c4), vc5 = svdup_f32(c5);
    svfloat32_t vc6 = svdup_f32(c6);
    svfloat32_t vclamp = svdup_f32(20.0f);
    svfloat32_t vmax_exp = svdup_f32(88.0f);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svld1_f32(pg, in + i);
        x = svmin_f32_z(pg, svmax_f32_z(pg, x, svneg_f32_z(pg, vclamp)), vclamp);

        // Compute exp(-x) inline
        svfloat32_t neg_x = svneg_f32_z(pg, x);
        neg_x = svmin_f32_z(pg, svmax_f32_z(pg, neg_x, svneg_f32_z(pg, vmax_exp)), vmax_exp);

        svfloat32_t t = svmul_f32_z(pg, neg_x, vlog2e);
        svfloat32_t k = svrintn_f32_z(pg, t);
        svfloat32_t f = svmls_f32_z(pg, neg_x, k, vln2);

        svfloat32_t p = svmad_f32_z(pg, vc6, f, vc5);
        p = svmad_f32_z(pg, p, f, vc4);
        p = svmad_f32_z(pg, p, f, vc3);
        p = svmad_f32_z(pg, p, f, vc2);
        p = svmad_f32_z(pg, p, f, vc1);
        p = svmad_f32_z(pg, p, f, vc0);

        svint32_t ki = svcvt_s32_f32_z(pg, k);
        ki = svadd_s32_z(pg, ki, svdup_s32(127));
        ki = svlsl_n_s32_z(pg, ki, 23);
        svfloat32_t scale = svreinterpret_f32_s32(ki);
        svfloat32_t exp_neg = svmul_f32_z(pg, p, scale);

        // sigmoid = 1 / (1 + exp(-x))
        svfloat32_t result = svdiv_f32_z(pg, vone, svadd_f32_z(pg, vone, exp_neg));
        svst1_f32(pg, out + i, result);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_fast_sigmoid_f32(out, in, n);
#endif
}

void sve2_fast_tanh_f32(float* out, const float* in, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // tanh(x) = 2*sigmoid(2x) - 1 — single-pass, no heap allocation
    const float log2e = 1.44269504088896341f;
    const float ln2   = 0.693147180559945309f;
    const float c0 = 1.0f;
    const float c1 = 0.693147182464599609f;
    const float c2 = 0.240226507186889648f;
    const float c3 = 0.055504187941551208f;
    const float c4 = 0.009618341922760010f;
    const float c5 = 0.001333355903625488f;
    const float c6 = 0.000154034309089184f;

    svfloat32_t vtwo    = svdup_f32(2.0f);
    svfloat32_t vone    = svdup_f32(1.0f);
    svfloat32_t vneg_one = svdup_f32(-1.0f);
    svfloat32_t vlog2e  = svdup_f32(log2e);
    svfloat32_t vln2    = svdup_f32(ln2);
    svfloat32_t vc0 = svdup_f32(c0), vc1 = svdup_f32(c1);
    svfloat32_t vc2 = svdup_f32(c2), vc3 = svdup_f32(c3);
    svfloat32_t vc4 = svdup_f32(c4), vc5 = svdup_f32(c5);
    svfloat32_t vc6 = svdup_f32(c6);
    svfloat32_t vclamp  = svdup_f32(10.0f);
    svfloat32_t vmax_exp = svdup_f32(88.0f);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svld1_f32(pg, in + i);
        x = svmin_f32_z(pg, svmax_f32_z(pg, x, svneg_f32_z(pg, vclamp)), vclamp);

        // Compute exp(-2x) inline
        svfloat32_t neg_2x = svneg_f32_z(pg, svmul_f32_z(pg, vtwo, x));
        neg_2x = svmin_f32_z(pg, svmax_f32_z(pg, neg_2x, svneg_f32_z(pg, vmax_exp)), vmax_exp);

        svfloat32_t t = svmul_f32_z(pg, neg_2x, vlog2e);
        svfloat32_t k = svrintn_f32_z(pg, t);
        svfloat32_t f = svmls_f32_z(pg, neg_2x, k, vln2);

        svfloat32_t p = svmad_f32_z(pg, vc6, f, vc5);
        p = svmad_f32_z(pg, p, f, vc4);
        p = svmad_f32_z(pg, p, f, vc3);
        p = svmad_f32_z(pg, p, f, vc2);
        p = svmad_f32_z(pg, p, f, vc1);
        p = svmad_f32_z(pg, p, f, vc0);

        svint32_t ki = svcvt_s32_f32_z(pg, k);
        ki = svadd_s32_z(pg, ki, svdup_s32(127));
        ki = svlsl_n_s32_z(pg, ki, 23);
        svfloat32_t scale = svreinterpret_f32_s32(ki);
        svfloat32_t exp_neg = svmul_f32_z(pg, p, scale);

        // sigmoid(2x) = 1/(1+exp(-2x)), then tanh = 2*sig - 1
        svfloat32_t sig = svdiv_f32_z(pg, vone, svadd_f32_z(pg, vone, exp_neg));
        svfloat32_t result = svmad_f32_z(pg, vtwo, sig, vneg_one);
        svst1_f32(pg, out + i, result);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_fast_tanh_f32(out, in, n);
#endif
}

// =========================================================================
// GEMM - Cache-blocked with runtime tuning parameters
// =========================================================================

#ifdef OPTMATH_USE_SVE2

// Microkernel dimensions (same as NEON for compatibility)
static constexpr size_t MR = 8;
static constexpr size_t NR = 8;

// Maximum blocking parameters (thread-local buffers sized for max)
static constexpr size_t MAX_MC = 256;
static constexpr size_t MAX_KC = 512;
static constexpr size_t MAX_NC = 1024;

// Aligned thread-local packed buffers
alignas(64) static thread_local float packed_A[MAX_MC * MAX_KC];
alignas(64) static thread_local float packed_B[MAX_KC * MAX_NC];

// Pack a panel of A for the microkernel
// A is M x K (column-major), pack as column-strips of MR rows
static void pack_A_panel_sve2(
    float* packed,
    const float* A,
    size_t lda,
    size_t m,
    size_t k) {

    for (size_t p = 0; p < k; ++p) {
        uint64_t ii = 0;
        svbool_t pg = svwhilelt_b32(ii, (uint64_t)m);

        // Use predicated loads for valid rows, storing into packed MR-strided buffer
        do {
            svfloat32_t va = svld1_f32(pg, A + ii + p * lda);
            svst1_f32(pg, packed + p * MR + ii, va);
            ii += svcntw();
            pg = svwhilelt_b32(ii, (uint64_t)m);
        } while (svptest_any(svptrue_b32(), pg));

        // Zero-pad remaining rows up to MR
        for (size_t r = m; r < MR; ++r) {
            packed[p * MR + r] = 0.0f;
        }
    }
}

// Pack a panel of B for the microkernel
// B is K x N (column-major), pack as row-strips of NR columns
static void pack_B_panel_sve2(
    float* packed,
    const float* B,
    size_t ldb,
    size_t k,
    size_t n_cols) {

    for (size_t p = 0; p < k; ++p) {
        uint64_t jj = 0;
        svbool_t pg = svwhilelt_b32(jj, (uint64_t)n_cols);

        // Gather B[p, j] = B[p + j*ldb] - not contiguous, must gather element by element
        // Since B is column-major and we need row p across columns, elements are strided
        do {
            // Manual element copy since stride is ldb (not contiguous for SVE gather)
            uint64_t end = jj + svcntw();
            if (end > n_cols) end = n_cols;
            for (uint64_t j = jj; j < end; ++j) {
                packed[p * NR + j] = B[p + j * ldb];
            }
            jj = end;
            pg = svwhilelt_b32(jj, (uint64_t)n_cols);
        } while (svptest_any(svptrue_b32(), pg));

        // Zero-pad remaining columns up to NR
        for (size_t j = n_cols; j < NR; ++j) {
            packed[p * NR + j] = 0.0f;
        }
    }
}

// 8x8 SVE2 microkernel for GEMM
// Accumulates C[0:8, 0:8] += A_packed[0:8, 0:k] * B_packed[0:k, 0:8]
// Uses column-oriented accumulators and SVE2 FMA for the A720 pipeline.
// On CIX P1 (128-bit SVE2, svcntw()=4), each svfloat32_t holds 4 floats,
// so each 8-row column needs lo+hi pairs — same structure as the NEON kernel.
static void micro_kernel_8x8_sve2(
    size_t k,
    const float* A_packed,  // packed: k panels of MR elements
    const float* B_packed,  // packed: k panels of NR elements
    float* C,
    size_t ldc) {

    // 8 column accumulators, each split into lo (rows 0-3) and hi (rows 4-7)
    svfloat32_t c0_lo = svdup_f32(0.0f), c0_hi = svdup_f32(0.0f);
    svfloat32_t c1_lo = svdup_f32(0.0f), c1_hi = svdup_f32(0.0f);
    svfloat32_t c2_lo = svdup_f32(0.0f), c2_hi = svdup_f32(0.0f);
    svfloat32_t c3_lo = svdup_f32(0.0f), c3_hi = svdup_f32(0.0f);
    svfloat32_t c4_lo = svdup_f32(0.0f), c4_hi = svdup_f32(0.0f);
    svfloat32_t c5_lo = svdup_f32(0.0f), c5_hi = svdup_f32(0.0f);
    svfloat32_t c6_lo = svdup_f32(0.0f), c6_hi = svdup_f32(0.0f);
    svfloat32_t c7_lo = svdup_f32(0.0f), c7_hi = svdup_f32(0.0f);

    svbool_t pt = svptrue_b32();

    for (size_t p = 0; p < k; ++p) {
        // Load A column panel (8 floats = rows 0-7 at k-index p)
        svfloat32_t a_lo = svld1_f32(pt, A_packed + p * MR);      // rows 0-3
        svfloat32_t a_hi = svld1_f32(pt, A_packed + p * MR + 4);  // rows 4-7

        // Load B row panel (8 floats = cols 0-7 at k-index p)
        // We need individual b[j] values for rank-1 broadcast
        const float* bp = B_packed + p * NR;

        // Rank-1 update: C[:,j] += A[:,p] * B[p,j] for j=0..7
        c0_lo = svmla_n_f32_z(pt, c0_lo, a_lo, bp[0]);
        c0_hi = svmla_n_f32_z(pt, c0_hi, a_hi, bp[0]);
        c1_lo = svmla_n_f32_z(pt, c1_lo, a_lo, bp[1]);
        c1_hi = svmla_n_f32_z(pt, c1_hi, a_hi, bp[1]);
        c2_lo = svmla_n_f32_z(pt, c2_lo, a_lo, bp[2]);
        c2_hi = svmla_n_f32_z(pt, c2_hi, a_hi, bp[2]);
        c3_lo = svmla_n_f32_z(pt, c3_lo, a_lo, bp[3]);
        c3_hi = svmla_n_f32_z(pt, c3_hi, a_hi, bp[3]);
        c4_lo = svmla_n_f32_z(pt, c4_lo, a_lo, bp[4]);
        c4_hi = svmla_n_f32_z(pt, c4_hi, a_hi, bp[4]);
        c5_lo = svmla_n_f32_z(pt, c5_lo, a_lo, bp[5]);
        c5_hi = svmla_n_f32_z(pt, c5_hi, a_hi, bp[5]);
        c6_lo = svmla_n_f32_z(pt, c6_lo, a_lo, bp[6]);
        c6_hi = svmla_n_f32_z(pt, c6_hi, a_hi, bp[6]);
        c7_lo = svmla_n_f32_z(pt, c7_lo, a_lo, bp[7]);
        c7_hi = svmla_n_f32_z(pt, c7_hi, a_hi, bp[7]);
    }

    // Store: load existing C, add accumulators, store back (column-major)
    #define SVE2_STORE_COL(j, lo, hi) \
        svst1_f32(pt, C + (j)*ldc,     svadd_f32_z(pt, svld1_f32(pt, C + (j)*ldc),     lo)); \
        svst1_f32(pt, C + (j)*ldc + 4, svadd_f32_z(pt, svld1_f32(pt, C + (j)*ldc + 4), hi));

    SVE2_STORE_COL(0, c0_lo, c0_hi);
    SVE2_STORE_COL(1, c1_lo, c1_hi);
    SVE2_STORE_COL(2, c2_lo, c2_hi);
    SVE2_STORE_COL(3, c3_lo, c3_hi);
    SVE2_STORE_COL(4, c4_lo, c4_hi);
    SVE2_STORE_COL(5, c5_lo, c5_hi);
    SVE2_STORE_COL(6, c6_lo, c6_hi);
    SVE2_STORE_COL(7, c7_lo, c7_hi);

    #undef SVE2_STORE_COL
}

#endif // OPTMATH_USE_SVE2

void sve2_gemm_blocked_f32(
    float* C,
    const float* A,
    const float* B,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc) {

#ifdef OPTMATH_USE_SVE2
    // Get runtime cache blocking parameters
    const size_t MC = platform::get_gemm_mc();
    const size_t KC = platform::get_gemm_kc();
    const size_t NC = platform::get_gemm_nc();

    // Initialize C to zero
    for (size_t j = 0; j < N; ++j) {
        uint64_t ii = 0;
        svbool_t pg = svwhilelt_b32(ii, (uint64_t)M);
        svfloat32_t vzero = svdup_f32(0.0f);
        do {
            svst1_f32(pg, C + ii + j * ldc, vzero);
            ii += svcntw();
            pg = svwhilelt_b32(ii, (uint64_t)M);
        } while (svptest_any(svptrue_b32(), pg));
    }

    // Loop over blocks of N (columns of B and C)
    for (size_t jc = 0; jc < N; jc += NC) {
        size_t nc = std::min(NC, N - jc);

        // Loop over blocks of K
        for (size_t pc = 0; pc < K; pc += KC) {
            size_t kc = std::min(KC, K - pc);

            // Pack B panel: B[pc:pc+kc, jc:jc+nc]
            for (size_t jr = 0; jr < nc; jr += NR) {
                size_t nr = std::min(NR, nc - jr);
                pack_B_panel_sve2(
                    packed_B + jr * kc,
                    B + pc + (jc + jr) * ldb,
                    ldb, kc, nr);
            }

            // Loop over blocks of M (rows of A and C)
            for (size_t ic = 0; ic < M; ic += MC) {
                size_t mc = std::min(MC, M - ic);

                // Pack A panel: A[ic:ic+mc, pc:pc+kc]
                for (size_t ir = 0; ir < mc; ir += MR) {
                    size_t mr = std::min(MR, mc - ir);
                    pack_A_panel_sve2(
                        packed_A + ir * kc,
                        A + (ic + ir) + pc * lda,
                        lda, mr, kc);
                }

                // Microkernel loop
                for (size_t jr = 0; jr < nc; jr += NR) {
                    size_t nr = std::min(NR, nc - jr);

                    for (size_t ir = 0; ir < mc; ir += MR) {
                        size_t mr = std::min(MR, mc - ir);

                        if (mr == MR && nr == NR) {
                            // Full microkernel
                            micro_kernel_8x8_sve2(
                                kc,
                                packed_A + ir * kc,
                                packed_B + jr * kc,
                                C + (ic + ir) + (jc + jr) * ldc,
                                ldc);
                        } else {
                            // Edge case: scalar fallback for partial blocks
                            for (size_t j = 0; j < nr; ++j) {
                                for (size_t i = 0; i < mr; ++i) {
                                    float sum = 0.0f;
                                    for (size_t p = 0; p < kc; ++p) {
                                        sum += packed_A[ir * kc + p * MR + i] *
                                               packed_B[jr * kc + p * NR + j];
                                    }
                                    C[(ic + ir + i) + (jc + jr + j) * ldc] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#else
    // Fallback: delegate to NEON blocked GEMM
    neon::neon_gemm_blocked_f32(C, A, B, M, N, K, lda, ldb, ldc);
#endif
}

// =========================================================================
// I8MM GEMM - Int8 Matrix Multiply with SVE2 I8MM instructions
// =========================================================================

void sve2_gemm_i8mm(
    float* C,
    const int8_t* A,
    const int8_t* B,
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t lda, std::size_t ldb, std::size_t ldc,
    float scale_a, float scale_b,
    int32_t zero_a, int32_t zero_b) {

#if defined(OPTMATH_USE_SVE2) && defined(OPTMATH_USE_I8MM)
    // Combined dequantization scale
    const float combined_scale = scale_a * scale_b;

    // Zero out C
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            C[i + j * ldc] = 0.0f;
        }
    }

    // Process in 2x2 output tiles (svmmla_s32 computes 2x2 from 2x8 * 8x2)
    // svmmla_s32 operates on groups of 8 int8 elements
    for (size_t j = 0; j < N; j += 2) {
        size_t nr = std::min((size_t)2, N - j);
        for (size_t i = 0; i < M; i += 2) {
            size_t mr = std::min((size_t)2, M - i);

            // Accumulate int32 results
            int32_t acc[2][2] = {{0, 0}, {0, 0}};

            // Process K dimension in chunks of 8 (I8MM granularity)
            for (size_t p = 0; p < K; p += 8) {
                size_t kk = std::min((size_t)8, K - p);

                // Pack 2 rows of A (8 elements each) into a 16-byte buffer
                int8_t a_pack[16];
                std::memset(a_pack, 0, sizeof(a_pack));
                for (size_t ki = 0; ki < kk; ++ki) {
                    if (i < M)
                        a_pack[ki] = A[i + (p + ki) * lda] - (int8_t)zero_a;
                    if (i + 1 < M)
                        a_pack[8 + ki] = A[(i + 1) + (p + ki) * lda] - (int8_t)zero_a;
                }

                // Pack 2 columns of B (8 elements each) into a 16-byte buffer
                int8_t b_pack[16];
                std::memset(b_pack, 0, sizeof(b_pack));
                for (size_t ki = 0; ki < kk; ++ki) {
                    if (j < N)
                        b_pack[ki] = B[(p + ki) + j * ldb] - (int8_t)zero_b;
                    if (j + 1 < N)
                        b_pack[8 + ki] = B[(p + ki) + (j + 1) * ldb] - (int8_t)zero_b;
                }

                // Use SVE2 I8MM: svmmla_s32 computes 2x2 += 2x8 * 8x2
                svbool_t pg8 = svwhilelt_b8((uint64_t)0, (uint64_t)16);
                svint8_t va = svld1_s8(pg8, a_pack);
                svint8_t vb = svld1_s8(pg8, b_pack);

                // Convert to unsigned for svmmla (it expects unsigned x signed or signed x signed)
                svint32_t vacc = svdup_s32(0);
                vacc = svmmla_s32(vacc, svreinterpret_s8_u8(svreinterpret_u8_s8(va)), vb);

                // Extract 2x2 result (packed as [c00, c01, c10, c11])
                int32_t result_buf[4] = {0, 0, 0, 0};
                svbool_t pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)4);
                svst1_s32(pg32, result_buf, vacc);

                acc[0][0] += result_buf[0];
                acc[0][1] += result_buf[1];
                acc[1][0] += result_buf[2];
                acc[1][1] += result_buf[3];
            }

            // Dequantize and store
            for (size_t jj = 0; jj < nr; ++jj) {
                for (size_t ii = 0; ii < mr; ++ii) {
                    C[(i + ii) + (j + jj) * ldc] = (float)acc[ii][jj] * combined_scale;
                }
            }
        }
    }

#elif defined(OPTMATH_USE_SVE2)
    // SVE2 without I8MM: scalar int8 GEMM with dequantization
    const float combined_scale = scale_a * scale_b;

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            int32_t acc = 0;
            for (size_t p = 0; p < K; ++p) {
                int32_t a_val = (int32_t)A[i + p * lda] - zero_a;
                int32_t b_val = (int32_t)B[p + j * ldb] - zero_b;
                acc += a_val * b_val;
            }
            C[i + j * ldc] = (float)acc * combined_scale;
        }
    }

#else
    // Scalar fallback
    const float combined_scale = scale_a * scale_b;

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            int32_t acc = 0;
            for (size_t p = 0; p < K; ++p) {
                int32_t a_val = (int32_t)A[i + p * lda] - zero_a;
                int32_t b_val = (int32_t)B[p + j * ldb] - zero_b;
                acc += a_val * b_val;
            }
            C[i + j * ldc] = (float)acc * combined_scale;
        }
    }
#endif
}

// =========================================================================
// FIR Filter
// =========================================================================

void sve2_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y) {
#ifdef OPTMATH_USE_SVE2
    // Output size = n_x - n_h + 1 (valid convolution)
    size_t n_y = (n_x >= n_h) ? (n_x - n_h + 1) : 0;

    for (size_t i = 0; i < n_y; ++i) {
        // Compute dot product of x[i..i+n_h-1] and h[0..n_h-1] using SVE2
        y[i] = sve2_dot_f32(x + i, h, n_h);
    }
#else
    neon::neon_fir_f32(x, n_x, h, n_h, y);
#endif
}

// =========================================================================
// Complex Number Operations: defined in sve2_complex.cpp
// Radar DSP Operations: defined in sve2_radar.cpp
// =========================================================================

// Eigen Wrappers for basic vector operations (non-complex, non-radar)
// Complex operations are in sve2_complex.cpp, radar ops in sve2_radar.cpp
// =========================================================================
// Eigen Wrappers
// =========================================================================

float sve2_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (a.size() != b.size()) return 0.0f;
    return sve2_dot_f32(a.data(), b.data(), a.size());
}

Eigen::VectorXf sve2_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    sve2_add_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf sve2_sub(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    sve2_sub_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf sve2_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    sve2_mul_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf sve2_div(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if (a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    sve2_div_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

float sve2_norm(const Eigen::VectorXf& a) {
    return sve2_norm_f32(a.data(), a.size());
}

float sve2_reduce_sum(const Eigen::VectorXf& a) {
    return sve2_reduce_sum_f32(a.data(), a.size());
}

float sve2_reduce_max(const Eigen::VectorXf& a) {
    return sve2_reduce_max_f32(a.data(), a.size());
}

float sve2_reduce_min(const Eigen::VectorXf& a) {
    return sve2_reduce_min_f32(a.data(), a.size());
}

Eigen::VectorXf sve2_fir(const Eigen::VectorXf& x, const Eigen::VectorXf& h) {
    if (x.size() < h.size()) return Eigen::VectorXf();
    long out_size = x.size() - h.size() + 1;
    Eigen::VectorXf y(out_size);
    sve2_fir_f32(x.data(), x.size(), h.data(), h.size(), y.data());
    return y;
}

Eigen::MatrixXf sve2_gemm_blocked(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    if (A.cols() != B.rows()) return Eigen::MatrixXf();
    Eigen::MatrixXf C(A.rows(), B.cols());
    sve2_gemm_blocked_f32(C.data(), A.data(), B.data(),
                          A.rows(), B.cols(), A.cols(),
                          A.outerStride(), B.outerStride(), C.outerStride());
    return C;
}

// Complex Eigen wrappers are in sve2_complex.cpp
// Radar Eigen wrappers (sve2_caf, etc.) are in sve2_radar.cpp

} // namespace sve2
} // namespace optmath
