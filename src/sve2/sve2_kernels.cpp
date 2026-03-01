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

bool is_available() {
#ifdef OPTMATH_USE_SVE2
    return true;
#else
    return false;
#endif
}

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
    // cos(x) = sin(x + pi/2)
    const float half_pi = 1.57079632679489661923f;
    svfloat32_t vhalf_pi = svdup_f32(half_pi);

    // Create temporary buffer with x + pi/2
    std::vector<float> temp(n);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svld1_f32(pg, in + i);
        svst1_f32(pg, temp.data() + i, svadd_f32_z(pg, x, vhalf_pi));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    sve2_fast_sin_f32(out, temp.data(), n);
#else
    neon::neon_fast_cos_f32(out, in, n);
#endif
}

void sve2_fast_sigmoid_f32(float* out, const float* in, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // Fast sigmoid: 1 / (1 + exp(-x))
    const float clamp_max = 20.0f;
    const float clamp_min = -20.0f;
    svfloat32_t vclamp_max = svdup_f32(clamp_max);
    svfloat32_t vclamp_min = svdup_f32(clamp_min);

    // Clamp and negate input for exp(-x)
    std::vector<float> neg_x(n);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svld1_f32(pg, in + i);
        // Clamp to [-20, 20]
        x = svmin_f32_z(pg, svmax_f32_z(pg, x, vclamp_min), vclamp_max);
        svst1_f32(pg, neg_x.data() + i, svneg_f32_z(pg, x));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    // Compute exp(-x)
    std::vector<float> exp_neg_x(n);
    sve2_fast_exp_f32(exp_neg_x.data(), neg_x.data(), n);

    // Compute 1 / (1 + exp(-x))
    svfloat32_t vone = svdup_f32(1.0f);
    i = 0;
    pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t e = svld1_f32(pg, exp_neg_x.data() + i);
        svfloat32_t denom = svadd_f32_z(pg, vone, e);
        svfloat32_t result = svdiv_f32_z(pg, vone, denom);
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
    // tanh(x) = 2 * sigmoid(2x) - 1
    svfloat32_t vtwo = svdup_f32(2.0f);

    // Compute 2x
    std::vector<float> two_x(n);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t x = svld1_f32(pg, in + i);
        svst1_f32(pg, two_x.data() + i, svmul_f32_z(pg, vtwo, x));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    // Compute sigmoid(2x)
    std::vector<float> sig(n);
    sve2_fast_sigmoid_f32(sig.data(), two_x.data(), n);

    // Compute 2*sigmoid(2x) - 1
    svfloat32_t vone = svdup_f32(1.0f);
    svfloat32_t vneg_one = svdup_f32(-1.0f);
    i = 0;
    pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t s = svld1_f32(pg, sig.data() + i);
        // result = 2*s - 1 = fma(2, s, -1)
        svfloat32_t result = svmad_f32_z(pg, vtwo, s, vneg_one);
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

// 8x8 scalar microkernel for SVE2 GEMM
// Accumulates C[0:mr, 0:nr] += A_packed * B_packed over k iterations
static void micro_kernel_8x8_sve2(
    size_t k,
    const float* A_packed,  // packed: k panels of MR elements
    const float* B_packed,  // packed: k panels of NR elements
    float* C,
    size_t ldc) {

    // Scalar accumulation (simple, correct, and SVE2 predication handles edges)
    float acc[MR][NR];
    std::memset(acc, 0, sizeof(acc));

    for (size_t p = 0; p < k; ++p) {
        for (size_t ii = 0; ii < MR; ++ii) {
            float a_val = A_packed[p * MR + ii];
            for (size_t jj = 0; jj < NR; ++jj) {
                acc[ii][jj] += a_val * B_packed[p * NR + jj];
            }
        }
    }

    // Store results back to column-major C
    for (size_t jj = 0; jj < NR; ++jj) {
        for (size_t ii = 0; ii < MR; ++ii) {
            C[ii + jj * ldc] += acc[ii][jj];
        }
    }
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
// Complex Number Operations (Separate real/imaginary format)
// =========================================================================

void sve2_complex_mul_f32(float* out_re, float* out_im,
                           const float* a_re, const float* a_im,
                           const float* b_re, const float* b_im,
                           std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // (a_re + j*a_im) * (b_re + j*b_im)
    // = (a_re*b_re - a_im*b_im) + j*(a_re*b_im + a_im*b_re)
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        // out_re = a_re*b_re - a_im*b_im
        svfloat32_t re = svmul_f32_z(pg, ar, br);
        re = svmls_f32_z(pg, re, ai, bi);

        // out_im = a_re*b_im + a_im*b_re
        svfloat32_t im = svmul_f32_z(pg, ar, bi);
        im = svmla_f32_z(pg, im, ai, br);

        svst1_f32(pg, out_re + i, re);
        svst1_f32(pg, out_im + i, im);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_complex_mul_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

void sve2_complex_conj_mul_f32(float* out_re, float* out_im,
                                const float* a_re, const float* a_im,
                                const float* b_re, const float* b_im,
                                std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // (a_re + j*a_im) * conj(b_re + j*b_im)
    // = (a_re*b_re + a_im*b_im) + j*(a_im*b_re - a_re*b_im)
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        // out_re = a_re*b_re + a_im*b_im
        svfloat32_t re = svmul_f32_z(pg, ar, br);
        re = svmla_f32_z(pg, re, ai, bi);

        // out_im = a_im*b_re - a_re*b_im
        svfloat32_t im = svmul_f32_z(pg, ai, br);
        im = svmls_f32_z(pg, im, ar, bi);

        svst1_f32(pg, out_re + i, re);
        svst1_f32(pg, out_im + i, im);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_complex_conj_mul_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

// =========================================================================
// Complex Operations with FCMA (Interleaved format)
// =========================================================================

void sve2_complex_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // Interleaved: [re0, im0, re1, im1, ...], array size = 2*n
    // Use FCMLA instructions for 2-instruction complex multiply
    uint64_t float_count = 2 * n;
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)float_count);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);

        // FCMLA: complex multiply accumulate
        // First rotation (0 degrees): re(a)*re(b), re(a)*im(b)
        svfloat32_t result = svdup_f32(0.0f);
        result = svcmla_f32_z(pg, result, va, vb, 0);
        // Second rotation (90 degrees): -im(a)*im(b), im(a)*re(b)
        result = svcmla_f32_z(pg, result, va, vb, 90);

        svst1_f32(pg, out + i, result);
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)float_count);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_complex_mul_interleaved_f32(out, a, b, n);
#endif
}

void sve2_complex_conj_mul_interleaved_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // a * conj(b) using FCMLA with 0 and 270 degree rotations
    uint64_t float_count = 2 * n;
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)float_count);

    do {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);

        svfloat32_t result = svdup_f32(0.0f);
        result = svcmla_f32_z(pg, result, va, vb, 0);
        result = svcmla_f32_z(pg, result, va, vb, 270);

        svst1_f32(pg, out + i, result);
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)float_count);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_complex_conj_mul_interleaved_f32(out, a, b, n);
#endif
}

void sve2_complex_dot_f32(float* out_re, float* out_im,
                           const float* a_re, const float* a_im,
                           const float* b_re, const float* b_im,
                           std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // Complex dot product: sum(a * conj(b))
    svfloat32_t sum_re = svdup_f32(0.0f);
    svfloat32_t sum_im = svdup_f32(0.0f);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        // conj_mul: re = a_re*b_re + a_im*b_im, im = a_im*b_re - a_re*b_im
        svfloat32_t re = svmul_f32_z(pg, ar, br);
        re = svmla_f32_z(pg, re, ai, bi);
        svfloat32_t im = svmul_f32_z(pg, ai, br);
        im = svmls_f32_z(pg, im, ar, bi);

        sum_re = svadd_f32_m(svptrue_b32(), sum_re, re);
        sum_im = svadd_f32_m(svptrue_b32(), sum_im, im);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));

    *out_re = svaddv_f32(svptrue_b32(), sum_re);
    *out_im = svaddv_f32(svptrue_b32(), sum_im);
#else
    neon::neon_complex_dot_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

void sve2_complex_magnitude_f32(float* out, const float* re, const float* im, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t vr = svld1_f32(pg, re + i);
        svfloat32_t vi = svld1_f32(pg, im + i);

        // magnitude = sqrt(re^2 + im^2)
        svfloat32_t mag_sq = svmul_f32_z(pg, vr, vr);
        mag_sq = svmla_f32_z(pg, mag_sq, vi, vi);
        svfloat32_t mag = svsqrt_f32_z(pg, mag_sq);

        svst1_f32(pg, out + i, mag);
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_complex_magnitude_f32(out, re, im, n);
#endif
}

void sve2_complex_phase_f32(float* out, const float* re, const float* im, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // atan2 has no SVE2 intrinsic, use scalar
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::atan2(im[i], re[i]);
    }
#else
    neon::neon_complex_phase_f32(out, re, im, n);
#endif
}

void sve2_complex_add_f32(float* out_re, float* out_im,
                           const float* a_re, const float* a_im,
                           const float* b_re, const float* b_im,
                           std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t ar = svld1_f32(pg, a_re + i);
        svfloat32_t ai = svld1_f32(pg, a_im + i);
        svfloat32_t br = svld1_f32(pg, b_re + i);
        svfloat32_t bi = svld1_f32(pg, b_im + i);

        svst1_f32(pg, out_re + i, svadd_f32_z(pg, ar, br));
        svst1_f32(pg, out_im + i, svadd_f32_z(pg, ai, bi));

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_complex_add_f32(out_re, out_im, a_re, a_im, b_re, b_im, n);
#endif
}

void sve2_complex_scale_f32(float* out_re, float* out_im,
                             const float* in_re, const float* in_im,
                             float scale_re, float scale_im,
                             std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // (in_re + j*in_im) * (scale_re + j*scale_im)
    svfloat32_t vsr = svdup_f32(scale_re);
    svfloat32_t vsi = svdup_f32(scale_im);

    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t ir = svld1_f32(pg, in_re + i);
        svfloat32_t ii = svld1_f32(pg, in_im + i);

        // out_re = in_re*scale_re - in_im*scale_im
        svfloat32_t re = svmul_f32_z(pg, ir, vsr);
        re = svmls_f32_z(pg, re, ii, vsi);

        // out_im = in_re*scale_im + in_im*scale_re
        svfloat32_t im = svmul_f32_z(pg, ir, vsi);
        im = svmla_f32_z(pg, im, ii, vsr);

        svst1_f32(pg, out_re + i, re);
        svst1_f32(pg, out_im + i, im);

        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::neon_complex_scale_f32(out_re, out_im, in_re, in_im, scale_re, scale_im, n);
#endif
}

void sve2_complex_exp_f32(float* out_re, float* out_im, const float* phase, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    // exp(j*phase) = cos(phase) + j*sin(phase)
    sve2_fast_cos_f32(out_re, phase, n);
    sve2_fast_sin_f32(out_im, phase, n);
#else
    neon::neon_complex_exp_f32(out_re, out_im, phase, n);
#endif
}

// =========================================================================
// Radar DSP Operations
// =========================================================================

void sve2_caf_f32(float* out_mag,
                   const float* ref_re, const float* ref_im,
                   const float* surv_re, const float* surv_im,
                   std::size_t n_samples,
                   std::size_t n_doppler_bins,
                   float doppler_start, float doppler_step,
                   float sample_rate,
                   std::size_t n_range_bins) {
#ifdef OPTMATH_USE_SVE2
    const float two_pi = 6.28318530717958647693f;

    // For each Doppler bin
    for (size_t d = 0; d < n_doppler_bins; ++d) {
        float freq = doppler_start + (float)d * doppler_step;
        float phase_rate = two_pi * freq / sample_rate;

        // For each range (delay) bin
        for (size_t r = 0; r < n_range_bins; ++r) {
            // Cross-correlate ref with Doppler-shifted surveillance at delay r
            size_t max_n = (n_samples > r) ? (n_samples - r) : 0;
            if (max_n == 0) {
                out_mag[d * n_range_bins + r] = 0.0f;
                continue;
            }

            // Accumulate complex correlation using FCMLA pattern
            svfloat32_t acc_re = svdup_f32(0.0f);
            svfloat32_t acc_im = svdup_f32(0.0f);

            uint64_t i = 0;
            svbool_t pg = svwhilelt_b32(i, (uint64_t)max_n);

            do {
                // Load reference signal
                svfloat32_t rr = svld1_f32(pg, ref_re + i);
                svfloat32_t ri = svld1_f32(pg, ref_im + i);

                // Load surveillance signal at delay r
                svfloat32_t sr = svld1_f32(pg, surv_re + i + r);
                svfloat32_t si = svld1_f32(pg, surv_im + i + r);

                // Generate Doppler steering vector: exp(-j * phase_rate * i)
                // For each element, compute cos/sin of -(phase_rate * (i + lane_index))
                // Since SVE2 does not have a vector index intrinsic, use scalar computation
                uint64_t vl = svcntw();
                float cos_buf[64], sin_buf[64]; // max VL/32 = 64 for 2048-bit SVE
                uint64_t end = i + vl;
                if (end > max_n) end = max_n;
                for (uint64_t lane = i; lane < end; ++lane) {
                    float phase = -phase_rate * (float)lane;
                    cos_buf[lane - i] = std::cos(phase);
                    sin_buf[lane - i] = std::sin(phase);
                }

                svfloat32_t dop_re = svld1_f32(pg, cos_buf);
                svfloat32_t dop_im = svld1_f32(pg, sin_buf);

                // Apply Doppler shift to surveillance: surv * exp(-j*phase_rate*i)
                // shifted_re = sr*dop_re - si*dop_im
                // shifted_im = sr*dop_im + si*dop_re
                svfloat32_t shifted_re = svmul_f32_z(pg, sr, dop_re);
                shifted_re = svmls_f32_z(pg, shifted_re, si, dop_im);
                svfloat32_t shifted_im = svmul_f32_z(pg, sr, dop_im);
                shifted_im = svmla_f32_z(pg, shifted_im, si, dop_re);

                // Conjugate multiply with reference: ref * conj(shifted)
                // re = rr*shifted_re + ri*shifted_im
                // im = ri*shifted_re - rr*shifted_im
                svfloat32_t prod_re = svmul_f32_z(pg, rr, shifted_re);
                prod_re = svmla_f32_z(pg, prod_re, ri, shifted_im);
                svfloat32_t prod_im = svmul_f32_z(pg, ri, shifted_re);
                prod_im = svmls_f32_z(pg, prod_im, rr, shifted_im);

                acc_re = svadd_f32_m(svptrue_b32(), acc_re, prod_re);
                acc_im = svadd_f32_m(svptrue_b32(), acc_im, prod_im);

                i += svcntw();
                pg = svwhilelt_b32(i, (uint64_t)max_n);
            } while (svptest_any(svptrue_b32(), pg));

            float total_re = svaddv_f32(svptrue_b32(), acc_re);
            float total_im = svaddv_f32(svptrue_b32(), acc_im);

            out_mag[d * n_range_bins + r] = std::sqrt(total_re * total_re + total_im * total_im);
        }
    }
#else
    // Scalar fallback
    const float two_pi = 6.28318530717958647693f;
    for (size_t d = 0; d < n_doppler_bins; ++d) {
        float freq = doppler_start + (float)d * doppler_step;
        float phase_rate = two_pi * freq / sample_rate;
        for (size_t r = 0; r < n_range_bins; ++r) {
            size_t max_n = (n_samples > r) ? (n_samples - r) : 0;
            float acc_re = 0.0f, acc_im = 0.0f;
            for (size_t i = 0; i < max_n; ++i) {
                float phase = -phase_rate * (float)i;
                float dop_re = std::cos(phase);
                float dop_im = std::sin(phase);
                float sr = surv_re[i + r] * dop_re - surv_im[i + r] * dop_im;
                float si = surv_re[i + r] * dop_im + surv_im[i + r] * dop_re;
                acc_re += ref_re[i] * sr + ref_im[i] * si;
                acc_im += ref_im[i] * sr - ref_re[i] * si;
            }
            out_mag[d * n_range_bins + r] = std::sqrt(acc_re * acc_re + acc_im * acc_im);
        }
    }
#endif
}

void sve2_xcorr_f32(float* out, const float* x, std::size_t nx,
                     const float* y, std::size_t ny) {
#ifdef OPTMATH_USE_SVE2
    // Cross-correlation: out[k] = sum(x[i] * y[i+k]) for valid range
    size_t out_len = (nx >= ny) ? (nx - ny + 1) : 0;
    for (size_t k = 0; k < out_len; ++k) {
        out[k] = sve2_dot_f32(x + k, y, ny);
    }
#else
    size_t out_len = (nx >= ny) ? (nx - ny + 1) : 0;
    for (size_t k = 0; k < out_len; ++k) {
        out[k] = neon::neon_dot_f32(x + k, y, ny);
    }
#endif
}

void sve2_xcorr_complex_f32(float* out_re, float* out_im,
                             const float* x_re, const float* x_im, std::size_t nx,
                             const float* y_re, const float* y_im, std::size_t ny) {
#ifdef OPTMATH_USE_SVE2
    size_t out_len = (nx >= ny) ? (nx - ny + 1) : 0;
    for (size_t k = 0; k < out_len; ++k) {
        sve2_complex_dot_f32(out_re + k, out_im + k,
                             x_re + k, x_im + k,
                             y_re, y_im, ny);
    }
#else
    size_t out_len = (nx >= ny) ? (nx - ny + 1) : 0;
    for (size_t k = 0; k < out_len; ++k) {
        neon::neon_complex_dot_f32(out_re + k, out_im + k,
                                   x_re + k, x_im + k,
                                   y_re, y_im, ny);
    }
#endif
}

void sve2_beamform_phase_f32(float* output_re, float* output_im,
                              const float* inputs_re, const float* inputs_im,
                              const float* phases,
                              const float* weights,
                              std::size_t n_channels,
                              std::size_t n_samples) {
#ifdef OPTMATH_USE_SVE2
    // Initialize output to zero
    std::memset(output_re, 0, n_samples * sizeof(float));
    std::memset(output_im, 0, n_samples * sizeof(float));

    for (size_t ch = 0; ch < n_channels; ++ch) {
        float w = weights[ch];
        float phase = phases[ch];
        float steer_re = std::cos(phase) * w;
        float steer_im = std::sin(phase) * w;

        svfloat32_t vsr = svdup_f32(steer_re);
        svfloat32_t vsi = svdup_f32(steer_im);

        const float* ch_re = inputs_re + ch * n_samples;
        const float* ch_im = inputs_im + ch * n_samples;

        uint64_t i = 0;
        svbool_t pg = svwhilelt_b32(i, (uint64_t)n_samples);

        do {
            svfloat32_t ir = svld1_f32(pg, ch_re + i);
            svfloat32_t ii = svld1_f32(pg, ch_im + i);

            // Rotate and weight: (ir + j*ii) * (steer_re + j*steer_im)
            svfloat32_t re = svmul_f32_z(pg, ir, vsr);
            re = svmls_f32_z(pg, re, ii, vsi);
            svfloat32_t im = svmul_f32_z(pg, ir, vsi);
            im = svmla_f32_z(pg, im, ii, vsr);

            // Accumulate into output
            svfloat32_t out_r = svld1_f32(pg, output_re + i);
            svfloat32_t out_i = svld1_f32(pg, output_im + i);
            svst1_f32(pg, output_re + i, svadd_f32_z(pg, out_r, re));
            svst1_f32(pg, output_im + i, svadd_f32_z(pg, out_i, im));

            i += svcntw();
            pg = svwhilelt_b32(i, (uint64_t)n_samples);
        } while (svptest_any(svptrue_b32(), pg));
    }
#else
    // Scalar fallback
    std::memset(output_re, 0, n_samples * sizeof(float));
    std::memset(output_im, 0, n_samples * sizeof(float));

    for (size_t ch = 0; ch < n_channels; ++ch) {
        float w = weights[ch];
        float steer_re = std::cos(phases[ch]) * w;
        float steer_im = std::sin(phases[ch]) * w;
        const float* ch_re = inputs_re + ch * n_samples;
        const float* ch_im = inputs_im + ch * n_samples;

        for (size_t i = 0; i < n_samples; ++i) {
            output_re[i] += ch_re[i] * steer_re - ch_im[i] * steer_im;
            output_im[i] += ch_re[i] * steer_im + ch_im[i] * steer_re;
        }
    }
#endif
}

void sve2_apply_window_f32(float* data, const float* window, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t vd = svld1_f32(pg, data + i);
        svfloat32_t vw = svld1_f32(pg, window + i);
        svst1_f32(pg, data + i, svmul_f32_z(pg, vd, vw));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    for (size_t i = 0; i < n; ++i) {
        data[i] *= window[i];
    }
#endif
}

void sve2_apply_window_complex_f32(float* data_re, float* data_im,
                                    const float* window, std::size_t n) {
#ifdef OPTMATH_USE_SVE2
    uint64_t i = 0;
    svbool_t pg = svwhilelt_b32(i, (uint64_t)n);

    do {
        svfloat32_t vw = svld1_f32(pg, window + i);
        svfloat32_t vr = svld1_f32(pg, data_re + i);
        svfloat32_t vi = svld1_f32(pg, data_im + i);
        svst1_f32(pg, data_re + i, svmul_f32_z(pg, vr, vw));
        svst1_f32(pg, data_im + i, svmul_f32_z(pg, vi, vw));
        i += svcntw();
        pg = svwhilelt_b32(i, (uint64_t)n);
    } while (svptest_any(svptrue_b32(), pg));
#else
    for (size_t i = 0; i < n; ++i) {
        data_re[i] *= window[i];
        data_im[i] *= window[i];
    }
#endif
}

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

Eigen::VectorXcf sve2_complex_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return Eigen::VectorXcf();
    size_t n = a.size();

    // Extract separate real/imaginary arrays from interleaved Eigen complex
    std::vector<float> a_re(n), a_im(n), b_re(n), b_im(n);
    std::vector<float> out_re(n), out_im(n);

    for (size_t i = 0; i < n; ++i) {
        a_re[i] = a[i].real();
        a_im[i] = a[i].imag();
        b_re[i] = b[i].real();
        b_im[i] = b[i].imag();
    }

    sve2_complex_mul_f32(out_re.data(), out_im.data(),
                         a_re.data(), a_im.data(),
                         b_re.data(), b_im.data(), n);

    Eigen::VectorXcf result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = std::complex<float>(out_re[i], out_im[i]);
    }
    return result;
}

Eigen::VectorXcf sve2_complex_conj_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return Eigen::VectorXcf();
    size_t n = a.size();

    std::vector<float> a_re(n), a_im(n), b_re(n), b_im(n);
    std::vector<float> out_re(n), out_im(n);

    for (size_t i = 0; i < n; ++i) {
        a_re[i] = a[i].real();
        a_im[i] = a[i].imag();
        b_re[i] = b[i].real();
        b_im[i] = b[i].imag();
    }

    sve2_complex_conj_mul_f32(out_re.data(), out_im.data(),
                              a_re.data(), a_im.data(),
                              b_re.data(), b_im.data(), n);

    Eigen::VectorXcf result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = std::complex<float>(out_re[i], out_im[i]);
    }
    return result;
}

std::complex<float> sve2_complex_dot(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    if (a.size() != b.size()) return std::complex<float>(0.0f, 0.0f);
    size_t n = a.size();

    std::vector<float> a_re(n), a_im(n), b_re(n), b_im(n);
    for (size_t i = 0; i < n; ++i) {
        a_re[i] = a[i].real();
        a_im[i] = a[i].imag();
        b_re[i] = b[i].real();
        b_im[i] = b[i].imag();
    }

    float out_re, out_im;
    sve2_complex_dot_f32(&out_re, &out_im,
                         a_re.data(), a_im.data(),
                         b_re.data(), b_im.data(), n);

    return std::complex<float>(out_re, out_im);
}

Eigen::VectorXf sve2_complex_magnitude(const Eigen::VectorXcf& a) {
    size_t n = a.size();

    std::vector<float> re(n), im(n);
    for (size_t i = 0; i < n; ++i) {
        re[i] = a[i].real();
        im[i] = a[i].imag();
    }

    Eigen::VectorXf result(n);
    sve2_complex_magnitude_f32(result.data(), re.data(), im.data(), n);
    return result;
}

Eigen::VectorXf sve2_complex_phase(const Eigen::VectorXcf& a) {
    size_t n = a.size();

    std::vector<float> re(n), im(n);
    for (size_t i = 0; i < n; ++i) {
        re[i] = a[i].real();
        im[i] = a[i].imag();
    }

    Eigen::VectorXf result(n);
    sve2_complex_phase_f32(result.data(), re.data(), im.data(), n);
    return result;
}

Eigen::MatrixXf sve2_caf(const Eigen::VectorXcf& ref,
                          const Eigen::VectorXcf& surv,
                          std::size_t n_doppler_bins,
                          float doppler_start, float doppler_step,
                          float sample_rate,
                          std::size_t n_range_bins) {
    size_t n_samples = std::min((size_t)ref.size(), (size_t)surv.size());

    // Extract separate real/imaginary from Eigen complex vectors
    std::vector<float> ref_re(n_samples), ref_im(n_samples);
    std::vector<float> surv_re(n_samples), surv_im(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        ref_re[i] = ref[i].real();
        ref_im[i] = ref[i].imag();
        surv_re[i] = surv[i].real();
        surv_im[i] = surv[i].imag();
    }

    Eigen::MatrixXf result(n_doppler_bins, n_range_bins);
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

} // namespace sve2
} // namespace optmath
