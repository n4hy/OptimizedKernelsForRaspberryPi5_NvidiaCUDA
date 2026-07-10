/**
 * OptMathKernels NEON Int8 Dot-Product Kernels
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Int8 GEMM using the Armv8.2 dot-product extension (SDOT / "asimddp"), tuned
 * for the Raspberry Pi 5's Cortex-A76.
 *
 * Microkernel:
 *   4x4 register tile with 16 int32x4 accumulators. Each SDOT consumes a
 *   16-int8 chunk of an A row and a 16-int8 chunk of a Bt row and accumulates
 *   4 partial dot products per 32-bit lane. Four A-row loads and four Bt-row
 *   loads feed sixteen SDOTs per K-chunk (high compute-to-load ratio). The
 *   four lane partials are horizontally reduced once at the end of the K loop,
 *   with a scalar tail for the K % 16 remainder. Edge tiles (M % 4, N % 4) use
 *   a scalar fallback.
 *
 * Threading:
 *   The row-tile loop is split across the 4 A76 cores with OpenMP; each thread
 *   writes disjoint rows of C.
 */
#include "optmath/neon_int8.hpp"
#include <vector>

#ifdef OPTMATH_USE_DOTPROD
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

#ifdef OPTMATH_USE_DOTPROD

// Cache-blocking parameters. A Bt column-panel of NC rows x K int8 plus the
// current 4-row A tile must stay L1-resident (64KB on the A76) so Bt is reused
// across the MC-row block instead of being re-streamed from L2/L3 for every
// row tile — that re-streaming is what made the naive kernel collapse at 4
// threads. NC*K + MC*K should sit comfortably inside L1/L2; these are capped
// per K at run time below.
static constexpr std::size_t INT8_MC = 64;
static constexpr std::size_t INT8_NC = 32;

// 4x4 SDOT microkernel: C[i..i+3][j..j+3] = dot over K. 16 int32x4 accumulators.
static inline void micro_4x4_s8(std::int32_t* C, std::size_t ldc,
                                const std::int8_t* a0, const std::int8_t* a1,
                                const std::int8_t* a2, const std::int8_t* a3,
                                const std::int8_t* b0, const std::int8_t* b1,
                                const std::int8_t* b2, const std::int8_t* b3,
                                std::size_t K) {
    int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0),
              acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
    int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0),
              acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
    int32x4_t acc20 = vdupq_n_s32(0), acc21 = vdupq_n_s32(0),
              acc22 = vdupq_n_s32(0), acc23 = vdupq_n_s32(0);
    int32x4_t acc30 = vdupq_n_s32(0), acc31 = vdupq_n_s32(0),
              acc32 = vdupq_n_s32(0), acc33 = vdupq_n_s32(0);
    std::size_t k = 0;
    for (; k + 16 <= K; k += 16) {
        int8x16_t va0 = vld1q_s8(a0 + k), va1 = vld1q_s8(a1 + k);
        int8x16_t va2 = vld1q_s8(a2 + k), va3 = vld1q_s8(a3 + k);
        int8x16_t vb0 = vld1q_s8(b0 + k), vb1 = vld1q_s8(b1 + k);
        int8x16_t vb2 = vld1q_s8(b2 + k), vb3 = vld1q_s8(b3 + k);
        acc00 = vdotq_s32(acc00, va0, vb0); acc01 = vdotq_s32(acc01, va0, vb1);
        acc02 = vdotq_s32(acc02, va0, vb2); acc03 = vdotq_s32(acc03, va0, vb3);
        acc10 = vdotq_s32(acc10, va1, vb0); acc11 = vdotq_s32(acc11, va1, vb1);
        acc12 = vdotq_s32(acc12, va1, vb2); acc13 = vdotq_s32(acc13, va1, vb3);
        acc20 = vdotq_s32(acc20, va2, vb0); acc21 = vdotq_s32(acc21, va2, vb1);
        acc22 = vdotq_s32(acc22, va2, vb2); acc23 = vdotq_s32(acc23, va2, vb3);
        acc30 = vdotq_s32(acc30, va3, vb0); acc31 = vdotq_s32(acc31, va3, vb1);
        acc32 = vdotq_s32(acc32, va3, vb2); acc33 = vdotq_s32(acc33, va3, vb3);
    }
    std::int32_t r[4][4] = {
        { vaddvq_s32(acc00), vaddvq_s32(acc01), vaddvq_s32(acc02), vaddvq_s32(acc03) },
        { vaddvq_s32(acc10), vaddvq_s32(acc11), vaddvq_s32(acc12), vaddvq_s32(acc13) },
        { vaddvq_s32(acc20), vaddvq_s32(acc21), vaddvq_s32(acc22), vaddvq_s32(acc23) },
        { vaddvq_s32(acc30), vaddvq_s32(acc31), vaddvq_s32(acc32), vaddvq_s32(acc33) },
    };
    const std::int8_t* ar[4] = { a0, a1, a2, a3 };
    const std::int8_t* br[4] = { b0, b1, b2, b3 };
    for (std::size_t kk = k; kk < K; ++kk)
        for (int rr = 0; rr < 4; ++rr)
            for (int cc = 0; cc < 4; ++cc)
                r[rr][cc] += (std::int32_t)ar[rr][kk] * (std::int32_t)br[cc][kk];
    for (int rr = 0; rr < 4; ++rr)
        for (int cc = 0; cc < 4; ++cc)
            C[rr * ldc + cc] = r[rr][cc];
}

// Scalar fallback for an arbitrary rectangular tile.
static inline void tile_scalar_s8(std::int32_t* C, std::size_t ldc,
                                  const std::int8_t* A, const std::int8_t* Bt,
                                  std::size_t i0, std::size_t i1,
                                  std::size_t j0, std::size_t j1, std::size_t K) {
    for (std::size_t ii = i0; ii < i1; ++ii)
        for (std::size_t jj = j0; jj < j1; ++jj) {
            std::int32_t s = 0;
            const std::int8_t* a = A + ii * K;
            const std::int8_t* b = Bt + jj * K;
            for (std::size_t k = 0; k < K; ++k)
                s += (std::int32_t)a[k] * (std::int32_t)b[k];
            C[ii * ldc + jj] = s;
        }
}

#endif // OPTMATH_USE_DOTPROD

void neon_gemm_s8s8s32(std::int32_t* C, std::size_t ldc,
                       const std::int8_t* A,
                       const std::int8_t* Bt,
                       std::size_t M, std::size_t N, std::size_t K) {
#ifdef OPTMATH_USE_DOTPROD
    // Cache-blocked, threaded over the MC row-blocks (disjoint C rows -> no
    // false sharing on the row-major C). Within a row-block, sweeping NC-wide
    // Bt panels keeps each panel L1-resident across the block's rows.
    const std::size_t MB = (M + INT8_MC - 1) / INT8_MC;
    #pragma omp parallel for schedule(dynamic) if(M >= 256)
    for (std::size_t ib = 0; ib < MB; ++ib) {
        const std::size_t ic = ib * INT8_MC;
        const std::size_t i_end = (ic + INT8_MC < M) ? ic + INT8_MC : M;
        for (std::size_t jc = 0; jc < N; jc += INT8_NC) {
            const std::size_t j_end = (jc + INT8_NC < N) ? jc + INT8_NC : N;
            std::size_t i = ic;
            for (; i + 4 <= i_end; i += 4) {
                const std::int8_t* a0 = A + (i + 0) * K;
                const std::int8_t* a1 = A + (i + 1) * K;
                const std::int8_t* a2 = A + (i + 2) * K;
                const std::int8_t* a3 = A + (i + 3) * K;
                std::size_t j = jc;
                for (; j + 4 <= j_end; j += 4) {
                    micro_4x4_s8(C + i * ldc + j, ldc, a0, a1, a2, a3,
                                 Bt + (j + 0) * K, Bt + (j + 1) * K,
                                 Bt + (j + 2) * K, Bt + (j + 3) * K, K);
                }
                if (j < j_end)  // column remainder for this 4-row strip
                    tile_scalar_s8(C, ldc, A, Bt, i, i + 4, j, j_end, K);
            }
            if (i < i_end)      // row remainder for this column block
                tile_scalar_s8(C, ldc, A, Bt, i, i_end, jc, j_end, K);
        }
    }
#else
    // Portable scalar fallback (no dot-product extension).
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            std::int32_t s = 0;
            const std::int8_t* a = A + i * K;
            const std::int8_t* b = Bt + j * K;
            for (std::size_t k = 0; k < K; ++k)
                s += (std::int32_t)a[k] * (std::int32_t)b[k];
            C[i * ldc + j] = s;
        }
    }
#endif
}

void neon_conv2d_s8s8s32(std::int32_t* out, const std::int8_t* in,
                         std::size_t in_rows, std::size_t in_cols,
                         const std::int8_t* kernel,
                         std::size_t k_rows, std::size_t k_cols) {
    if (in_rows < k_rows || in_cols < k_cols) return;
    const std::size_t out_rows = in_rows - k_rows + 1;
    const std::size_t out_cols = in_cols - k_cols + 1;

#ifdef OPTMATH_USE_DOTPROD
    #pragma omp parallel for schedule(static) if(out_rows >= 64)
    for (std::size_t r = 0; r < out_rows; ++r) {
        std::size_t c = 0;
        // 8 output columns at a time. Two int32x4 accumulators (low/high halves).
        for (; c + 8 <= out_cols; c += 8) {
            int32x4_t acc_lo = vdupq_n_s32(0), acc_hi = vdupq_n_s32(0);
            for (std::size_t kr = 0; kr < k_rows; ++kr) {
                const std::int8_t* in_row = in + (r + kr) * in_cols + c;
                const std::int8_t* k_row = kernel + kr * k_cols;
                for (std::size_t kc = 0; kc < k_cols; ++kc) {
                    int8x8_t vin = vld1_s8(in_row + kc);        // 8 inputs
                    int8x8_t vk = vdup_n_s8(k_row[kc]);         // broadcast tap
                    int16x8_t prod = vmull_s8(vin, vk);         // int16 products
                    acc_lo = vaddw_s16(acc_lo, vget_low_s16(prod));
                    acc_hi = vaddw_s16(acc_hi, vget_high_s16(prod));
                }
            }
            vst1q_s32(out + r * out_cols + c,     acc_lo);
            vst1q_s32(out + r * out_cols + c + 4, acc_hi);
        }
        // Scalar tail
        for (; c < out_cols; ++c) {
            std::int32_t s = 0;
            for (std::size_t kr = 0; kr < k_rows; ++kr)
                for (std::size_t kc = 0; kc < k_cols; ++kc)
                    s += (std::int32_t)in[(r + kr) * in_cols + (c + kc)] *
                         (std::int32_t)kernel[kr * k_cols + kc];
            out[r * out_cols + c] = s;
        }
    }
#else
    for (std::size_t r = 0; r < out_rows; ++r)
        for (std::size_t c = 0; c < out_cols; ++c) {
            std::int32_t s = 0;
            for (std::size_t kr = 0; kr < k_rows; ++kr)
                for (std::size_t kc = 0; kc < k_cols; ++kc)
                    s += (std::int32_t)in[(r + kr) * in_cols + (c + kc)] *
                         (std::int32_t)kernel[kr * k_cols + kc];
            out[r * out_cols + c] = s;
        }
#endif
}

Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>
neon_gemm_int8(const Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic>& A,
               const Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic>& B) {
    using I32Mat = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>;
    if (A.cols() != B.rows()) return I32Mat();

    const std::size_t M = static_cast<std::size_t>(A.rows());
    const std::size_t K = static_cast<std::size_t>(A.cols());
    const std::size_t N = static_cast<std::size_t>(B.cols());

    // Pack A into M x K row-major and B into N x K row-major (transposed), both
    // K-contiguous as the core kernel requires.
    std::vector<std::int8_t> Arm(M * K), Bt(N * K);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            Arm[i * K + k] = A(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k));
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t k = 0; k < K; ++k)
            Bt[j * K + k] = B(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(j));

    // Result stored row-major in a temp, then copied into the (column-major) Eigen matrix.
    std::vector<std::int32_t> Crm(M * N);
    neon_gemm_s8s8s32(Crm.data(), N, Arm.data(), Bt.data(), M, N, K);

    I32Mat C(static_cast<Eigen::Index>(M), static_cast<Eigen::Index>(N));
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            C(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = Crm[i * N + j];
    return C;
}

} // namespace neon
} // namespace optmath
