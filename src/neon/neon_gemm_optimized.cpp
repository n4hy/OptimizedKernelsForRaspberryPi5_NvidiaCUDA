/**
 * OptMathKernels NEON Optimized GEMM
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * High-performance cache-blocked GEMM implementation using NEON intrinsics,
 * following the Goto BLAS three-level blocking strategy with an 8x8
 * microkernel optimized for Cortex-A76 and Cortex-A720 cache hierarchies.
 *
 * Cache Blocking Infrastructure:
 *   Runtime MC/KC/NC parameters tuned per target: Cortex-A76 (Pi 5: 64KB
 *   L1, 512KB L2, 2MB L3) uses MC=384, KC=512, NC=4096. Cortex-A720
 *   (CIX P1: 12MB L3) uses MC=512, KC=768, NC=8192. Thread-local aligned
 *   buffers for packed panels.
 *
 * 8x8 Microkernel:
 *   micro_kernel_8x8 uses column-oriented accumulators for efficient
 *   column-major store. Rank-1 updates via vmlaq_laneq_f32 (FMA with
 *   scalar lane broadcast). 16 vector stores vs 64 scalar stores.
 *   MR = NR = 8.
 *
 * Data Packing:
 *   pack_A_panel packs MR=8 row strips for contiguous sequential access.
 *   pack_B_panel packs NR=8 column strips. Both handle edge cases with
 *   zero-fill padding.
 *
 * Three-Level Cache-Blocked GEMM:
 *   neon_gemm_blocked_f32 implements Goto-style GEMM: Level 1 (NC x KC)
 *   blocks B into L2, Level 2 (MC x KC) blocks A into L3, Level 3
 *   invokes 8x8 microkernels. Edge panels handled with scalar fallback.
 *
 * Eigen Wrapper:
 *   neon_gemm_blocked with automatic column-major data pointer extraction
 *   from Eigen::MatrixXf.
 */
#include "optmath/neon_kernels.hpp"
#include "optmath/platform.hpp"
#include <cstring>
#include <algorithm>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

// =========================================================================
// Optimized GEMM with Cache Blocking
// =========================================================================
//
// This implementation uses:
// - Cache blocking with MC, KC, NC parameters tuned at runtime
// - Data packing for contiguous memory access
// - 8x8 register-blocked microkernel
//
// Memory hierarchy targeting:
// - Cortex-A76 (Pi 5): L2=512KB, L3=2MB → MC=128, KC=256, NC=512
// - Cortex-A720 (CIX P1): L2=512KB, L3=12MB → MC=256, KC=512, NC=1024

// Runtime-selected cache blocking parameters
static size_t get_mc() { return platform::get_gemm_mc(); }
static size_t get_kc() { return platform::get_gemm_kc(); }
static size_t get_nc() { return platform::get_gemm_nc(); }

// Maximum possible values (for static buffer sizing)
constexpr size_t MC_MAX = 256;
constexpr size_t KC_MAX = 512;
constexpr size_t NC_MAX = 1024;

// Microkernel dimensions
constexpr size_t MR = 8;    // Rows per microkernel
constexpr size_t NR = 8;    // Cols per microkernel

// Aligned buffer for packed matrices (sized to max values)
alignas(64) static thread_local float packed_A[MC_MAX * KC_MAX];
alignas(64) static thread_local float packed_B[KC_MAX * NC_MAX];

#ifdef OPTMATH_USE_NEON

// 8x8 microkernel using NEON intrinsics
// Computes C[0:8, 0:8] += A_packed[0:8, 0:KC] * B_packed[0:KC, 0:8]
// Uses column-oriented accumulators for efficient vector store to column-major C.
static void micro_kernel_8x8(
    size_t k,
    const float* A_packed,  // 8 x k, packed row-major in register panels
    const float* B_packed,  // k x 8, packed column-major in register panels
    float* C,
    size_t ldc) {

    // Column-oriented accumulators: c_colJ_lo = C[0:3, J], c_colJ_hi = C[4:7, J]
    // This allows contiguous vector store to column-major C.
    float32x4_t c0_lo = vdupq_n_f32(0.0f), c0_hi = vdupq_n_f32(0.0f);
    float32x4_t c1_lo = vdupq_n_f32(0.0f), c1_hi = vdupq_n_f32(0.0f);
    float32x4_t c2_lo = vdupq_n_f32(0.0f), c2_hi = vdupq_n_f32(0.0f);
    float32x4_t c3_lo = vdupq_n_f32(0.0f), c3_hi = vdupq_n_f32(0.0f);
    float32x4_t c4_lo = vdupq_n_f32(0.0f), c4_hi = vdupq_n_f32(0.0f);
    float32x4_t c5_lo = vdupq_n_f32(0.0f), c5_hi = vdupq_n_f32(0.0f);
    float32x4_t c6_lo = vdupq_n_f32(0.0f), c6_hi = vdupq_n_f32(0.0f);
    float32x4_t c7_lo = vdupq_n_f32(0.0f), c7_hi = vdupq_n_f32(0.0f);

    // Main loop over k dimension
    for (size_t p = 0; p < k; ++p) {
        // Load A column (8 elements: rows 0-7 of A at k-index p)
        float32x4_t a0 = vld1q_f32(A_packed + p * MR);
        float32x4_t a1 = vld1q_f32(A_packed + p * MR + 4);

        // Load B row (8 elements: cols 0-7 of B at k-index p)
        float32x4_t b0 = vld1q_f32(B_packed + p * NR);
        float32x4_t b1 = vld1q_f32(B_packed + p * NR + 4);

        // Rank-1 update by columns: C[:,j] += A[:,p] * B[p,j]
        // Column 0-3 (elements from b0)
        c0_lo = vmlaq_laneq_f32(c0_lo, a0, b0, 0);
        c0_hi = vmlaq_laneq_f32(c0_hi, a1, b0, 0);
        c1_lo = vmlaq_laneq_f32(c1_lo, a0, b0, 1);
        c1_hi = vmlaq_laneq_f32(c1_hi, a1, b0, 1);
        c2_lo = vmlaq_laneq_f32(c2_lo, a0, b0, 2);
        c2_hi = vmlaq_laneq_f32(c2_hi, a1, b0, 2);
        c3_lo = vmlaq_laneq_f32(c3_lo, a0, b0, 3);
        c3_hi = vmlaq_laneq_f32(c3_hi, a1, b0, 3);
        // Column 4-7 (elements from b1)
        c4_lo = vmlaq_laneq_f32(c4_lo, a0, b1, 0);
        c4_hi = vmlaq_laneq_f32(c4_hi, a1, b1, 0);
        c5_lo = vmlaq_laneq_f32(c5_lo, a0, b1, 1);
        c5_hi = vmlaq_laneq_f32(c5_hi, a1, b1, 1);
        c6_lo = vmlaq_laneq_f32(c6_lo, a0, b1, 2);
        c6_hi = vmlaq_laneq_f32(c6_hi, a1, b1, 2);
        c7_lo = vmlaq_laneq_f32(c7_lo, a0, b1, 3);
        c7_hi = vmlaq_laneq_f32(c7_hi, a1, b1, 3);
    }

    // Store results with vector load+add+store (16 vector ops vs 64 scalar)
    // C is column-major: column j starts at C + j*ldc, contiguous for 8 rows
    #define STORE_COL(j, lo, hi) \
        vst1q_f32(C + (j)*ldc,     vaddq_f32(vld1q_f32(C + (j)*ldc),     lo)); \
        vst1q_f32(C + (j)*ldc + 4, vaddq_f32(vld1q_f32(C + (j)*ldc + 4), hi));

    STORE_COL(0, c0_lo, c0_hi);
    STORE_COL(1, c1_lo, c1_hi);
    STORE_COL(2, c2_lo, c2_hi);
    STORE_COL(3, c3_lo, c3_hi);
    STORE_COL(4, c4_lo, c4_hi);
    STORE_COL(5, c5_lo, c5_hi);
    STORE_COL(6, c6_lo, c6_hi);
    STORE_COL(7, c7_lo, c7_hi);

    #undef STORE_COL
}

// Pack a panel of A for the microkernel
// A is M x K (column-major), pack as column-strips of MR rows
static void pack_A_panel(
    float* packed,
    const float* A,
    size_t lda,
    size_t m,
    size_t k) {

    for (size_t p = 0; p < k; ++p) {
        for (size_t i = 0; i < m; ++i) {
            packed[p * MR + i] = A[i + p * lda];
        }
        // Zero-pad if m < MR
        for (size_t i = m; i < MR; ++i) {
            packed[p * MR + i] = 0.0f;
        }
    }
}

// Pack a panel of B for the microkernel
// B is K x N (column-major), pack as row-strips of NR columns
static void pack_B_panel(
    float* packed,
    const float* B,
    size_t ldb,
    size_t k,
    size_t n) {

    for (size_t p = 0; p < k; ++p) {
        for (size_t j = 0; j < n; ++j) {
            packed[p * NR + j] = B[p + j * ldb];
        }
        // Zero-pad if n < NR
        for (size_t j = n; j < NR; ++j) {
            packed[p * NR + j] = 0.0f;
        }
    }
}

#endif // OPTMATH_USE_NEON

void neon_gemm_blocked_f32(
    float* C,
    const float* A,
    const float* B,
    size_t M, size_t N, size_t K,
    size_t lda, size_t ldb, size_t ldc) {

#ifdef OPTMATH_USE_NEON
    // Runtime-selected cache blocking parameters
    const size_t MC = std::min(get_mc(), MC_MAX);
    const size_t KC = std::min(get_kc(), KC_MAX);
    const size_t NC = std::min(get_nc(), NC_MAX);

    // Initialize C to zero
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            C[i + j * ldc] = 0.0f;
        }
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
                pack_B_panel(
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
                    pack_A_panel(
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
                            micro_kernel_8x8(
                                kc,
                                packed_A + ir * kc,
                                packed_B + jr * kc,
                                C + (ic + ir) + (jc + jr) * ldc,
                                ldc);
                        } else {
                            // Edge case: scalar fallback
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
    // Scalar fallback
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i + k * lda] * B[k + j * ldb];
            }
            C[i + j * ldc] = sum;
        }
    }
#endif
}

// Eigen wrapper for optimized blocked GEMM with 8x8 microkernel
Eigen::MatrixXf neon_gemm_blocked(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    if (A.cols() != B.rows()) {
        return Eigen::MatrixXf();
    }

    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(A.rows(), B.cols());
    neon_gemm_blocked_f32(C.data(), A.data(), B.data(),
                           static_cast<size_t>(A.rows()),
                           static_cast<size_t>(B.cols()),
                           static_cast<size_t>(A.cols()),
                           static_cast<size_t>(A.outerStride()),
                           static_cast<size_t>(B.outerStride()),
                           static_cast<size_t>(C.outerStride()));
    return C;
}

} // namespace neon
} // namespace optmath
