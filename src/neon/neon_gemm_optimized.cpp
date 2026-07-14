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
 *   L1, 512KB L2, 2MB L3) uses MC=128, KC=256, NC=512. Cortex-A720
 *   (CIX P1: 512KB L2, 12MB L3) uses MC=256, KC=512, NC=2048. Thread-
 *   local aligned buffers for packed panels.
 *
 * 8x8 Microkernel:
 *   micro_kernel_8x8 uses column-oriented accumulators for efficient
 *   column-major store. Rank-1 updates via vfmaq_laneq_f32 (FMA with
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
#include <cstdlib>
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
// Memory hierarchy targeting. MC/KC/NC are NOT constants -- they are computed at
// run time from the detected cache sizes by platform::get_gemm_mc/kc/nc(), then
// capped by MC_MAX/KC_MAX/NC_MAX below. Measured on this Pi 5 (A76, L2=512KB,
// L3=2MB): MC=128, KC=256, NC=256.
//
// NC=256, not 512: platform.cpp:432-434 records that NC=512 made the packed-B
// panel 256*512*4 = 512KB -- the entire L2 -- evicting the A panel and C tiles
// every pass. It sizes B to ~half of L2 instead (Goto blocking). This header
// advertised the old 512 long after that fix landed; if you change the blocking,
// change platform.cpp and re-measure, do not edit this comment to match a guess.

// Runtime-selected cache blocking parameters
static size_t get_mc() { return platform::get_gemm_mc(); }
static size_t get_kc() { return platform::get_gemm_kc(); }
static size_t get_nc() { return platform::get_gemm_nc(); }

// Upper bounds on the runtime blocking parameters. These CAP whatever
// platform::get_gemm_*() reports so a mis-detected topology can never request an
// unbounded panel; they no longer size any static storage (see PackBuffer below).
// CIX P1 (12MB L3) is the largest supported target: KC*NC*4 = 512*2048*4 = 4MB.
constexpr size_t MC_MAX = 256;
constexpr size_t KC_MAX = 512;
constexpr size_t NC_MAX = 2048;

// Microkernel dimensions
constexpr size_t MR = 8;    // Rows per microkernel
constexpr size_t NR = 8;    // Cols per microkernel

// Below this much work (M*N*K), Eigen's small-matrix GEMM beats this kernel and
// we hand off to it. The 8x8 microkernel amortizes its prologue (zeroing 16
// accumulators) and epilogue (16 load-add-stores to C) over the k loop, so a
// short k makes that overhead dominate; Eigen has dedicated small paths.
// Measured on an idle Pi 5, this kernel vs Eigen (GFLOPS, best-of-7):
//     N=32   11.8 vs 23.0  -> Eigen 1.95x
//     N=48   18.3 vs 35.0  -> Eigen 1.92x
//     N=64   60.3 vs 61.6  -> Eigen 1.02x   (crossover ~N=72)
//     N=80   73.1 vs 72.9  -> NEON  1.00x
//     N=96   89.9 vs 84.4  -> NEON  1.07x
//     N=256 123.4 vs 68.0  -> NEON  1.82x
// Use each where it wins. 80^3 sits just past the crossover, so neither path is
// ever the slower choice. Re-measure if MR/NR or the microkernel change.
constexpr size_t OPTMATH_GEMM_EIGEN_MAX = 80 * 80 * 80;

// 64-byte-aligned, thread-local scratch for the packed A/B panels, sized to the
// ACTUAL runtime blocking rather than the worst-case max. On a Pi 5 (MC=128,
// KC=256, NC=256) this is ~128KB (A) + ~256KB (B) per thread; the old static
// max-sized arrays reserved 4.5MB per thread (~18MB across the 4 A76 cores),
// touching far more pages/TLB entries than the kernel ever uses. The buffer
// grows on demand and frees itself on thread exit.
struct PackBuffer {
    float* ptr = nullptr;
    size_t cap = 0;  // capacity in floats
    ~PackBuffer() { std::free(ptr); }
    float* get(size_t n_floats) {
        if (n_floats > cap) {
            std::free(ptr);
            // std::aligned_alloc requires size to be a multiple of the alignment.
            const size_t bytes = ((n_floats * sizeof(float) + 63) / 64) * 64;
            ptr = static_cast<float*>(std::aligned_alloc(64, bytes));
            cap = ptr ? bytes / sizeof(float) : 0;
        }
        return ptr;
    }
};
static thread_local PackBuffer packed_A_buf;
static thread_local PackBuffer packed_B_buf;

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

    // Prefetch the 8 output columns for write; they are read-modify-written
    // after the k-loop, so warming them now hides the load-add latency.
    for (int j = 0; j < 8; ++j) {
        __builtin_prefetch(C + j * ldc, 1 /*write*/, 3 /*high locality*/);
    }

    // Main loop over k dimension
    for (size_t p = 0; p < k; ++p) {
        // Prefetch packed A/B a few k-steps ahead to hide L1/L2 latency on the
        // A76 (out-of-range prefetch is harmless — it never faults).
        __builtin_prefetch(A_packed + (p + 8) * MR, 0, 3);
        __builtin_prefetch(B_packed + (p + 8) * NR, 0, 3);

        // Load A column (8 elements: rows 0-7 of A at k-index p)
        float32x4_t a0 = vld1q_f32(A_packed + p * MR);
        float32x4_t a1 = vld1q_f32(A_packed + p * MR + 4);

        // Load B row (8 elements: cols 0-7 of B at k-index p)
        float32x4_t b0 = vld1q_f32(B_packed + p * NR);
        float32x4_t b1 = vld1q_f32(B_packed + p * NR + 4);

        // Rank-1 update by columns: C[:,j] += A[:,p] * B[p,j]
        // Column 0-3 (elements from b0)
        c0_lo = vfmaq_laneq_f32(c0_lo, a0, b0, 0);
        c0_hi = vfmaq_laneq_f32(c0_hi, a1, b0, 0);
        c1_lo = vfmaq_laneq_f32(c1_lo, a0, b0, 1);
        c1_hi = vfmaq_laneq_f32(c1_hi, a1, b0, 1);
        c2_lo = vfmaq_laneq_f32(c2_lo, a0, b0, 2);
        c2_hi = vfmaq_laneq_f32(c2_hi, a1, b0, 2);
        c3_lo = vfmaq_laneq_f32(c3_lo, a0, b0, 3);
        c3_hi = vfmaq_laneq_f32(c3_hi, a1, b0, 3);
        // Column 4-7 (elements from b1)
        c4_lo = vfmaq_laneq_f32(c4_lo, a0, b1, 0);
        c4_hi = vfmaq_laneq_f32(c4_hi, a1, b1, 0);
        c5_lo = vfmaq_laneq_f32(c5_lo, a0, b1, 1);
        c5_hi = vfmaq_laneq_f32(c5_hi, a1, b1, 1);
        c6_lo = vfmaq_laneq_f32(c6_lo, a0, b1, 2);
        c6_hi = vfmaq_laneq_f32(c6_hi, a1, b1, 2);
        c7_lo = vfmaq_laneq_f32(c7_lo, a0, b1, 3);
        c7_hi = vfmaq_laneq_f32(c7_hi, a1, b1, 3);
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

    // B is column-major, so column j (B + j*ldb) is contiguous in p. Iterate
    // columns on the outside and rows on the inside so B is read sequentially
    // (the A76 HW prefetcher loves this) — the old p-outer/j-inner order strode
    // B by ldb every element, touching a fresh cache line each time. The strided
    // writes now land in the small, L1-resident packed buffer, which is cheap.
    for (size_t j = 0; j < n; ++j) {
        const float* bcol = B + j * ldb;
        if (j + 1 < n) {
            __builtin_prefetch(B + (j + 1) * ldb, 0, 3);  // next column, read
        }
        for (size_t p = 0; p < k; ++p) {
            packed[p * NR + j] = bcol[p];
        }
    }
    // Zero-pad columns n..NR-1
    for (size_t j = n; j < NR; ++j) {
        for (size_t p = 0; p < k; ++p) {
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
    // Small GEMM: hand off to Eigen, which wins below the crossover (see
    // OPTMATH_GEMM_EIGEN_MAX). Same result; this only picks the faster path.
    // Mapped with the caller's strides, so it works for sub-blocks too.
    if (M * N * K < OPTMATH_GEMM_EIGEN_MAX) {
        using OS = Eigen::OuterStride<>;
        Eigen::Map<const Eigen::MatrixXf, 0, OS> Am(A, M, K, OS((Eigen::Index)lda));
        Eigen::Map<const Eigen::MatrixXf, 0, OS> Bm(B, K, N, OS((Eigen::Index)ldb));
        Eigen::Map<Eigen::MatrixXf, 0, OS>       Cm(C, M, N, OS((Eigen::Index)ldc));
        Cm.noalias() = Am * Bm;
        return;
    }

    // Runtime-selected cache blocking parameters
    const size_t MC = std::min(get_mc(), MC_MAX);
    const size_t KC = std::min(get_kc(), KC_MAX);
    const size_t NC = std::min(get_nc(), NC_MAX);

    // Initialize C to zero. Parallel over columns (each is a disjoint, contiguous
    // run in column-major C); serial this is pure Amdahl overhead at large N.
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; ++j) {
        std::memset(C + j * ldc, 0, M * sizeof(float));
    }

    // Loop over blocks of N (columns of B and C).
    //
    // Threading note (v0.6.2): this loop used to carry the `#pragma omp parallel
    // for`, which gave ceil(N/NC) units of parallelism -- and NC is 256 on a Pi 5.
    // So a 256x256 GEMM produced exactly ONE block and ran fully serial on one of
    // the four A76s, while 512 produced two (and 4 threads then ran *slower* than
    // 2: 43.4 vs 54.0 GFLOPS, from scheduling four workers over two blocks).
    // Measured: Blocked/256 was 32.0 / 32.0 / 29.2 GFLOPS at 1 / 2 / 4 threads --
    // zero scaling. The microkernel was never the problem; per core it already
    // beat Eigen (32.0 vs 31.3 GFLOPS single-threaded).
    //
    // The parallelism now lives on the (jr, ir) microkernel loops below, which
    // yield (nc/NR)*(mc/MR) independent tiles -- 512 for a 256x256 GEMM, ample
    // for 4 cores at any size that reaches the microkernel.
    for (size_t jc = 0; jc < N; jc += NC) {
        size_t nc = std::min(NC, N - jc);

        // Packed panels, sized to the actual blocking (grow-on-demand). These MUST
        // be acquired OUTSIDE the parallel region below: the buffers are
        // thread_local, so a worker calling .get() itself would get its own
        // separate scratch -- it would pack into one buffer and the microkernel
        // would read another. Taking the pointers here gives every worker the same
        // (master's) buffers, which they fill cooperatively via `omp for` and then
        // read back after the barrier.
        float* packed_A = packed_A_buf.get(MC * KC);
        float* packed_B = packed_B_buf.get(KC * NC);

        // ONE parallel region for the whole jc block. Every thread runs the pc/ic
        // loops redundantly (identical, cheap index math) while the `omp for`s
        // split the actual work -- so the team forks/joins once per jc instead of
        // once per pack_B and once per ic block. That overhead is invisible at
        // N>=256 but dominated small problems: with a region per stage, N=128 ran
        // 57.9 GFLOPS against Eigen's 81.7.
        //
        // The implicit barrier ending each `omp for` is load-bearing:
        //   - after pack_B: packed_B is complete before any thread reads it;
        //   - after pack_A: likewise for packed_A;
        //   - after the microkernel sweep: every thread is done reading packed_A/
        //     packed_B before the next ic/pc iteration repacks them.
        // Do not add `nowait` to any of them.
        //
        // No `if()` guard is needed: anything small enough for fork/join to hurt
        // already went to Eigen above (OPTMATH_GEMM_EIGEN_MAX).
        #pragma omp parallel
        {
            // Loop over blocks of K
            for (size_t pc = 0; pc < K; pc += KC) {
                size_t kc = std::min(KC, K - pc);

                // Pack B panel: B[pc:pc+kc, jc:jc+nc]. Parallel over jr: each
                // strip writes its own packed_B + jr*kc, so writes are disjoint.
                // Packing must be parallel too -- left serial it is ~25% of a
                // 256x256 GEMM, and Amdahl then caps the kernel near 2.3x on 4
                // cores (measured: 82 GFLOPS vs 117 once parallelized).
                #pragma omp for schedule(static)
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

                    // Pack A panel: A[ic:ic+mc, pc:pc+kc]. Disjoint per ir.
                    #pragma omp for schedule(static)
                    for (size_t ir = 0; ir < mc; ir += MR) {
                        size_t mr = std::min(MR, mc - ir);
                        pack_A_panel(
                            packed_A + ir * kc,
                            A + (ic + ir) + pc * lda,
                            lda, mr, kc);
                    }

                    // Microkernel loop, parallel over the (jr, ir) tile grid.
                    // Tile (jr, ir) writes C[ic+ir .. ic+ir+MR, jc+jr .. jc+jr+NR],
                    // disjoint for distinct (jr, ir), so the workers never race.
                    // packed_A/packed_B are read-only here. static: tiles are equal
                    // cost (edge tiles are cheaper, and there are at most one row
                    // and one column of them).
                    #pragma omp for collapse(2) schedule(static)
                    for (size_t jr = 0; jr < nc; jr += NR) {
                        for (size_t ir = 0; ir < mc; ir += MR) {
                            size_t nr = std::min(NR, nc - jr);
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
