/**
 * OptMathKernels NEON Int8 Dot-Product Kernels
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Int8 GEMM built on the Armv8.2 dot-product extension (FEAT_DotProd /
 * "asimddp": SDOT), available on the Raspberry Pi 5's Cortex-A76. A single
 * SDOT does 16 int8 multiply-accumulates (4 per 32-bit lane), roughly 4x the
 * MAC throughput of the fp32 FMA path and a quarter of the memory traffic.
 *
 * These kernels are only compiled when the target advertises the dot-product
 * extension (OPTMATH_USE_DOTPROD, set by the CMake probe). Use
 * optmath::platform::detect_cpu_info().has_dotprod for a runtime gate.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <Eigen/Dense>

namespace optmath {
namespace neon {

/**
 * @brief Signed int8 GEMM with int32 accumulation: C = A * B.
 *
 * K-contiguous layout (the natural layout for quantized inference weights):
 *   A  is M x K, row-major   (element A[i*K + k])
 *   Bt is N x K, row-major   (i.e. B transposed: element Bt[j*K + k] == B[k][j])
 *   C  is M x N, row-major with leading dimension ldc (element C[i*ldc + j])
 *
 * Both A rows and Bt rows are contiguous in K, so no packing is needed and the
 * inner loop is a chain of SDOT instructions.
 */
void neon_gemm_s8s8s32(std::int32_t* C, std::size_t ldc,
                       const std::int8_t* A,
                       const std::int8_t* Bt,
                       std::size_t M, std::size_t N, std::size_t K);

/**
 * @brief Eigen convenience wrapper. A is M x K, B is K x N (both int8, any
 * storage order). Returns M x N int32. B is transposed internally to the
 * K-contiguous layout the core kernel expects.
 */
Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>
neon_gemm_int8(const Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic>& A,
               const Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic>& B);

/**
 * @brief Int8 valid-mode 2D convolution with int32 accumulation.
 *   in     : in_rows x in_cols, row-major (int8)
 *   kernel : k_rows x k_cols,   row-major (int8)
 *   out    : (in_rows-k_rows+1) x (in_cols-k_cols+1), row-major (int32)
 * Uses widening multiply-accumulate (vmull_s8 -> int32) over 8 output columns at
 * a time, threaded over output rows. Products are int16 (127*127 fits) and
 * accumulated in int32, so no overflow for realistic kernel sizes.
 *
 * Performance note: this is roughly compute-parity with the fp32 conv on the
 * A76 — the convolution access pattern (a scalar tap broadcast over a window)
 * does not fit SDOT, so it uses widening multiplies, which are no cheaper than
 * fp32 FMA here. Its value is native int8 dataflow (no int8<->fp32 round-trip
 * in an all-int8 pipeline), not raw speed. For a real int8 speedup use the
 * SDOT GEMM above (e.g. via im2col at the call site).
 */
void neon_conv2d_s8s8s32(std::int32_t* out, const std::int8_t* in,
                         std::size_t in_rows, std::size_t in_cols,
                         const std::int8_t* kernel,
                         std::size_t k_rows, std::size_t k_cols);

} // namespace neon
} // namespace optmath
