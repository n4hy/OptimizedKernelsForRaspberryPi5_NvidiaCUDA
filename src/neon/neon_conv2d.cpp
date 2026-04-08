/**
 * OptMathKernels NEON 2D Convolution
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * NEON-accelerated 2D convolution kernels including general, separable,
 * and fixed-size optimized variants for image and signal processing.
 *
 * General 2D Convolution:
 *   neon_conv2d_f32 performs valid-mode row-major convolution. Vectorizes
 *   4 output columns per iteration via vld1q_f32/vmlaq_f32 with scalar
 *   tail fallback for non-aligned widths.
 *
 * Separable 2D Convolution:
 *   neon_conv2d_separable_f32 decomposes a KxL kernel into K+L: row pass
 *   then column pass using neon_fir_f32. Reduces computational complexity
 *   from O(K*L) to O(K+L) per output pixel.
 *
 * Optimized 3x3 Convolution:
 *   neon_conv2d_3x3_f32 fully unrolls all 9 tap coefficients with
 *   pre-broadcast via vdupq_n_f32. Processes 4 output columns
 *   simultaneously for maximum throughput.
 *
 * Optimized 5x5 Convolution:
 *   neon_conv2d_5x5_f32 fully unrolls all 25 tap coefficients using
 *   the same vectorization strategy as the 3x3 variant.
 *
 * Eigen Wrapper:
 *   neon_conv2d handles column-major to row-major conversion for
 *   Eigen MatrixXf compatibility.
 */
#include "optmath/neon_kernels.hpp"
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

// =========================================================================
// General 2D Convolution (valid mode, row-major layout)
// =========================================================================

void neon_conv2d_f32(float* out, const float* in,
                     std::size_t in_rows, std::size_t in_cols,
                     const float* kernel, std::size_t kernel_rows, std::size_t kernel_cols) {
    if (in_rows < kernel_rows || in_cols < kernel_cols) return;

    const std::size_t out_rows = in_rows - kernel_rows + 1;
    const std::size_t out_cols = in_cols - kernel_cols + 1;

#ifdef OPTMATH_USE_NEON
    for (std::size_t r = 0; r < out_rows; ++r) {
        std::size_t c = 0;

        // NEON path: process 4 output columns at a time
        for (; c + 3 < out_cols; c += 4) {
            float32x4_t vsum = vdupq_n_f32(0.0f);

            for (std::size_t kr = 0; kr < kernel_rows; ++kr) {
                const float* in_row = in + (r + kr) * in_cols + c;
                const float* k_row = kernel + kr * kernel_cols;

                for (std::size_t kc = 0; kc < kernel_cols; ++kc) {
                    float32x4_t vin = vld1q_f32(in_row + kc);
                    float32x4_t vk = vdupq_n_f32(k_row[kc]);
                    vsum = vmlaq_f32(vsum, vin, vk);
                }
            }

            vst1q_f32(out + r * out_cols + c, vsum);
        }

        // Scalar tail
        for (; c < out_cols; ++c) {
            float sum = 0.0f;
            for (std::size_t kr = 0; kr < kernel_rows; ++kr) {
                for (std::size_t kc = 0; kc < kernel_cols; ++kc) {
                    sum += in[(r + kr) * in_cols + (c + kc)] * kernel[kr * kernel_cols + kc];
                }
            }
            out[r * out_cols + c] = sum;
        }
    }
#else
    for (std::size_t r = 0; r < out_rows; ++r) {
        for (std::size_t c = 0; c < out_cols; ++c) {
            float sum = 0.0f;
            for (std::size_t kr = 0; kr < kernel_rows; ++kr) {
                for (std::size_t kc = 0; kc < kernel_cols; ++kc) {
                    sum += in[(r + kr) * in_cols + (c + kc)] * kernel[kr * kernel_cols + kc];
                }
            }
            out[r * out_cols + c] = sum;
        }
    }
#endif
}

// =========================================================================
// Separable 2D Convolution
// =========================================================================

void neon_conv2d_separable_f32(float* out, const float* in,
                                std::size_t in_rows, std::size_t in_cols,
                                const float* row_kernel, std::size_t row_kernel_len,
                                const float* col_kernel, std::size_t col_kernel_len) {
    if (in_cols < row_kernel_len || in_rows < col_kernel_len) return;

    const std::size_t mid_rows = in_rows;
    const std::size_t mid_cols = in_cols - row_kernel_len + 1;
    const std::size_t out_rows = in_rows - col_kernel_len + 1;
    const std::size_t out_cols = mid_cols;

    // Intermediate buffer: apply row kernel along each row
    std::vector<float> mid(mid_rows * mid_cols);

    // Pass 1: Row convolution (1D FIR along each row)
    for (std::size_t r = 0; r < mid_rows; ++r) {
        const float* row_in = in + r * in_cols;
        float* row_out = mid.data() + r * mid_cols;
        neon_fir_f32(row_in, in_cols, row_kernel, row_kernel_len, row_out);
    }

    // Pass 2: Column convolution on the intermediate result
#ifdef OPTMATH_USE_NEON
    for (std::size_t r = 0; r < out_rows; ++r) {
        std::size_t c = 0;

        for (; c + 3 < out_cols; c += 4) {
            float32x4_t vsum = vdupq_n_f32(0.0f);

            for (std::size_t k = 0; k < col_kernel_len; ++k) {
                float32x4_t vin = vld1q_f32(&mid[(r + k) * mid_cols + c]);
                float32x4_t vk = vdupq_n_f32(col_kernel[k]);
                vsum = vmlaq_f32(vsum, vin, vk);
            }

            vst1q_f32(out + r * out_cols + c, vsum);
        }

        for (; c < out_cols; ++c) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < col_kernel_len; ++k) {
                sum += mid[(r + k) * mid_cols + c] * col_kernel[k];
            }
            out[r * out_cols + c] = sum;
        }
    }
#else
    for (std::size_t r = 0; r < out_rows; ++r) {
        for (std::size_t c = 0; c < out_cols; ++c) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < col_kernel_len; ++k) {
                sum += mid[(r + k) * mid_cols + c] * col_kernel[k];
            }
            out[r * out_cols + c] = sum;
        }
    }
#endif
}

// =========================================================================
// Optimized 3x3 Convolution (fully unrolled)
// =========================================================================

void neon_conv2d_3x3_f32(float* out, const float* in,
                          std::size_t in_rows, std::size_t in_cols,
                          const float kernel[9]) {
    if (in_rows < 3 || in_cols < 3) return;

    const std::size_t out_rows = in_rows - 2;
    const std::size_t out_cols = in_cols - 2;

#ifdef OPTMATH_USE_NEON
    // Load all 9 kernel values as broadcast vectors
    float32x4_t vk00 = vdupq_n_f32(kernel[0]);
    float32x4_t vk01 = vdupq_n_f32(kernel[1]);
    float32x4_t vk02 = vdupq_n_f32(kernel[2]);
    float32x4_t vk10 = vdupq_n_f32(kernel[3]);
    float32x4_t vk11 = vdupq_n_f32(kernel[4]);
    float32x4_t vk12 = vdupq_n_f32(kernel[5]);
    float32x4_t vk20 = vdupq_n_f32(kernel[6]);
    float32x4_t vk21 = vdupq_n_f32(kernel[7]);
    float32x4_t vk22 = vdupq_n_f32(kernel[8]);

    for (std::size_t r = 0; r < out_rows; ++r) {
        const float* r0 = in + r * in_cols;
        const float* r1 = in + (r + 1) * in_cols;
        const float* r2 = in + (r + 2) * in_cols;

        std::size_t c = 0;
        for (; c + 3 < out_cols; c += 4) {
            float32x4_t vsum;
            // Row 0
            vsum = vmulq_f32(vld1q_f32(r0 + c), vk00);
            vsum = vmlaq_f32(vsum, vld1q_f32(r0 + c + 1), vk01);
            vsum = vmlaq_f32(vsum, vld1q_f32(r0 + c + 2), vk02);
            // Row 1
            vsum = vmlaq_f32(vsum, vld1q_f32(r1 + c), vk10);
            vsum = vmlaq_f32(vsum, vld1q_f32(r1 + c + 1), vk11);
            vsum = vmlaq_f32(vsum, vld1q_f32(r1 + c + 2), vk12);
            // Row 2
            vsum = vmlaq_f32(vsum, vld1q_f32(r2 + c), vk20);
            vsum = vmlaq_f32(vsum, vld1q_f32(r2 + c + 1), vk21);
            vsum = vmlaq_f32(vsum, vld1q_f32(r2 + c + 2), vk22);

            vst1q_f32(out + r * out_cols + c, vsum);
        }

        // Scalar tail
        for (; c < out_cols; ++c) {
            float sum = 0.0f;
            sum += r0[c]     * kernel[0] + r0[c + 1] * kernel[1] + r0[c + 2] * kernel[2];
            sum += r1[c]     * kernel[3] + r1[c + 1] * kernel[4] + r1[c + 2] * kernel[5];
            sum += r2[c]     * kernel[6] + r2[c + 1] * kernel[7] + r2[c + 2] * kernel[8];
            out[r * out_cols + c] = sum;
        }
    }
#else
    neon_conv2d_f32(out, in, in_rows, in_cols, kernel, 3, 3);
#endif
}

// =========================================================================
// Optimized 5x5 Convolution (unrolled)
// =========================================================================

void neon_conv2d_5x5_f32(float* out, const float* in,
                          std::size_t in_rows, std::size_t in_cols,
                          const float kernel[25]) {
    if (in_rows < 5 || in_cols < 5) return;

    const std::size_t out_rows = in_rows - 4;
    const std::size_t out_cols = in_cols - 4;

#ifdef OPTMATH_USE_NEON
    // Load all 25 kernel values
    float32x4_t vk[5][5];
    for (int kr = 0; kr < 5; ++kr) {
        for (int kc = 0; kc < 5; ++kc) {
            vk[kr][kc] = vdupq_n_f32(kernel[kr * 5 + kc]);
        }
    }

    for (std::size_t r = 0; r < out_rows; ++r) {
        std::size_t c = 0;

        for (; c + 3 < out_cols; c += 4) {
            float32x4_t vsum = vdupq_n_f32(0.0f);

            for (int kr = 0; kr < 5; ++kr) {
                const float* row = in + (r + kr) * in_cols + c;
                vsum = vmlaq_f32(vsum, vld1q_f32(row),     vk[kr][0]);
                vsum = vmlaq_f32(vsum, vld1q_f32(row + 1), vk[kr][1]);
                vsum = vmlaq_f32(vsum, vld1q_f32(row + 2), vk[kr][2]);
                vsum = vmlaq_f32(vsum, vld1q_f32(row + 3), vk[kr][3]);
                vsum = vmlaq_f32(vsum, vld1q_f32(row + 4), vk[kr][4]);
            }

            vst1q_f32(out + r * out_cols + c, vsum);
        }

        for (; c < out_cols; ++c) {
            float sum = 0.0f;
            for (int kr = 0; kr < 5; ++kr) {
                for (int kc = 0; kc < 5; ++kc) {
                    sum += in[(r + kr) * in_cols + (c + kc)] * kernel[kr * 5 + kc];
                }
            }
            out[r * out_cols + c] = sum;
        }
    }
#else
    neon_conv2d_f32(out, in, in_rows, in_cols, kernel, 5, 5);
#endif
}

// =========================================================================
// Eigen Wrapper
// =========================================================================

Eigen::MatrixXf neon_conv2d(const Eigen::MatrixXf& in, const Eigen::MatrixXf& kernel) {
    if (in.rows() < kernel.rows() || in.cols() < kernel.cols()) {
        return Eigen::MatrixXf();
    }

    const long out_rows = in.rows() - kernel.rows() + 1;
    const long out_cols = in.cols() - kernel.cols() + 1;

    // Convert Eigen column-major to row-major for the 2D conv
    const long ir = in.rows();
    const long ic = in.cols();
    const long kr = kernel.rows();
    const long kc = kernel.cols();

    std::vector<float> in_rm(ir * ic);
    std::vector<float> k_rm(kr * kc);

    for (long r = 0; r < ir; ++r)
        for (long c = 0; c < ic; ++c)
            in_rm[r * ic + c] = in(r, c);

    for (long r = 0; r < kr; ++r)
        for (long c = 0; c < kc; ++c)
            k_rm[r * kc + c] = kernel(r, c);

    std::vector<float> out_rm(out_rows * out_cols);
    neon_conv2d_f32(out_rm.data(), in_rm.data(), ir, ic,
                    k_rm.data(), kr, kc);

    // Convert back to Eigen column-major
    Eigen::MatrixXf result(out_rows, out_cols);
    for (long r = 0; r < out_rows; ++r)
        for (long c = 0; c < out_cols; ++c)
            result(r, c) = out_rm[r * out_cols + c];

    return result;
}

}
}
