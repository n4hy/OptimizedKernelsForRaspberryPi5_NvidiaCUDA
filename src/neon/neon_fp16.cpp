/**
 * OptMathKernels NEON Half-Precision (FP16) Kernels
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * IEEE binary16 kernels for ARMv8.2-A FEAT_FP16 cores (Raspberry Pi 5
 * Cortex-A76). 8 lanes per 128-bit vector double the elementwise throughput of
 * the fp32 paths. Elementwise ops (add/mul/relu) are exact in fp16; the dot
 * product multiplies in fp16 but accumulates in fp32 to avoid fp16 accumulation
 * drift over long vectors.
 *
 * Compiled only when the target supports FP16 vector arithmetic (guarded by
 * OPTMATH_USE_FP16, set by CMake). The A76 lacks FEAT_FHM (fp16->fp32 fused
 * multiply-add, asimdfhm), so the dot widens fp16 products with vcvt and
 * accumulates with ordinary fp32 adds.
 */
#include "optmath/neon_fp16.hpp"

#if defined(OPTMATH_USE_FP16) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include <arm_neon.h>

namespace optmath {
namespace neon {

bool fp16_available() { return true; }

void neon_add_f16(__fp16* out, const __fp16* a, const __fp16* b, std::size_t n) {
    const float16_t* pa = reinterpret_cast<const float16_t*>(a);
    const float16_t* pb = reinterpret_cast<const float16_t*>(b);
    float16_t* po = reinterpret_cast<float16_t*>(out);
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        vst1q_f16(po + i, vaddq_f16(vld1q_f16(pa + i), vld1q_f16(pb + i)));
    }
    for (; i < n; ++i) out[i] = static_cast<__fp16>(a[i] + b[i]);
}

void neon_mul_f16(__fp16* out, const __fp16* a, const __fp16* b, std::size_t n) {
    const float16_t* pa = reinterpret_cast<const float16_t*>(a);
    const float16_t* pb = reinterpret_cast<const float16_t*>(b);
    float16_t* po = reinterpret_cast<float16_t*>(out);
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        vst1q_f16(po + i, vmulq_f16(vld1q_f16(pa + i), vld1q_f16(pb + i)));
    }
    for (; i < n; ++i) out[i] = static_cast<__fp16>(a[i] * b[i]);
}

void neon_relu_f16(__fp16* data, std::size_t n) {
    float16_t* p = reinterpret_cast<float16_t*>(data);
    const float16x8_t vzero = vdupq_n_f16(0.0f);
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        vst1q_f16(p + i, vmaxq_f16(vzero, vld1q_f16(p + i)));
    }
    for (; i < n; ++i) {
        if (data[i] < static_cast<__fp16>(0)) data[i] = static_cast<__fp16>(0);
    }
}

float neon_dot_f16(const __fp16* a, const __fp16* b, std::size_t n) {
    const float16_t* pa = reinterpret_cast<const float16_t*>(a);
    const float16_t* pb = reinterpret_cast<const float16_t*>(b);

    // Multiply in fp16 (8 lanes), widen the two halves to fp32 and accumulate
    // in fp32 with two independent accumulators for the A76 dual-FMA pipes.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t prod = vmulq_f16(vld1q_f16(pa + i), vld1q_f16(pb + i));
        acc0 = vaddq_f32(acc0, vcvt_f32_f16(vget_low_f16(prod)));
        acc1 = vaddq_f32(acc1, vcvt_f32_f16(vget_high_f16(prod)));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) {
        sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return sum;
}

// fp16 GEMM/GEMV intentionally omitted: on the Cortex-A76 (no FEAT_FHM) the
// mandatory vcvt widening makes fp16-with-fp32-accumulation slower than the
// fp32 FMA GEMM. See the note in neon_fp16.hpp. Use the int8 SDOT GEMM
// (neon_int8.hpp) for a genuine quantized-matmul speedup.

} // namespace neon
} // namespace optmath

#endif // OPTMATH_USE_FP16 && __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
