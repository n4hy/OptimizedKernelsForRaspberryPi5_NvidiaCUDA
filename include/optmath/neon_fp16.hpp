#pragma once

#include <cstddef>

// Half-precision (IEEE binary16) NEON kernels for ARMv8.2-A FEAT_FP16 cores
// such as the Raspberry Pi 5 Cortex-A76 (cpuinfo: fphp/asimdhp). These process
// 8 lanes per 128-bit vector (vs 4 for fp32), roughly doubling throughput where
// reduced precision is acceptable — e.g. ML inference activations and GEMV.
//
// The declarations are only visible when compiling for an FP16-capable target
// (the compiler defines __ARM_FEATURE_FP16_VECTOR_ARITHMETIC). The library is
// built with -mcpu=native on the Pi 5, which enables it.
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace optmath {
namespace neon {

    /// True when the half-precision kernels are compiled in (always true here,
    /// since this header is only visible on FP16-capable builds).
    bool fp16_available();

    /// out[i] = a[i] + b[i], half precision.
    void neon_add_f16(__fp16* out, const __fp16* a, const __fp16* b, std::size_t n);

    /// out[i] = a[i] * b[i], half precision.
    void neon_mul_f16(__fp16* out, const __fp16* a, const __fp16* b, std::size_t n);

    /// data[i] = max(0, data[i]) in place, half precision (ReLU).
    void neon_relu_f16(__fp16* data, std::size_t n);

    /// Dot product of two fp16 vectors. Products are formed in fp16 but
    /// accumulated in fp32, so the result is accurate to ~fp16 input precision
    /// without fp16 accumulation drift over long vectors. Returns a float.
    float neon_dot_f16(const __fp16* a, const __fp16* b, std::size_t n);

    // NOTE: fp16 GEMM/GEMV were prototyped here and removed. The Cortex-A76 has
    // no FEAT_FHM (fused fp16->fp32 MAC), so any fp16 dot/GEMM needing fp32
    // accumulation must issue explicit vcvt widenings and ends up SLOWER than
    // the fp32 FMA path (measured ~3.3x slower for 512^3 GEMM). fp16 wins only
    // for the pure elementwise ops above (no reduction/widening). Quantized
    // matmul speedups come from the int8 SDOT path (neon_int8.hpp) instead.

} // namespace neon
} // namespace optmath

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
