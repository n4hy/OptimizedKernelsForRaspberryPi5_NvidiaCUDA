/**
 * OptMathKernels SVE2 Runtime Hardware Detection
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Runtime SVE2 hardware detection module. Compiled WITHOUT -march=sve2 flags
 * so it can execute safely on any AArch64 hardware.
 *
 * Runtime SVE2 Hardware Detection:
 *   is_available() uses getauxval(AT_HWCAP2) & HWCAP2_SVE2 bitmask to detect
 *   SVE2 support at runtime. The result is cached in a static const bool for
 *   one-time detection. Returns true only when both the OPTMATH_USE_SVE2
 *   compile flag is set AND hardware reports SVE2 capability.
 */
#include "optmath/sve2_kernels.hpp"

#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#endif

namespace optmath {
namespace sve2 {

bool is_available() {
#ifdef OPTMATH_USE_SVE2
    // Compiled with SVE2 support — check if hardware actually has it.
#if defined(__aarch64__)
    static const bool has_sve2 = (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
    return has_sve2;
#else
    return false;
#endif
#else
    return false;
#endif
}

} // namespace sve2
} // namespace optmath
