/**
 * OptMathKernels NEON IIR Biquad Filters
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Biquad IIR filter implementation with cascade support and filter
 * design functions based on the Robert Bristow-Johnson Audio EQ Cookbook.
 *
 * Biquad IIR Filter (Direct Form II Transposed):
 *   neon_biquad_f32 implements y[n] = b0*x[n] + s1, s1 = b1*x[n] -
 *   a1*y[n] + s2, s2 = b2*x[n] - a2*y[n]. Scalar implementation only
 *   as the feedback dependency prevents SIMD vectorization across samples.
 *
 * Biquad Cascade:
 *   neon_biquad_cascade_f32 chains multiple biquad stages in series,
 *   each maintaining a 2-element state vector {s1, s2}.
 *
 * Biquad Filter Design (Robert Bristow-Johnson Audio EQ Cookbook):
 *   neon_biquad_lowpass - Low-pass filter coefficients.
 *   neon_biquad_highpass - High-pass filter coefficients.
 *   neon_biquad_bandpass - Band-pass filter coefficients.
 *   neon_biquad_notch - Notch (band-reject) filter coefficients.
 *   Each returns {b0, b1, b2, a1, a2} as normalized coefficients.
 *
 * Eigen Wrapper:
 *   neon_biquad for VectorXf with state management.
 */
#include "optmath/neon_kernels.hpp"
#include <cmath>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

// =========================================================================
// Biquad IIR Processing (Direct Form II Transposed)
// =========================================================================
//
// Recurrence:
//   y[n] = b0 * x[n] + s1
//   s1   = b1 * x[n] - a1 * y[n] + s2
//   s2   = b2 * x[n] - a2 * y[n]
//
// The feedback dependency on y[n] prevents SIMD vectorization across time
// for a single channel. NEON is used within the cascade for coefficient
// broadcast operations when beneficial.

void neon_biquad_f32(float* out, const float* in, std::size_t n,
                     const BiquadCoeffs& coeffs, BiquadState& state) {
    const float b0 = coeffs.b0;
    const float b1 = coeffs.b1;
    const float b2 = coeffs.b2;
    const float a1 = coeffs.a1;
    const float a2 = coeffs.a2;
    float s1 = state.s1;
    float s2 = state.s2;

    for (std::size_t i = 0; i < n; ++i) {
        float x = in[i];
        float y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        out[i] = y;
    }

    state.s1 = s1;
    state.s2 = s2;
}

void neon_biquad_cascade_f32(float* out, const float* in, std::size_t n,
                              const BiquadCoeffs* coeffs, BiquadState* states,
                              std::size_t n_sections) {
    if (n_sections == 0 || n == 0) return;

    // First section reads from in, writes to out
    neon_biquad_f32(out, in, n, coeffs[0], states[0]);

    // Subsequent sections process in-place on out
    for (std::size_t s = 1; s < n_sections; ++s) {
        neon_biquad_f32(out, out, n, coeffs[s], states[s]);
    }
}

// =========================================================================
// Biquad Design Helpers (Audio EQ Cookbook by Robert Bristow-Johnson)
// =========================================================================

// Helper to validate IIR filter design parameters
// Returns true if valid, false otherwise (and returns unity pass-through coeffs)
static bool validate_iir_params(float fc, float fs, float Q, BiquadCoeffs& fallback) {
    // Initialize fallback to unity pass-through (no filtering)
    fallback.b0 = 1.0f;
    fallback.b1 = 0.0f;
    fallback.b2 = 0.0f;
    fallback.a1 = 0.0f;
    fallback.a2 = 0.0f;

    // Sample rate must be positive (check first since fs/2 depends on it)
    if (fs <= 0.0f) {
        return false;
    }
    // Q must be positive to avoid division by zero
    if (Q <= 0.0f) {
        return false;
    }
    // Cutoff frequency must be in valid range (0, fs/2)
    if (fc <= 0.0f || fc >= fs * 0.5f) {
        return false;
    }
    return true;
}

BiquadCoeffs neon_biquad_lowpass(float fc, float fs, float Q) {
    BiquadCoeffs c;
    if (!validate_iir_params(fc, fs, Q, c)) {
        return c;  // Return unity pass-through on invalid params
    }

    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float cosw0 = std::cos(w0);
    const float sinw0 = std::sin(w0);
    const float alpha = sinw0 / (2.0f * Q);

    const float a0 = 1.0f + alpha;
    const float inv_a0 = 1.0f / a0;

    c.b0 = ((1.0f - cosw0) * 0.5f) * inv_a0;
    c.b1 = (1.0f - cosw0) * inv_a0;
    c.b2 = ((1.0f - cosw0) * 0.5f) * inv_a0;
    c.a1 = (-2.0f * cosw0) * inv_a0;
    c.a2 = (1.0f - alpha) * inv_a0;
    return c;
}

BiquadCoeffs neon_biquad_highpass(float fc, float fs, float Q) {
    BiquadCoeffs c;
    if (!validate_iir_params(fc, fs, Q, c)) {
        return c;  // Return unity pass-through on invalid params
    }

    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float cosw0 = std::cos(w0);
    const float sinw0 = std::sin(w0);
    const float alpha = sinw0 / (2.0f * Q);

    const float a0 = 1.0f + alpha;
    const float inv_a0 = 1.0f / a0;

    c.b0 = ((1.0f + cosw0) * 0.5f) * inv_a0;
    c.b1 = -(1.0f + cosw0) * inv_a0;
    c.b2 = ((1.0f + cosw0) * 0.5f) * inv_a0;
    c.a1 = (-2.0f * cosw0) * inv_a0;
    c.a2 = (1.0f - alpha) * inv_a0;
    return c;
}

BiquadCoeffs neon_biquad_bandpass(float fc, float fs, float Q) {
    BiquadCoeffs c;
    if (!validate_iir_params(fc, fs, Q, c)) {
        return c;  // Return unity pass-through on invalid params
    }

    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float cosw0 = std::cos(w0);
    const float sinw0 = std::sin(w0);
    const float alpha = sinw0 / (2.0f * Q);

    const float a0 = 1.0f + alpha;
    const float inv_a0 = 1.0f / a0;

    c.b0 = alpha * inv_a0;
    c.b1 = 0.0f;
    c.b2 = -alpha * inv_a0;
    c.a1 = (-2.0f * cosw0) * inv_a0;
    c.a2 = (1.0f - alpha) * inv_a0;
    return c;
}

BiquadCoeffs neon_biquad_notch(float fc, float fs, float Q) {
    BiquadCoeffs c;
    if (!validate_iir_params(fc, fs, Q, c)) {
        return c;  // Return unity pass-through on invalid params
    }

    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float cosw0 = std::cos(w0);
    const float sinw0 = std::sin(w0);
    const float alpha = sinw0 / (2.0f * Q);

    const float a0 = 1.0f + alpha;
    const float inv_a0 = 1.0f / a0;

    c.b0 = 1.0f * inv_a0;
    c.b1 = (-2.0f * cosw0) * inv_a0;
    c.b2 = 1.0f * inv_a0;
    c.a1 = (-2.0f * cosw0) * inv_a0;
    c.a2 = (1.0f - alpha) * inv_a0;
    return c;
}

// Eigen wrapper
Eigen::VectorXf neon_biquad(const Eigen::VectorXf& in,
                             const BiquadCoeffs& coeffs) {
    if (in.size() == 0) return Eigen::VectorXf();
    Eigen::VectorXf result(in.size());
    BiquadState state;
    neon_biquad_f32(result.data(), in.data(), in.size(), coeffs, state);
    return result;
}

}
}
