/**
 * OptMathKernels NEON Polyphase Resampler
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * NEON-accelerated polyphase rational rate resampler for sample rate
 * conversion of real-valued signals.
 *
 * Polyphase Filter Initialization:
 *   neon_resample_init decomposes a prototype FIR filter into L polyphase
 *   phases, storing coefficients in time-reversed order for direct dot
 *   product computation without index reversal at runtime.
 *
 * Polyphase Rational Resampler:
 *   neon_resample_f32 implements L:M rational rate conversion. Uses a
 *   phase accumulator to drive the output rate. Delay line managed via
 *   memmove shift buffer. Per-output-sample operation: advance phase,
 *   shift new input into delay line when phase wraps, compute polyphase
 *   FIR output via neon_dot_f32.
 *
 * Eigen Wrapper:
 *   neon_resample with automatic output size calculation (n_in * L / M).
 */
#include "optmath/neon_kernels.hpp"
#include <cstring>
#include <algorithm>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

void neon_resample_init(PolyphaseResamplerState& state,
                        const float* filter, std::size_t filter_len,
                        std::size_t L, std::size_t M) {
    state.L = L;
    state.M = M;
    state.n_taps = (filter_len + L - 1) / L;
    state.phase_acc = 0;

    // Decompose filter into L polyphase phases
    // phase_coeffs[p][k] = filter[p + k*L]
    // Stored in reversed order for efficient dot product:
    // reversed_phase[p][k] = filter[p + (n_taps - 1 - k)*L]
    state.phases.resize(L);
    for (std::size_t p = 0; p < L; ++p) {
        state.phases[p].resize(state.n_taps, 0.0f);
        for (std::size_t k = 0; k < state.n_taps; ++k) {
            std::size_t filt_idx = p + (state.n_taps - 1 - k) * L;
            if (filt_idx < filter_len) {
                state.phases[p][k] = filter[filt_idx];
            }
        }
    }

    // Initialize delay line (double-sized for contiguous NEON access)
    state.delay.resize(state.n_taps, 0.0f);
    state.delay_pos = 0;
}

std::size_t neon_resample_f32(float* out, const float* in, std::size_t input_len,
                               PolyphaseResamplerState& state) {
    std::size_t n_out = 0;
    const std::size_t L = state.L;
    const std::size_t M = state.M;
    const std::size_t n_taps = state.n_taps;

    for (std::size_t i = 0; i < input_len; ++i) {
        // Push input sample into delay line (shift left, new sample at end)
        // This keeps the delay line in order: oldest at [0], newest at [n_taps-1]
        if (n_taps > 1) {
            std::memmove(state.delay.data(), state.delay.data() + 1,
                         (n_taps - 1) * sizeof(float));
        }
        state.delay[n_taps - 1] = in[i];

        // Produce output samples while phase accumulator is within range
        while (state.phase_acc < L) {
            // y = dot(reversed_phase[phase_acc], delay_line, n_taps)
            out[n_out++] = neon_dot_f32(state.phases[state.phase_acc].data(),
                                         state.delay.data(), n_taps);
            state.phase_acc += M;
        }
        state.phase_acc -= L;
    }

    return n_out;
}

void neon_resample_oneshot_f32(float* out, std::size_t* output_len,
                                const float* in, std::size_t input_len,
                                const float* filter, std::size_t filter_len,
                                std::size_t L, std::size_t M) {
    PolyphaseResamplerState state;
    neon_resample_init(state, filter, filter_len, L, M);
    *output_len = neon_resample_f32(out, in, input_len, state);
}

// Eigen wrapper
Eigen::VectorXf neon_resample(const Eigen::VectorXf& in,
                               const Eigen::VectorXf& filter,
                               std::size_t L, std::size_t M) {
    if (in.size() == 0 || filter.size() == 0 || L == 0 || M == 0) {
        return Eigen::VectorXf();
    }

    // Upper bound on output size
    std::size_t max_out = (static_cast<std::uint64_t>(in.size()) * L + M - 1) / M + 1;
    Eigen::VectorXf result(max_out);

    std::size_t actual_len = 0;
    neon_resample_oneshot_f32(result.data(), &actual_len,
                               in.data(), in.size(),
                               filter.data(), filter.size(),
                               L, M);

    result.conservativeResize(actual_len);
    return result;
}

}
}
