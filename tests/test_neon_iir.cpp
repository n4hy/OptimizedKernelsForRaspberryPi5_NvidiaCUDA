#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

using namespace optmath::neon;

TEST(NeonBiquadTest, DCGainLowpass) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // A lowpass filter should pass DC (gain = 1.0 at 0 Hz)
    float fs = 48000.0f;
    float fc = 1000.0f;
    BiquadCoeffs coeffs = neon_biquad_lowpass(fc, fs);

    const std::size_t N = 1000;
    std::vector<float> in(N, 1.0f);  // DC signal
    std::vector<float> out(N);

    BiquadState state;
    neon_biquad_f32(out.data(), in.data(), N, coeffs, state);

    // After settling, output should be ~1.0
    EXPECT_NEAR(out[N - 1], 1.0f, 0.01f);
    EXPECT_NEAR(out[N - 10], 1.0f, 0.01f);
}

TEST(NeonBiquadTest, HighpassBlocksDC) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 1000.0f;
    BiquadCoeffs coeffs = neon_biquad_highpass(fc, fs);

    const std::size_t N = 2000;
    std::vector<float> in(N, 1.0f);
    std::vector<float> out(N);

    BiquadState state;
    neon_biquad_f32(out.data(), in.data(), N, coeffs, state);

    // Highpass should attenuate DC to near zero
    EXPECT_NEAR(out[N - 1], 0.0f, 0.01f);
}

TEST(NeonBiquadTest, LowpassAttenuatesHighFreq) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 100.0f;
    BiquadCoeffs coeffs = neon_biquad_lowpass(fc, fs, 0.707f);

    const std::size_t N = 4000;
    const float freq = 10000.0f;  // Well above cutoff
    std::vector<float> in(N);
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = std::sin(2.0f * static_cast<float>(M_PI) * freq * i / fs);
    }

    std::vector<float> out(N);
    BiquadState state;
    neon_biquad_f32(out.data(), in.data(), N, coeffs, state);

    // Measure RMS of output in steady state (skip transient)
    float rms = 0.0f;
    std::size_t start = N / 2;
    for (std::size_t i = start; i < N; ++i) {
        rms += out[i] * out[i];
    }
    rms = std::sqrt(rms / (N - start));

    // Should be heavily attenuated (input RMS is ~0.707)
    EXPECT_LT(rms, 0.01f);
}

TEST(NeonBiquadTest, BandpassCenterFreq) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 1000.0f;
    float Q = 5.0f;
    BiquadCoeffs coeffs = neon_biquad_bandpass(fc, fs, Q);

    const std::size_t N = 8000;

    // Signal at center frequency should pass
    std::vector<float> in_center(N);
    for (std::size_t i = 0; i < N; ++i) {
        in_center[i] = std::sin(2.0f * static_cast<float>(M_PI) * fc * i / fs);
    }

    std::vector<float> out_center(N);
    BiquadState state1;
    neon_biquad_f32(out_center.data(), in_center.data(), N, coeffs, state1);

    // Signal far from center should be attenuated
    std::vector<float> in_off(N);
    float off_freq = 10000.0f;
    for (std::size_t i = 0; i < N; ++i) {
        in_off[i] = std::sin(2.0f * static_cast<float>(M_PI) * off_freq * i / fs);
    }

    std::vector<float> out_off(N);
    BiquadState state2;
    neon_biquad_f32(out_off.data(), in_off.data(), N, coeffs, state2);

    // Measure RMS in steady state
    std::size_t start = N / 2;
    float rms_center = 0.0f, rms_off = 0.0f;
    for (std::size_t i = start; i < N; ++i) {
        rms_center += out_center[i] * out_center[i];
        rms_off += out_off[i] * out_off[i];
    }
    rms_center = std::sqrt(rms_center / (N - start));
    rms_off = std::sqrt(rms_off / (N - start));

    // Center frequency should have much higher output than off-frequency
    EXPECT_GT(rms_center, rms_off * 5.0f);
}

TEST(NeonBiquadTest, NotchRemovesTargetFreq) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 1000.0f;
    BiquadCoeffs coeffs = neon_biquad_notch(fc, fs, 10.0f);

    const std::size_t N = 8000;
    std::vector<float> in(N);
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = std::sin(2.0f * static_cast<float>(M_PI) * fc * i / fs);
    }

    std::vector<float> out(N);
    BiquadState state;
    neon_biquad_f32(out.data(), in.data(), N, coeffs, state);

    // RMS at notch frequency should be very small
    float rms = 0.0f;
    std::size_t start = N / 2;
    for (std::size_t i = start; i < N; ++i) {
        rms += out[i] * out[i];
    }
    rms = std::sqrt(rms / (N - start));

    EXPECT_LT(rms, 0.05f);
}

TEST(NeonBiquadTest, CascadeHigherOrder) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // Cascade of 2 lowpass sections = 4th order Butterworth-like
    float fs = 48000.0f;
    float fc = 1000.0f;
    BiquadCoeffs coeffs[2];
    BiquadState states[2];
    coeffs[0] = neon_biquad_lowpass(fc, fs);
    coeffs[1] = neon_biquad_lowpass(fc, fs);

    const std::size_t N = 2000;
    std::vector<float> in(N, 1.0f);
    std::vector<float> out(N);

    neon_biquad_cascade_f32(out.data(), in.data(), N, coeffs, states, 2);

    // DC gain should still be 1.0
    EXPECT_NEAR(out[N - 1], 1.0f, 0.01f);
}

TEST(NeonBiquadTest, ImpulseResponse) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 5000.0f;
    BiquadCoeffs coeffs = neon_biquad_lowpass(fc, fs);

    const std::size_t N = 100;
    std::vector<float> in(N, 0.0f);
    in[0] = 1.0f;  // Impulse

    std::vector<float> out(N);
    BiquadState state;
    neon_biquad_f32(out.data(), in.data(), N, coeffs, state);

    // First output should be b0
    EXPECT_NEAR(out[0], coeffs.b0, 1e-6f);

    // Response should decay
    float max_tail = 0.0f;
    for (std::size_t i = 50; i < N; ++i) {
        max_tail = std::max(max_tail, std::fabs(out[i]));
    }
    EXPECT_LT(max_tail, 0.01f);
}

TEST(NeonBiquadTest, BlockProcessingConsistency) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 2000.0f;
    BiquadCoeffs coeffs = neon_biquad_lowpass(fc, fs);

    const std::size_t N = 200;
    std::vector<float> in(N);
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = std::sin(0.3f * i) + 0.5f * std::cos(1.7f * i);
    }

    // Process all at once
    std::vector<float> out_all(N);
    BiquadState state_all;
    neon_biquad_f32(out_all.data(), in.data(), N, coeffs, state_all);

    // Process in two blocks
    std::vector<float> out_blocks(N);
    BiquadState state_blocks;
    std::size_t block1 = 73;
    neon_biquad_f32(out_blocks.data(), in.data(), block1, coeffs, state_blocks);
    neon_biquad_f32(out_blocks.data() + block1, in.data() + block1,
                    N - block1, coeffs, state_blocks);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_blocks[i], out_all[i], 1e-6f) << "at index " << i;
    }
}

TEST(NeonBiquadTest, InPlaceProcessing) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 2000.0f;
    BiquadCoeffs coeffs = neon_biquad_lowpass(fc, fs);

    const std::size_t N = 100;
    std::vector<float> in(N);
    for (std::size_t i = 0; i < N; ++i) in[i] = std::sin(0.5f * i);

    // Out-of-place
    std::vector<float> out_separate(N);
    BiquadState state1;
    neon_biquad_f32(out_separate.data(), in.data(), N, coeffs, state1);

    // In-place
    std::vector<float> buf(in);
    BiquadState state2;
    neon_biquad_f32(buf.data(), buf.data(), N, coeffs, state2);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(buf[i], out_separate[i], 1e-6f) << "at index " << i;
    }
}

TEST(NeonBiquadTest, X4MatchesScalarPerChannel) {
    const std::size_t n = 4096;
    float fs = 48000.0f;
    BiquadCoeffs co[4] = {
        neon_biquad_lowpass(1000.0f, fs, 0.7f),
        neon_biquad_highpass(2000.0f, fs, 0.9f),
        neon_biquad_bandpass(3000.0f, fs, 1.5f),
        neon_biquad_notch(5000.0f, fs, 2.0f),
    };
    std::vector<float> chin[4], ref[4], inter(4 * n), outer(4 * n);
    for (int c = 0; c < 4; ++c) { chin[c].resize(n); ref[c].resize(n); }
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 4; ++c) {
            float v = std::sin(0.01f * i * (c + 1)) + 0.3f * std::cos(0.13f * i);
            chin[c][i] = v; inter[4 * i + c] = v;
        }
    for (int c = 0; c < 4; ++c) { BiquadState st{}; neon_biquad_f32(ref[c].data(), chin[c].data(), n, co[c], st); }

    BiquadState st4[4] = {};
    neon_biquad_x4_f32(outer.data(), inter.data(), n, co, st4);

    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 4; ++c)
            EXPECT_NEAR(outer[4 * i + c], ref[c][i], 1e-4f) << "ch " << c << " idx " << i;
}

TEST(NeonBiquadTest, EigenWrapper) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    float fs = 48000.0f;
    float fc = 1000.0f;
    BiquadCoeffs coeffs = neon_biquad_lowpass(fc, fs);

    Eigen::VectorXf in = Eigen::VectorXf::Ones(500);
    Eigen::VectorXf result = neon_biquad(in, coeffs);

    ASSERT_EQ(result.size(), 500);
    // DC should pass through
    EXPECT_NEAR(result[499], 1.0f, 0.01f);
}
