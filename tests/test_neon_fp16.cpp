// Tests for the NEON half-precision (FP16) kernels (Cortex-A76 FEAT_FP16).
#include <gtest/gtest.h>
#include <optmath/neon_fp16.hpp>

#include <vector>
#include <cmath>
#include <random>

using optmath::neon::neon_add_f16;
using optmath::neon::neon_mul_f16;
using optmath::neon::neon_relu_f16;
using optmath::neon::neon_dot_f16;
using optmath::neon::fp16_available;

namespace {
// fp16 has ~3 decimal digits; relative tolerance ~2^-10.
constexpr float kFp16Rel = 1.0f / 1024.0f;

std::vector<__fp16> to_f16(const std::vector<float>& v) {
    std::vector<__fp16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<__fp16>(v[i]);
    return out;
}
}  // namespace

TEST(NeonFp16Test, Available) {
    EXPECT_TRUE(fp16_available());
}

TEST(NeonFp16Test, AddMatchesScalar) {
    // Sizes that exercise both the 8-wide body and the scalar tail.
    for (size_t n : {1u, 3u, 8u, 15u, 64u, 100u}) {
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) { a[i] = 0.5f * i - 3.0f; b[i] = 2.0f - 0.25f * i; }
        auto fa = to_f16(a), fb = to_f16(b);
        std::vector<__fp16> out(n);
        neon_add_f16(out.data(), fa.data(), fb.data(), n);
        for (size_t i = 0; i < n; ++i) {
            // Reference must add the fp16-rounded inputs, matching the kernel.
            float expected = static_cast<float>(
                static_cast<__fp16>(static_cast<float>(fa[i]) + static_cast<float>(fb[i])));
            EXPECT_EQ(static_cast<float>(out[i]), expected) << "n=" << n << " i=" << i;
        }
    }
}

TEST(NeonFp16Test, MulMatchesScalar) {
    for (size_t n : {1u, 7u, 8u, 33u, 128u}) {
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) { a[i] = 0.1f * i + 0.5f; b[i] = 1.5f - 0.05f * i; }
        auto fa = to_f16(a), fb = to_f16(b);
        std::vector<__fp16> out(n);
        neon_mul_f16(out.data(), fa.data(), fb.data(), n);
        for (size_t i = 0; i < n; ++i) {
            // Reference multiplies the fp16-rounded inputs, matching the kernel.
            float expected = static_cast<float>(
                static_cast<__fp16>(static_cast<float>(fa[i]) * static_cast<float>(fb[i])));
            EXPECT_EQ(static_cast<float>(out[i]), expected) << "n=" << n << " i=" << i;
        }
    }
}

TEST(NeonFp16Test, ReluClampsNegatives) {
    std::vector<float> a = {-3.0f, -0.5f, 0.0f, 0.25f, 2.0f, -10.0f, 7.5f, 1.0f, -0.01f, 4.0f, -2.0f};
    auto fa = to_f16(a);
    neon_relu_f16(fa.data(), fa.size());
    for (size_t i = 0; i < a.size(); ++i) {
        float expected = a[i] > 0.0f ? a[i] : 0.0f;
        EXPECT_NEAR(static_cast<float>(fa[i]), expected, 1e-3f) << "i=" << i;
    }
}

TEST(NeonFp16Test, DotAccumulatesInFp32) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t n : {4u, 8u, 17u, 256u, 1000u}) {
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) { a[i] = dist(rng); b[i] = dist(rng); }
        auto fa = to_f16(a), fb = to_f16(b);

        // Reference: same fp16 rounding of inputs, exact fp32 accumulation.
        double ref = 0.0;
        for (size_t i = 0; i < n; ++i)
            ref += static_cast<double>(fa[i]) * static_cast<double>(fb[i]);

        float got = neon_dot_f16(fa.data(), fb.data(), n);
        // fp16 products rounded to fp16 then summed in fp32: tolerance scales
        // with n * per-term fp16 rounding.
        float tol = static_cast<float>(n) * kFp16Rel + 1e-3f;
        EXPECT_NEAR(got, static_cast<float>(ref), tol) << "n=" << n;
    }
}
