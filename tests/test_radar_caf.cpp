#include <gtest/gtest.h>
#include <optmath/radar_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <complex>

using namespace optmath::radar;

static void expect_approx_equal(const Eigen::VectorXf& a, const Eigen::VectorXf& b, float tol = 1e-4) {
    ASSERT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "at index " << i;
    }
}

TEST(RadarCAFTest, WindowGeneration) {
    size_t N = 256;

    // Test each window type
    std::vector<WindowType> types = {
        WindowType::RECTANGULAR,
        WindowType::HAMMING,
        WindowType::HANNING,
        WindowType::BLACKMAN,
        WindowType::BLACKMAN_HARRIS,
        WindowType::KAISER
    };

    for (auto type : types) {
        Eigen::VectorXf window = generate_window(N, type);

        ASSERT_EQ(window.size(), N);

        // All windows should have values in [0, 1] (approximately)
        EXPECT_GE(window.minCoeff(), -0.01f);
        EXPECT_LE(window.maxCoeff(), 1.01f);

        // Rectangular should be all 1s
        if (type == WindowType::RECTANGULAR) {
            EXPECT_NEAR(window.minCoeff(), 1.0f, 1e-5f);
            EXPECT_NEAR(window.maxCoeff(), 1.0f, 1e-5f);
        }

        // Other windows should have max near center
        if (type != WindowType::RECTANGULAR) {
            size_t max_idx;
            window.maxCoeff(&max_idx);
            EXPECT_NEAR(static_cast<float>(max_idx), N / 2.0f, N / 10.0f);
        }
    }
}

TEST(RadarCAFTest, WindowApplication) {
    size_t N = 128;
    Eigen::VectorXf data = Eigen::VectorXf::Ones(N);
    Eigen::VectorXf window = generate_window(N, WindowType::HAMMING);

    apply_window(data, window);

    // After applying window, data should equal window
    expect_approx_equal(data, window);
}

TEST(RadarCAFTest, CrossCorrelation) {
    // Create a known signal with a peak
    int N = 64;
    Eigen::VectorXf x = Eigen::VectorXf::Zero(N);
    Eigen::VectorXf y = Eigen::VectorXf::Zero(N);

    // Simple impulse test
    x(N/2) = 1.0f;
    y(N/2 + 5) = 1.0f;  // Shifted by 5

    Eigen::VectorXf corr = xcorr(x, y);

    EXPECT_EQ(corr.size(), 2*N - 1);

    // Peak should be at lag = 5
    Eigen::Index peak_idx;
    corr.maxCoeff(&peak_idx);
    EXPECT_EQ(peak_idx, N - 1 + 5);  // Center + offset
}

TEST(RadarCAFTest, ComplexCrossCorrelation) {
    int N = 64;
    Eigen::VectorXcf x = Eigen::VectorXcf::Zero(N);
    Eigen::VectorXcf y = Eigen::VectorXcf::Zero(N);

    // Complex impulse
    x(N/2) = std::complex<float>(1.0f, 0.0f);
    y(N/2 + 3) = std::complex<float>(1.0f, 0.0f);

    Eigen::VectorXcf corr = xcorr(x, y);

    EXPECT_EQ(corr.size(), 2*N - 1);

    // Find peak in magnitude
    Eigen::VectorXf mag = corr.array().abs();
    Eigen::Index peak_idx;
    mag.maxCoeff(&peak_idx);
    EXPECT_EQ(peak_idx, N - 1 + 3);
}

TEST(RadarCAFTest, CAFWithSimulatedTarget) {
    // Simulate a target at known range and Doppler
    // Note: CAF computes correlation between reference and surveillance
    // The peak should appear at the range/Doppler of the target

    size_t n_samples = 2048;
    float sample_rate = 1e6f;  // 1 MHz
    float target_delay_samples = 30;  // Range delay
    float target_doppler = 200.0f;    // 200 Hz Doppler

    const float two_pi = 6.28318530717958647693f;

    // Create reference signal (simple tone)
    Eigen::VectorXcf ref(n_samples);
    float carrier_freq = 10000.0f;
    for (size_t i = 0; i < n_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float phase = two_pi * carrier_freq * t;
        ref(i) = std::complex<float>(std::cos(phase), std::sin(phase));
    }

    // Create surveillance signal (delayed and Doppler-shifted reference)
    Eigen::VectorXcf surv(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        // Doppler shift
        float doppler_phase = two_pi * target_doppler * t;
        std::complex<float> doppler_shift(std::cos(doppler_phase), std::sin(doppler_phase));

        // Delayed reference
        int delay_idx = static_cast<int>(i) - static_cast<int>(target_delay_samples);
        if (delay_idx >= 0 && delay_idx < static_cast<int>(n_samples)) {
            surv(i) = ref(delay_idx) * doppler_shift;
        } else {
            surv(i) = std::complex<float>(0, 0);
        }
    }

    // Compute CAF
    size_t n_doppler = 41;
    size_t n_range = 60;
    float doppler_start = 0.0f;
    float doppler_step = 10.0f;  // 10 Hz steps

    Eigen::MatrixXf caf_mag = caf(ref, surv, n_doppler, doppler_start, doppler_step,
                                  sample_rate, n_range);

    ASSERT_EQ(caf_mag.rows(), n_doppler);
    ASSERT_EQ(caf_mag.cols(), n_range);

    // Find peak
    Eigen::Index max_doppler, max_range;
    caf_mag.maxCoeff(&max_doppler, &max_range);

    // Expected Doppler bin
    int expected_doppler_bin = static_cast<int>((target_doppler - doppler_start) / doppler_step);

    // Check peak location (with relaxed tolerance since this is a simplified simulation)
    // The CAF algorithm may have some offset due to discrete sampling and
    // frequency resolution limitations with the short simulation length
    EXPECT_NEAR(static_cast<float>(max_doppler), expected_doppler_bin, 10);
    EXPECT_NEAR(static_cast<float>(max_range), target_delay_samples, 5);
}

TEST(RadarCAFTest, CAFZeroDoppler) {
    // Test with zero Doppler (just cross-correlation)
    size_t n_samples = 512;
    float sample_rate = 1e6f;

    // Random reference
    Eigen::VectorXcf ref = Eigen::VectorXcf::Random(n_samples);

    // Surveillance = delayed reference (no Doppler)
    size_t delay = 20;
    Eigen::VectorXcf surv = Eigen::VectorXcf::Zero(n_samples);
    for (size_t i = delay; i < n_samples; ++i) {
        surv(i) = ref(i - delay);
    }

    // Single Doppler bin at 0 Hz
    Eigen::MatrixXf caf_mag = caf(ref, surv, 1, 0.0f, 1.0f, sample_rate, 50);

    // Peak should be at range = delay
    Eigen::Index max_col;
    caf_mag.row(0).maxCoeff(&max_col);
    EXPECT_NEAR(static_cast<float>(max_col), delay, 2);
}
