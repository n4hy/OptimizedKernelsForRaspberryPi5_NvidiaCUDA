#include <gtest/gtest.h>
#include <optmath/radar_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>

using namespace optmath::radar;

TEST(RadarCFARTest, CA_CFAR_SingleTarget) {
    // Create data with background noise and a single target
    size_t N = 128;
    float noise_level = 1.0f;
    float target_snr = 20.0f;  // 20 dB above noise

    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0, noise_level);

    Eigen::VectorXf data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::abs(noise(gen));  // Power-like values (positive)
    }

    // Add target at center
    size_t target_idx = N / 2;
    data[target_idx] = noise_level * std::pow(10.0f, target_snr / 10.0f);

    // Run CFAR
    size_t guard_cells = 2;
    size_t reference_cells = 8;
    float pfa_factor = 10.0f;  // High threshold

    auto detections = cfar_ca(data, guard_cells, reference_cells, pfa_factor);

    ASSERT_EQ(detections.size(), N);

    // Target should be detected
    EXPECT_EQ(detections[target_idx], 1);

    // Count total detections (should be low due to high threshold)
    int total_detections = detections.sum();
    EXPECT_GE(total_detections, 1);  // At least the target
    EXPECT_LE(total_detections, 5);  // Not too many false alarms
}

TEST(RadarCFARTest, CA_CFAR_NoTarget) {
    // Pure noise - should have few false alarms with proper threshold
    size_t N = 256;
    float noise_level = 1.0f;

    std::mt19937 gen(123);
    std::exponential_distribution<float> noise(1.0f / noise_level);

    Eigen::VectorXf data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = noise(gen);
    }

    size_t guard_cells = 2;
    size_t reference_cells = 16;
    float pfa_factor = 20.0f;  // Very high threshold for low Pfa

    auto detections = cfar_ca(data, guard_cells, reference_cells, pfa_factor);

    // Very few false alarms expected
    int total_detections = detections.sum();
    float pfa = static_cast<float>(total_detections) / N;
    EXPECT_LT(pfa, 0.1f);  // Less than 10% false alarm rate
}

TEST(RadarCFARTest, CA_CFAR_EdgeHandling) {
    // Test that edges are handled correctly
    size_t N = 64;
    Eigen::VectorXf data = Eigen::VectorXf::Ones(N);

    // Put targets at edges
    data[0] = 100.0f;
    data[N-1] = 100.0f;

    size_t guard_cells = 2;
    size_t reference_cells = 8;
    float pfa_factor = 2.0f;

    auto detections = cfar_ca(data, guard_cells, reference_cells, pfa_factor);

    // Edge targets should be detected (or gracefully handled)
    EXPECT_EQ(detections[0], 1);
    EXPECT_EQ(detections[N-1], 1);
}

TEST(RadarCFARTest, CFAR_2D_SingleTarget) {
    size_t n_doppler = 32;
    size_t n_range = 64;

    // Background noise
    Eigen::MatrixXf data = Eigen::MatrixXf::Random(n_doppler, n_range).cwiseAbs();

    // Add target
    size_t target_d = n_doppler / 2;
    size_t target_r = n_range / 2;
    data(target_d, target_r) = 50.0f;  // Strong target

    size_t guard_range = 2, guard_doppler = 2;
    size_t ref_range = 4, ref_doppler = 4;
    float pfa_factor = 10.0f;

    auto detections = cfar_2d(data, guard_range, guard_doppler, ref_range, ref_doppler, pfa_factor);

    ASSERT_EQ(detections.rows(), n_doppler);
    ASSERT_EQ(detections.cols(), n_range);

    // Target should be detected
    EXPECT_EQ(detections(target_d, target_r), 1);
}

TEST(RadarCFARTest, CFAR_2D_MultipleTargets) {
    size_t n_doppler = 64;
    size_t n_range = 64;

    // Low background
    Eigen::MatrixXf data = Eigen::MatrixXf::Constant(n_doppler, n_range, 0.1f);

    // Add multiple targets
    std::vector<std::pair<size_t, size_t>> targets = {
        {10, 20}, {30, 40}, {50, 50}
    };

    for (auto& [d, r] : targets) {
        data(d, r) = 10.0f;
    }

    size_t guard_range = 1, guard_doppler = 1;
    size_t ref_range = 3, ref_doppler = 3;
    float pfa_factor = 5.0f;

    auto detections = cfar_2d(data, guard_range, guard_doppler, ref_range, ref_doppler, pfa_factor);

    // All targets should be detected
    for (auto& [d, r] : targets) {
        EXPECT_EQ(detections(d, r), 1) << "Target at (" << d << ", " << r << ") not detected";
    }
}

TEST(RadarCFARTest, NLMSFilterZeroLength) {
    // BUG-11: filter_length==0 previously caused SIZE_MAX wraparound loop
    size_t N = 64;
    Eigen::VectorXf input = Eigen::VectorXf::Random(N);
    Eigen::VectorXf reference = Eigen::VectorXf::Random(N);

    // Should return empty vector without crashing
    Eigen::VectorXf output = nlms_filter(input, reference, 0, 0.1f, 1e-6f);
    // The Eigen wrapper creates a VectorXf(N) then calls the raw function which returns early,
    // so output may be uninitialized but should not crash
    EXPECT_EQ(output.size(), N);
}

TEST(RadarCFARTest, NLMSFilter) {
    size_t N = 256;
    size_t filter_length = 16;
    float mu = 0.1f;
    float eps = 1e-6f;

    // Create reference signal (clutter)
    Eigen::VectorXf reference(N);
    for (size_t i = 0; i < N; ++i) {
        reference[i] = std::sin(0.1f * i);
    }

    // Input = clutter + target
    Eigen::VectorXf input = reference;
    // Add a target (impulse)
    input[N/2] += 5.0f;

    Eigen::VectorXf output = nlms_filter(input, reference, filter_length, mu, eps);

    ASSERT_EQ(output.size(), N);

    // After adaptation, clutter should be suppressed
    // The target should remain visible
    // Check that output at target location is significant
    size_t target_region_start = N/2 - 5;
    size_t target_region_end = N/2 + 5;

    float target_region_max = output.segment(target_region_start, target_region_end - target_region_start).cwiseAbs().maxCoeff();
    float background_mean = (output.head(target_region_start).cwiseAbs().sum() +
                            output.tail(N - target_region_end).cwiseAbs().sum()) /
                           (target_region_start + N - target_region_end);

    // Target region should be stronger than background
    EXPECT_GT(target_region_max, 2.0f * background_mean);
}

TEST(RadarCFARTest, MTIFilter) {
    // Test 2-pulse MTI canceller
    size_t n_pulses = 32;
    size_t n_range = 64;

    // Create stationary clutter (same across pulses)
    Eigen::MatrixXf data = Eigen::MatrixXf::Ones(n_pulses, n_range) * 10.0f;

    // Add moving target (varies across pulses) - larger amplitude
    for (size_t p = 0; p < n_pulses; ++p) {
        data(p, n_range/2) += 5.0f * std::sin(0.8f * p);
    }

    // 2-pulse canceller: [1, -1]
    Eigen::VectorXf coeffs(2);
    coeffs << 1.0f, -1.0f;

    Eigen::MatrixXf output = mti_filter(data, coeffs);

    ASSERT_EQ(output.rows(), n_pulses - 1);
    ASSERT_EQ(output.cols(), n_range);

    // Stationary clutter should be cancelled (output near zero away from target)
    float clutter_residual = output.col(10).cwiseAbs().mean();  // Away from target
    EXPECT_LT(clutter_residual, 1.0f);  // Near zero after differencing

    // Moving target should remain significant
    float target_signal = output.col(n_range/2).cwiseAbs().mean();
    // Target signal should exist (positive mean)
    EXPECT_GT(target_signal, 0.1f);
}

TEST(RadarCFARTest, BeamformingDelaySum) {
    size_t n_channels = 4;
    size_t n_samples = 128;

    // Create coherent signals with known delays
    Eigen::MatrixXf inputs(n_channels, n_samples);
    for (size_t ch = 0; ch < n_channels; ++ch) {
        for (size_t i = 0; i < n_samples; ++i) {
            // Sinusoid with phase offset per channel
            inputs(ch, i) = std::sin(0.2f * i - 0.5f * ch);
        }
    }

    // Delays to align signals
    Eigen::VectorXi delays(n_channels);
    delays << 0, 2, 5, 7;  // Sample delays

    // Unity weights
    Eigen::VectorXf weights = Eigen::VectorXf::Ones(n_channels);

    Eigen::VectorXf output = beamform_delay_sum(inputs, delays, weights);

    ASSERT_EQ(output.size(), n_samples);

    // Beamformed output should have higher power than individual channels
    float beamformed_power = output.squaredNorm() / n_samples;
    float single_channel_power = inputs.row(0).squaredNorm() / n_samples;

    EXPECT_GT(beamformed_power, single_channel_power);
}

TEST(RadarCFARTest, SteeringVectorULA) {
    size_t n_elements = 8;
    float d_lambda = 0.5f;  // Half-wavelength spacing
    float theta_rad = 0.0f;  // Broadside

    Eigen::VectorXcf steering = steering_vector_ula(n_elements, d_lambda, theta_rad);

    ASSERT_EQ(steering.size(), n_elements);

    // At broadside, all phases should be equal (exp(0) = 1 for all elements)
    for (size_t i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(steering[i].real(), 1.0f, 1e-5f);
        EXPECT_NEAR(steering[i].imag(), 0.0f, 1e-5f);
    }

    // Test non-zero steering angle
    float theta = 0.5f;  // ~30 degrees
    steering = steering_vector_ula(n_elements, d_lambda, theta);

    // Phases should progress linearly across elements
    float expected_phase_step = 2.0f * 3.14159f * d_lambda * std::sin(theta);
    for (size_t i = 1; i < n_elements; ++i) {
        float phase_diff = std::arg(steering[i]) - std::arg(steering[i-1]);
        // Normalize to [-pi, pi]
        while (phase_diff > 3.14159f) phase_diff -= 2.0f * 3.14159f;
        while (phase_diff < -3.14159f) phase_diff += 2.0f * 3.14159f;
        EXPECT_NEAR(phase_diff, expected_phase_step, 0.1f);
    }
}
