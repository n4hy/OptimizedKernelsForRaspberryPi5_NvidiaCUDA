/**
 * @file test_cuda_radar.cpp
 * @brief Tests for CUDA radar signal processing operations
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <vector>

#ifdef OPTMATH_USE_CUDA
#include "optmath/cuda_backend.hpp"
#endif

constexpr float TOLERANCE = 1e-4f;
constexpr float PI = 3.14159265358979323846f;

// ============================================================================
// Test Fixture
// ============================================================================

class CudaRadarTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef OPTMATH_USE_CUDA
        if (!optmath::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        optmath::cuda::init();
#else
        GTEST_SKIP() << "CUDA not enabled in build";
#endif
    }

    void TearDown() override {
#ifdef OPTMATH_USE_CUDA
        if (optmath::cuda::is_available()) {
            optmath::cuda::synchronize();
        }
#endif
    }
};

#ifdef OPTMATH_USE_CUDA

// ============================================================================
// Window Function Tests
// ============================================================================

TEST_F(CudaRadarTest, HammingWindow) {
    const int n = 256;

    Eigen::VectorXf window = optmath::cuda::cuda_generate_window(n, optmath::cuda::WindowType::HAMMING);

    EXPECT_EQ(window.size(), n);

    // Check Hamming window properties
    // First and last should be approximately 0.08
    EXPECT_NEAR(window(0), 0.08f, 0.01f);
    EXPECT_NEAR(window(n-1), 0.08f, 0.01f);

    // Center should be approximately 1.0
    EXPECT_NEAR(window(n/2), 1.0f, 0.01f);
}

TEST_F(CudaRadarTest, HanningWindow) {
    const int n = 256;

    Eigen::VectorXf window = optmath::cuda::cuda_generate_window(n, optmath::cuda::WindowType::HANNING);

    EXPECT_EQ(window.size(), n);

    // Hanning window starts and ends at 0
    EXPECT_NEAR(window(0), 0.0f, 0.01f);

    // Center should be approximately 1.0
    EXPECT_NEAR(window(n/2), 1.0f, 0.01f);
}

TEST_F(CudaRadarTest, BlackmanWindow) {
    const int n = 256;

    Eigen::VectorXf window = optmath::cuda::cuda_generate_window(n, optmath::cuda::WindowType::BLACKMAN);

    EXPECT_EQ(window.size(), n);

    // Blackman window has very low sidelobes, starts/ends near 0
    EXPECT_NEAR(window(0), 0.0f, 0.01f);

    // Center should be approximately 1.0
    EXPECT_NEAR(window(n/2), 1.0f, 0.05f);
}

TEST_F(CudaRadarTest, BlackmanHarrisWindow) {
    const int n = 256;

    Eigen::VectorXf window = optmath::cuda::cuda_generate_window(n, optmath::cuda::WindowType::BLACKMAN_HARRIS);

    EXPECT_EQ(window.size(), n);

    // Blackman-Harris has even lower sidelobes
    EXPECT_NEAR(window(0), 0.0f, 0.001f);
    EXPECT_NEAR(window(n/2), 1.0f, 0.05f);
}

TEST_F(CudaRadarTest, ApplyWindow) {
    const int n = 256;
    Eigen::VectorXcf signal = Eigen::VectorXcf::Ones(n);
    Eigen::VectorXf window = optmath::cuda::cuda_generate_window(n, optmath::cuda::WindowType::HAMMING);

    // cuda_apply_window modifies in-place
    optmath::cuda::cuda_apply_window(signal, window);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(signal(i).real(), window(i), TOLERANCE);
        EXPECT_NEAR(signal(i).imag(), 0.0f, TOLERANCE);
    }
}

// ============================================================================
// CAF (Cross-Ambiguity Function) Tests
// ============================================================================

TEST_F(CudaRadarTest, CAFZeroDoppler) {
    // Test CAF with a simple delayed copy (zero Doppler shift)
    const int n = 256;
    const int delay = 10;

    // Reference signal (chirp-like)
    Eigen::VectorXcf ref(n);
    for (int i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / n;
        ref(i) = std::complex<float>(std::cos(2*PI*t*10), std::sin(2*PI*t*10));
    }

    // Surveillance = delayed reference
    Eigen::VectorXcf surv = Eigen::VectorXcf::Zero(n);
    for (int i = delay; i < n; ++i) {
        surv(i) = ref(i - delay);
    }

    const int n_doppler = 64;
    const int max_range = 128;
    float doppler_step = 10.0f;  // Hz

    Eigen::MatrixXf caf = optmath::cuda::cuda_caf(ref, surv, n_doppler, max_range, doppler_step);

    EXPECT_EQ(caf.rows(), n_doppler);
    EXPECT_EQ(caf.cols(), max_range);

    // Peak should be at zero Doppler, delay range bin
    int peak_doppler = -1, peak_range = -1;
    float peak_val = -std::numeric_limits<float>::infinity();
    for (int d = 0; d < n_doppler; ++d) {
        for (int r = 0; r < max_range; ++r) {
            if (caf(d, r) > peak_val) {
                peak_val = caf(d, r);
                peak_doppler = d;
                peak_range = r;
            }
        }
    }

    // Zero Doppler is at center
    EXPECT_NEAR(peak_doppler, n_doppler / 2, 2);
    EXPECT_NEAR(peak_range, delay, 2);
}

TEST_F(CudaRadarTest, CAFWithDoppler) {
    // Test CAF with Doppler shift
    const int n = 512;
    const float fs = 1000.0f;  // Sample rate
    const float doppler_hz = 25.0f;

    // Reference signal
    Eigen::VectorXcf ref(n);
    for (int i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / fs;
        ref(i) = std::complex<float>(std::cos(2*PI*100*t), std::sin(2*PI*100*t));
    }

    // Surveillance = Doppler-shifted reference
    Eigen::VectorXcf surv(n);
    for (int i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / fs;
        // Apply Doppler shift
        std::complex<float> shift(std::cos(2*PI*doppler_hz*t), std::sin(2*PI*doppler_hz*t));
        surv(i) = ref(i) * shift;
    }

    const int n_doppler = 128;
    const int max_range = 64;
    float doppler_step = 1.0f;  // Hz

    Eigen::MatrixXf caf = optmath::cuda::cuda_caf(ref, surv, n_doppler, max_range, doppler_step);

    // Find peak
    int peak_doppler = -1;
    float peak_val = -std::numeric_limits<float>::infinity();
    for (int d = 0; d < n_doppler; ++d) {
        for (int r = 0; r < max_range; ++r) {
            if (caf(d, r) > peak_val) {
                peak_val = caf(d, r);
                peak_doppler = d;
            }
        }
    }

    // Expected Doppler bin
    int expected_doppler = n_doppler / 2 + static_cast<int>(doppler_hz / doppler_step);

    EXPECT_NEAR(peak_doppler, expected_doppler, 3);
}

// ============================================================================
// CFAR Detector Tests
// ============================================================================

TEST_F(CudaRadarTest, CFAR2DNoiseOnly) {
    // Test CFAR with noise-only input (should have few detections)
    const int n_doppler = 64;
    const int n_range = 128;

    // Generate exponential noise (radar power data)
    Eigen::MatrixXf data(n_doppler, n_range);
    for (int i = 0; i < n_doppler; ++i) {
        for (int j = 0; j < n_range; ++j) {
            data(i, j) = -std::log(static_cast<float>(rand()) / RAND_MAX + 1e-10f);  // Exponential(1)
        }
    }

    const int guard = 2;
    const int ref_cells = 4;
    const float pfa = 1e-4f;  // Very low false alarm rate

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> detections =
        optmath::cuda::cuda_cfar_2d(data, guard, ref_cells, pfa);

    EXPECT_EQ(detections.rows(), n_doppler);
    EXPECT_EQ(detections.cols(), n_range);

    // Count detections - should be very few with low Pfa
    int n_detections = 0;
    for (int i = 0; i < n_doppler; ++i) {
        for (int j = 0; j < n_range; ++j) {
            if (detections(i, j) > 0) n_detections++;
        }
    }

    // With Pfa = 1e-4, expect ~0.01% false alarms = ~0.8 detections on average
    // Allow up to 10 for statistical variance
    EXPECT_LT(n_detections, 10);
}

TEST_F(CudaRadarTest, CFAR2DWithTarget) {
    // Test CFAR with a strong target embedded in noise
    const int n_doppler = 64;
    const int n_range = 128;
    const int target_doppler = 32;
    const int target_range = 64;
    const float target_snr = 20.0f;  // dB

    // Generate exponential noise
    Eigen::MatrixXf data(n_doppler, n_range);
    for (int i = 0; i < n_doppler; ++i) {
        for (int j = 0; j < n_range; ++j) {
            data(i, j) = -std::log(static_cast<float>(rand()) / RAND_MAX + 1e-10f);
        }
    }

    // Add target
    float target_power = std::pow(10.0f, target_snr / 10.0f);
    data(target_doppler, target_range) = target_power;

    const int guard = 2;
    const int ref_cells = 4;
    const float pfa = 1e-3f;

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> detections =
        optmath::cuda::cuda_cfar_2d(data, guard, ref_cells, pfa);

    // Target should be detected
    EXPECT_EQ(detections(target_doppler, target_range), 1);
}

TEST_F(CudaRadarTest, CFAR1D) {
    const int n = 512;
    const int target_idx = 256;
    const float target_snr = 15.0f;  // dB

    // Generate exponential noise
    Eigen::VectorXf data(n);
    for (int i = 0; i < n; ++i) {
        data(i) = -std::log(static_cast<float>(rand()) / RAND_MAX + 1e-10f);
    }

    // Add target
    data(target_idx) = std::pow(10.0f, target_snr / 10.0f);

    const int guard = 2;
    const int ref_cells = 8;
    const float pfa = 1e-3f;

    Eigen::VectorXi detections = optmath::cuda::cuda_cfar_1d(data, guard, ref_cells, pfa);

    EXPECT_EQ(detections.size(), n);
    EXPECT_EQ(detections(target_idx), 1);
}

// ============================================================================
// Beamforming Tests
// ============================================================================

TEST_F(CudaRadarTest, SteeringVectorULA) {
    const int n_elements = 4;
    const float d_lambda = 0.5f;  // Half wavelength spacing
    const float angle_rad = 0.0f;  // Broadside

    Eigen::VectorXcf sv = optmath::cuda::cuda_steering_vector_ula(n_elements, d_lambda, angle_rad);

    EXPECT_EQ(sv.size(), n_elements);

    // At broadside, all elements should have the same phase (all ones)
    for (int i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(sv(i).real(), 1.0f, 0.01f);
        EXPECT_NEAR(sv(i).imag(), 0.0f, 0.01f);
    }
}

TEST_F(CudaRadarTest, SteeringVectorULA30Deg) {
    const int n_elements = 4;
    const float d_lambda = 0.5f;
    const float angle_rad = PI / 6.0f;  // 30 degrees

    Eigen::VectorXcf sv = optmath::cuda::cuda_steering_vector_ula(n_elements, d_lambda, angle_rad);

    // Check phase progression
    float expected_phase_step = 2.0f * PI * d_lambda * std::sin(angle_rad);
    for (int i = 1; i < n_elements; ++i) {
        float phase_diff = std::arg(sv(i)) - std::arg(sv(i-1));
        // Handle phase wrapping
        while (phase_diff > PI) phase_diff -= 2*PI;
        while (phase_diff < -PI) phase_diff += 2*PI;
        EXPECT_NEAR(phase_diff, -expected_phase_step, 0.1f);
    }
}

TEST_F(CudaRadarTest, BartlettBeamformer) {
    const int n_elements = 4;
    const int n_angles = 181;  // -90 to +90 degrees
    const float d_lambda = 0.5f;

    // Create array response from broadside (0 degrees)
    Eigen::VectorXcf sv_true = optmath::cuda::cuda_steering_vector_ula(n_elements, d_lambda, 0.0f);

    // Add some noise
    Eigen::VectorXcf array_data = sv_true;
    for (int i = 0; i < n_elements; ++i) {
        array_data(i) += std::complex<float>(0.1f * (rand() / (float)RAND_MAX - 0.5f),
                                               0.1f * (rand() / (float)RAND_MAX - 0.5f));
    }

    Eigen::VectorXf spectrum = optmath::cuda::cuda_bartlett_spectrum(array_data, d_lambda, n_angles);

    EXPECT_EQ(spectrum.size(), n_angles);

    // Find peak
    int peak_idx = 0;
    float peak_val = spectrum(0);
    for (int i = 1; i < n_angles; ++i) {
        if (spectrum(i) > peak_val) {
            peak_val = spectrum(i);
            peak_idx = i;
        }
    }

    // Peak should be at center (broadside, 0 degrees)
    int center_idx = n_angles / 2;
    EXPECT_NEAR(peak_idx, center_idx, 5);  // Within 5 degrees
}

// ============================================================================
// NLMS Filter Tests
// ============================================================================

TEST_F(CudaRadarTest, NLMSConvergence) {
    const int n = 1024;
    const int filter_length = 32;
    const float mu = 0.5f;
    const float eps = 1e-6f;

    // Reference signal
    Eigen::VectorXf ref(n);
    for (int i = 0; i < n; ++i) {
        ref(i) = std::sin(2.0f * PI * 0.1f * i) + 0.5f * std::sin(2.0f * PI * 0.05f * i);
    }

    // Surveillance = scaled reference + noise
    Eigen::VectorXf surv(n);
    for (int i = 0; i < n; ++i) {
        surv(i) = 2.0f * ref(i) + 0.1f * (rand() / (float)RAND_MAX - 0.5f);
    }

    Eigen::VectorXf output(n);
    Eigen::VectorXf weights(filter_length);

    optmath::cuda::cuda_nlms_filter(ref.data(), surv.data(), output.data(),
                                     weights.data(), n, filter_length, mu, eps);

    // After convergence, output should have lower power than input
    float input_power = surv.squaredNorm();
    float output_power = output.squaredNorm();

    // Should achieve at least 10 dB reduction
    float reduction_db = 10.0f * std::log10(input_power / (output_power + 1e-10f));
    EXPECT_GT(reduction_db, 10.0f);
}

// ============================================================================
// Multi-GPU Test (if available)
// ============================================================================

TEST_F(CudaRadarTest, DeviceInfo) {
    auto info = optmath::cuda::get_device_info();

    EXPECT_GT(info.name.length(), 0u);
    EXPECT_GT(info.compute_capability_major, 0);
    EXPECT_GT(info.total_memory, 0u);
    EXPECT_GT(info.multiprocessor_count, 0);

    std::cout << "CUDA Device: " << info.name << std::endl;
    std::cout << "  Compute: " << info.compute_capability_major << "." << info.compute_capability_minor << std::endl;
    std::cout << "  Memory: " << info.total_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << info.multiprocessor_count << std::endl;
    std::cout << "  Tensor Cores: " << (info.tensor_cores ? "Yes" : "No") << std::endl;
}

TEST_F(CudaRadarTest, DeviceCount) {
    int count = optmath::cuda::get_device_count();
    EXPECT_GE(count, 1);

    std::cout << "Found " << count << " CUDA device(s)" << std::endl;
}

#endif // OPTMATH_USE_CUDA

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
