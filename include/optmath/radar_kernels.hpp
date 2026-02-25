#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <Eigen/Dense>

namespace optmath {
namespace radar {

// =========================================================================
// Window Function Types
// =========================================================================

enum class WindowType {
    RECTANGULAR,
    HAMMING,
    HANNING,
    BLACKMAN,
    BLACKMAN_HARRIS,
    KAISER
};

// =========================================================================
// Window Functions
// =========================================================================

/**
 * @brief Generate a window function of specified type
 * @param window Output array of size n
 * @param n Window length
 * @param type Type of window function
 * @param beta Kaiser window parameter (only used for KAISER type)
 */
void generate_window_f32(float* window, std::size_t n, WindowType type, float beta = 5.0f);

/**
 * @brief Apply window function in-place to data
 */
void apply_window_f32(float* data, const float* window, std::size_t n);

/**
 * @brief Apply window to complex data (separate real/imag arrays)
 */
void apply_window_complex_f32(float* data_re, float* data_im,
                               const float* window, std::size_t n);

// Eigen wrappers
Eigen::VectorXf generate_window(std::size_t n, WindowType type, float beta = 5.0f);
void apply_window(Eigen::VectorXf& data, const Eigen::VectorXf& window);
void apply_window(Eigen::VectorXcf& data, const Eigen::VectorXf& window);

// =========================================================================
// Cross-Correlation
// =========================================================================

/**
 * @brief Real-valued cross-correlation
 * @param out Output array of size nx + ny - 1 (full correlation)
 * @param x First signal
 * @param nx Length of x
 * @param y Second signal
 * @param ny Length of y
 */
void xcorr_f32(float* out, const float* x, std::size_t nx,
               const float* y, std::size_t ny);

/**
 * @brief Complex cross-correlation (separate real/imag)
 * @param out_re, out_im Output arrays of size nx + ny - 1
 */
void xcorr_complex_f32(float* out_re, float* out_im,
                       const float* x_re, const float* x_im, std::size_t nx,
                       const float* y_re, const float* y_im, std::size_t ny);

// Eigen wrappers
Eigen::VectorXf xcorr(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
Eigen::VectorXcf xcorr(const Eigen::VectorXcf& x, const Eigen::VectorXcf& y);

// =========================================================================
// Cross-Ambiguity Function (CAF)
// =========================================================================

/**
 * @brief Compute Cross-Ambiguity Function for passive radar
 *
 * The CAF measures correlation between reference and surveillance signals
 * across multiple Doppler shifts and range delays.
 *
 * @param out_mag Output magnitude array [n_doppler_bins x n_range_bins]
 * @param ref_re, ref_im Reference signal (transmitter)
 * @param surv_re, surv_im Surveillance signal (receiver)
 * @param n_samples Number of samples in signals
 * @param n_doppler_bins Number of Doppler frequency bins
 * @param doppler_start Starting Doppler frequency (Hz)
 * @param doppler_step Doppler step size (Hz)
 * @param sample_rate Sample rate (Hz)
 * @param n_range_bins Number of range (delay) bins
 */
void caf_f32(float* out_mag,
             const float* ref_re, const float* ref_im,
             const float* surv_re, const float* surv_im,
             std::size_t n_samples,
             std::size_t n_doppler_bins,
             float doppler_start, float doppler_step,
             float sample_rate,
             std::size_t n_range_bins);

// Eigen wrapper
Eigen::MatrixXf caf(const Eigen::VectorXcf& ref,
                    const Eigen::VectorXcf& surv,
                    std::size_t n_doppler_bins,
                    float doppler_start, float doppler_step,
                    float sample_rate,
                    std::size_t n_range_bins);

// =========================================================================
// CFAR Detection
// =========================================================================

/**
 * @brief 1D Cell-Averaging CFAR detector
 *
 * @param detections Output binary detection mask (1 = detection, 0 = no detection)
 * @param threshold Output threshold values for each cell
 * @param input Input power/magnitude data
 * @param n Number of input samples
 * @param guard_cells Number of guard cells on each side
 * @param reference_cells Number of reference/training cells on each side
 * @param pfa_factor Scale factor for threshold (related to Pfa)
 */
void cfar_ca_f32(std::uint8_t* detections, float* threshold,
                 const float* input, std::size_t n,
                 std::size_t guard_cells, std::size_t reference_cells,
                 float pfa_factor);

/**
 * @brief 2D Cell-Averaging CFAR detector for range-Doppler maps
 *
 * @param detections Output binary detection mask [n_doppler x n_range]
 * @param input Input power array [n_doppler x n_range] (row-major)
 * @param n_doppler Number of Doppler bins
 * @param n_range Number of range bins
 * @param guard_range Guard cells in range dimension
 * @param guard_doppler Guard cells in Doppler dimension
 * @param ref_range Reference cells in range dimension
 * @param ref_doppler Reference cells in Doppler dimension
 * @param pfa_factor Threshold scale factor
 */
void cfar_2d_f32(std::uint8_t* detections,
                 const float* input,
                 std::size_t n_doppler, std::size_t n_range,
                 std::size_t guard_range, std::size_t guard_doppler,
                 std::size_t ref_range, std::size_t ref_doppler,
                 float pfa_factor);

/**
 * @brief OS-CFAR (Ordered Statistic) 1D detector
 * More robust to clutter edges than CA-CFAR
 */
void cfar_os_f32(std::uint8_t* detections, float* threshold,
                 const float* input, std::size_t n,
                 std::size_t guard_cells, std::size_t reference_cells,
                 std::size_t k_select,  // Which ordered statistic to use
                 float pfa_factor);

// Eigen wrappers
Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 1> cfar_ca(
    const Eigen::VectorXf& input,
    std::size_t guard_cells, std::size_t reference_cells,
    float pfa_factor);

Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> cfar_2d(
    const Eigen::MatrixXf& input,
    std::size_t guard_range, std::size_t guard_doppler,
    std::size_t ref_range, std::size_t ref_doppler,
    float pfa_factor);

// =========================================================================
// Clutter Filtering
// =========================================================================

/**
 * @brief Normalized LMS adaptive filter for clutter cancellation
 *
 * @param output Filtered output signal
 * @param weights Filter weights (updated in place, size = filter_length)
 * @param input Signal to be filtered (surveillance)
 * @param reference Reference signal for adaptation
 * @param n Number of samples
 * @param filter_length Number of filter taps
 * @param mu Adaptation step size (0 < mu < 2)
 * @param eps Regularization constant to prevent division by zero
 */
void nlms_filter_f32(float* output, float* weights,
                     const float* input, const float* reference,
                     std::size_t n, std::size_t filter_length,
                     float mu, float eps);

/**
 * @brief Projection-based clutter cancellation
 *
 * Projects surveillance signal onto orthogonal complement of clutter subspace
 *
 * @param output Clutter-cancelled output
 * @param input Input surveillance signal
 * @param clutter_subspace Clutter subspace basis vectors [n x subspace_dim] (col-major)
 * @param n Signal length
 * @param subspace_dim Number of basis vectors in clutter subspace
 */
void projection_clutter_f32(float* output,
                            const float* input,
                            const float* clutter_subspace,
                            std::size_t n, std::size_t subspace_dim);

// Eigen wrappers
Eigen::VectorXf nlms_filter(const Eigen::VectorXf& input,
                            const Eigen::VectorXf& reference,
                            std::size_t filter_length,
                            float mu, float eps);

Eigen::VectorXf projection_clutter(const Eigen::VectorXf& input,
                                   const Eigen::MatrixXf& clutter_subspace);

// =========================================================================
// Doppler Processing
// =========================================================================

/**
 * @brief Apply Doppler FFT across pulses for each range bin
 *
 * @param output_re, output_im Output [fft_size x n_range]
 * @param input_re, input_im Input [n_pulses x n_range] (row-major)
 * @param n_pulses Number of pulses (slow-time samples)
 * @param n_range Number of range bins
 * @param fft_size FFT size (must be >= n_pulses, typically power of 2)
 */
void doppler_fft_f32(float* output_re, float* output_im,
                     const float* input_re, const float* input_im,
                     std::size_t n_pulses, std::size_t n_range,
                     std::size_t fft_size);

/**
 * @brief Moving Target Indicator (MTI) filter
 *
 * Applies FIR filter across slow-time (pulses) to suppress stationary clutter
 *
 * @param output Output [n_pulses - n_coeffs + 1 x n_range]
 * @param input Input [n_pulses x n_range] (row-major, complex)
 * @param n_pulses Number of pulses
 * @param n_range Number of range bins
 * @param coeffs MTI filter coefficients (e.g., [1, -1] for 2-pulse canceller)
 * @param n_coeffs Number of coefficients
 */
void mti_filter_f32(float* output, const float* input,
                    std::size_t n_pulses, std::size_t n_range,
                    const float* coeffs, std::size_t n_coeffs);

// Eigen wrappers
Eigen::MatrixXcf doppler_fft(const Eigen::MatrixXcf& input, std::size_t fft_size);
Eigen::MatrixXf mti_filter(const Eigen::MatrixXf& input, const Eigen::VectorXf& coeffs);

// =========================================================================
// Beamforming Primitives
// =========================================================================

/**
 * @brief Delay-and-sum beamformer
 *
 * @param output Output beamformed signal
 * @param inputs Array of input signals [n_channels x n_samples]
 * @param delays Delay in samples for each channel [n_channels]
 * @param weights Weights for each channel [n_channels] (optional, nullptr for unity)
 * @param n_channels Number of input channels
 * @param n_samples Number of samples per channel
 */
void beamform_delay_sum_f32(float* output,
                            const float* inputs,
                            const int* delays,
                            const float* weights,
                            std::size_t n_channels,
                            std::size_t n_samples);

/**
 * @brief Phase-shift beamformer for narrowband signals
 *
 * @param output Output beamformed complex signal
 * @param inputs_re, inputs_im Input complex signals [n_channels x n_samples]
 * @param phases Phase shifts in radians for each channel [n_channels]
 * @param weights Magnitude weights for each channel [n_channels]
 * @param n_channels Number of input channels
 * @param n_samples Number of samples per channel
 */
void beamform_phase_f32(float* output_re, float* output_im,
                        const float* inputs_re, const float* inputs_im,
                        const float* phases,
                        const float* weights,
                        std::size_t n_channels,
                        std::size_t n_samples);

/**
 * @brief Compute steering vector for uniform linear array
 *
 * @param steering_re, steering_im Output steering vector [n_elements]
 * @param n_elements Number of array elements
 * @param d_lambda Element spacing in wavelengths
 * @param theta_rad Steering angle in radians from broadside
 */
void steering_vector_ula_f32(float* steering_re, float* steering_im,
                             std::size_t n_elements,
                             float d_lambda,
                             float theta_rad);

// Eigen wrappers
Eigen::VectorXf beamform_delay_sum(const Eigen::MatrixXf& inputs,
                                   const Eigen::VectorXi& delays,
                                   const Eigen::VectorXf& weights);

Eigen::VectorXcf beamform_phase(const Eigen::MatrixXcf& inputs,
                                const Eigen::VectorXf& phases,
                                const Eigen::VectorXf& weights);

Eigen::VectorXcf steering_vector_ula(std::size_t n_elements,
                                     float d_lambda,
                                     float theta_rad);

} // namespace radar
} // namespace optmath
