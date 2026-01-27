/**
 * OptMathKernels CUDA Backend
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * High-performance NVIDIA CUDA backend for OptMathKernels.
 * Leverages cuBLAS, cuFFT, cuSOLVER, Thrust, and custom CUDA kernels
 * for maximum GPU acceleration.
 *
 * Features:
 * - Tensor Core acceleration (Ampere, Ada Lovelace, Hopper architectures)
 * - Mixed-precision computing (FP16, TF32, FP32, FP64)
 * - Unified Memory for simplified data management
 * - Multi-GPU support
 * - Asynchronous execution with CUDA streams
 * - Pinned memory for fast CPU-GPU transfers
 */

#pragma once

#include <complex>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <Eigen/Dense>

#ifdef OPTMATH_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#endif

namespace optmath {
namespace cuda {

// =============================================================================
// Device Information and Capability Queries
// =============================================================================

/**
 * @brief Check if CUDA is available at runtime
 * @return true if at least one CUDA device is available
 */
bool is_available();

/**
 * @brief Get number of CUDA devices
 */
int get_device_count();

/**
 * @brief GPU capability information
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    size_t free_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    bool tensor_cores;           // Volta+ (SM 7.0+)
    bool tf32_support;           // Ampere+ (SM 8.0+)
    bool fp16_support;           // Pascal+ (SM 6.0+)
    bool fp8_support;            // Blackwell (SM 10.0+) - RTX 5090
    bool blackwell;              // Blackwell architecture (SM 10.0+)
    bool unified_memory;
    int memory_bus_width;
    float memory_bandwidth_gbps;
    int l2_cache_size;
    size_t shared_memory_per_block;      // Blackwell has 228KB per SM
    size_t shared_memory_per_multiprocessor;

    // Short aliases for convenience
    int compute_major() const { return compute_capability_major; }
    int compute_minor() const { return compute_capability_minor; }
    int multiprocessors() const { return multiprocessor_count; }
    bool has_tensor_cores() const { return tensor_cores; }
    bool has_tf32() const { return tf32_support; }
    bool has_fp16() const { return fp16_support; }
    bool has_fp8() const { return fp8_support; }
    bool is_blackwell() const { return blackwell; }

    // Architecture detection helpers
    bool is_volta_or_newer() const { return compute_capability_major >= 7; }
    bool is_ampere_or_newer() const { return compute_capability_major >= 8; }
    bool is_ada_or_newer() const { return compute_capability_major >= 8 && compute_capability_minor >= 9; }
    bool is_hopper_or_newer() const { return compute_capability_major >= 9; }
    bool is_blackwell_or_newer() const { return compute_capability_major >= 10; }
};

/**
 * @brief Get device information
 * @param device_id Device index (default 0)
 */
DeviceInfo get_device_info(int device_id = 0);

/**
 * @brief Print device capabilities to stdout
 */
void print_device_info(int device_id = 0);

// =============================================================================
// CUDA Context Management
// =============================================================================

/**
 * @brief CUDA execution stream wrapper
 */
class CudaStream {
public:
    CudaStream();
    ~CudaStream();
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    void synchronize();
    bool query() const;  // Returns true if stream is idle

#ifdef OPTMATH_USE_CUDA
    cudaStream_t get() const { return m_stream; }
private:
    cudaStream_t m_stream = nullptr;
#endif
};

/**
 * @brief Singleton CUDA context managing handles and resources
 */
class CudaContext {
public:
    static CudaContext& get();

    bool init(int device_id = 0);
    void cleanup();
    bool is_initialized() const { return m_initialized; }

    // Get current device
    int device_id() const { return m_device_id; }

    // Memory info
    size_t get_free_memory() const;
    size_t get_total_memory() const;

    // Default stream operations
    void synchronize();

    // Precision mode for Tensor Cores
    enum class PrecisionMode {
        FP32,           // Standard single precision
        TF32,           // TensorFloat-32 (Ampere+, 19-bit mantissa)
        FP16,           // Half precision
        FP64,           // Double precision
        MIXED_FP16_FP32 // FP16 compute, FP32 accumulate
    };

    void set_precision_mode(PrecisionMode mode);
    PrecisionMode get_precision_mode() const { return m_precision_mode; }

#ifdef OPTMATH_USE_CUDA
    cublasHandle_t cublas() const { return m_cublas_handle; }
    cusolverDnHandle_t cusolver() const { return m_cusolver_handle; }
    cudaStream_t default_stream() const { return m_default_stream; }
#endif

private:
    CudaContext() = default;
    ~CudaContext();
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    bool m_initialized = false;
    int m_device_id = 0;
    PrecisionMode m_precision_mode = PrecisionMode::FP32;

#ifdef OPTMATH_USE_CUDA
    cublasHandle_t m_cublas_handle = nullptr;
    cusolverDnHandle_t m_cusolver_handle = nullptr;
    cudaStream_t m_default_stream = nullptr;
#endif
};

// =============================================================================
// Memory Management
// =============================================================================

/**
 * @brief RAII wrapper for CUDA device memory
 */
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void allocate(size_t count);
    void free();
    void copy_from_host(const T* host_data, size_t count);
    void copy_to_host(T* host_data, size_t count) const;
    void copy_from_host_async(const T* host_data, size_t count, CudaStream& stream);
    void copy_to_host_async(T* host_data, size_t count, CudaStream& stream) const;

    T* data() { return m_data; }
    const T* data() const { return m_data; }
    size_t size() const { return m_size; }
    size_t bytes() const { return m_size * sizeof(T); }
    bool empty() const { return m_data == nullptr; }

private:
    T* m_data = nullptr;
    size_t m_size = 0;
};

/**
 * @brief Pinned (page-locked) host memory for fast transfers
 */
template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count);
    ~PinnedBuffer();

    void allocate(size_t count);
    void free();

    T* data() { return m_data; }
    const T* data() const { return m_data; }
    size_t size() const { return m_size; }

private:
    T* m_data = nullptr;
    size_t m_size = 0;
};

/**
 * @brief Unified Memory buffer (accessible from CPU and GPU)
 */
template<typename T>
class UnifiedBuffer {
public:
    UnifiedBuffer() = default;
    explicit UnifiedBuffer(size_t count);
    ~UnifiedBuffer();

    void allocate(size_t count);
    void free();
    void prefetch_to_device(int device_id = 0);
    void prefetch_to_host();

    T* data() { return m_data; }
    const T* data() const { return m_data; }
    size_t size() const { return m_size; }

private:
    T* m_data = nullptr;
    size_t m_size = 0;
};

// =============================================================================
// Vector Operations
// =============================================================================

// Low-level pointer-based operations
void cuda_vec_add_f32(float* out, const float* a, const float* b, size_t n);
void cuda_vec_mul_f32(float* out, const float* a, const float* b, size_t n);
void cuda_vec_scale_f32(float* out, const float* a, float scalar, size_t n);
float cuda_vec_dot_f32(const float* a, const float* b, size_t n);
float cuda_vec_sum_f32(const float* a, size_t n);
float cuda_vec_max_f32(const float* a, size_t n);
float cuda_vec_min_f32(const float* a, size_t n);
float cuda_vec_norm_f32(const float* a, size_t n);
void cuda_vec_abs_f32(float* out, const float* a, size_t n);
void cuda_vec_sqrt_f32(float* out, const float* a, size_t n);

// Eigen wrappers
Eigen::VectorXf cuda_vec_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
Eigen::VectorXf cuda_vec_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
Eigen::VectorXf cuda_vec_scale(const Eigen::VectorXf& a, float scalar);
float cuda_vec_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
float cuda_reduce_sum(const Eigen::VectorXf& a);
float cuda_reduce_max(const Eigen::VectorXf& a);
float cuda_reduce_min(const Eigen::VectorXf& a);
float cuda_vec_norm(const Eigen::VectorXf& a);

// =============================================================================
// Vectorized Transcendental Functions (CUDA Fast Math)
// =============================================================================

void cuda_exp_f32(float* out, const float* in, size_t n);
void cuda_log_f32(float* out, const float* in, size_t n);
void cuda_sin_f32(float* out, const float* in, size_t n);
void cuda_cos_f32(float* out, const float* in, size_t n);
void cuda_sincos_f32(float* sin_out, float* cos_out, const float* in, size_t n);
void cuda_tan_f32(float* out, const float* in, size_t n);
void cuda_atan2_f32(float* out, const float* y, const float* x, size_t n);
void cuda_pow_f32(float* out, const float* base, const float* exp, size_t n);

// Activation functions (for ML applications)
void cuda_sigmoid_f32(float* out, const float* in, size_t n);
void cuda_tanh_f32(float* out, const float* in, size_t n);
void cuda_relu_f32(float* out, const float* in, size_t n);
void cuda_leaky_relu_f32(float* out, const float* in, float alpha, size_t n);
void cuda_gelu_f32(float* out, const float* in, size_t n);
void cuda_softmax_f32(float* out, const float* in, size_t n);

// Eigen wrappers
Eigen::VectorXf cuda_exp(const Eigen::VectorXf& x);
Eigen::VectorXf cuda_log(const Eigen::VectorXf& x);
Eigen::VectorXf cuda_sin(const Eigen::VectorXf& x);
Eigen::VectorXf cuda_cos(const Eigen::VectorXf& x);
Eigen::VectorXf cuda_sigmoid(const Eigen::VectorXf& x);
Eigen::VectorXf cuda_tanh(const Eigen::VectorXf& x);
Eigen::VectorXf cuda_relu(const Eigen::VectorXf& x);

// =============================================================================
// Matrix Operations (cuBLAS accelerated)
// =============================================================================

// Low-level operations
void cuda_mat_mul_f32(float* C, const float* A, const float* B,
                      int M, int N, int K,
                      bool transA = false, bool transB = false);
void cuda_mat_add_f32(float* C, const float* A, const float* B, int M, int N);
void cuda_mat_scale_f32(float* out, const float* A, float scalar, int M, int N);
void cuda_mat_transpose_f32(float* out, const float* A, int M, int N);
void cuda_mat_vec_mul_f32(float* out, const float* A, const float* x, int M, int N);

// Tensor Core GEMM (Ampere+)
void cuda_mat_mul_tensorcore_f32(float* C, const float* A, const float* B,
                                  int M, int N, int K);
void cuda_mat_mul_tensorcore_fp16(void* C, const void* A, const void* B,
                                   int M, int N, int K);

// Eigen wrappers
Eigen::MatrixXf cuda_mat_mul(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);
Eigen::MatrixXf cuda_mat_add(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);
Eigen::MatrixXf cuda_mat_scale(const Eigen::MatrixXf& A, float scalar);
Eigen::MatrixXf cuda_mat_transpose(const Eigen::MatrixXf& A);
Eigen::VectorXf cuda_mat_vec_mul(const Eigen::MatrixXf& A, const Eigen::VectorXf& x);

// Batch operations
void cuda_batched_mat_mul_f32(float** C, float** A, float** B,
                               int M, int N, int K, int batch_count);

// =============================================================================
// Linear Algebra (cuSOLVER)
// =============================================================================

// Cholesky decomposition
Eigen::MatrixXf cuda_cholesky(const Eigen::MatrixXf& A);

// LU decomposition
std::pair<Eigen::MatrixXf, Eigen::VectorXi> cuda_lu(const Eigen::MatrixXf& A);

// QR decomposition
std::pair<Eigen::MatrixXf, Eigen::MatrixXf> cuda_qr(const Eigen::MatrixXf& A);

// SVD
struct SVDResult {
    Eigen::MatrixXf U;
    Eigen::VectorXf S;
    Eigen::MatrixXf Vt;
};
SVDResult cuda_svd(const Eigen::MatrixXf& A);

// Eigenvalue decomposition
std::pair<Eigen::VectorXf, Eigen::MatrixXf> cuda_eig(const Eigen::MatrixXf& A);

// Linear solve Ax = b
Eigen::VectorXf cuda_solve(const Eigen::MatrixXf& A, const Eigen::VectorXf& b);

// Matrix inverse
Eigen::MatrixXf cuda_inverse(const Eigen::MatrixXf& A);

// =============================================================================
// Complex Number Operations
// =============================================================================

// Interleaved complex format (re, im, re, im, ...)
void cuda_complex_mul_f32(float* out_re, float* out_im,
                          const float* a_re, const float* a_im,
                          const float* b_re, const float* b_im, size_t n);
void cuda_complex_conj_mul_f32(float* out_re, float* out_im,
                               const float* a_re, const float* a_im,
                               const float* b_re, const float* b_im, size_t n);
void cuda_complex_dot_f32(float* out_re, float* out_im,
                          const float* a_re, const float* a_im,
                          const float* b_re, const float* b_im, size_t n);
void cuda_complex_magnitude_f32(float* out, const float* re, const float* im, size_t n);
void cuda_complex_phase_f32(float* out, const float* re, const float* im, size_t n);
void cuda_complex_exp_f32(float* out_re, float* out_im, const float* phase, size_t n);

// Eigen wrappers
Eigen::VectorXcf cuda_complex_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
Eigen::VectorXcf cuda_complex_conj_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
std::complex<float> cuda_complex_dot(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b);
Eigen::VectorXf cuda_complex_magnitude(const Eigen::VectorXcf& a);
Eigen::VectorXf cuda_complex_phase(const Eigen::VectorXcf& a);

// =============================================================================
// FFT Operations (cuFFT)
// =============================================================================

/**
 * @brief FFT plan cache for efficient repeated transforms
 */
class CudaFFTPlan {
public:
    CudaFFTPlan() = default;
    ~CudaFFTPlan();

    // 1D FFT plan
    bool create_1d(size_t n, bool inverse = false);
    // 1D batched FFT plan
    bool create_1d_batch(size_t n, size_t batch, bool inverse = false);
    // 2D FFT plan
    bool create_2d(size_t nx, size_t ny, bool inverse = false);

    void execute(float* inout);
    void execute(const float* in, float* out);

    void destroy();

private:
#ifdef OPTMATH_USE_CUDA
    cufftHandle m_plan = 0;
#endif
    bool m_valid = false;
};

// One-shot FFT operations (creates plan internally, less efficient for repeated use)
void cuda_fft_1d_f32(float* inout, size_t n, bool inverse = false);
void cuda_fft_1d_batch_f32(float* inout, size_t n, size_t batch, bool inverse = false);
void cuda_fft_2d_f32(float* inout, size_t nx, size_t ny, bool inverse = false);

// Eigen wrappers
Eigen::VectorXcf cuda_fft(const Eigen::VectorXcf& x);
Eigen::VectorXcf cuda_ifft(const Eigen::VectorXcf& x);
Eigen::MatrixXcf cuda_fft2(const Eigen::MatrixXcf& x);
Eigen::MatrixXcf cuda_ifft2(const Eigen::MatrixXcf& x);

// Real-to-complex FFT
Eigen::VectorXcf cuda_rfft(const Eigen::VectorXf& x);
Eigen::VectorXf cuda_irfft(const Eigen::VectorXcf& x, size_t n);

// =============================================================================
// Convolution
// =============================================================================

// 1D convolution
void cuda_conv1d_f32(float* out, const float* signal, const float* kernel,
                     size_t signal_len, size_t kernel_len);

// 2D convolution
void cuda_conv2d_f32(float* out, const float* image, const float* kernel,
                     int img_h, int img_w, int kern_h, int kern_w);

// FFT-based convolution (for large kernels)
void cuda_fftconv1d_f32(float* out, const float* signal, const float* kernel,
                        size_t signal_len, size_t kernel_len);

// Eigen wrappers
Eigen::VectorXf cuda_conv1d(const Eigen::VectorXf& signal, const Eigen::VectorXf& kernel);
Eigen::MatrixXf cuda_conv2d(const Eigen::MatrixXf& image, const Eigen::MatrixXf& kernel);

// =============================================================================
// Radar Signal Processing (GPU-accelerated)
// =============================================================================

/**
 * @brief Cross-Ambiguity Function (CAF) with GPU acceleration
 *
 * Computes full range-Doppler map using batched FFT and complex operations.
 * Significantly faster than CPU for large sample counts.
 */
Eigen::MatrixXf cuda_caf(const Eigen::VectorXcf& ref,
                          const Eigen::VectorXcf& surv,
                          size_t n_doppler_bins,
                          float doppler_start,
                          float doppler_step,
                          float sample_rate,
                          size_t n_range_bins);

/**
 * @brief GPU-accelerated 2D CFAR detector
 */
Eigen::MatrixXi cuda_cfar_2d(const Eigen::MatrixXf& power_map,
                              int guard_range, int guard_doppler,
                              int ref_range, int ref_doppler,
                              float pfa_factor);

/**
 * @brief GPU-accelerated 1D CA-CFAR
 */
Eigen::VectorXi cuda_cfar_ca(const Eigen::VectorXf& power,
                              int guard_cells, int ref_cells,
                              float pfa_factor);

/**
 * @brief GPU Doppler processing (range-Doppler map generation)
 */
Eigen::MatrixXcf cuda_doppler_process(const Eigen::MatrixXcf& pulse_data,
                                       size_t fft_size,
                                       int window_type = 1);  // 0=rect, 1=hamming

/**
 * @brief GPU Bartlett beamformer
 */
Eigen::VectorXf cuda_bartlett_spectrum(const Eigen::VectorXcf& array_data,
                                        float d_lambda,
                                        int n_angles = 181);

/**
 * @brief Generate steering vectors on GPU
 */
Eigen::MatrixXcf cuda_steering_vectors_ula(int n_elements,
                                            float d_lambda,
                                            const Eigen::VectorXf& angles);

/**
 * @brief GPU NLMS adaptive filter
 */
Eigen::VectorXcf cuda_nlms_filter(const Eigen::VectorXcf& surv,
                                   const Eigen::VectorXcf& ref,
                                   int filter_length,
                                   float mu = 0.1f,
                                   float eps = 1e-6f);

/**
 * @brief GPU projection-based clutter cancellation
 */
Eigen::VectorXcf cuda_projection_clutter(const Eigen::VectorXcf& surv,
                                          const Eigen::MatrixXcf& clutter_subspace);

// =============================================================================
// Window Functions (GPU-generated)
// =============================================================================

enum class WindowType {
    RECTANGULAR,
    HAMMING,
    HANNING,
    BLACKMAN,
    BLACKMAN_HARRIS,
    KAISER,
    GAUSSIAN,
    TUKEY
};

Eigen::VectorXf cuda_generate_window(size_t n, WindowType type, float param = 0.0f);
void cuda_apply_window(Eigen::VectorXf& data, const Eigen::VectorXf& window);
void cuda_apply_window(Eigen::VectorXcf& data, const Eigen::VectorXf& window);

// =============================================================================
// Convenient Shorthand Aliases (match NEON API naming)
// =============================================================================

// Initialization
inline void init(int device_id = 0) { CudaContext::get().init(device_id); }
inline void synchronize() { CudaContext::get().synchronize(); }

// Vector ops
inline Eigen::VectorXf cuda_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b) { return cuda_vec_add(a, b); }
inline Eigen::VectorXf cuda_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b) { return cuda_vec_mul(a, b); }
inline Eigen::VectorXf cuda_scale(const Eigen::VectorXf& a, float s) { return cuda_vec_scale(a, s); }
inline float cuda_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) { return cuda_vec_dot(a, b); }
inline float cuda_sum(const Eigen::VectorXf& a) { return cuda_reduce_sum(a); }
inline float cuda_max(const Eigen::VectorXf& a) { return cuda_reduce_max(a); }
inline float cuda_min(const Eigen::VectorXf& a) { return cuda_reduce_min(a); }
inline Eigen::VectorXf cuda_sqrt(const Eigen::VectorXf& a) {
    Eigen::VectorXf result(a.size());
    DeviceBuffer<float> d_in(a.size()), d_out(a.size());
    d_in.copy_from_host(a.data(), a.size());
    cuda_vec_sqrt_f32(d_out.data(), d_in.data(), a.size());
    d_out.copy_to_host(result.data(), result.size());
    return result;
}

// Matrix ops (shorthand)
inline Eigen::MatrixXf cuda_gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) { return cuda_mat_mul(A, B); }
inline Eigen::VectorXf cuda_gemv(const Eigen::MatrixXf& A, const Eigen::VectorXf& x) { return cuda_mat_vec_mul(A, x); }
inline Eigen::MatrixXf cuda_transpose(const Eigen::MatrixXf& A) { return cuda_mat_transpose(A); }

// Activation functions (Eigen wrappers)
inline Eigen::VectorXf cuda_leaky_relu(const Eigen::VectorXf& x, float alpha) {
    Eigen::VectorXf result(x.size());
    DeviceBuffer<float> d_in(x.size()), d_out(x.size());
    d_in.copy_from_host(x.data(), x.size());
    cuda_leaky_relu_f32(d_out.data(), d_in.data(), alpha, x.size());
    d_out.copy_to_host(result.data(), result.size());
    return result;
}

// Complex ops (shorthand) - cuda_complex_conj_mul already declared above
inline Eigen::VectorXf cuda_complex_abs(const Eigen::VectorXcf& a) { return cuda_complex_magnitude(a); }
inline Eigen::VectorXf cuda_complex_arg(const Eigen::VectorXcf& a) { return cuda_complex_phase(a); }

// Convolution (1D shorthand)
inline Eigen::VectorXf cuda_convolve_1d(const Eigen::VectorXf& signal, const Eigen::VectorXf& kernel) {
    return cuda_conv1d(signal, kernel);
}

// Radar shortcuts
Eigen::MatrixXf cuda_caf(const Eigen::VectorXcf& ref, const Eigen::VectorXcf& surv,
                          int n_doppler, int max_range, float doppler_step);
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> cuda_cfar_2d(
    const Eigen::MatrixXf& data, int guard, int ref_cells, float pfa);
Eigen::VectorXi cuda_cfar_1d(const Eigen::VectorXf& data, int guard, int ref_cells, float pfa);
Eigen::VectorXcf cuda_steering_vector_ula(int n_elements, float d_lambda, float angle_rad);
Eigen::VectorXf cuda_bartlett_spectrum(const Eigen::VectorXcf& array_data, float d_lambda, int n_angles);
void cuda_nlms_filter(const float* ref, const float* surv, float* output,
                      float* weights, size_t n, int filter_len, float mu, float eps);

// =============================================================================
// Multi-GPU Support
// =============================================================================

/**
 * @brief Set the active CUDA device
 */
void set_device(int device_id);

/**
 * @brief Get current active device
 */
int get_device();

/**
 * @brief Enable peer-to-peer access between GPUs
 */
bool enable_peer_access(int device_from, int device_to);

/**
 * @brief Distribute workload across multiple GPUs
 */
template<typename Func>
void parallel_for_devices(const std::vector<int>& devices, Func&& func);

// =============================================================================
// Performance Profiling
// =============================================================================

/**
 * @brief CUDA event-based timer
 */
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();

    void start();
    void stop();
    float elapsed_ms() const;

private:
#ifdef OPTMATH_USE_CUDA
    cudaEvent_t m_start, m_stop;
#endif
    bool m_running = false;
};

/**
 * @brief Memory bandwidth profiling
 */
struct BandwidthStats {
    float host_to_device_gbps;
    float device_to_host_gbps;
    float device_to_device_gbps;
};

BandwidthStats measure_bandwidth(size_t bytes = 256 * 1024 * 1024);

// =============================================================================
// Error Handling
// =============================================================================

/**
 * @brief Get last CUDA error as string
 */
std::string get_last_error();

/**
 * @brief Check CUDA operation result
 */
bool check_cuda_error(const char* operation = nullptr);

} // namespace cuda
} // namespace optmath
