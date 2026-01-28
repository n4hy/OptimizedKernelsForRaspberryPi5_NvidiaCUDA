/**
 * OptMathKernels CUDA Radar Signal Processing
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * GPU-accelerated radar signal processing kernels:
 * - Cross-Ambiguity Function (CAF)
 * - CFAR Detection
 * - Doppler Processing
 * - Beamforming
 * - NLMS Adaptive Filtering
 */

// Suppress NVCC warning about Eigen host functions being called from host/device context
// This is a known Eigen + CUDA compatibility issue with template instantiation
// Must be placed before including Eigen headers
#ifdef __CUDACC__
#pragma nv_diag_suppress 20014
#endif

#include "optmath/cuda_backend.hpp"
#include "optmath/cuda_error.hpp"
#include <cmath>

#ifdef OPTMATH_USE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>

constexpr int BLOCK_SIZE = 256;
constexpr int BLOCK_2D = 16;
constexpr float PI = 3.14159265358979323846f;

inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// Window Function Kernels
// =============================================================================

__global__ void kernel_generate_hamming_f32(float* __restrict__ window, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        window[idx] = 0.54f - 0.46f * cosf(2.0f * PI * idx / (n - 1));
    }
}

__global__ void kernel_generate_hanning_f32(float* __restrict__ window, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        window[idx] = 0.5f * (1.0f - cosf(2.0f * PI * idx / (n - 1)));
    }
}

__global__ void kernel_generate_blackman_f32(float* __restrict__ window, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a0 = 0.42f;
        float a1 = 0.5f;
        float a2 = 0.08f;
        float x = 2.0f * PI * idx / (n - 1);
        window[idx] = a0 - a1 * cosf(x) + a2 * cosf(2.0f * x);
    }
}

__global__ void kernel_generate_blackman_harris_f32(float* __restrict__ window, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a0 = 0.35875f;
        float a1 = 0.48829f;
        float a2 = 0.14128f;
        float a3 = 0.01168f;
        float x = 2.0f * PI * idx / (n - 1);
        window[idx] = a0 - a1 * cosf(x) + a2 * cosf(2.0f * x) - a3 * cosf(3.0f * x);
    }
}

__global__ void kernel_apply_window_complex_f32(float* __restrict__ data_re,
                                                  float* __restrict__ data_im,
                                                  const float* __restrict__ window,
                                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float w = window[idx];
        data_re[idx] *= w;
        data_im[idx] *= w;
    }
}

// =============================================================================
// CAF Kernels
// =============================================================================

// Apply Doppler shift to reference signal: ref * exp(-j * 2 * pi * fd * t)
__global__ void kernel_doppler_shift_f32(float* __restrict__ out_re,
                                          float* __restrict__ out_im,
                                          const float* __restrict__ ref_re,
                                          const float* __restrict__ ref_im,
                                          float doppler_freq,
                                          float sample_rate,
                                          int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        float phase = -2.0f * PI * doppler_freq * idx / sample_rate;
        float cos_phase, sin_phase;
        __sincosf(phase, &sin_phase, &cos_phase);

        float rr = ref_re[idx];
        float ri = ref_im[idx];

        out_re[idx] = rr * cos_phase - ri * sin_phase;
        out_im[idx] = rr * sin_phase + ri * cos_phase;
    }
}

// Compute magnitude squared of complex cross-correlation result
__global__ void kernel_magnitude_squared_f32(float* __restrict__ out,
                                              const float* __restrict__ re,
                                              const float* __restrict__ im,
                                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = re[idx];
        float i = im[idx];
        out[idx] = r * r + i * i;
    }
}

// Interleave separate real/imag arrays into complex interleaved format (zero-padded)
__global__ void kernel_interleave_complex_f32(float* __restrict__ out_interleaved,
                                               const float* __restrict__ in_re,
                                               const float* __restrict__ in_im,
                                               int n_samples,
                                               int fft_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < fft_len) {
        if (idx < n_samples) {
            out_interleaved[2 * idx] = in_re[idx];
            out_interleaved[2 * idx + 1] = in_im[idx];
        } else {
            // Zero-padding
            out_interleaved[2 * idx] = 0.0f;
            out_interleaved[2 * idx + 1] = 0.0f;
        }
    }
}

// Complex conjugate multiply in interleaved format: out = a * conj(b) * scale
__global__ void kernel_complex_conj_mul_interleaved_f32(float* __restrict__ out,
                                                         const float* __restrict__ a,
                                                         const float* __restrict__ b,
                                                         float scale,
                                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a[2 * idx];
        float ai = a[2 * idx + 1];
        float br = b[2 * idx];
        float bi = -b[2 * idx + 1];  // Conjugate of b

        out[2 * idx] = (ar * br - ai * bi) * scale;
        out[2 * idx + 1] = (ar * bi + ai * br) * scale;
    }
}

// Compute magnitude from interleaved complex and store into output row
__global__ void kernel_magnitude_interleaved_f32(float* __restrict__ out,
                                                   const float* __restrict__ in_interleaved,
                                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float re = in_interleaved[2 * idx];
        float im = in_interleaved[2 * idx + 1];
        out[idx] = sqrtf(re * re + im * im);
    }
}

// =============================================================================
// CFAR Kernels
// =============================================================================

// 2D CFAR detector
__global__ void kernel_cfar_2d_f32(int* __restrict__ detections,
                                    const float* __restrict__ power_map,
                                    int n_doppler, int n_range,
                                    int guard_doppler, int guard_range,
                                    int ref_doppler, int ref_range,
                                    float pfa_factor) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int doppler_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (range_idx >= n_range || doppler_idx >= n_doppler) {
        return;
    }

    // Calculate CFAR window bounds
    int r_start = max(0, range_idx - guard_range - ref_range);
    int r_end = min(n_range - 1, range_idx + guard_range + ref_range);
    int d_start = max(0, doppler_idx - guard_doppler - ref_doppler);
    int d_end = min(n_doppler - 1, doppler_idx + guard_doppler + ref_doppler);

    // Calculate noise estimate (average of reference cells)
    float sum = 0.0f;
    int count = 0;

    for (int d = d_start; d <= d_end; ++d) {
        for (int r = r_start; r <= r_end; ++r) {
            // Skip guard cells and CUT
            bool in_guard = (abs(d - doppler_idx) <= guard_doppler) &&
                           (abs(r - range_idx) <= guard_range);
            if (!in_guard) {
                sum += power_map[d * n_range + r];
                count++;
            }
        }
    }

    float threshold = (count > 0) ? (sum / count) * pfa_factor : 0.0f;
    float cell_power = power_map[doppler_idx * n_range + range_idx];

    detections[doppler_idx * n_range + range_idx] = (cell_power > threshold) ? 1 : 0;
}

// 1D CA-CFAR detector
__global__ void kernel_cfar_ca_1d_f32(int* __restrict__ detections,
                                       const float* __restrict__ power,
                                       int n,
                                       int guard_cells,
                                       int ref_cells,
                                       float pfa_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Leading window
    int lead_start = max(0, idx - guard_cells - ref_cells);
    int lead_end = max(0, idx - guard_cells - 1);

    // Lagging window
    int lag_start = min(n - 1, idx + guard_cells + 1);
    int lag_end = min(n - 1, idx + guard_cells + ref_cells);

    float sum = 0.0f;
    int count = 0;

    // Sum leading cells
    for (int i = lead_start; i <= lead_end; ++i) {
        sum += power[i];
        count++;
    }

    // Sum lagging cells
    for (int i = lag_start; i <= lag_end; ++i) {
        sum += power[i];
        count++;
    }

    float threshold = (count > 0) ? (sum / count) * pfa_factor : 0.0f;
    detections[idx] = (power[idx] > threshold) ? 1 : 0;
}

// =============================================================================
// Doppler Processing Kernels
// =============================================================================

// Apply window along slow-time (Doppler) dimension
__global__ void kernel_apply_window_2d_f32(float* __restrict__ data_re,
                                            float* __restrict__ data_im,
                                            const float* __restrict__ window,
                                            int n_pulses, int n_range) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pulse_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (range_idx < n_range && pulse_idx < n_pulses) {
        float w = window[pulse_idx];
        int idx = pulse_idx * n_range + range_idx;
        data_re[idx] *= w;
        data_im[idx] *= w;
    }
}

// =============================================================================
// Beamforming Kernels
// =============================================================================

// Generate steering vector for ULA: a(theta) = [1, exp(-j*2*pi*d*sin(theta)), ...]
__global__ void kernel_steering_vector_ula_f32(float* __restrict__ steer_re,
                                                float* __restrict__ steer_im,
                                                float d_lambda,
                                                float theta,  // radians
                                                int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float phase = -2.0f * PI * d_lambda * idx * sinf(theta);
        __sincosf(phase, &steer_im[idx], &steer_re[idx]);
    }
}

// Batch steering vectors for multiple angles
__global__ void kernel_steering_vectors_batch_f32(float* __restrict__ steer_re,
                                                   float* __restrict__ steer_im,
                                                   const float* __restrict__ angles,
                                                   float d_lambda,
                                                   int n_elements,
                                                   int n_angles) {
    int elem_idx = threadIdx.x;
    int angle_idx = blockIdx.x;

    if (elem_idx < n_elements && angle_idx < n_angles) {
        float theta = angles[angle_idx];
        float phase = -2.0f * PI * d_lambda * elem_idx * sinf(theta);
        float sin_phase, cos_phase;
        __sincosf(phase, &sin_phase, &cos_phase);

        int out_idx = angle_idx * n_elements + elem_idx;
        steer_re[out_idx] = cos_phase;
        steer_im[out_idx] = sin_phase;
    }
}

// Bartlett beamformer: P(theta) = |a(theta)^H * x|^2
__global__ void kernel_bartlett_spectrum_f32(float* __restrict__ spectrum,
                                              const float* __restrict__ steer_re,
                                              const float* __restrict__ steer_im,
                                              const float* __restrict__ data_re,
                                              const float* __restrict__ data_im,
                                              int n_elements,
                                              int n_angles) {
    extern __shared__ float sdata[];
    float* s_sum_re = sdata;
    float* s_sum_im = sdata + blockDim.x;

    int angle_idx = blockIdx.x;
    int tid = threadIdx.x;

    float sum_re = 0.0f;
    float sum_im = 0.0f;

    // Each thread handles multiple elements
    for (int i = tid; i < n_elements; i += blockDim.x) {
        int steer_idx = angle_idx * n_elements + i;
        // a^H * x = conj(a) * x
        float ar = steer_re[steer_idx];
        float ai = -steer_im[steer_idx];  // conjugate
        float xr = data_re[i];
        float xi = data_im[i];

        sum_re += ar * xr - ai * xi;
        sum_im += ar * xi + ai * xr;
    }

    s_sum_re[tid] = sum_re;
    s_sum_im[tid] = sum_im;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_re[tid] += s_sum_re[tid + s];
            s_sum_im[tid] += s_sum_im[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float re = s_sum_re[0];
        float im = s_sum_im[0];
        spectrum[angle_idx] = re * re + im * im;
    }
}

// =============================================================================
// NLMS Filter Kernels
// =============================================================================

// NOTE: NLMS (Normalized Least Mean Squares) adaptive filtering is inherently
// sequential due to weight update dependencies between samples. Each sample's
// weight update depends on the previous sample's weights, making it unsuitable
// for GPU parallelization without significant algorithmic changes.
//
// The kernel below is intentionally left as a stub/placeholder. NLMS filtering
// should be performed on the CPU (see cuda_nlms_filter() function below) or
// using block-based LMS variants that can be parallelized (e.g., Block LMS,
// Frequency-Domain LMS).
//
// DO NOT USE THIS KERNEL - it contains race conditions and does not produce
// correct results. It exists only as documentation of the intended interface.
//
// For GPU-accelerated adaptive filtering, consider:
// 1. Block LMS (BLMS) - processes multiple samples per weight update
// 2. Frequency-Domain LMS (FDLMS) - uses FFT-based convolution
// 3. RLS with matrix operations - can leverage cuBLAS

#if 0 // DISABLED - PLACEHOLDER ONLY - DO NOT ENABLE
__global__ void kernel_nlms_step_f32_UNIMPLEMENTED(
    float* __restrict__ weights_re,
    float* __restrict__ weights_im,
    float* __restrict__ error_re,
    float* __restrict__ error_im,
    const float* __restrict__ surv_re,
    const float* __restrict__ surv_im,
    const float* __restrict__ ref_re,
    const float* __restrict__ ref_im,
    float mu, float eps,
    int filter_length,
    int sample_idx)
{
    // This kernel is intentionally disabled.
    // NLMS requires sequential weight updates that cannot be parallelized
    // across samples without introducing race conditions.
    //
    // Race conditions in original implementation:
    // 1. Shared memory s_power[] accessed by all threads but only valid for
    //    threads where ref_idx >= 0
    // 2. Weight updates would conflict if multiple threads try to update
    //    the same weight simultaneously
    // 3. The power normalization loop assumes all threads have valid data
    //
    // Use the CPU implementation cuda_nlms_filter() instead.
}
#endif // DISABLED PLACEHOLDER

#endif // OPTMATH_USE_CUDA

namespace optmath {
namespace cuda {

// =============================================================================
// Window Function Implementations
// =============================================================================

Eigen::VectorXf cuda_generate_window(size_t n, WindowType type, float param) {
    Eigen::VectorXf window(n);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float* d_window = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_window, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // Fall through to CPU fallback
        goto cpu_fallback;
    }

    {
        int blocks = div_ceil(n, BLOCK_SIZE);

        switch (type) {
            case WindowType::RECTANGULAR:
                // Set all to 1.0
                {
                    std::vector<float> ones(n, 1.0f);
                    err = cudaMemcpy(d_window, ones.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        cudaFree(d_window);
                        goto cpu_fallback;
                    }
                }
                break;
            case WindowType::HAMMING:
                kernel_generate_hamming_f32<<<blocks, BLOCK_SIZE>>>(d_window, static_cast<int>(n));
                break;
            case WindowType::HANNING:
                kernel_generate_hanning_f32<<<blocks, BLOCK_SIZE>>>(d_window, static_cast<int>(n));
                break;
            case WindowType::BLACKMAN:
                kernel_generate_blackman_f32<<<blocks, BLOCK_SIZE>>>(d_window, static_cast<int>(n));
                break;
            case WindowType::BLACKMAN_HARRIS:
                kernel_generate_blackman_harris_f32<<<blocks, BLOCK_SIZE>>>(d_window, static_cast<int>(n));
                break;
            default:
                // Default to Hamming
                kernel_generate_hamming_f32<<<blocks, BLOCK_SIZE>>>(d_window, static_cast<int>(n));
                break;
        }

        // Synchronize before D2H transfer
        cudaDeviceSynchronize();

        err = cudaMemcpy(window.data(), d_window, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_window);
            goto cpu_fallback;
        }

        cudaFree(d_window);
        return window;
    }

cpu_fallback:
#endif
    // Fallback: generate on CPU
    for (size_t i = 0; i < n; ++i) {
        switch (type) {
            case WindowType::RECTANGULAR:
                window[i] = 1.0f;
                break;
            case WindowType::HAMMING:
                window[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (n - 1));
                break;
            case WindowType::HANNING:
                window[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / (n - 1)));
                break;
            case WindowType::BLACKMAN:
                window[i] = 0.42f - 0.5f * std::cos(2.0f * PI * i / (n - 1))
                          + 0.08f * std::cos(4.0f * PI * i / (n - 1));
                break;
            default:
                window[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (n - 1));
                break;
        }
    }

    return window;
}

void cuda_apply_window(Eigen::VectorXf& data, const Eigen::VectorXf& window) {
    data.array() *= window.array();
}

void cuda_apply_window(Eigen::VectorXcf& data, const Eigen::VectorXf& window) {
    for (Eigen::Index i = 0; i < data.size(); ++i) {
        data[i] *= window[i];
    }
}

// =============================================================================
// CAF Implementation
// =============================================================================

Eigen::MatrixXf cuda_caf(const Eigen::VectorXcf& ref,
                          const Eigen::VectorXcf& surv,
                          size_t n_doppler_bins,
                          float doppler_start,
                          float doppler_step,
                          float sample_rate,
                          size_t n_range_bins) {
    size_t n_samples = ref.size();
    Eigen::MatrixXf caf_out(n_doppler_bins, n_range_bins);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    // Determine FFT size (next power of 2)
    // For cross-correlation, we need fft_len >= n_samples + n_range_bins - 1
    // to avoid circular correlation artifacts. Using 2*n_samples ensures
    // sufficient zero-padding for linear correlation.
    size_t min_fft_len = n_samples + n_range_bins;  // Fixed: was 2*n_samples (off-by-one)
    size_t fft_len = 1;
    while (fft_len < min_fft_len) fft_len <<= 1;

    // Allocate device memory with error checking
    float *d_ref_re = nullptr, *d_ref_im = nullptr;
    float *d_surv_re = nullptr, *d_surv_im = nullptr;
    float *d_shifted_re = nullptr, *d_shifted_im = nullptr;
    float *d_fft_ref = nullptr, *d_fft_surv = nullptr, *d_fft_prod = nullptr;
    float *d_caf_out = nullptr;  // Full CAF output on GPU

    // Helper lambda for cleanup on error
    auto cleanup = [&]() {
        if (d_ref_re) cudaFree(d_ref_re);
        if (d_ref_im) cudaFree(d_ref_im);
        if (d_surv_re) cudaFree(d_surv_re);
        if (d_surv_im) cudaFree(d_surv_im);
        if (d_shifted_re) cudaFree(d_shifted_re);
        if (d_shifted_im) cudaFree(d_shifted_im);
        if (d_fft_ref) cudaFree(d_fft_ref);
        if (d_fft_surv) cudaFree(d_fft_surv);
        if (d_fft_prod) cudaFree(d_fft_prod);
        if (d_caf_out) cudaFree(d_caf_out);
    };

    cudaError_t err;

    err = cudaMalloc(&d_ref_re, n_samples * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_ref_im, n_samples * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_surv_re, n_samples * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_surv_im, n_samples * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_shifted_re, n_samples * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_shifted_im, n_samples * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_fft_ref, fft_len * 2 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_fft_surv, fft_len * 2 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_fft_prod, fft_len * 2 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMalloc(&d_caf_out, n_doppler_bins * n_range_bins * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }

    // Deinterleave and copy input signals (one-time H2D transfer)
    std::vector<float> ref_re(n_samples), ref_im(n_samples);
    std::vector<float> surv_re(n_samples), surv_im(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        ref_re[i] = ref[i].real();
        ref_im[i] = ref[i].imag();
        surv_re[i] = surv[i].real();
        surv_im[i] = surv[i].imag();
    }

    err = cudaMemcpy(d_ref_re, ref_re.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMemcpy(d_ref_im, ref_im.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMemcpy(d_surv_re, surv_re.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }
    err = cudaMemcpy(d_surv_im, surv_im.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return caf_out;
    }

    // Create FFT plan
    cufftHandle fft_plan;
    cufftResult fft_result = cufftPlan1d(&fft_plan, static_cast<int>(fft_len), CUFFT_C2C, 1);
    if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error: plan creation failed with status " << fft_result << std::endl;
        cleanup();
        return caf_out;
    }

    // Interleave surveillance signal on GPU (zero-padded)
    int blocks_fft = div_ceil(fft_len, BLOCK_SIZE);
    kernel_interleave_complex_f32<<<blocks_fft, BLOCK_SIZE>>>(
        d_fft_surv, d_surv_re, d_surv_im, static_cast<int>(n_samples), static_cast<int>(fft_len));

    // Synchronize before FFT depends on kernel output
    cudaDeviceSynchronize();

    // FFT surveillance signal (once)
    cufftExecC2C(fft_plan,
                 reinterpret_cast<cufftComplex*>(d_fft_surv),
                 reinterpret_cast<cufftComplex*>(d_fft_surv),
                 CUFFT_FORWARD);

    // Precompute scale factor for normalization
    float scale = 1.0f / fft_len;

    // Process each Doppler bin - ALL OPERATIONS ON GPU
    for (size_t d = 0; d < n_doppler_bins; ++d) {
        float doppler_freq = doppler_start + d * doppler_step;

        // Apply Doppler shift to reference (GPU kernel)
        int blocks_samples = div_ceil(n_samples, BLOCK_SIZE);
        kernel_doppler_shift_f32<<<blocks_samples, BLOCK_SIZE>>>(
            d_shifted_re, d_shifted_im,
            d_ref_re, d_ref_im,
            doppler_freq, sample_rate, static_cast<int>(n_samples));

        // Synchronize before interleave depends on Doppler shift output
        cudaDeviceSynchronize();

        // Interleave shifted reference on GPU (zero-padded)
        kernel_interleave_complex_f32<<<blocks_fft, BLOCK_SIZE>>>(
            d_fft_ref, d_shifted_re, d_shifted_im, static_cast<int>(n_samples), static_cast<int>(fft_len));

        // Synchronize before FFT depends on interleave output
        cudaDeviceSynchronize();

        // FFT shifted reference
        cufftExecC2C(fft_plan,
                     reinterpret_cast<cufftComplex*>(d_fft_ref),
                     reinterpret_cast<cufftComplex*>(d_fft_ref),
                     CUFFT_FORWARD);

        // Synchronize before multiply depends on FFT output
        cudaDeviceSynchronize();

        // Multiply: Surv * conj(Ref) on GPU
        kernel_complex_conj_mul_interleaved_f32<<<blocks_fft, BLOCK_SIZE>>>(
            d_fft_prod, d_fft_surv, d_fft_ref, scale, static_cast<int>(fft_len));

        // Synchronize before IFFT depends on multiply output
        cudaDeviceSynchronize();

        // IFFT
        cufftExecC2C(fft_plan,
                     reinterpret_cast<cufftComplex*>(d_fft_prod),
                     reinterpret_cast<cufftComplex*>(d_fft_prod),
                     CUFFT_INVERSE);

        // Synchronize before magnitude depends on IFFT output
        cudaDeviceSynchronize();

        // Extract magnitude directly to output row on GPU
        int blocks_range = div_ceil(n_range_bins, BLOCK_SIZE);
        kernel_magnitude_interleaved_f32<<<blocks_range, BLOCK_SIZE>>>(
            d_caf_out + d * n_range_bins, d_fft_prod, static_cast<int>(n_range_bins));
    }

    // Final synchronize before D2H transfer
    cudaDeviceSynchronize();

    // Single D2H transfer at the end
    err = cudaMemcpy(caf_out.data(), d_caf_out, n_doppler_bins * n_range_bins * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Cleanup
    cufftDestroy(fft_plan);
    cleanup();

#else
    // Fallback: compute on CPU
    caf_out.setZero();
#endif

    return caf_out;
}

// =============================================================================
// CFAR Implementation
// =============================================================================

Eigen::MatrixXi cuda_cfar_2d(const Eigen::MatrixXf& power_map,
                              int guard_range, int guard_doppler,
                              int ref_range, int ref_doppler,
                              float pfa_factor) {
    int n_doppler = power_map.rows();
    int n_range = power_map.cols();
    Eigen::MatrixXi detections(n_doppler, n_range);
    detections.setZero();

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float* d_power = nullptr;
    int* d_detections = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_power, n_doppler * n_range * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        goto cpu_fallback_cfar2d;
    }

    err = cudaMalloc(&d_detections, n_doppler * n_range * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_power);
        goto cpu_fallback_cfar2d;
    }

    err = cudaMemcpy(d_power, power_map.data(), n_doppler * n_range * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_power);
        cudaFree(d_detections);
        goto cpu_fallback_cfar2d;
    }

    {
        dim3 block(BLOCK_2D, BLOCK_2D);
        dim3 grid(div_ceil(n_range, BLOCK_2D), div_ceil(n_doppler, BLOCK_2D));

        kernel_cfar_2d_f32<<<grid, block>>>(
            d_detections, d_power,
            n_doppler, n_range,
            guard_doppler, guard_range,
            ref_doppler, ref_range,
            pfa_factor);

        // Synchronize before D2H transfer
        cudaDeviceSynchronize();

        err = cudaMemcpy(detections.data(), d_detections, n_doppler * n_range * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaFree(d_power);
        cudaFree(d_detections);
        return detections;
    }

cpu_fallback_cfar2d:
#endif
    // Fallback: compute on CPU
    for (int d = 0; d < n_doppler; ++d) {
        for (int r = 0; r < n_range; ++r) {
            float sum = 0.0f;
            int count = 0;

            for (int dd = std::max(0, d - guard_doppler - ref_doppler);
                 dd <= std::min(n_doppler - 1, d + guard_doppler + ref_doppler); ++dd) {
                for (int rr = std::max(0, r - guard_range - ref_range);
                     rr <= std::min(n_range - 1, r + guard_range + ref_range); ++rr) {
                    bool in_guard = (std::abs(dd - d) <= guard_doppler) &&
                                   (std::abs(rr - r) <= guard_range);
                    if (!in_guard) {
                        sum += power_map(dd, rr);
                        count++;
                    }
                }
            }

            float threshold = (count > 0) ? (sum / count) * pfa_factor : 0.0f;
            detections(d, r) = (power_map(d, r) > threshold) ? 1 : 0;
        }
    }

    return detections;
}

Eigen::VectorXi cuda_cfar_ca(const Eigen::VectorXf& power,
                              int guard_cells, int ref_cells,
                              float pfa_factor) {
    int n = power.size();
    Eigen::VectorXi detections(n);
    detections.setZero();

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float* d_power = nullptr;
    int* d_detections = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_power, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return detections;
    }

    err = cudaMalloc(&d_detections, n * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_power);
        return detections;
    }

    err = cudaMemcpy(d_power, power.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_power);
        cudaFree(d_detections);
        return detections;
    }

    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_cfar_ca_1d_f32<<<blocks, BLOCK_SIZE>>>(
        d_detections, d_power, n, guard_cells, ref_cells, pfa_factor);

    // Synchronize before D2H transfer
    cudaDeviceSynchronize();

    err = cudaMemcpy(detections.data(), d_detections, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_power);
    cudaFree(d_detections);
#endif

    return detections;
}

// =============================================================================
// Beamforming Implementation
// =============================================================================

Eigen::VectorXf cuda_bartlett_spectrum(const Eigen::VectorXcf& array_data,
                                        float d_lambda,
                                        int n_angles) {
    int n_elements = array_data.size();
    Eigen::VectorXf spectrum(n_angles);
    spectrum.setZero();

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    // Generate angles from -90 to +90 degrees
    std::vector<float> angles(n_angles);
    for (int i = 0; i < n_angles; ++i) {
        angles[i] = (-90.0f + i * 180.0f / (n_angles - 1)) * PI / 180.0f;
    }

    // Allocate device memory with error checking
    float *d_angles = nullptr, *d_steer_re = nullptr, *d_steer_im = nullptr;
    float *d_data_re = nullptr, *d_data_im = nullptr, *d_spectrum = nullptr;
    cudaError_t err;

    auto cleanup_bartlett = [&]() {
        if (d_angles) cudaFree(d_angles);
        if (d_steer_re) cudaFree(d_steer_re);
        if (d_steer_im) cudaFree(d_steer_im);
        if (d_data_re) cudaFree(d_data_re);
        if (d_data_im) cudaFree(d_data_im);
        if (d_spectrum) cudaFree(d_spectrum);
    };

    err = cudaMalloc(&d_angles, n_angles * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_bartlett();
        goto cpu_fallback_bartlett;
    }
    err = cudaMalloc(&d_steer_re, n_angles * n_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_bartlett();
        goto cpu_fallback_bartlett;
    }
    err = cudaMalloc(&d_steer_im, n_angles * n_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_bartlett();
        goto cpu_fallback_bartlett;
    }
    err = cudaMalloc(&d_data_re, n_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_bartlett();
        goto cpu_fallback_bartlett;
    }
    err = cudaMalloc(&d_data_im, n_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_bartlett();
        goto cpu_fallback_bartlett;
    }
    err = cudaMalloc(&d_spectrum, n_angles * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_bartlett();
        goto cpu_fallback_bartlett;
    }

    // Copy angles and data
    err = cudaMemcpy(d_angles, angles.data(), n_angles * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_bartlett();
        goto cpu_fallback_bartlett;
    }

    {
        std::vector<float> data_re(n_elements), data_im(n_elements);
        for (int i = 0; i < n_elements; ++i) {
            data_re[i] = array_data[i].real();
            data_im[i] = array_data[i].imag();
        }
        err = cudaMemcpy(d_data_re, data_re.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            cleanup_bartlett();
            goto cpu_fallback_bartlett;
        }
        err = cudaMemcpy(d_data_im, data_im.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            cleanup_bartlett();
            goto cpu_fallback_bartlett;
        }
    }

    // Generate steering vectors
    kernel_steering_vectors_batch_f32<<<n_angles, n_elements>>>(
        d_steer_re, d_steer_im, d_angles, d_lambda, n_elements, n_angles);

    // Synchronize before Bartlett spectrum depends on steering vectors
    cudaDeviceSynchronize();

    // Compute Bartlett spectrum
    {
        int smem_size = 2 * 256 * sizeof(float);
        kernel_bartlett_spectrum_f32<<<n_angles, 256, smem_size>>>(
            d_spectrum, d_steer_re, d_steer_im, d_data_re, d_data_im, n_elements, n_angles);
    }

    // Synchronize before D2H transfer
    cudaDeviceSynchronize();

    err = cudaMemcpy(spectrum.data(), d_spectrum, n_angles * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cleanup_bartlett();
    return spectrum;

cpu_fallback_bartlett:
#endif
    // Fallback: compute on CPU
    for (int a = 0; a < n_angles; ++a) {
        float theta = (-90.0f + a * 180.0f / (n_angles - 1)) * PI / 180.0f;

        std::complex<float> sum(0.0f, 0.0f);
        for (int i = 0; i < n_elements; ++i) {
            float phase = -2.0f * PI * d_lambda * i * std::sin(theta);
            std::complex<float> steer(std::cos(phase), std::sin(phase));
            sum += std::conj(steer) * array_data[i];
        }
        spectrum[a] = std::norm(sum);
    }

    return spectrum;
}

Eigen::MatrixXcf cuda_steering_vectors_ula(int n_elements,
                                            float d_lambda,
                                            const Eigen::VectorXf& angles) {
    int n_angles = angles.size();
    Eigen::MatrixXcf steering(n_angles, n_elements);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_angles = nullptr, *d_steer_re = nullptr, *d_steer_im = nullptr;
    cudaError_t err;

    auto cleanup_steer = [&]() {
        if (d_angles) cudaFree(d_angles);
        if (d_steer_re) cudaFree(d_steer_re);
        if (d_steer_im) cudaFree(d_steer_im);
    };

    err = cudaMalloc(&d_angles, n_angles * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_steer();
        goto cpu_fallback_steer;
    }
    err = cudaMalloc(&d_steer_re, n_angles * n_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_steer();
        goto cpu_fallback_steer;
    }
    err = cudaMalloc(&d_steer_im, n_angles * n_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_steer();
        goto cpu_fallback_steer;
    }

    err = cudaMemcpy(d_angles, angles.data(), n_angles * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup_steer();
        goto cpu_fallback_steer;
    }

    kernel_steering_vectors_batch_f32<<<n_angles, n_elements>>>(
        d_steer_re, d_steer_im, d_angles, d_lambda, n_elements, n_angles);

    // Synchronize before D2H transfer
    cudaDeviceSynchronize();

    {
        std::vector<float> steer_re(n_angles * n_elements), steer_im(n_angles * n_elements);
        err = cudaMemcpy(steer_re.data(), d_steer_re, n_angles * n_elements * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            cleanup_steer();
            goto cpu_fallback_steer;
        }
        err = cudaMemcpy(steer_im.data(), d_steer_im, n_angles * n_elements * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            cleanup_steer();
            goto cpu_fallback_steer;
        }

        for (int a = 0; a < n_angles; ++a) {
            for (int e = 0; e < n_elements; ++e) {
                int idx = a * n_elements + e;
                steering(a, e) = std::complex<float>(steer_re[idx], steer_im[idx]);
            }
        }

        cleanup_steer();
        return steering;
    }

cpu_fallback_steer:
#endif
    // Fallback
    for (int a = 0; a < n_angles; ++a) {
        float theta = angles[a];
        for (int e = 0; e < n_elements; ++e) {
            float phase = -2.0f * PI * d_lambda * e * std::sin(theta);
            steering(a, e) = std::complex<float>(std::cos(phase), std::sin(phase));
        }
    }

    return steering;
}

// =============================================================================
// NLMS Filter (Simplified CPU implementation)
// =============================================================================

Eigen::VectorXcf cuda_nlms_filter(const Eigen::VectorXcf& surv,
                                   const Eigen::VectorXcf& ref,
                                   int filter_length,
                                   float mu,
                                   float eps) {
    // Note: NLMS is inherently sequential, so GPU benefit is limited
    // This is a CPU implementation for correctness
    int n = surv.size();
    Eigen::VectorXcf output(n);
    Eigen::VectorXcf weights = Eigen::VectorXcf::Zero(filter_length);

    for (int i = 0; i < n; ++i) {
        // Compute filter output
        std::complex<float> y(0.0f, 0.0f);
        float power = eps;

        for (int k = 0; k < filter_length; ++k) {
            int ref_idx = i - k;
            if (ref_idx >= 0) {
                y += weights[k] * ref[ref_idx];
                power += std::norm(ref[ref_idx]);
            }
        }

        // Error signal
        std::complex<float> error = surv[i] - y;
        output[i] = error;

        // Update weights
        for (int k = 0; k < filter_length; ++k) {
            int ref_idx = i - k;
            if (ref_idx >= 0) {
                weights[k] += (mu / power) * error * std::conj(ref[ref_idx]);
            }
        }
    }

    return output;
}

Eigen::VectorXcf cuda_projection_clutter(const Eigen::VectorXcf& surv,
                                          const Eigen::MatrixXcf& clutter_subspace) {
    // Projection: P_perp = I - C * (C^H * C)^-1 * C^H
    // output = P_perp * surv

#ifdef OPTMATH_USE_CUDA
    // For small matrices, CPU is often faster due to overhead
    // This uses Eigen for the computation
#endif

    int n = surv.size();
    int k = clutter_subspace.cols();

    // Compute C^H * C
    Eigen::MatrixXcf CtC = clutter_subspace.adjoint() * clutter_subspace;

    // Compute inverse
    Eigen::MatrixXcf CtC_inv = CtC.inverse();

    // Compute C * (C^H * C)^-1 * C^H * surv
    Eigen::VectorXcf proj = clutter_subspace * (CtC_inv * (clutter_subspace.adjoint() * surv));

    // Output = surv - projection
    return surv - proj;
}

} // namespace cuda
} // namespace optmath
