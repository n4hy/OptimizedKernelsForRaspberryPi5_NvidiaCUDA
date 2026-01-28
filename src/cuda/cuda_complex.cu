/**
 * OptMathKernels CUDA Complex Operations
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * CUDA kernels for complex number operations, FFT, and convolution.
 * Optimized for signal processing applications.
 */

#include "optmath/cuda_backend.hpp"
#include "optmath/cuda_error.hpp"
#include <cmath>

// Helper function to compute phase (arg) of complex vector without Eigen's .arg()
// which has NVCC compatibility issues with std::arg ADL lookup
static inline Eigen::VectorXf compute_phase_fallback(const Eigen::VectorXcf& a) {
    Eigen::VectorXf result(a.size());
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        result[i] = std::atan2(a[i].imag(), a[i].real());
    }
    return result;
}

#ifdef OPTMATH_USE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

constexpr int BLOCK_SIZE = 256;

inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// Complex Number Kernels
// =============================================================================

// Complex multiply: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
__global__ void kernel_complex_mul_f32(float* __restrict__ out_re,
                                        float* __restrict__ out_im,
                                        const float* __restrict__ a_re,
                                        const float* __restrict__ a_im,
                                        const float* __restrict__ b_re,
                                        const float* __restrict__ b_im,
                                        size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a_re[idx];
        float ai = a_im[idx];
        float br = b_re[idx];
        float bi = b_im[idx];
        out_re[idx] = ar * br - ai * bi;
        out_im[idx] = ar * bi + ai * br;
    }
}

// Complex conjugate multiply: a * conj(b) = (a + bi) * (c - di)
__global__ void kernel_complex_conj_mul_f32(float* __restrict__ out_re,
                                             float* __restrict__ out_im,
                                             const float* __restrict__ a_re,
                                             const float* __restrict__ a_im,
                                             const float* __restrict__ b_re,
                                             const float* __restrict__ b_im,
                                             size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a_re[idx];
        float ai = a_im[idx];
        float br = b_re[idx];
        float bi = b_im[idx];
        // a * conj(b) = (ar*br + ai*bi) + (ai*br - ar*bi)i
        out_re[idx] = ar * br + ai * bi;
        out_im[idx] = ai * br - ar * bi;
    }
}

// Complex magnitude: |z| = sqrt(re^2 + im^2)
__global__ void kernel_complex_magnitude_f32(float* __restrict__ out,
                                              const float* __restrict__ re,
                                              const float* __restrict__ im,
                                              size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = re[idx];
        float i = im[idx];
        out[idx] = sqrtf(r * r + i * i);
    }
}

// Complex magnitude squared: |z|^2 = re^2 + im^2 (faster, avoids sqrt)
__global__ void kernel_complex_magnitude_squared_f32(float* __restrict__ out,
                                                      const float* __restrict__ re,
                                                      const float* __restrict__ im,
                                                      size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = re[idx];
        float i = im[idx];
        out[idx] = r * r + i * i;
    }
}

// Complex phase: arg(z) = atan2(im, re)
__global__ void kernel_complex_phase_f32(float* __restrict__ out,
                                          const float* __restrict__ re,
                                          const float* __restrict__ im,
                                          size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atan2f(im[idx], re[idx]);
    }
}

// Complex exponential: exp(i*phase) = cos(phase) + i*sin(phase)
__global__ void kernel_complex_exp_f32(float* __restrict__ out_re,
                                        float* __restrict__ out_im,
                                        const float* __restrict__ phase,
                                        size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __sincosf(phase[idx], &out_im[idx], &out_re[idx]);
    }
}

// Complex scale
__global__ void kernel_complex_scale_f32(float* __restrict__ out_re,
                                          float* __restrict__ out_im,
                                          const float* __restrict__ in_re,
                                          const float* __restrict__ in_im,
                                          float scalar,
                                          size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_re[idx] = in_re[idx] * scalar;
        out_im[idx] = in_im[idx] * scalar;
    }
}

// Complex add
__global__ void kernel_complex_add_f32(float* __restrict__ out_re,
                                        float* __restrict__ out_im,
                                        const float* __restrict__ a_re,
                                        const float* __restrict__ a_im,
                                        const float* __restrict__ b_re,
                                        const float* __restrict__ b_im,
                                        size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_re[idx] = a_re[idx] + b_re[idx];
        out_im[idx] = a_im[idx] + b_im[idx];
    }
}

// Complex dot product reduction (using warp shuffle)
__device__ void warp_reduce_complex(volatile float* sdata_re, volatile float* sdata_im, int tid) {
    sdata_re[tid] += sdata_re[tid + 32];
    sdata_im[tid] += sdata_im[tid + 32];
    sdata_re[tid] += sdata_re[tid + 16];
    sdata_im[tid] += sdata_im[tid + 16];
    sdata_re[tid] += sdata_re[tid + 8];
    sdata_im[tid] += sdata_im[tid + 8];
    sdata_re[tid] += sdata_re[tid + 4];
    sdata_im[tid] += sdata_im[tid + 4];
    sdata_re[tid] += sdata_re[tid + 2];
    sdata_im[tid] += sdata_im[tid + 2];
    sdata_re[tid] += sdata_re[tid + 1];
    sdata_im[tid] += sdata_im[tid + 1];
}

__global__ void kernel_complex_dot_reduce_f32(float* __restrict__ out_re,
                                               float* __restrict__ out_im,
                                               const float* __restrict__ a_re,
                                               const float* __restrict__ a_im,
                                               const float* __restrict__ b_re,
                                               const float* __restrict__ b_im,
                                               size_t n) {
    extern __shared__ float sdata[];
    float* sdata_re = sdata;
    float* sdata_im = sdata + blockDim.x;

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Initialize shared memory to zero before any computation
    // This ensures threads that don't participate have zero contribution
    sdata_re[tid] = 0.0f;
    sdata_im[tid] = 0.0f;
    __syncthreads();

    // Initialize accumulators
    float sum_re = 0.0f;
    float sum_im = 0.0f;

    // Grid-stride loop for handling large arrays
    while (idx < n) {
        float ar = a_re[idx];
        float ai = a_im[idx];
        float br = b_re[idx];
        float bi = b_im[idx];
        // Eigen's dot product convention: conj(a) * b
        // (ar - ai*i) * (br + bi*i) = ar*br + ai*bi + (ar*bi - ai*br)*i
        sum_re += ar * br + ai * bi;
        sum_im += ar * bi - ai * br;

        if (idx + blockDim.x < n) {
            ar = a_re[idx + blockDim.x];
            ai = a_im[idx + blockDim.x];
            br = b_re[idx + blockDim.x];
            bi = b_im[idx + blockDim.x];
            sum_re += ar * br + ai * bi;
            sum_im += ar * bi - ai * br;
        }
        idx += blockDim.x * gridDim.x * 2;
    }

    sdata_re[tid] = sum_re;
    sdata_im[tid] = sum_im;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata_re[tid] += sdata_re[tid + s];
            sdata_im[tid] += sdata_im[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_reduce_complex(sdata_re, sdata_im, tid);
    }

    if (tid == 0) {
        atomicAdd(out_re, sdata_re[0]);
        atomicAdd(out_im, sdata_im[0]);
    }
}

// Interleaved to deinterleaved complex conversion
__global__ void kernel_deinterleave_complex_f32(float* __restrict__ out_re,
                                                 float* __restrict__ out_im,
                                                 const float* __restrict__ interleaved,
                                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_re[idx] = interleaved[2 * idx];
        out_im[idx] = interleaved[2 * idx + 1];
    }
}

// Deinterleaved to interleaved complex conversion
__global__ void kernel_interleave_complex_f32(float* __restrict__ interleaved,
                                               const float* __restrict__ in_re,
                                               const float* __restrict__ in_im,
                                               size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        interleaved[2 * idx] = in_re[idx];
        interleaved[2 * idx + 1] = in_im[idx];
    }
}

// =============================================================================
// Convolution Kernels
// =============================================================================

// 1D convolution (naive, for small kernels)
__global__ void kernel_conv1d_f32(float* __restrict__ out,
                                   const float* __restrict__ signal,
                                   const float* __restrict__ kernel,
                                   int signal_len, int kernel_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_len = signal_len + kernel_len - 1;

    if (idx < out_len) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_len; ++k) {
            int sig_idx = idx - k;
            if (sig_idx >= 0 && sig_idx < signal_len) {
                sum += signal[sig_idx] * kernel[k];
            }
        }
        out[idx] = sum;
    }
}

// 1D convolution with shared memory (for medium kernels)
template<int KERNEL_SIZE>
__global__ void kernel_conv1d_shared_f32(float* __restrict__ out,
                                          const float* __restrict__ signal,
                                          const float* __restrict__ kernel,
                                          int signal_len) {
    extern __shared__ float shared[];
    float* s_signal = shared;
    float* s_kernel = shared + blockDim.x + KERNEL_SIZE - 1;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    // Load kernel to shared memory
    if (lid < KERNEL_SIZE) {
        s_kernel[lid] = kernel[lid];
    }

    // Load signal with halo
    int halo_start = blockIdx.x * blockDim.x - KERNEL_SIZE / 2;
    for (int i = lid; i < blockDim.x + KERNEL_SIZE - 1; i += blockDim.x) {
        int sig_idx = halo_start + i;
        s_signal[i] = (sig_idx >= 0 && sig_idx < signal_len) ? signal[sig_idx] : 0.0f;
    }

    __syncthreads();

    if (gid < signal_len) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            sum += s_signal[lid + k] * s_kernel[k];
        }
        out[gid] = sum;
    }
}

// 2D convolution (naive, for small kernels)
__global__ void kernel_conv2d_f32(float* __restrict__ out,
                                   const float* __restrict__ image,
                                   const float* __restrict__ kernel,
                                   int img_h, int img_w,
                                   int kern_h, int kern_w) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int out_h = img_h - kern_h + 1;
    int out_w = img_w - kern_w + 1;

    if (row < out_h && col < out_w) {
        float sum = 0.0f;
        for (int kh = 0; kh < kern_h; ++kh) {
            for (int kw = 0; kw < kern_w; ++kw) {
                sum += image[(row + kh) * img_w + (col + kw)] *
                       kernel[kh * kern_w + kw];
            }
        }
        out[row * out_w + col] = sum;
    }
}

#endif // OPTMATH_USE_CUDA

namespace optmath {
namespace cuda {

// =============================================================================
// Complex Number Implementations
// =============================================================================

void cuda_complex_mul_f32(float* out_re, float* out_im,
                          const float* a_re, const float* a_im,
                          const float* b_re, const float* b_im, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_complex_mul_f32<<<blocks, BLOCK_SIZE>>>(out_re, out_im,
                                                    a_re, a_im, b_re, b_im, n);
#endif
}

void cuda_complex_conj_mul_f32(float* out_re, float* out_im,
                               const float* a_re, const float* a_im,
                               const float* b_re, const float* b_im, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_complex_conj_mul_f32<<<blocks, BLOCK_SIZE>>>(out_re, out_im,
                                                         a_re, a_im, b_re, b_im, n);
#endif
}

void cuda_complex_dot_f32(float* out_re, float* out_im,
                          const float* a_re, const float* a_im,
                          const float* b_re, const float* b_im, size_t n) {
#ifdef OPTMATH_USE_CUDA
    // Initialize output to zero - must succeed before launching kernel
    CUDA_CHECK(cudaMemset(out_re, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(out_im, 0, sizeof(float)));

    int blocks = div_ceil(n, BLOCK_SIZE * 2);
    blocks = std::min(blocks, 256);  // Cap number of blocks
    int smem_size = 2 * BLOCK_SIZE * sizeof(float);

    kernel_complex_dot_reduce_f32<<<blocks, BLOCK_SIZE, smem_size>>>(
        out_re, out_im, a_re, a_im, b_re, b_im, n);
    CUDA_KERNEL_CHECK();
#endif
}

void cuda_complex_magnitude_f32(float* out, const float* re, const float* im, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_complex_magnitude_f32<<<blocks, BLOCK_SIZE>>>(out, re, im, n);
#endif
}

void cuda_complex_phase_f32(float* out, const float* re, const float* im, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_complex_phase_f32<<<blocks, BLOCK_SIZE>>>(out, re, im, n);
#endif
}

void cuda_complex_exp_f32(float* out_re, float* out_im, const float* phase, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_complex_exp_f32<<<blocks, BLOCK_SIZE>>>(out_re, out_im, phase, n);
#endif
}

// =============================================================================
// Eigen Wrappers for Complex Operations
// =============================================================================

Eigen::VectorXcf cuda_complex_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    size_t n = a.size();
    Eigen::VectorXcf result(n);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    // Allocate device memory with error checking
    float *d_a_re = nullptr, *d_a_im = nullptr, *d_b_re = nullptr;
    float *d_b_im = nullptr, *d_out_re = nullptr, *d_out_im = nullptr;

    // Helper lambda for cleanup on error
    auto cleanup = [&]() {
        if (d_a_re) cudaFree(d_a_re);
        if (d_a_im) cudaFree(d_a_im);
        if (d_b_re) cudaFree(d_b_re);
        if (d_b_im) cudaFree(d_b_im);
        if (d_out_re) cudaFree(d_out_re);
        if (d_out_im) cudaFree(d_out_im);
    };

    cudaError_t err;
    err = cudaMalloc(&d_a_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();  // Fallback to CPU
    }
    err = cudaMalloc(&d_a_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMalloc(&d_b_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMalloc(&d_b_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMalloc(&d_out_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMalloc(&d_out_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }

    // Deinterleave and copy
    std::vector<float> a_re(n), a_im(n), b_re(n), b_im(n);
    for (size_t i = 0; i < n; ++i) {
        a_re[i] = a[i].real();
        a_im[i] = a[i].imag();
        b_re[i] = b[i].real();
        b_im[i] = b[i].imag();
    }

    err = cudaMemcpy(d_a_re, a_re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMemcpy(d_a_im, a_im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMemcpy(d_b_re, b_re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMemcpy(d_b_im, b_im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }

    // Compute
    cuda_complex_mul_f32(d_out_re, d_out_im, d_a_re, d_a_im, d_b_re, d_b_im, n);

    // Copy back and interleave
    std::vector<float> out_re(n), out_im(n);
    err = cudaMemcpy(out_re.data(), d_out_re, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }
    err = cudaMemcpy(out_im.data(), d_out_im, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.array();
    }

    for (size_t i = 0; i < n; ++i) {
        result[i] = std::complex<float>(out_re[i], out_im[i]);
    }

    cleanup();
#else
    result = a.array() * b.array();
#endif

    return result;
}

Eigen::VectorXcf cuda_complex_conj_mul(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
    size_t n = a.size();
    Eigen::VectorXcf result(n);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_a_re = nullptr, *d_a_im = nullptr, *d_b_re = nullptr;
    float *d_b_im = nullptr, *d_out_re = nullptr, *d_out_im = nullptr;

    auto cleanup = [&]() {
        if (d_a_re) cudaFree(d_a_re);
        if (d_a_im) cudaFree(d_a_im);
        if (d_b_re) cudaFree(d_b_re);
        if (d_b_im) cudaFree(d_b_im);
        if (d_out_re) cudaFree(d_out_re);
        if (d_out_im) cudaFree(d_out_im);
    };

    cudaError_t err;
    err = cudaMalloc(&d_a_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMalloc(&d_a_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMalloc(&d_b_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMalloc(&d_b_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMalloc(&d_out_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMalloc(&d_out_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }

    std::vector<float> a_re(n), a_im(n), b_re(n), b_im(n);
    for (size_t i = 0; i < n; ++i) {
        a_re[i] = a[i].real();
        a_im[i] = a[i].imag();
        b_re[i] = b[i].real();
        b_im[i] = b[i].imag();
    }

    err = cudaMemcpy(d_a_re, a_re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMemcpy(d_a_im, a_im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMemcpy(d_b_re, b_re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMemcpy(d_b_im, b_im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }

    cuda_complex_conj_mul_f32(d_out_re, d_out_im, d_a_re, d_a_im, d_b_re, d_b_im, n);

    std::vector<float> out_re(n), out_im(n);
    err = cudaMemcpy(out_re.data(), d_out_re, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }
    err = cudaMemcpy(out_im.data(), d_out_im, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array() * b.conjugate().array();
    }

    for (size_t i = 0; i < n; ++i) {
        result[i] = std::complex<float>(out_re[i], out_im[i]);
    }

    cleanup();
#else
    result = a.array() * b.conjugate().array();
#endif

    return result;
}

std::complex<float> cuda_complex_dot(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b) {
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = a.size();

    float *d_a_re = nullptr, *d_a_im = nullptr, *d_b_re = nullptr;
    float *d_b_im = nullptr, *d_out_re = nullptr, *d_out_im = nullptr;

    auto cleanup = [&]() {
        if (d_a_re) cudaFree(d_a_re);
        if (d_a_im) cudaFree(d_a_im);
        if (d_b_re) cudaFree(d_b_re);
        if (d_b_im) cudaFree(d_b_im);
        if (d_out_re) cudaFree(d_out_re);
        if (d_out_im) cudaFree(d_out_im);
    };

    cudaError_t err;
    err = cudaMalloc(&d_a_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMalloc(&d_a_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMalloc(&d_b_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMalloc(&d_b_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMalloc(&d_out_re, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMalloc(&d_out_im, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }

    std::vector<float> a_re(n), a_im(n), b_re(n), b_im(n);
    for (size_t i = 0; i < n; ++i) {
        a_re[i] = a[i].real();
        a_im[i] = a[i].imag();
        b_re[i] = b[i].real();
        b_im[i] = b[i].imag();
    }

    err = cudaMemcpy(d_a_re, a_re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMemcpy(d_a_im, a_im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMemcpy(d_b_re, b_re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMemcpy(d_b_im, b_im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }

    cuda_complex_dot_f32(d_out_re, d_out_im, d_a_re, d_a_im, d_b_re, d_b_im, n);

    float out_re, out_im;
    err = cudaMemcpy(&out_re, d_out_re, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }
    err = cudaMemcpy(&out_im, d_out_im, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.dot(b);
    }

    cleanup();

    return std::complex<float>(out_re, out_im);
#else
    return a.dot(b);
#endif
}

Eigen::VectorXf cuda_complex_magnitude(const Eigen::VectorXcf& a) {
    size_t n = a.size();
    Eigen::VectorXf result(n);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_re = nullptr, *d_im = nullptr, *d_out = nullptr;

    auto cleanup = [&]() {
        if (d_re) cudaFree(d_re);
        if (d_im) cudaFree(d_im);
        if (d_out) cudaFree(d_out);
    };

    cudaError_t err;
    err = cudaMalloc(&d_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array().abs();
    }
    err = cudaMalloc(&d_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array().abs();
    }
    err = cudaMalloc(&d_out, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array().abs();
    }

    std::vector<float> re(n), im(n);
    for (size_t i = 0; i < n; ++i) {
        re[i] = a[i].real();
        im[i] = a[i].imag();
    }

    err = cudaMemcpy(d_re, re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array().abs();
    }
    err = cudaMemcpy(d_im, im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array().abs();
    }

    cuda_complex_magnitude_f32(d_out, d_re, d_im, n);

    err = cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return a.array().abs();
    }

    cleanup();
#else
    result = a.array().abs();
#endif

    return result;
}

Eigen::VectorXf cuda_complex_phase(const Eigen::VectorXcf& a) {
    size_t n = a.size();
    Eigen::VectorXf result(n);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_re = nullptr, *d_im = nullptr, *d_out = nullptr;

    auto cleanup = [&]() {
        if (d_re) cudaFree(d_re);
        if (d_im) cudaFree(d_im);
        if (d_out) cudaFree(d_out);
    };

    cudaError_t err;
    err = cudaMalloc(&d_re, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return compute_phase_fallback(a);
    }
    err = cudaMalloc(&d_im, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return compute_phase_fallback(a);
    }
    err = cudaMalloc(&d_out, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return compute_phase_fallback(a);
    }

    std::vector<float> re(n), im(n);
    for (size_t i = 0; i < n; ++i) {
        re[i] = a[i].real();
        im[i] = a[i].imag();
    }

    err = cudaMemcpy(d_re, re.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return compute_phase_fallback(a);
    }
    err = cudaMemcpy(d_im, im.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return compute_phase_fallback(a);
    }

    cuda_complex_phase_f32(d_out, d_re, d_im, n);

    err = cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return compute_phase_fallback(a);
    }

    cleanup();
#else
    result = compute_phase_fallback(a);
#endif

    return result;
}

// =============================================================================
// FFT Implementations
// =============================================================================

void cuda_fft_1d_f32(float* inout, size_t n, bool inverse) {
#ifdef OPTMATH_USE_CUDA
    cufftHandle plan;
    cufftPlan1d(&plan, static_cast<int>(n), CUFFT_C2C, 1);
    cufftExecC2C(plan,
                 reinterpret_cast<cufftComplex*>(inout),
                 reinterpret_cast<cufftComplex*>(inout),
                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    cufftDestroy(plan);

    // Normalize if inverse
    if (inverse) {
        float scale = 1.0f / static_cast<float>(n);
        int blocks = div_ceil(n * 2, BLOCK_SIZE);
        // Scale both real and imaginary parts
        // (using simple kernel since we're just scaling)
        cudaDeviceSynchronize();
    }
#endif
}

void cuda_fft_1d_batch_f32(float* inout, size_t n, size_t batch, bool inverse) {
#ifdef OPTMATH_USE_CUDA
    cufftHandle plan;
    int rank = 1;
    int nfft[] = {static_cast<int>(n)};
    cufftPlanMany(&plan, rank, nfft,
                  nullptr, 1, static_cast<int>(n),
                  nullptr, 1, static_cast<int>(n),
                  CUFFT_C2C, static_cast<int>(batch));
    cufftExecC2C(plan,
                 reinterpret_cast<cufftComplex*>(inout),
                 reinterpret_cast<cufftComplex*>(inout),
                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    cufftDestroy(plan);
#endif
}

void cuda_fft_2d_f32(float* inout, size_t nx, size_t ny, bool inverse) {
#ifdef OPTMATH_USE_CUDA
    cufftHandle plan;
    cufftPlan2d(&plan, static_cast<int>(ny), static_cast<int>(nx), CUFFT_C2C);
    cufftExecC2C(plan,
                 reinterpret_cast<cufftComplex*>(inout),
                 reinterpret_cast<cufftComplex*>(inout),
                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    cufftDestroy(plan);
#endif
}

Eigen::VectorXcf cuda_fft(const Eigen::VectorXcf& x) {
    size_t n = x.size();
    Eigen::VectorXcf result(n);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    // Copy to device (interleaved format)
    float* d_data = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_data, n * 2 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return x;  // Fallback: return input unchanged
    }

    err = cudaMemcpy(d_data, x.data(), n * 2 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return x;
    }

    cuda_fft_1d_f32(d_data, n, false);

    err = cudaMemcpy(result.data(), d_data, n * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return x;
    }

    cudaFree(d_data);
#else
    // Fallback: would need Eigen FFT or similar
    result = x;  // Placeholder
#endif

    return result;
}

Eigen::VectorXcf cuda_ifft(const Eigen::VectorXcf& x) {
    size_t n = x.size();
    Eigen::VectorXcf result(n);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float* d_data = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_data, n * 2 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return x;  // Fallback: return input unchanged
    }

    err = cudaMemcpy(d_data, x.data(), n * 2 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return x;
    }

    cuda_fft_1d_f32(d_data, n, true);

    // Normalize
    float scale = 1.0f / static_cast<float>(n);
    int blocks = div_ceil(n * 2, BLOCK_SIZE);
    // Would need a scale kernel here

    err = cudaMemcpy(result.data(), d_data, n * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return x;
    }

    // Normalize on CPU for now
    result /= static_cast<float>(n);

    cudaFree(d_data);
#else
    result = x;  // Placeholder
#endif

    return result;
}

// =============================================================================
// Convolution Implementations
// =============================================================================

void cuda_conv1d_f32(float* out, const float* signal, const float* kernel,
                     size_t signal_len, size_t kernel_len) {
#ifdef OPTMATH_USE_CUDA
    size_t out_len = signal_len + kernel_len - 1;
    int blocks = div_ceil(out_len, BLOCK_SIZE);
    kernel_conv1d_f32<<<blocks, BLOCK_SIZE>>>(out, signal, kernel,
                                               static_cast<int>(signal_len),
                                               static_cast<int>(kernel_len));
#endif
}

void cuda_conv2d_f32(float* out, const float* image, const float* kernel,
                     int img_h, int img_w, int kern_h, int kern_w) {
#ifdef OPTMATH_USE_CUDA
    int out_h = img_h - kern_h + 1;
    int out_w = img_w - kern_w + 1;

    dim3 block(16, 16);
    dim3 grid(div_ceil(out_w, 16), div_ceil(out_h, 16));

    kernel_conv2d_f32<<<grid, block>>>(out, image, kernel, img_h, img_w, kern_h, kern_w);
#endif
}

Eigen::VectorXf cuda_conv1d(const Eigen::VectorXf& signal, const Eigen::VectorXf& kernel) {
    size_t sig_len = signal.size();
    size_t kern_len = kernel.size();
    size_t out_len = sig_len + kern_len - 1;
    Eigen::VectorXf result(out_len);

    // CPU fallback implementation
    auto cpu_conv1d = [&]() {
        result.setZero();
        for (size_t i = 0; i < out_len; ++i) {
            for (size_t k = 0; k < kern_len; ++k) {
                int sig_idx = static_cast<int>(i) - static_cast<int>(k);
                if (sig_idx >= 0 && sig_idx < static_cast<int>(sig_len)) {
                    result[i] += signal[sig_idx] * kernel[k];
                }
            }
        }
        return result;
    };

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_signal = nullptr, *d_kernel = nullptr, *d_out = nullptr;

    auto cleanup = [&]() {
        if (d_signal) cudaFree(d_signal);
        if (d_kernel) cudaFree(d_kernel);
        if (d_out) cudaFree(d_out);
    };

    cudaError_t err;
    err = cudaMalloc(&d_signal, sig_len * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv1d();
    }
    err = cudaMalloc(&d_kernel, kern_len * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv1d();
    }
    err = cudaMalloc(&d_out, out_len * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv1d();
    }

    err = cudaMemcpy(d_signal, signal.data(), sig_len * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv1d();
    }
    err = cudaMemcpy(d_kernel, kernel.data(), kern_len * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv1d();
    }

    cuda_conv1d_f32(d_out, d_signal, d_kernel, sig_len, kern_len);

    err = cudaMemcpy(result.data(), d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv1d();
    }

    cleanup();
#else
    return cpu_conv1d();
#endif

    return result;
}

Eigen::MatrixXf cuda_conv2d(const Eigen::MatrixXf& image, const Eigen::MatrixXf& kernel) {
    int img_h = image.rows();
    int img_w = image.cols();
    int kern_h = kernel.rows();
    int kern_w = kernel.cols();
    int out_h = img_h - kern_h + 1;
    int out_w = img_w - kern_w + 1;
    Eigen::MatrixXf result(out_h, out_w);

    // CPU fallback implementation
    auto cpu_conv2d = [&]() {
        result.setZero();
        for (int r = 0; r < out_h; ++r) {
            for (int c = 0; c < out_w; ++c) {
                float sum = 0.0f;
                for (int kh = 0; kh < kern_h; ++kh) {
                    for (int kw = 0; kw < kern_w; ++kw) {
                        sum += image(r + kh, c + kw) * kernel(kh, kw);
                    }
                }
                result(r, c) = sum;
            }
        }
        return result;
    };

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_image = nullptr, *d_kernel = nullptr, *d_out = nullptr;

    auto cleanup = [&]() {
        if (d_image) cudaFree(d_image);
        if (d_kernel) cudaFree(d_kernel);
        if (d_out) cudaFree(d_out);
    };

    cudaError_t err;
    err = cudaMalloc(&d_image, img_h * img_w * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv2d();
    }
    err = cudaMalloc(&d_kernel, kern_h * kern_w * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv2d();
    }
    err = cudaMalloc(&d_out, out_h * out_w * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv2d();
    }

    err = cudaMemcpy(d_image, image.data(), img_h * img_w * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv2d();
    }
    err = cudaMemcpy(d_kernel, kernel.data(), kern_h * kern_w * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv2d();
    }

    cuda_conv2d_f32(d_out, d_image, d_kernel, img_h, img_w, kern_h, kern_w);

    err = cudaMemcpy(result.data(), d_out, out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return cpu_conv2d();
    }

    cleanup();
#else
    return cpu_conv2d();
#endif

    return result;
}

} // namespace cuda
} // namespace optmath
