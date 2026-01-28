/**
 * OptMathKernels CUDA Kernels
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Custom CUDA kernels and cuBLAS wrappers for vector/matrix operations.
 * Optimized for NVIDIA GPUs from Pascal to Hopper architecture.
 */

#include "optmath/cuda_backend.hpp"
#include "optmath/cuda_error.hpp"
#include <algorithm>
#include <cmath>

#ifdef OPTMATH_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Use error checking macros from cuda_error.hpp
using namespace optmath::cuda;

// =============================================================================
// Kernel Configuration Helpers
// =============================================================================

constexpr int BLOCK_SIZE = 256;
[[maybe_unused]] constexpr int WARP_SIZE = 32;

inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// Vector Kernels
// =============================================================================

__global__ void kernel_vec_add_f32(float* __restrict__ out,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void kernel_vec_mul_f32(float* __restrict__ out,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_vec_scale_f32(float* __restrict__ out,
                                      const float* __restrict__ a,
                                      float scalar,
                                      size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void kernel_vec_abs_f32(float* __restrict__ out,
                                    const float* __restrict__ a,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(a[idx]);
    }
}

__global__ void kernel_vec_sqrt_f32(float* __restrict__ out,
                                     const float* __restrict__ a,
                                     size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(a[idx]);
    }
}

// Vectorized operations (4 floats at a time)
__global__ void kernel_vec_add_f32_vec4(float4* __restrict__ out,
                                         const float4* __restrict__ a,
                                         const float4* __restrict__ b,
                                         size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        out[idx] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}

// =============================================================================
// Transcendental Kernels (using CUDA fast math)
// =============================================================================

__global__ void kernel_exp_f32(float* __restrict__ out,
                                const float* __restrict__ in,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __expf(in[idx]);  // Fast exp
    }
}

__global__ void kernel_log_f32(float* __restrict__ out,
                                const float* __restrict__ in,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __logf(in[idx]);  // Fast log
    }
}

__global__ void kernel_sin_f32(float* __restrict__ out,
                                const float* __restrict__ in,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __sinf(in[idx]);  // Fast sin
    }
}

__global__ void kernel_cos_f32(float* __restrict__ out,
                                const float* __restrict__ in,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __cosf(in[idx]);  // Fast cos
    }
}

__global__ void kernel_sincos_f32(float* __restrict__ sin_out,
                                   float* __restrict__ cos_out,
                                   const float* __restrict__ in,
                                   size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __sincosf(in[idx], &sin_out[idx], &cos_out[idx]);  // Fused sincos
    }
}

__global__ void kernel_tan_f32(float* __restrict__ out,
                                const float* __restrict__ in,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __tanf(in[idx]);
    }
}

__global__ void kernel_atan2_f32(float* __restrict__ out,
                                  const float* __restrict__ y,
                                  const float* __restrict__ x,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atan2f(y[idx], x[idx]);
    }
}

__global__ void kernel_pow_f32(float* __restrict__ out,
                                const float* __restrict__ base,
                                const float* __restrict__ exp,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __powf(base[idx], exp[idx]);
    }
}

// =============================================================================
// Activation Function Kernels
// =============================================================================

__global__ void kernel_sigmoid_f32(float* __restrict__ out,
                                    const float* __restrict__ in,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + __expf(-in[idx]));
    }
}

__global__ void kernel_tanh_f32(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float e2x = __expf(2.0f * in[idx]);
        out[idx] = (e2x - 1.0f) / (e2x + 1.0f);
    }
}

__global__ void kernel_relu_f32(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

__global__ void kernel_leaky_relu_f32(float* __restrict__ out,
                                       const float* __restrict__ in,
                                       float alpha,
                                       size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = (x > 0.0f) ? x : alpha * x;
    }
}

__global__ void kernel_gelu_f32(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/pi)
        float tanh_val = tanhf(inner);
        out[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

// Softmax with shared memory reduction
__global__ void kernel_softmax_f32(float* __restrict__ out,
                                    const float* __restrict__ in,
                                    size_t n) {
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and find max
    float val = (idx < n) ? in[idx] : -INFINITY;
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction for max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Compute exp(x - max)
    float exp_val = (idx < n) ? __expf(val - max_val) : 0.0f;
    sdata[tid] = exp_val;
    __syncthreads();

    // Parallel reduction for sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum = sdata[0];

    // Normalize
    if (idx < n) {
        out[idx] = exp_val / sum;
    }
}

// =============================================================================
// Matrix Kernels
// =============================================================================

__global__ void kernel_mat_add_f32(float* __restrict__ C,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void kernel_mat_scale_f32(float* __restrict__ out,
                                      const float* __restrict__ A,
                                      float scalar,
                                      int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        out[idx] = A[idx] * scalar;
    }
}

// Naive transpose (for small matrices)
__global__ void kernel_mat_transpose_naive_f32(float* __restrict__ out,
                                                const float* __restrict__ A,
                                                int M, int N) {
    // Column-major transpose: A(M,N) -> out(N,M)
    // A(i,j) at index i + j*M goes to out(j,i) at index j + i*N
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row in A
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col in A

    if (i < M && j < N) {
        out[j + i * N] = A[i + j * M];
    }
}

// Tiled transpose with shared memory (for large matrices)
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void kernel_mat_transpose_tiled_f32(float* __restrict__ out,
                                                const float* __restrict__ A,
                                                int M, int N) {
    // Column-major tiled transpose: A(M,N) -> out(N,M)
    // A(i,j) at index i + j*M goes to out(j,i) at index j + i*N
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    // Block (bx, by) handles a tile of A starting at row by*TILE_DIM, col bx*TILE_DIM
    int col_A = blockIdx.x * TILE_DIM + threadIdx.x;  // column index in A
    int row_A = blockIdx.y * TILE_DIM + threadIdx.y;  // row index in A

    // Load tile from column-major A
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (col_A < N && (row_A + j) < M) {
            // A(row_A+j, col_A) in column-major: index = (row_A+j) + col_A*M
            tile[threadIdx.y + j][threadIdx.x] = A[(row_A + j) + col_A * M];
        }
    }

    __syncthreads();

    // Write transposed tile to out
    // out(j,i) where i was row in A, j was col in A
    // Tile at (by, bx) in A becomes tile at (bx, by) in out
    int col_out = blockIdx.y * TILE_DIM + threadIdx.x;  // column index in out (was row in A)
    int row_out = blockIdx.x * TILE_DIM + threadIdx.y;  // row index in out (was col in A)

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (col_out < M && (row_out + j) < N) {
            // out(row_out+j, col_out) in column-major: index = (row_out+j) + col_out*N
            out[(row_out + j) + col_out * N] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

#endif // OPTMATH_USE_CUDA

namespace optmath {
namespace cuda {

// =============================================================================
// Vector Operation Implementations
// =============================================================================

void cuda_vec_add_f32(float* out, const float* a, const float* b, size_t n) {
#ifdef OPTMATH_USE_CUDA
    // Use vectorized version if aligned and large enough
    if (n >= 4 && (reinterpret_cast<uintptr_t>(out) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(b) % 16 == 0)) {
        size_t n4 = n / 4;
        int blocks = div_ceil(n4, BLOCK_SIZE);
        kernel_vec_add_f32_vec4<<<blocks, BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(out),
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            n4);

        // Handle remainder
        size_t remainder = n % 4;
        if (remainder > 0) {
            kernel_vec_add_f32<<<1, remainder>>>(
                out + n4 * 4, a + n4 * 4, b + n4 * 4, remainder);
        }
    } else {
        int blocks = div_ceil(n, BLOCK_SIZE);
        kernel_vec_add_f32<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    }
#endif
}

void cuda_vec_mul_f32(float* out, const float* a, const float* b, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_vec_mul_f32<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
#endif
}

void cuda_vec_scale_f32(float* out, const float* a, float scalar, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_vec_scale_f32<<<blocks, BLOCK_SIZE>>>(out, a, scalar, n);
#endif
}

float cuda_vec_dot_f32(const float* a, const float* b, size_t n) {
#ifdef OPTMATH_USE_CUDA
    CudaContext& ctx = CudaContext::get();
    if (!ctx.is_initialized()) {
        return 0.0f;
    }

    float result = 0.0f;
    cublasSdot(ctx.cublas(), static_cast<int>(n), a, 1, b, 1, &result);
    return result;
#else
    return 0.0f;
#endif
}

float cuda_vec_sum_f32(const float* a, size_t n) {
#ifdef OPTMATH_USE_CUDA
    CudaContext& ctx = CudaContext::get();
    if (!ctx.is_initialized()) {
        return 0.0f;
    }

    float result = 0.0f;
    float* d_result = nullptr;
    void* d_temp_storage = nullptr;

    // Allocate device result
    cudaError_t err = cudaMalloc(&d_result, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating d_result: " << cudaGetErrorString(err) << std::endl;
        return 0.0f;
    }

    // Query temp storage size
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, a, d_result, n);

    // Allocate temp storage
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating temp storage: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_result);
        return 0.0f;
    }

    // Perform reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, a, d_result, n);

    err = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in memcpy: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_temp_storage);
    cudaFree(d_result);

    return result;
#else
    return 0.0f;
#endif
}

float cuda_vec_max_f32(const float* a, size_t n) {
#ifdef OPTMATH_USE_CUDA
    float result = 0.0f;
    float* d_result = nullptr;
    void* d_temp_storage = nullptr;

    cudaError_t err = cudaMalloc(&d_result, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating d_result: " << cudaGetErrorString(err) << std::endl;
        return 0.0f;
    }

    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes, a, d_result, n);

    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating temp storage: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_result);
        return 0.0f;
    }

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, a, d_result, n);

    err = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in memcpy: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_temp_storage);
    cudaFree(d_result);

    return result;
#else
    return 0.0f;
#endif
}

float cuda_vec_min_f32(const float* a, size_t n) {
#ifdef OPTMATH_USE_CUDA
    float result = 0.0f;
    float* d_result = nullptr;
    void* d_temp_storage = nullptr;

    cudaError_t err = cudaMalloc(&d_result, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating d_result: " << cudaGetErrorString(err) << std::endl;
        return 0.0f;
    }

    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Min(nullptr, temp_storage_bytes, a, d_result, n);

    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating temp storage: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_result);
        return 0.0f;
    }

    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, a, d_result, n);

    err = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in memcpy: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_temp_storage);
    cudaFree(d_result);

    return result;
#else
    return 0.0f;
#endif
}

float cuda_vec_norm_f32(const float* a, size_t n) {
#ifdef OPTMATH_USE_CUDA
    CudaContext& ctx = CudaContext::get();
    if (!ctx.is_initialized()) {
        return 0.0f;
    }

    float result = 0.0f;
    cublasSnrm2(ctx.cublas(), static_cast<int>(n), a, 1, &result);
    return result;
#else
    return 0.0f;
#endif
}

void cuda_vec_abs_f32(float* out, const float* a, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_vec_abs_f32<<<blocks, BLOCK_SIZE>>>(out, a, n);
#endif
}

void cuda_vec_sqrt_f32(float* out, const float* a, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_vec_sqrt_f32<<<blocks, BLOCK_SIZE>>>(out, a, n);
#endif
}

// =============================================================================
// Eigen Vector Wrappers
// =============================================================================

Eigen::VectorXf cuda_vec_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    Eigen::VectorXf result(a.size());

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    cudaError_t err;

    auto cleanup = [&]() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_out) cudaFree(d_out);
    };

    err = cudaMalloc(&d_a, n * sizeof(float));
    if (err != cudaSuccess) { cleanup(); result = a + b; return result; }

    err = cudaMalloc(&d_b, n * sizeof(float));
    if (err != cudaSuccess) { cleanup(); result = a + b; return result; }

    err = cudaMalloc(&d_out, n * sizeof(float));
    if (err != cudaSuccess) { cleanup(); result = a + b; return result; }

    err = cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cleanup(); result = a + b; return result; }

    err = cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cleanup(); result = a + b; return result; }

    cuda_vec_add_f32(d_out, d_a, d_b, n);

    err = cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cleanup(); result = a + b; return result; }

    cleanup();
#else
    result = a + b;
#endif

    return result;
}

Eigen::VectorXf cuda_vec_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    Eigen::VectorXf result(a.size());

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a;
    float* d_b;
    float* d_out;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_vec_mul_f32(d_out, d_a, d_b, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
#else
    result = a.array() * b.array();
#endif

    return result;
}

Eigen::VectorXf cuda_vec_scale(const Eigen::VectorXf& a, float scalar) {
    Eigen::VectorXf result(a.size());

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a;
    float* d_out;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_vec_scale_f32(d_out, d_a, scalar, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_out);
#else
    result = a * scalar;
#endif

    return result;
}

float cuda_vec_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a;
    float* d_b;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float result = cuda_vec_dot_f32(d_a, d_b, n);

    cudaFree(d_a);
    cudaFree(d_b);

    return result;
#else
    return a.dot(b);
#endif
}

float cuda_reduce_sum(const Eigen::VectorXf& a) {
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float result = cuda_vec_sum_f32(d_a, n);

    cudaFree(d_a);
    return result;
#else
    return a.sum();
#endif
}

float cuda_reduce_max(const Eigen::VectorXf& a) {
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float result = cuda_vec_max_f32(d_a, n);

    cudaFree(d_a);
    return result;
#else
    return a.maxCoeff();
#endif
}

float cuda_reduce_min(const Eigen::VectorXf& a) {
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float result = cuda_vec_min_f32(d_a, n);

    cudaFree(d_a);
    return result;
#else
    return a.minCoeff();
#endif
}

float cuda_vec_norm(const Eigen::VectorXf& a) {
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) {
        CudaContext::get().init();
    }

    size_t n = a.size();
    float* d_a;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float result = cuda_vec_norm_f32(d_a, n);

    cudaFree(d_a);
    return result;
#else
    return a.norm();
#endif
}

// =============================================================================
// Transcendental Function Implementations
// =============================================================================

void cuda_exp_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_exp_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_log_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_log_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_sin_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_sin_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_cos_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_cos_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_sincos_f32(float* sin_out, float* cos_out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_sincos_f32<<<blocks, BLOCK_SIZE>>>(sin_out, cos_out, in, n);
#endif
}

void cuda_tan_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_tan_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_atan2_f32(float* out, const float* y, const float* x, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_atan2_f32<<<blocks, BLOCK_SIZE>>>(out, y, x, n);
#endif
}

void cuda_pow_f32(float* out, const float* base, const float* exp, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_pow_f32<<<blocks, BLOCK_SIZE>>>(out, base, exp, n);
#endif
}

void cuda_sigmoid_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_sigmoid_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_tanh_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_tanh_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_relu_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_relu_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_leaky_relu_f32(float* out, const float* in, float alpha, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_leaky_relu_f32<<<blocks, BLOCK_SIZE>>>(out, in, alpha, n);
#endif
}

void cuda_gelu_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    int blocks = div_ceil(n, BLOCK_SIZE);
    kernel_gelu_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
#endif
}

void cuda_softmax_f32(float* out, const float* in, size_t n) {
#ifdef OPTMATH_USE_CUDA
    // Note: This is a simplified single-block softmax
    // For production, use a more sophisticated multi-block approach
    kernel_softmax_f32<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(out, in, n);
#endif
}

// Eigen wrappers for transcendentals
Eigen::VectorXf cuda_exp(const Eigen::VectorXf& x) {
    Eigen::VectorXf result(x.size());
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = x.size();
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_exp_f32(d_out, d_in, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
#else
    result = x.array().exp();
#endif
    return result;
}

Eigen::VectorXf cuda_log(const Eigen::VectorXf& x) {
    Eigen::VectorXf result(x.size());
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = x.size();
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_log_f32(d_out, d_in, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
#else
    result = x.array().log();
#endif
    return result;
}

Eigen::VectorXf cuda_sin(const Eigen::VectorXf& x) {
    Eigen::VectorXf result(x.size());
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = x.size();
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_sin_f32(d_out, d_in, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
#else
    result = x.array().sin();
#endif
    return result;
}

Eigen::VectorXf cuda_cos(const Eigen::VectorXf& x) {
    Eigen::VectorXf result(x.size());
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = x.size();
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_cos_f32(d_out, d_in, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
#else
    result = x.array().cos();
#endif
    return result;
}

Eigen::VectorXf cuda_sigmoid(const Eigen::VectorXf& x) {
    Eigen::VectorXf result(x.size());
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = x.size();
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_sigmoid_f32(d_out, d_in, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
#else
    result = 1.0f / (1.0f + (-x.array()).exp());
#endif
    return result;
}

Eigen::VectorXf cuda_tanh(const Eigen::VectorXf& x) {
    Eigen::VectorXf result(x.size());
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = x.size();
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_tanh_f32(d_out, d_in, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
#else
    result = x.array().tanh();
#endif
    return result;
}

Eigen::VectorXf cuda_relu(const Eigen::VectorXf& x) {
    Eigen::VectorXf result(x.size());
#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    size_t n = x.size();
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda_relu_f32(d_out, d_in, n);

    cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
#else
    result = x.array().max(0.0f);
#endif
    return result;
}

// =============================================================================
// Matrix Operation Implementations (cuBLAS)
// =============================================================================

void cuda_mat_mul_f32(float* C, const float* A, const float* B,
                      int M, int N, int K, bool transA, bool transB) {
#ifdef OPTMATH_USE_CUDA
    CudaContext& ctx = CudaContext::get();
    if (!ctx.is_initialized()) return;

    float alpha = 1.0f;
    float beta = 0.0f;

    // Eigen uses column-major storage, same as cuBLAS
    // C(M,N) = A(M,K) * B(K,N)
    // Direct cuBLAS call with proper leading dimensions
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // For column-major: lda = #rows of A, ldb = #rows of B, ldc = #rows of C
    int lda = transA ? K : M;
    int ldb = transB ? N : K;
    int ldc = M;

    cublasSgemm(ctx.cublas(),
                opA, opB,
                M, N, K,
                &alpha,
                A, lda,
                B, ldb,
                &beta,
                C, ldc);
#endif
}

void cuda_mat_add_f32(float* C, const float* A, const float* B, int M, int N) {
#ifdef OPTMATH_USE_CUDA
    int total = M * N;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_mat_add_f32<<<blocks, BLOCK_SIZE>>>(C, A, B, M, N);
#endif
}

void cuda_mat_scale_f32(float* out, const float* A, float scalar, int M, int N) {
#ifdef OPTMATH_USE_CUDA
    int total = M * N;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_mat_scale_f32<<<blocks, BLOCK_SIZE>>>(out, A, scalar, M, N);
#endif
}

void cuda_mat_transpose_f32(float* out, const float* A, int M, int N) {
#ifdef OPTMATH_USE_CUDA
    // Use tiled version for large matrices
    if (M >= 32 && N >= 32) {
        dim3 grid(div_ceil(N, TILE_DIM), div_ceil(M, TILE_DIM));
        dim3 block(TILE_DIM, BLOCK_ROWS);
        kernel_mat_transpose_tiled_f32<<<grid, block>>>(out, A, M, N);
    } else {
        dim3 block(16, 16);
        dim3 grid(div_ceil(N, 16), div_ceil(M, 16));
        kernel_mat_transpose_naive_f32<<<grid, block>>>(out, A, M, N);
    }
#endif
}

void cuda_mat_vec_mul_f32(float* out, const float* A, const float* x, int M, int N) {
#ifdef OPTMATH_USE_CUDA
    CudaContext& ctx = CudaContext::get();
    if (!ctx.is_initialized()) return;

    float alpha = 1.0f;
    float beta = 0.0f;

    // Eigen uses column-major, same as cuBLAS
    // y(M) = A(M,N) * x(N)
    // cuBLAS GEMV: y = alpha * A * x + beta * y
    cublasSgemv(ctx.cublas(),
                CUBLAS_OP_N,  // No transpose - column-major A
                M, N,
                &alpha,
                A, M,  // lda = M for column-major
                x, 1,
                &beta,
                out, 1);
#endif
}

// Eigen wrappers
Eigen::MatrixXf cuda_mat_mul(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    int M = A.rows();
    int K = A.cols();
    int N = B.cols();
    Eigen::MatrixXf C(M, N);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaError_t err;

    auto cleanup = [&]() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    };

    err = cudaMalloc(&d_A, M * K * sizeof(float));
    if (err != cudaSuccess) { cleanup(); C = A * B; return C; }

    err = cudaMalloc(&d_B, K * N * sizeof(float));
    if (err != cudaSuccess) { cleanup(); C = A * B; return C; }

    err = cudaMalloc(&d_C, M * N * sizeof(float));
    if (err != cudaSuccess) { cleanup(); C = A * B; return C; }

    err = cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cleanup(); C = A * B; return C; }

    err = cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cleanup(); C = A * B; return C; }

    cuda_mat_mul_f32(d_C, d_A, d_B, M, N, K);

    err = cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cleanup(); C = A * B; return C; }

    cleanup();
#else
    C = A * B;
#endif

    return C;
}

Eigen::MatrixXf cuda_mat_add(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    int M = A.rows();
    int N = A.cols();
    Eigen::MatrixXf C(M, N);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_A, *d_B, *d_C;
    size_t size = M * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    cuda_mat_add_f32(d_C, d_A, d_B, M, N);

    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
#else
    C = A + B;
#endif

    return C;
}

Eigen::MatrixXf cuda_mat_scale(const Eigen::MatrixXf& A, float scalar) {
    int M = A.rows();
    int N = A.cols();
    Eigen::MatrixXf C(M, N);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_A, *d_C;
    size_t size = M * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);

    cuda_mat_scale_f32(d_C, d_A, scalar, M, N);

    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
#else
    C = A * scalar;
#endif

    return C;
}

Eigen::MatrixXf cuda_mat_transpose(const Eigen::MatrixXf& A) {
    int M = A.rows();
    int N = A.cols();
    Eigen::MatrixXf C(N, M);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_A, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    cuda_mat_transpose_f32(d_C, d_A, M, N);

    cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
#else
    C = A.transpose();
#endif

    return C;
}

Eigen::VectorXf cuda_mat_vec_mul(const Eigen::MatrixXf& A, const Eigen::VectorXf& x) {
    int M = A.rows();
    int N = A.cols();
    Eigen::VectorXf y(M);

#ifdef OPTMATH_USE_CUDA
    if (!CudaContext::get().is_initialized()) CudaContext::get().init();

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cuda_mat_vec_mul_f32(d_y, d_A, d_x, M, N);

    cudaMemcpy(y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
#else
    y = A * x;
#endif

    return y;
}

} // namespace cuda
} // namespace optmath
