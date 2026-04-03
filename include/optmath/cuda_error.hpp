/**
 * OptMathKernels CUDA Error Handling Utilities
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Common CUDA error checking macros for use across all CUDA source files.
 */

#pragma once

#ifdef OPTMATH_USE_CUDA
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace optmath {
namespace cuda {

/**
 * @brief CUDA error checking macro - logs and returns void on error
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " (" << __func__ << "): "                              \
                      << cudaGetErrorString(err) << std::endl;                 \
            return;                                                            \
        }                                                                      \
    } while (0)

/**
 * @brief CUDA error checking macro - logs and returns specified value on error
 */
#define CUDA_CHECK_RETURN(call, ret)                                           \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " (" << __func__ << "): "                              \
                      << cudaGetErrorString(err) << std::endl;                 \
            return ret;                                                        \
        }                                                                      \
    } while (0)

/**
 * @brief CUDA error checking macro - logs and throws exception on error
 */
#define CUDA_CHECK_THROW(call)                                                 \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::string msg = std::string("CUDA error in ") + __FILE__ + ":" + \
                              std::to_string(__LINE__) + " (" + __func__ +     \
                              "): " + cudaGetErrorString(err);                  \
            std::cerr << msg << std::endl;                                     \
            throw std::runtime_error(msg);                                     \
        }                                                                      \
    } while (0)

/**
 * @brief CUDA error checking macro with cleanup - executes cleanup code on error
 * Usage: CUDA_CHECK_CLEANUP(cudaMalloc(...), { cudaFree(ptr1); cudaFree(ptr2); })
 */
#define CUDA_CHECK_CLEANUP(call, cleanup)                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " (" << __func__ << "): "                              \
                      << cudaGetErrorString(err) << std::endl;                 \
            cleanup;                                                           \
            return;                                                            \
        }                                                                      \
    } while (0)

/**
 * @brief CUDA error checking macro with cleanup and return value
 */
#define CUDA_CHECK_CLEANUP_RETURN(call, cleanup, ret)                          \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " (" << __func__ << "): "                              \
                      << cudaGetErrorString(err) << std::endl;                 \
            cleanup;                                                           \
            return ret;                                                        \
        }                                                                      \
    } while (0)

/**
 * @brief cuBLAS error checking macro
 */
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__      \
                      << " (" << __func__ << "): status " << status            \
                      << std::endl;                                            \
            return;                                                            \
        }                                                                      \
    } while (0)

/**
 * @brief cuBLAS error checking macro with return value
 */
#define CUBLAS_CHECK_RETURN(call, ret)                                         \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__      \
                      << " (" << __func__ << "): status " << status            \
                      << std::endl;                                            \
            return ret;                                                        \
        }                                                                      \
    } while (0)

/**
 * @brief cuSOLVER error checking macro
 */
#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t status = call;                                        \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__    \
                      << " (" << __func__ << "): status " << status            \
                      << std::endl;                                            \
            return;                                                            \
        }                                                                      \
    } while (0)

/**
 * @brief cuSOLVER error checking macro with return value
 */
#define CUSOLVER_CHECK_RETURN(call, ret)                                       \
    do {                                                                       \
        cusolverStatus_t status = call;                                        \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__    \
                      << " (" << __func__ << "): status " << status            \
                      << std::endl;                                            \
            return ret;                                                        \
        }                                                                      \
    } while (0)

/**
 * @brief cuFFT error checking macro
 */
#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult status = call;                                             \
        if (status != CUFFT_SUCCESS) {                                         \
            std::cerr << "cuFFT error in " << __FILE__ << ":" << __LINE__       \
                      << " (" << __func__ << "): status " << status            \
                      << std::endl;                                            \
            return;                                                            \
        }                                                                      \
    } while (0)

/**
 * @brief Synchronize and check for kernel errors
 */
#define CUDA_KERNEL_CHECK()                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA kernel error in " << __FILE__ << ":" << __LINE__ \
                      << " (" << __func__ << "): "                              \
                      << cudaGetErrorString(err) << std::endl;                 \
            return;                                                            \
        }                                                                      \
    } while (0)

/**
 * @brief Safe division helper to avoid division by zero in window functions
 * Returns 0.0f when n <= 1, otherwise returns (n - 1)
 */
inline __host__ __device__ float safe_window_divisor(size_t n) {
    return (n <= 1) ? 1.0f : static_cast<float>(n - 1);
}

} // namespace cuda
} // namespace optmath

#endif // OPTMATH_USE_CUDA
