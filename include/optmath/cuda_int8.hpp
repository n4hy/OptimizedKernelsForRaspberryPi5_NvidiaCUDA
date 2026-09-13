/**
 * @file cuda_int8.hpp
 * @brief int8 x int8 -> int32 GEMM on NVIDIA tensor cores (cuBLAS IMMA), Eigen-free.
 *
 * Row-major convention, matching optmath::neon::neon_gemm_s8s8s32:
 *     C[M x N] (int32, leading dim ldc) = A[M x K] (int8) * Bt[N x K]^T (int8)
 * i.e. C[m][n] = sum_k A[m][k] * Bt[n][k], exact 32-bit integer accumulation.
 *
 * Two entry points:
 *   - cuda_gemm_s8s8s32(): one-shot, host pointers in / host pointer out (uploads A and Bt every call).
 *   - CudaInt8Gemm: keeps Bt (the weights of an inference layer) resident on the device across calls, so a
 *     forward pass costs one H2D of A, the GEMM, and one D2H of C.
 *
 * Constraints handled internally: cuBLAS int8 GEMM needs K % 4 == 0 and leading dimensions % 4 == 0; N is
 * padded to a multiple of 4 on the device (zero rows) and the padding is dropped on the way back. K % 4 != 0 is
 * refused (returns false) rather than silently computed wrong.
 *
 * All functions return false (and leave C untouched) if the CUDA context is down, an allocation or a cuBLAS
 * call fails, or the constraints are not met -- never garbage.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace optmath {
namespace cuda {

/// True if the library was built with CUDA and a device can be initialised.
bool cuda_int8_available();

bool cuda_gemm_s8s8s32(std::int32_t* C, std::size_t ldc,
                       const std::int8_t* A, const std::int8_t* Bt,
                       std::size_t M, std::size_t N, std::size_t K);

class CudaInt8Gemm {
public:
    CudaInt8Gemm();
    ~CudaInt8Gemm();
    CudaInt8Gemm(const CudaInt8Gemm&) = delete;
    CudaInt8Gemm& operator=(const CudaInt8Gemm&) = delete;

    /// Upload Bt [N x K] row-major. Returns false if K % 4 != 0 or CUDA is unavailable.
    bool set_weights(const std::int8_t* Bt, std::size_t N, std::size_t K);
    /// C[M x N] (row-major, ldc >= N) = A[M x K] * Bt^T. Returns false if weights unset or a CUDA call fails.
    bool run(std::int32_t* C, std::size_t ldc, const std::int8_t* A, std::size_t M);
    bool ready() const;
    std::size_t N() const;
    std::size_t K() const;

private:
    struct Impl;
    Impl* p_;
};

}  // namespace cuda
}  // namespace optmath
