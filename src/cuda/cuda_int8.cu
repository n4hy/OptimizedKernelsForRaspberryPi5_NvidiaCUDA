/**
 * @file cuda_int8.cu
 * @brief int8 GEMM via cublasGemmEx (CUBLAS_COMPUTE_32I, IMMA tensor cores on Turing+). See cuda_int8.hpp.
 *
 * Layout mapping. Our matrices are row-major; cuBLAS is column-major. A row-major X[r x c] is the column-major
 * matrix X^T[c x r] with leading dimension c. We want row-major C[M x N] = A[M x K] * Bt[N x K]^T, i.e.
 * column-major C^T[N x M] = Bt[N x K] * A^T[K x M] where Bt^T (col-major view of Bt, K x N, ld K) must be
 * transposed and A^T (col-major view of A, K x M, ld K) used as is:
 *     cublasGemmEx(op(A_cb)=T on Bt-view, op(B_cb)=N on A-view, m=N, n=M, k=K, lda=K, ldb=K, ldc=N)
 * This is the "TN" configuration that cuBLAS supports for CUDA_R_8I inputs with CUDA_R_32I output.
 */
#include "optmath/cuda_int8.hpp"

#ifdef OPTMATH_USE_CUDA
#include "optmath/cuda_backend.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#endif

namespace optmath {
namespace cuda {

#ifdef OPTMATH_USE_CUDA

namespace {
inline std::size_t pad4(std::size_t n) { return (n + 3) & ~std::size_t(3); }

bool ensure_ctx() {
    CudaContext& ctx = CudaContext::get();
    if (!ctx.is_initialized() && !ctx.init(0)) return false;
    return ctx.is_initialized();
}

// C_dev [M x Npad] row-major = A_dev [M x K] * Bt_dev [Npad x K]^T
bool gemm_device(std::int32_t* C_dev, const std::int8_t* A_dev, const std::int8_t* Bt_dev,
                 std::size_t M, std::size_t Npad, std::size_t K) {
    const int alpha = 1, beta = 0;
    cublasStatus_t st = cublasGemmEx(CudaContext::get().cublas(),
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     static_cast<int>(Npad), static_cast<int>(M), static_cast<int>(K),
                                     &alpha,
                                     Bt_dev, CUDA_R_8I, static_cast<int>(K),
                                     A_dev,  CUDA_R_8I, static_cast<int>(K),
                                     &beta,
                                     C_dev,  CUDA_R_32I, static_cast<int>(Npad),
                                     CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    return st == CUBLAS_STATUS_SUCCESS;
}

// Copy the M x N window of a row-major [M x Npad] device matrix into host C with leading dim ldc.
bool download_c(std::int32_t* C, std::size_t ldc, const std::int32_t* C_dev, std::size_t M, std::size_t N, std::size_t Npad) {
    if (ldc == Npad) return cudaMemcpy(C, C_dev, M * Npad * sizeof(std::int32_t), cudaMemcpyDeviceToHost) == cudaSuccess;
    return cudaMemcpy2D(C, ldc * sizeof(std::int32_t), C_dev, Npad * sizeof(std::int32_t),
                        N * sizeof(std::int32_t), M, cudaMemcpyDeviceToHost) == cudaSuccess;
}
}  // namespace

bool cuda_int8_available() { return ensure_ctx(); }

struct CudaInt8Gemm::Impl {
    std::int8_t*  Bt = nullptr;  std::size_t N = 0, Npad = 0, K = 0;
    std::int8_t*  A  = nullptr;  std::size_t A_cap = 0;   // bytes
    std::int32_t* C  = nullptr;  std::size_t C_cap = 0;   // elements
    ~Impl() { if (Bt) cudaFree(Bt); if (A) cudaFree(A); if (C) cudaFree(C); }
    bool reserve(std::size_t M) {
        const std::size_t a_need = M * K, c_need = M * Npad;
        if (a_need > A_cap) { if (A) cudaFree(A); A = nullptr; A_cap = 0;
            if (cudaMalloc(&A, a_need) != cudaSuccess) return false; A_cap = a_need; }
        if (c_need > C_cap) { if (C) cudaFree(C); C = nullptr; C_cap = 0;
            if (cudaMalloc(&C, c_need * sizeof(std::int32_t)) != cudaSuccess) return false; C_cap = c_need; }
        return true;
    }
};

CudaInt8Gemm::CudaInt8Gemm() : p_(new Impl) {}
CudaInt8Gemm::~CudaInt8Gemm() { delete p_; }
bool CudaInt8Gemm::ready() const { return p_->Bt != nullptr; }
std::size_t CudaInt8Gemm::N() const { return p_->N; }
std::size_t CudaInt8Gemm::K() const { return p_->K; }

bool CudaInt8Gemm::set_weights(const std::int8_t* Bt, std::size_t N, std::size_t K) {
    if (!ensure_ctx() || K % 4 != 0 || N == 0 || K == 0) return false;
    const std::size_t Npad = pad4(N);
    if (p_->Bt) { cudaFree(p_->Bt); p_->Bt = nullptr; }
    if (cudaMalloc(&p_->Bt, Npad * K) != cudaSuccess) { p_->Bt = nullptr; return false; }
    if (cudaMemset(p_->Bt, 0, Npad * K) != cudaSuccess ||
        cudaMemcpy(p_->Bt, Bt, N * K, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(p_->Bt); p_->Bt = nullptr; return false;
    }
    p_->N = N; p_->Npad = Npad; p_->K = K;
    return true;
}

bool CudaInt8Gemm::run(std::int32_t* C, std::size_t ldc, const std::int8_t* A, std::size_t M) {
    if (!ready() || M == 0 || ldc < p_->N || !p_->reserve(M)) return false;
    if (cudaMemcpy(p_->A, A, M * p_->K, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (!gemm_device(p_->C, p_->A, p_->Bt, M, p_->Npad, p_->K)) return false;
    return download_c(C, ldc, p_->C, M, p_->N, p_->Npad);
}

bool cuda_gemm_s8s8s32(std::int32_t* C, std::size_t ldc, const std::int8_t* A, const std::int8_t* Bt,
                       std::size_t M, std::size_t N, std::size_t K) {
    CudaInt8Gemm g;
    return g.set_weights(Bt, N, K) && g.run(C, ldc, A, M);
}

#else  // !OPTMATH_USE_CUDA

bool cuda_int8_available() { return false; }
bool cuda_gemm_s8s8s32(std::int32_t*, std::size_t, const std::int8_t*, const std::int8_t*, std::size_t, std::size_t, std::size_t) { return false; }
struct CudaInt8Gemm::Impl {};
CudaInt8Gemm::CudaInt8Gemm() : p_(nullptr) {}
CudaInt8Gemm::~CudaInt8Gemm() {}
bool CudaInt8Gemm::set_weights(const std::int8_t*, std::size_t, std::size_t) { return false; }
bool CudaInt8Gemm::run(std::int32_t*, std::size_t, const std::int8_t*, std::size_t) { return false; }
bool CudaInt8Gemm::ready() const { return false; }
std::size_t CudaInt8Gemm::N() const { return 0; }
std::size_t CudaInt8Gemm::K() const { return 0; }

#endif

}  // namespace cuda
}  // namespace optmath
