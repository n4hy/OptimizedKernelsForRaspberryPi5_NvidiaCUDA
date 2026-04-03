/**
 * OptMathKernels CUDA Backend Implementation
 * Copyright (c) 2026 Dr Robert W McGwier, PhD
 * SPDX-License-Identifier: MIT
 *
 * Core CUDA context management, memory operations, and library integration.
 */

#include "optmath/cuda_backend.hpp"
#include <iostream>
#include <sstream>
#include <cstring>

#ifdef OPTMATH_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>

// Error checking macros
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            return;                                                            \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_RETURN(call, ret)                                           \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            return ret;                                                        \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__      \
                      << ": " << status << std::endl;                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t status = call;                                        \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__    \
                      << ": " << status << std::endl;                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult status = call;                                             \
        if (status != CUFFT_SUCCESS) {                                         \
            std::cerr << "cuFFT error in " << __FILE__ << ":" << __LINE__       \
                      << ": " << status << std::endl;                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#endif // OPTMATH_USE_CUDA

namespace optmath {
namespace cuda {

// =============================================================================
// Device Information
// =============================================================================

bool is_available() {
#ifdef OPTMATH_USE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

int get_device_count() {
#ifdef OPTMATH_USE_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
#else
    return 0;
#endif
}

DeviceInfo get_device_info(int device_id) {
    DeviceInfo info = {};
    info.device_id = device_id;

#ifdef OPTMATH_USE_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return info;
    }

    info.name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_memory = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.warp_size = prop.warpSize;
    info.memory_bus_width = prop.memoryBusWidth;
    info.l2_cache_size = prop.l2CacheSize;

    // Memory bandwidth in GB/s
    // Note: memoryClockRate was removed in CUDA 13.0, use peak memory bandwidth if available
#if CUDART_VERSION >= 13000
    // CUDA 13+ doesn't expose memory clock rate directly
    // Use a reasonable estimate based on memory bus width and modern memory speeds
    // For HBM2e: ~2.4 GT/s, for GDDR6X: ~21 Gbps per pin
    info.memory_bandwidth_gbps = static_cast<float>(prop.memoryBusWidth) * 2.0f / 8.0f; // Approximate
#else
    info.memory_bandwidth_gbps = 2.0f * prop.memoryClockRate *
                                  (prop.memoryBusWidth / 8) / 1.0e6f;
#endif

    // Shared memory info
    info.shared_memory_per_block = prop.sharedMemPerBlock;
    info.shared_memory_per_multiprocessor = prop.sharedMemPerMultiprocessor;

    // Feature detection based on compute capability
    int cc = prop.major * 10 + prop.minor;
    info.fp16_support = (cc >= 60);        // Pascal+ (SM 6.0+)
    info.tensor_cores = (cc >= 70);        // Volta+ (SM 7.0+)
    info.tf32_support = (cc >= 80);        // Ampere+ (SM 8.0+)
    info.fp8_support = (cc >= 100);        // Blackwell+ (SM 10.0+)
    info.is_blackwell_arch = (cc >= 100);  // Blackwell architecture
    info.unified_memory = (prop.managedMemory != 0);

    // Get free memory
    size_t free_mem, total_mem;
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);
    cudaSetDevice(current_device);
    info.free_memory = free_mem;
#endif

    return info;
}

bool is_device_supported(int device_id) {
#ifdef OPTMATH_USE_CUDA
    DeviceInfo info = get_device_info(device_id);
    return info.is_supported_by_toolkit();
#else
    return false;
#endif
}

void print_device_info(int device_id) {
    DeviceInfo info = get_device_info(device_id);

    std::cout << "=== CUDA Device " << device_id << " ===" << std::endl;
    std::cout << "Name: " << info.name << std::endl;
    std::cout << "Compute Capability: " << info.compute_capability_major << "."
              << info.compute_capability_minor << std::endl;

    // Architecture generation
    std::string arch_name;
    if (info.is_blackwell()) arch_name = "Blackwell";
    else if (info.is_hopper()) arch_name = "Hopper";
    else if (info.is_ada()) arch_name = "Ada Lovelace";
    else if (info.is_ampere()) arch_name = "Ampere";
    else if (info.is_turing()) arch_name = "Turing";
    else if (info.is_volta()) arch_name = "Volta";
    else if (info.is_pascal()) arch_name = "Pascal";
    else if (info.is_maxwell()) arch_name = "Maxwell";
    else arch_name = "Unknown/Legacy";
    std::cout << "Architecture: " << arch_name << std::endl;

    std::cout << "Total Memory: " << (info.total_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Free Memory: " << (info.free_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << info.multiprocessor_count << std::endl;
    std::cout << "Max Threads/Block: " << info.max_threads_per_block << std::endl;
    std::cout << "Warp Size: " << info.warp_size << std::endl;
    std::cout << "Memory Bandwidth: " << info.memory_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "L2 Cache: " << (info.l2_cache_size / 1024) << " KB" << std::endl;
    std::cout << "Shared Memory/Block: " << (info.shared_memory_per_block / 1024) << " KB" << std::endl;
    std::cout << "Shared Memory/SM: " << (info.shared_memory_per_multiprocessor / 1024) << " KB" << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  FP16 Arithmetic: " << (info.supports_fp16_arithmetic() ? "Yes" : "No") << " (Pascal+)" << std::endl;
    std::cout << "  Tensor Cores: " << (info.supports_tensor_cores() ? "Yes" : "No") << " (Volta+)" << std::endl;
    std::cout << "  TF32 Tensor: " << (info.supports_tf32() ? "Yes" : "No") << " (Ampere+)" << std::endl;
    std::cout << "  FP8: " << (info.supports_fp8() ? "Yes" : "No") << " (Blackwell+)" << std::endl;
    std::cout << "  Unified Memory: " << (info.supports_unified_memory() ? "Yes" : "No") << std::endl;
    std::cout << "  Supported by Toolkit: " << (info.is_supported_by_toolkit() ? "Yes" : "No") << std::endl;
    if (!info.is_supported_by_toolkit()) {
        std::cout << "    Recommended CUDA version: " << (info.recommended_cuda_version() / 1000)
                  << "." << ((info.recommended_cuda_version() % 1000) / 10) << std::endl;
    }
}

// =============================================================================
// CUDA Stream
// =============================================================================

CudaStream::CudaStream() {
#ifdef OPTMATH_USE_CUDA
    cudaStreamCreate(&m_stream);
#endif
}

CudaStream::~CudaStream() {
#ifdef OPTMATH_USE_CUDA
    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }
#endif
}

CudaStream::CudaStream(CudaStream&& other) noexcept {
#ifdef OPTMATH_USE_CUDA
    m_stream = other.m_stream;
    other.m_stream = nullptr;
#endif
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
#ifdef OPTMATH_USE_CUDA
    if (this != &other) {
        if (m_stream) {
            cudaStreamDestroy(m_stream);
        }
        m_stream = other.m_stream;
        other.m_stream = nullptr;
    }
#endif
    return *this;
}

void CudaStream::synchronize() {
#ifdef OPTMATH_USE_CUDA
    cudaStreamSynchronize(m_stream);
#endif
}

bool CudaStream::query() const {
#ifdef OPTMATH_USE_CUDA
    return cudaStreamQuery(m_stream) == cudaSuccess;
#else
    return true;
#endif
}

// =============================================================================
// CUDA Context
// =============================================================================

CudaContext& CudaContext::get() {
    static CudaContext instance;
    return instance;
}

CudaContext::~CudaContext() {
    cleanup();
}

bool CudaContext::init(int device_id) {
    if (m_initialized) {
        return true;
    }

#ifdef OPTMATH_USE_CUDA
    // Check device availability
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA: No devices available" << std::endl;
        return false;
    }

    if (device_id >= device_count) {
        std::cerr << "CUDA: Invalid device ID " << device_id << std::endl;
        return false;
    }

    // Set device
    CUDA_CHECK_RETURN(cudaSetDevice(device_id), false);
    m_device_id = device_id;

    // Check if GPU architecture is supported by compiled toolkit
    DeviceInfo info = get_device_info(device_id);
    if (!info.is_supported_by_toolkit()) {
        std::cerr << "CUDA: GPU architecture SM " << info.compute_capability_major << "."
                  << info.compute_capability_minor << " is not supported by CUDA toolkit "
                  << (CUDART_VERSION / 1000) << "." << ((CUDART_VERSION % 1000) / 10) << std::endl;
        std::cerr << "  Blackwell (SM 10.0+) requires CUDA 13.x or later" << std::endl;
        std::cerr << "  Falling back to CPU implementation" << std::endl;
        return false;
    }

    // Create cuBLAS handle
    cublasStatus_t cublas_status = cublasCreate(&m_cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS: Failed to create handle" << std::endl;
        return false;
    }

    // Create cuSOLVER handle
    cusolverStatus_t cusolver_status = cusolverDnCreate(&m_cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER: Failed to create handle" << std::endl;
        cublasDestroy(m_cublas_handle);
        return false;
    }

    // Create default stream
    CUDA_CHECK_RETURN(cudaStreamCreate(&m_default_stream), false);

    // Set cuBLAS to use default stream
    cublasSetStream(m_cublas_handle, m_default_stream);
    cusolverDnSetStream(m_cusolver_handle, m_default_stream);

    // Print initialization info (reuse info from architecture check above)
    std::cout << "CUDA Context initialized on " << info.name
              << " (SM " << info.compute_capability_major << "."
              << info.compute_capability_minor << ")" << std::endl;

    // Enable TF32 on Ampere+ for better performance
    if (info.tf32_support) {
        cublasSetMathMode(m_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        std::cout << "  TF32 Tensor Core math enabled" << std::endl;
    }

    m_initialized = true;
    return true;
#else
    std::cerr << "CUDA support not compiled" << std::endl;
    return false;
#endif
}

void CudaContext::cleanup() {
#ifdef OPTMATH_USE_CUDA
    if (m_initialized) {
        if (m_default_stream) {
            cudaStreamDestroy(m_default_stream);
            m_default_stream = nullptr;
        }
        if (m_cusolver_handle) {
            cusolverDnDestroy(m_cusolver_handle);
            m_cusolver_handle = nullptr;
        }
        if (m_cublas_handle) {
            cublasDestroy(m_cublas_handle);
            m_cublas_handle = nullptr;
        }
        m_initialized = false;
    }
#endif
}

size_t CudaContext::get_free_memory() const {
#ifdef OPTMATH_USE_CUDA
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
#else
    return 0;
#endif
}

size_t CudaContext::get_total_memory() const {
#ifdef OPTMATH_USE_CUDA
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
#else
    return 0;
#endif
}

void CudaContext::synchronize() {
#ifdef OPTMATH_USE_CUDA
    if (m_default_stream) {
        cudaStreamSynchronize(m_default_stream);
    }
#endif
}

void CudaContext::set_precision_mode(PrecisionMode mode) {
#ifdef OPTMATH_USE_CUDA
    m_precision_mode = mode;

    switch (mode) {
        case PrecisionMode::FP32:
            cublasSetMathMode(m_cublas_handle, CUBLAS_DEFAULT_MATH);
            break;
        case PrecisionMode::TF32:
            cublasSetMathMode(m_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
            break;
        case PrecisionMode::FP16:
        case PrecisionMode::MIXED_FP16_FP32:
            cublasSetMathMode(m_cublas_handle, CUBLAS_TENSOR_OP_MATH);
            break;
        case PrecisionMode::FP64:
            cublasSetMathMode(m_cublas_handle, CUBLAS_DEFAULT_MATH);
            break;
    }
#endif
}

// =============================================================================
// Device Buffer Template Implementations
// =============================================================================

template<typename T>
DeviceBuffer<T>::DeviceBuffer(size_t count) {
    allocate(count);
}

template<typename T>
DeviceBuffer<T>::~DeviceBuffer() {
    free();
}

template<typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer&& other) noexcept
    : m_data(other.m_data), m_size(other.m_size) {
    other.m_data = nullptr;
    other.m_size = 0;
}

template<typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        m_data = other.m_data;
        m_size = other.m_size;
        other.m_data = nullptr;
        other.m_size = 0;
    }
    return *this;
}

template<typename T>
void DeviceBuffer<T>::allocate(size_t count) {
#ifdef OPTMATH_USE_CUDA
    free();
    if (count > 0) {
        cudaError_t err = cudaMalloc(&m_data, count * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in DeviceBuffer::allocate: "
                      << cudaGetErrorString(err) << std::endl;
            m_data = nullptr;
            m_size = 0;
            return;
        }
        m_size = count;
    }
#endif
}

template<typename T>
void DeviceBuffer<T>::free() {
#ifdef OPTMATH_USE_CUDA
    if (m_data) {
        cudaFree(m_data);
        m_data = nullptr;
        m_size = 0;
    }
#endif
}

template<typename T>
void DeviceBuffer<T>::copy_from_host(const T* host_data, size_t count) {
#ifdef OPTMATH_USE_CUDA
    if (m_data && count <= m_size) {
        cudaError_t err = cudaMemcpy(m_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in DeviceBuffer::copy_from_host: "
                      << cudaGetErrorString(err) << std::endl;
        }
    }
#endif
}

template<typename T>
void DeviceBuffer<T>::copy_to_host(T* host_data, size_t count) const {
#ifdef OPTMATH_USE_CUDA
    if (m_data && count <= m_size) {
        cudaError_t err = cudaMemcpy(host_data, m_data, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in DeviceBuffer::copy_to_host: "
                      << cudaGetErrorString(err) << std::endl;
        }
    }
#endif
}

template<typename T>
void DeviceBuffer<T>::copy_from_host_async(const T* host_data, size_t count, CudaStream& stream) {
#ifdef OPTMATH_USE_CUDA
    if (m_data && count <= m_size) {
        cudaMemcpyAsync(m_data, host_data, count * sizeof(T),
                        cudaMemcpyHostToDevice, stream.get());
    }
#endif
}

template<typename T>
void DeviceBuffer<T>::copy_to_host_async(T* host_data, size_t count, CudaStream& stream) const {
#ifdef OPTMATH_USE_CUDA
    if (m_data && count <= m_size) {
        cudaMemcpyAsync(host_data, m_data, count * sizeof(T),
                        cudaMemcpyDeviceToHost, stream.get());
    }
#endif
}

// Explicit template instantiations
template class DeviceBuffer<float>;
template class DeviceBuffer<double>;
template class DeviceBuffer<int>;
template class DeviceBuffer<unsigned int>;

// =============================================================================
// Pinned Buffer Template Implementations
// =============================================================================

template<typename T>
PinnedBuffer<T>::PinnedBuffer(size_t count) {
    allocate(count);
}

template<typename T>
PinnedBuffer<T>::~PinnedBuffer() {
    free();
}

template<typename T>
void PinnedBuffer<T>::allocate(size_t count) {
#ifdef OPTMATH_USE_CUDA
    free();
    if (count > 0) {
        cudaError_t err = cudaMallocHost(&m_data, count * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] PinnedBuffer allocation failed: " << cudaGetErrorString(err) << std::endl;
            m_data = nullptr;
            m_size = 0;
            return;
        }
        m_size = count;
    }
#endif
}

template<typename T>
void PinnedBuffer<T>::free() {
#ifdef OPTMATH_USE_CUDA
    if (m_data) {
        cudaFreeHost(m_data);
        m_data = nullptr;
        m_size = 0;
    }
#endif
}

template class PinnedBuffer<float>;
template class PinnedBuffer<double>;

// =============================================================================
// Unified Buffer Template Implementations
// =============================================================================

template<typename T>
UnifiedBuffer<T>::UnifiedBuffer(size_t count) {
    allocate(count);
}

template<typename T>
UnifiedBuffer<T>::~UnifiedBuffer() {
    free();
}

template<typename T>
void UnifiedBuffer<T>::allocate(size_t count) {
#ifdef OPTMATH_USE_CUDA
    free();
    if (count > 0) {
        cudaError_t err = cudaMallocManaged(&m_data, count * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] UnifiedBuffer allocation failed: " << cudaGetErrorString(err) << std::endl;
            m_data = nullptr;
            m_size = 0;
            return;
        }
        m_size = count;
    }
#endif
}

template<typename T>
void UnifiedBuffer<T>::free() {
#ifdef OPTMATH_USE_CUDA
    if (m_data) {
        cudaFree(m_data);
        m_data = nullptr;
        m_size = 0;
    }
#endif
}

template<typename T>
void UnifiedBuffer<T>::prefetch_to_device(int device_id) {
#ifdef OPTMATH_USE_CUDA
    if (m_data && m_size > 0) {
#if CUDART_VERSION >= 13000
        // CUDA 13+ uses cudaMemLocation for device specification
        cudaMemLocation location;
        location.type = cudaMemLocationTypeDevice;
        location.id = device_id;
        cudaMemPrefetchAsync(m_data, m_size * sizeof(T), location, 0);
#else
        cudaMemPrefetchAsync(m_data, m_size * sizeof(T), device_id);
#endif
    }
#endif
}

template<typename T>
void UnifiedBuffer<T>::prefetch_to_host() {
#ifdef OPTMATH_USE_CUDA
    if (m_data && m_size > 0) {
#if CUDART_VERSION >= 13000
        // CUDA 13+ uses cudaMemLocation for host specification
        cudaMemLocation location;
        location.type = cudaMemLocationTypeHost;
        location.id = 0;
        cudaMemPrefetchAsync(m_data, m_size * sizeof(T), location, 0);
#else
        cudaMemPrefetchAsync(m_data, m_size * sizeof(T), cudaCpuDeviceId);
#endif
    }
#endif
}

template class UnifiedBuffer<float>;
template class UnifiedBuffer<double>;

// =============================================================================
// FFT Plan
// =============================================================================

CudaFFTPlan::~CudaFFTPlan() {
    destroy();
}

bool CudaFFTPlan::create_1d(size_t n, bool inverse) {
#ifdef OPTMATH_USE_CUDA
    destroy();
    cufftResult result = cufftPlan1d(&m_plan, static_cast<int>(n), CUFFT_C2C, 1);
    m_valid = (result == CUFFT_SUCCESS);
    return m_valid;
#else
    return false;
#endif
}

bool CudaFFTPlan::create_1d_batch(size_t n, size_t batch, bool inverse) {
#ifdef OPTMATH_USE_CUDA
    destroy();
    int rank = 1;
    int nfft[] = {static_cast<int>(n)};
    cufftResult result = cufftPlanMany(&m_plan, rank, nfft,
                                        nullptr, 1, static_cast<int>(n),
                                        nullptr, 1, static_cast<int>(n),
                                        CUFFT_C2C, static_cast<int>(batch));
    m_valid = (result == CUFFT_SUCCESS);
    return m_valid;
#else
    return false;
#endif
}

bool CudaFFTPlan::create_2d(size_t nx, size_t ny, bool inverse) {
#ifdef OPTMATH_USE_CUDA
    destroy();
    cufftResult result = cufftPlan2d(&m_plan, static_cast<int>(ny),
                                      static_cast<int>(nx), CUFFT_C2C);
    m_valid = (result == CUFFT_SUCCESS);
    return m_valid;
#else
    return false;
#endif
}

void CudaFFTPlan::execute(float* inout) {
#ifdef OPTMATH_USE_CUDA
    if (m_valid) {
        cufftResult result = cufftExecC2C(m_plan, reinterpret_cast<cufftComplex*>(inout),
                     reinterpret_cast<cufftComplex*>(inout), CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "[cuFFT] Execute failed with error: " << result << std::endl;
        }
    }
#endif
}

void CudaFFTPlan::execute(const float* in, float* out) {
#ifdef OPTMATH_USE_CUDA
    if (m_valid) {
        cufftResult result = cufftExecC2C(m_plan,
                     const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(in)),
                     reinterpret_cast<cufftComplex*>(out), CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "[cuFFT] Execute failed with error: " << result << std::endl;
        }
    }
#endif
}

void CudaFFTPlan::destroy() {
#ifdef OPTMATH_USE_CUDA
    if (m_valid) {
        cufftDestroy(m_plan);
        m_plan = 0;
        m_valid = false;
    }
#endif
}

// =============================================================================
// CUDA Timer
// =============================================================================

CudaTimer::CudaTimer() {
#ifdef OPTMATH_USE_CUDA
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
#endif
}

CudaTimer::~CudaTimer() {
#ifdef OPTMATH_USE_CUDA
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
#endif
}

void CudaTimer::start() {
#ifdef OPTMATH_USE_CUDA
    cudaEventRecord(m_start);
    m_running = true;
#endif
}

void CudaTimer::stop() {
#ifdef OPTMATH_USE_CUDA
    cudaEventRecord(m_stop);
    cudaEventSynchronize(m_stop);
    m_running = false;
#endif
}

float CudaTimer::elapsed_ms() const {
#ifdef OPTMATH_USE_CUDA
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, m_start, m_stop);
    return ms;
#else
    return 0.0f;
#endif
}

// =============================================================================
// Multi-GPU Support
// =============================================================================

void set_device(int device_id) {
#ifdef OPTMATH_USE_CUDA
    cudaSetDevice(device_id);
#endif
}

int get_device() {
#ifdef OPTMATH_USE_CUDA
    int device;
    cudaGetDevice(&device);
    return device;
#else
    return -1;
#endif
}

bool enable_peer_access(int device_from, int device_to) {
#ifdef OPTMATH_USE_CUDA
    int can_access;
    cudaDeviceCanAccessPeer(&can_access, device_from, device_to);
    if (!can_access) {
        return false;
    }

    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_from);
    cudaError_t err = cudaDeviceEnablePeerAccess(device_to, 0);
    cudaSetDevice(current_device);

    return (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
#else
    return false;
#endif
}

// =============================================================================
// Bandwidth Measurement
// =============================================================================

BandwidthStats measure_bandwidth(size_t bytes) {
    BandwidthStats stats = {};

#ifdef OPTMATH_USE_CUDA
    void* d_data;
    void* h_data;

    cudaMalloc(&d_data, bytes);
    cudaMallocHost(&h_data, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // Host to Device
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    stats.host_to_device_gbps = (bytes / 1e9f) / (ms / 1000.0f);

    // Device to Host
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    stats.device_to_host_gbps = (bytes / 1e9f) / (ms / 1000.0f);

    // Device to Device (if we have a second buffer)
    void* d_data2;
    cudaMalloc(&d_data2, bytes);
    cudaEventRecord(start);
    cudaMemcpy(d_data2, d_data, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    stats.device_to_device_gbps = (bytes / 1e9f) / (ms / 1000.0f);

    cudaFree(d_data2);
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    return stats;
}

// =============================================================================
// Error Handling
// =============================================================================

std::string get_last_error() {
#ifdef OPTMATH_USE_CUDA
    cudaError_t err = cudaGetLastError();
    return std::string(cudaGetErrorString(err));
#else
    return "CUDA not available";
#endif
}

bool check_cuda_error(const char* operation) {
#ifdef OPTMATH_USE_CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error";
        if (operation) {
            std::cerr << " in " << operation;
        }
        std::cerr << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
#else
    return true;
#endif
}

} // namespace cuda
} // namespace optmath
