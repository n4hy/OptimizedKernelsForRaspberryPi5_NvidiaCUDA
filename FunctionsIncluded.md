# OptMathKernels API Reference

Complete API documentation for OptMathKernels - High-Performance Numerical Library for Raspberry Pi 5 and NVIDIA GPUs.

**Version**: 1.0.0
**Total Functions**: 417+
**Backends**: NEON (ARM), CUDA (NVIDIA), Vulkan (Cross-platform)

---

## Table of Contents

- [NEON Backend (ARM SIMD)](#neon-backend-arm-simd)
- [CUDA Backend (NVIDIA GPU)](#cuda-backend-nvidia-gpu)
- [Vulkan Backend (Cross-platform GPU)](#vulkan-backend-cross-platform-gpu)
- [Radar Kernels (Signal Processing)](#radar-kernels-signal-processing)
- [Quick Reference Tables](#quick-reference-tables)

---

## NEON Backend (ARM SIMD)

**Header**: `#include <optmath/neon_kernels.hpp>`
**Namespace**: `optmath::neon`
**Target**: ARM Cortex-A76 (Raspberry Pi 5), ARMv8-A with NEON

### Availability Check

```cpp
bool is_available();
```
Returns `true` if NEON acceleration was compiled in and is available.

---

### Core Vector Operations (Low-Level)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_dot_f32` | `float` | `const float* a, const float* b, std::size_t n` | Dot product of two float32 arrays |
| `neon_dot_f64` | `double` | `const double* a, const double* b, std::size_t n` | Dot product of two float64 arrays |
| `neon_add_f32` | `void` | `float* out, const float* a, const float* b, std::size_t n` | Element-wise addition: `out = a + b` |
| `neon_sub_f32` | `void` | `float* out, const float* a, const float* b, std::size_t n` | Element-wise subtraction: `out = a - b` |
| `neon_mul_f32` | `void` | `float* out, const float* a, const float* b, std::size_t n` | Element-wise multiplication: `out = a * b` |
| `neon_div_f32` | `void` | `float* out, const float* a, const float* b, std::size_t n` | Element-wise division: `out = a / b` |

---

### Reductions (Low-Level)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_norm_f32` | `float` | `const float* a, std::size_t n` | L2 norm: `sqrt(sum(a[i]^2))` |
| `neon_reduce_sum_f32` | `float` | `const float* a, std::size_t n` | Sum all elements |
| `neon_reduce_max_f32` | `float` | `const float* a, std::size_t n` | Maximum element |
| `neon_reduce_min_f32` | `float` | `const float* a, std::size_t n` | Minimum element |

---

### Matrix Operations (Low-Level)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_gemm_4x4_f32` | `void` | `float* C, const float* A, std::size_t lda, const float* B, std::size_t ldb, std::size_t ldc` | 4x4 GEMM microkernel: `C += A * B` |
| `neon_gemm_blocked_f32` | `void` | `float* C, const float* A, const float* B, std::size_t M, std::size_t N, std::size_t K, std::size_t lda, std::size_t ldb, std::size_t ldc` | Cache-blocked GEMM (MC=128, KC=256, NC=512) |

**Cache Blocking Parameters** (optimized for Cortex-A76):
- MC = 128 (rows of A per block)
- KC = 256 (columns of A / rows of B per block)
- NC = 512 (columns of B per block)
- 8x8 microkernel with 4x4 register tiles

---

### DSP / Filter Operations (Low-Level)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_fir_f32` | `void` | `const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y` | FIR filter: `y = x * h` (convolution) |

---

### Polyphase Resampler (Low-Level)

Rational sample rate conversion by L/M using polyphase decomposition with NEON-optimized FIR per phase.

**Structures:**
```cpp
struct PolyphaseResamplerState {
    std::vector<std::vector<float>> phases;  // Polyphase decomposition [L][n_taps]
    std::size_t L;              // Interpolation factor
    std::size_t M;              // Decimation factor
    std::size_t n_taps;         // Taps per phase
    std::vector<float> delay;   // Delay line for streaming
    std::size_t delay_pos;      // Write position in circular delay line
    std::size_t phase_acc;      // Phase accumulator
};
```

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_resample_init` | `void` | `PolyphaseResamplerState& state, const float* filter, std::size_t filter_len, std::size_t L, std::size_t M` | Initialize resampler with prototype lowpass filter |
| `neon_resample_f32` | `std::size_t` | `float* out, const float* in, std::size_t input_len, PolyphaseResamplerState& state` | Streaming resampler (returns output sample count) |
| `neon_resample_oneshot_f32` | `void` | `float* out, std::size_t* output_len, const float* in, std::size_t input_len, const float* filter, std::size_t filter_len, std::size_t L, std::size_t M` | One-shot resampler (non-streaming) |

---

### Biquad IIR Filter (Low-Level)

Direct Form II Transposed biquad filter with cascade support and design helpers.

**Structures:**
```cpp
struct BiquadCoeffs {
    float b0, b1, b2;  // Numerator (feedforward)
    float a1, a2;       // Denominator (feedback), a0 normalized to 1
};

struct BiquadState {
    float s1 = 0.0f;   // DF2T state variable 1
    float s2 = 0.0f;   // DF2T state variable 2
};
```

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_biquad_f32` | `void` | `float* out, const float* in, std::size_t n, const BiquadCoeffs& coeffs, BiquadState& state` | Process single biquad section (in-place OK) |
| `neon_biquad_cascade_f32` | `void` | `float* out, const float* in, std::size_t n, const BiquadCoeffs* coeffs, BiquadState* states, std::size_t n_sections` | Process cascade of biquad sections |
| `neon_biquad_lowpass` | `BiquadCoeffs` | `float fc, float fs, float Q = 0.707` | Design 2nd-order Butterworth lowpass |
| `neon_biquad_highpass` | `BiquadCoeffs` | `float fc, float fs, float Q = 0.707` | Design 2nd-order Butterworth highpass |
| `neon_biquad_bandpass` | `BiquadCoeffs` | `float fc, float fs, float Q = 1.0` | Design 2nd-order bandpass (constant 0dB peak) |
| `neon_biquad_notch` | `BiquadCoeffs` | `float fc, float fs, float Q = 1.0` | Design 2nd-order notch (band-reject) |

---

### 2D Convolution (Low-Level)

NEON-vectorized 2D convolution with row-major layout. Valid mode (no padding).
Output size: `(in_rows - kernel_rows + 1) x (in_cols - kernel_cols + 1)`.

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_conv2d_f32` | `void` | `float* out, const float* in, std::size_t in_rows, std::size_t in_cols, const float* kernel, std::size_t kernel_rows, std::size_t kernel_cols` | General NxM 2D convolution |
| `neon_conv2d_separable_f32` | `void` | `float* out, const float* in, std::size_t in_rows, std::size_t in_cols, const float* row_kernel, std::size_t row_kernel_len, const float* col_kernel, std::size_t col_kernel_len` | Separable 2D convolution (row then column pass) |
| `neon_conv2d_3x3_f32` | `void` | `float* out, const float* in, std::size_t in_rows, std::size_t in_cols, const float kernel[9]` | Optimized 3x3 convolution (fully unrolled) |
| `neon_conv2d_5x5_f32` | `void` | `float* out, const float* in, std::size_t in_rows, std::size_t in_cols, const float kernel[25]` | Optimized 5x5 convolution (unrolled) |

---

### Activation Functions (In-Place)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_relu_f32` | `void` | `float* data, std::size_t n` | ReLU: `max(0, x)` |
| `neon_sigmoid_f32` | `void` | `float* data, std::size_t n` | Sigmoid: `1/(1+exp(-x))` (scalar) |
| `neon_tanh_f32` | `void` | `float* data, std::size_t n` | Hyperbolic tangent (scalar) |

---

### Vectorized Transcendentals (Fast Approximations)

These functions use NEON SIMD for 4-8x speedup, trading accuracy for speed.
Typical accuracy: exp ~12%, sin/cos ~1e-5, sigmoid ~3%, tanh ~6%.

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_fast_exp_f32` | `void` | `float* out, const float* in, std::size_t n` | Vectorized exp (6th-order polynomial, ~12% error) |
| `neon_fast_sin_f32` | `void` | `float* out, const float* in, std::size_t n` | Vectorized sin (Chebyshev polynomial, ~1e-5 error) |
| `neon_fast_cos_f32` | `void` | `float* out, const float* in, std::size_t n` | Vectorized cos (~1e-5 error) |
| `neon_fast_sigmoid_f32` | `void` | `float* out, const float* in, std::size_t n` | Fast vectorized sigmoid (~3% error) |
| `neon_fast_tanh_f32` | `void` | `float* out, const float* in, std::size_t n` | Fast vectorized tanh (~6% error) |

**Performance** (Raspberry Pi 5):
- `exp`: ~13 GFLOPS (45x faster than scalar)
- `sin/cos`: ~10 GFLOPS (30x faster than scalar)
- `sigmoid/tanh`: ~8 GFLOPS (25x faster than scalar)

---

### Complex Number Operations (Separate Real/Imag)

For C/ctypes interop where complex data is in separate arrays.

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_complex_mul_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, std::size_t n` | Complex multiply: `out = a * b` |
| `neon_complex_conj_mul_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, std::size_t n` | Complex conjugate multiply: `out = a * conj(b)` |
| `neon_complex_dot_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, std::size_t n` | Complex dot product: `sum(a * conj(b))` |
| `neon_complex_magnitude_f32` | `void` | `float* out, const float* re, const float* im, std::size_t n` | Magnitude: `sqrt(re^2 + im^2)` |
| `neon_complex_magnitude_squared_f32` | `void` | `float* out, const float* re, const float* im, std::size_t n` | Squared magnitude: `re^2 + im^2` |
| `neon_complex_phase_f32` | `void` | `float* out, const float* re, const float* im, std::size_t n` | Phase angle: `atan2(im, re)` |
| `neon_complex_add_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, std::size_t n` | Complex addition |
| `neon_complex_scale_f32` | `void` | `float* out_re, float* out_im, const float* in_re, const float* in_im, float scale_re, float scale_im, std::size_t n` | Complex scalar multiply |
| `neon_complex_exp_f32` | `void` | `float* out_re, float* out_im, const float* phase, std::size_t n` | Complex exponential: `exp(j*phase)` |

---

### Complex Number Operations (Interleaved)

For IQ data format: `[re0, im0, re1, im1, ...]`

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_complex_mul_interleaved_f32` | `void` | `float* out, const float* a, const float* b, std::size_t n` | Complex multiply (interleaved format) |
| `neon_complex_conj_mul_interleaved_f32` | `void` | `float* out, const float* a, const float* b, std::size_t n` | Complex conjugate multiply (interleaved) |

---

### Eigen Vector Wrappers

High-level C++ interface using Eigen types.

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_dot` | `float` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Dot product |
| `neon_dot` | `double` | `const Eigen::VectorXd& a, const Eigen::VectorXd& b` | Double-precision dot product |
| `neon_add` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector addition |
| `neon_sub` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector subtraction |
| `neon_mul` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise multiplication |
| `neon_div` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise division |
| `neon_norm` | `float` | `const Eigen::VectorXf& a` | L2 norm |
| `neon_reduce_sum` | `float` | `const Eigen::VectorXf& a` | Sum all elements |
| `neon_reduce_max` | `float` | `const Eigen::VectorXf& a` | Maximum element |
| `neon_reduce_min` | `float` | `const Eigen::VectorXf& a` | Minimum element |
| `neon_fir` | `Eigen::VectorXf` | `const Eigen::VectorXf& x, const Eigen::VectorXf& h` | FIR filter |
| `neon_relu` | `void` | `Eigen::VectorXf& x` | In-place ReLU |
| `neon_sigmoid` | `void` | `Eigen::VectorXf& x` | In-place sigmoid |
| `neon_tanh` | `void` | `Eigen::VectorXf& x` | In-place tanh |

---

### Eigen Matrix Wrappers

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_gemm` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A, const Eigen::MatrixXf& B` | Matrix multiply: `A * B` |
| `neon_gemm_blocked` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A, const Eigen::MatrixXf& B` | Optimized blocked GEMM |
| `neon_mat_scale` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A, float s` | Scalar multiply: `A * s` |
| `neon_mat_transpose` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Matrix transpose |
| `neon_mat_vec_mul` | `Eigen::VectorXf` | `const Eigen::MatrixXf& A, const Eigen::VectorXf& v` | Matrix-vector multiply: `A * v` |

---

### Eigen Complex Wrappers

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_complex_mul` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& a, const Eigen::VectorXcf& b` | Complex multiply |
| `neon_complex_conj_mul` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& a, const Eigen::VectorXcf& b` | Complex conjugate multiply |
| `neon_complex_dot` | `std::complex<float>` | `const Eigen::VectorXcf& a, const Eigen::VectorXcf& b` | Complex dot product |
| `neon_complex_magnitude` | `Eigen::VectorXf` | `const Eigen::VectorXcf& a` | Magnitude of complex vector |
| `neon_complex_phase` | `Eigen::VectorXf` | `const Eigen::VectorXcf& a` | Phase of complex vector |

---

### Eigen DSP Wrappers

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_resample` | `Eigen::VectorXf` | `const Eigen::VectorXf& in, const Eigen::VectorXf& filter, std::size_t L, std::size_t M` | Polyphase resampler (one-shot) |
| `neon_biquad` | `Eigen::VectorXf` | `const Eigen::VectorXf& in, const BiquadCoeffs& coeffs` | Biquad IIR filter (single section) |
| `neon_conv2d` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& in, const Eigen::MatrixXf& kernel` | 2D convolution (handles col/row-major conversion) |

---

### Dense Linear Algebra (Low-Level)

Column-major layout. All operations are in-place unless noted. NEON-vectorized AXPY/dot/scale for contiguous column data.

#### Triangular Solve

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_trsv_lower_f32` | `void` | `float* b, const float* L, std::size_t n, std::size_t ldl` | Forward substitution: solve L*x = b |
| `neon_trsv_upper_f32` | `void` | `float* b, const float* U, std::size_t n, std::size_t ldu` | Backward substitution: solve U*x = b |
| `neon_trsv_lower_unit_f32` | `void` | `float* b, const float* L, std::size_t n, std::size_t ldl` | Unit-diagonal forward substitution |
| `neon_trsv_lower_trans_f32` | `void` | `float* b, const float* L, std::size_t n, std::size_t ldl` | Solve L^T*x = b using lower L |
| `neon_trsm_lower_f32` | `void` | `float* B, const float* L, std::size_t n, std::size_t nrhs, std::size_t ldl, std::size_t ldb` | Multi-RHS lower triangular solve |
| `neon_trsm_upper_f32` | `void` | `float* B, const float* U, std::size_t n, std::size_t nrhs, std::size_t ldu, std::size_t ldb` | Multi-RHS upper triangular solve |

#### Decompositions

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_cholesky_f32` | `int` | `float* A, std::size_t n, std::size_t lda` | Cholesky A = L*L^T (returns 0 or failing pivot) |
| `neon_lu_f32` | `int` | `float* A, int* piv, std::size_t m, std::size_t n, std::size_t lda` | LU with partial pivoting (returns 0 or failing pivot) |
| `neon_qr_f32` | `void` | `float* A, float* tau, std::size_t m, std::size_t n, std::size_t lda` | QR via Householder reflections |
| `neon_qr_extract_q_f32` | `void` | `float* Q, const float* A, const float* tau, std::size_t m, std::size_t n, std::size_t lda, std::size_t ldq` | Extract explicit Q from Householder vectors |

#### Solvers

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_solve_f32` | `int` | `float* A, float* b, std::size_t n, std::size_t lda` | General solve via LU |
| `neon_solve_spd_f32` | `int` | `float* A, float* b, std::size_t n, std::size_t lda` | SPD solve via Cholesky |
| `neon_inverse_f32` | `int` | `float* Ainv, const float* A, std::size_t n, std::size_t lda, std::size_t ldinv` | Matrix inverse via LU |

---

### Eigen Dense Linear Algebra Wrappers

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `neon_cholesky` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Cholesky: returns L (empty on failure) |
| `neon_lu` | `pair<MatrixXf, VectorXi>` | `const Eigen::MatrixXf& A` | LU: returns (LU combined, pivot vector) |
| `neon_qr` | `pair<MatrixXf, MatrixXf>` | `const Eigen::MatrixXf& A` | QR: returns (Q, R) |
| `neon_trsv_lower` | `Eigen::VectorXf` | `const Eigen::MatrixXf& L, const Eigen::VectorXf& b` | Solve L*x = b |
| `neon_trsv_upper` | `Eigen::VectorXf` | `const Eigen::MatrixXf& U, const Eigen::VectorXf& b` | Solve U*x = b |
| `neon_solve` | `Eigen::VectorXf` | `const Eigen::MatrixXf& A, const Eigen::VectorXf& b` | General solve A*x = b |
| `neon_solve_spd` | `Eigen::VectorXf` | `const Eigen::MatrixXf& A, const Eigen::VectorXf& b` | SPD solve A*x = b |
| `neon_inverse` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Matrix inverse (empty on failure) |

---

## CUDA Backend (NVIDIA GPU)

**Header**: `#include <optmath/cuda_backend.hpp>`
**Namespace**: `optmath::cuda`
**Target**: NVIDIA GPUs with Compute Capability 7.0+ (Turing, Ampere, Ada Lovelace, Hopper, Blackwell)

### Device Information

```cpp
bool is_available();
int get_device_count();
DeviceInfo get_device_info(int device_id = 0);
void print_device_info(int device_id = 0);
```

**DeviceInfo Structure**:
```cpp
struct DeviceInfo {
    int device_id;
    std::string name;
    int compute_capability_major;    // e.g., 8 for Ampere
    int compute_capability_minor;    // e.g., 6 for RTX 3090
    size_t total_memory;             // Total GPU memory in bytes
    size_t free_memory;              // Available GPU memory
    int multiprocessor_count;        // Number of SMs
    int max_threads_per_block;       // Max threads per block (1024)
    int warp_size;                   // Warp size (32)
    bool tensor_cores;               // Volta+ (SM 7.0+)
    bool tf32_support;               // Ampere+ (SM 8.0+)
    bool fp16_support;               // Pascal+ (SM 6.0+)
    bool fp8_support;                // Blackwell (SM 10.0+)
    bool blackwell;                  // Blackwell architecture
    bool unified_memory;             // Unified Memory support
    float memory_bandwidth_gbps;     // Memory bandwidth
    size_t shared_memory_per_block;  // Shared memory per block

    // Convenience methods
    int compute_major() const;
    int compute_minor() const;
    bool has_tensor_cores() const;
    bool is_ampere_or_newer() const;
    bool is_blackwell_or_newer() const;
};
```

---

### CUDA Context Management

**CudaContext** (Singleton):
```cpp
class CudaContext {
    static CudaContext& get();
    bool init(int device_id = 0);
    void cleanup();
    bool is_initialized() const;
    int device_id() const;
    size_t get_free_memory() const;
    size_t get_total_memory() const;
    void synchronize();

    enum class PrecisionMode { FP32, TF32, FP16, FP64, MIXED_FP16_FP32 };
    void set_precision_mode(PrecisionMode mode);
    PrecisionMode get_precision_mode() const;
};
```

**CudaStream**:
```cpp
class CudaStream {
    CudaStream();
    ~CudaStream();
    void synchronize();
    bool query() const;  // Returns true if stream is idle
    cudaStream_t get() const;
};
```

---

### Memory Management Templates

**DeviceBuffer<T>** - GPU memory:
```cpp
template<typename T>
class DeviceBuffer {
    DeviceBuffer();
    explicit DeviceBuffer(size_t count);
    void allocate(size_t count);
    void free();
    void copy_from_host(const T* host_data, size_t count);
    void copy_to_host(T* host_data, size_t count) const;
    void copy_from_host_async(const T* host_data, size_t count, CudaStream& stream);
    void copy_to_host_async(T* host_data, size_t count, CudaStream& stream) const;
    T* data();
    const T* data() const;
    size_t size() const;
    size_t bytes() const;
    bool empty() const;
};
```

**PinnedBuffer<T>** - Page-locked host memory:
```cpp
template<typename T>
class PinnedBuffer {
    PinnedBuffer();
    explicit PinnedBuffer(size_t count);
    void allocate(size_t count);
    void free();
    T* data();
    size_t size() const;
};
```

**UnifiedBuffer<T>** - Unified Memory (CPU/GPU accessible):
```cpp
template<typename T>
class UnifiedBuffer {
    UnifiedBuffer();
    explicit UnifiedBuffer(size_t count);
    void allocate(size_t count);
    void free();
    void prefetch_to_device(int device_id = 0);
    void prefetch_to_host();
    T* data();
    size_t size() const;
};
```

---

### Vector Operations (Low-Level)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_vec_add_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Vector addition |
| `cuda_vec_mul_f32` | `void` | `float* out, const float* a, const float* b, size_t n` | Element-wise multiplication |
| `cuda_vec_scale_f32` | `void` | `float* out, const float* a, float scalar, size_t n` | Scalar multiplication |
| `cuda_vec_dot_f32` | `float` | `const float* a, const float* b, size_t n` | Dot product |
| `cuda_vec_sum_f32` | `float` | `const float* a, size_t n` | Sum all elements |
| `cuda_vec_max_f32` | `float` | `const float* a, size_t n` | Maximum element |
| `cuda_vec_min_f32` | `float` | `const float* a, size_t n` | Minimum element |
| `cuda_vec_norm_f32` | `float` | `const float* a, size_t n` | L2 norm |
| `cuda_vec_abs_f32` | `void` | `float* out, const float* a, size_t n` | Absolute value |
| `cuda_vec_sqrt_f32` | `void` | `float* out, const float* a, size_t n` | Square root |

---

### Transcendental Functions (CUDA Fast Math)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_exp_f32` | `void` | `float* out, const float* in, size_t n` | Exponential |
| `cuda_log_f32` | `void` | `float* out, const float* in, size_t n` | Natural logarithm |
| `cuda_sin_f32` | `void` | `float* out, const float* in, size_t n` | Sine |
| `cuda_cos_f32` | `void` | `float* out, const float* in, size_t n` | Cosine |
| `cuda_sincos_f32` | `void` | `float* sin_out, float* cos_out, const float* in, size_t n` | Simultaneous sin/cos |
| `cuda_tan_f32` | `void` | `float* out, const float* in, size_t n` | Tangent |
| `cuda_atan2_f32` | `void` | `float* out, const float* y, const float* x, size_t n` | atan2(y, x) |
| `cuda_pow_f32` | `void` | `float* out, const float* base, const float* exp, size_t n` | Power function |

---

### Activation Functions (ML)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_sigmoid_f32` | `void` | `float* out, const float* in, size_t n` | Sigmoid: `1/(1+exp(-x))` |
| `cuda_tanh_f32` | `void` | `float* out, const float* in, size_t n` | Hyperbolic tangent |
| `cuda_relu_f32` | `void` | `float* out, const float* in, size_t n` | ReLU: `max(0, x)` |
| `cuda_leaky_relu_f32` | `void` | `float* out, const float* in, float alpha, size_t n` | Leaky ReLU |
| `cuda_gelu_f32` | `void` | `float* out, const float* in, size_t n` | GELU activation |
| `cuda_softmax_f32` | `void` | `float* out, const float* in, size_t n` | Softmax |

---

### Matrix Operations (cuBLAS)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_mat_mul_f32` | `void` | `float* C, const float* A, const float* B, int M, int N, int K, bool transA, bool transB` | GEMM: `C = A * B` |
| `cuda_mat_add_f32` | `void` | `float* C, const float* A, const float* B, int M, int N` | Matrix addition |
| `cuda_mat_scale_f32` | `void` | `float* out, const float* A, float scalar, int M, int N` | Scalar multiply |
| `cuda_mat_transpose_f32` | `void` | `float* out, const float* A, int M, int N` | Transpose |
| `cuda_mat_vec_mul_f32` | `void` | `float* out, const float* A, const float* x, int M, int N` | Matrix-vector multiply |
| `cuda_mat_mul_tensorcore_f32` | `void` | `float* C, const float* A, const float* B, int M, int N, int K` | Tensor Core GEMM (Ampere+) |
| `cuda_mat_mul_tensorcore_fp16` | `void` | `void* C, const void* A, const void* B, int M, int N, int K` | FP16 Tensor Core GEMM |
| `cuda_batched_mat_mul_f32` | `void` | `float** C, float** A, float** B, int M, int N, int K, int batch` | Batched GEMM |

---

### Linear Algebra (cuSOLVER)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_cholesky` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Cholesky decomposition |
| `cuda_lu` | `pair<Eigen::MatrixXf, Eigen::VectorXi>` | `const Eigen::MatrixXf& A` | LU decomposition with pivots |
| `cuda_qr` | `pair<Eigen::MatrixXf, Eigen::MatrixXf>` | `const Eigen::MatrixXf& A` | QR decomposition (Q, R) |
| `cuda_svd` | `SVDResult` | `const Eigen::MatrixXf& A` | SVD: U, S, Vt |
| `cuda_eig` | `pair<Eigen::VectorXf, Eigen::MatrixXf>` | `const Eigen::MatrixXf& A` | Eigendecomposition |
| `cuda_solve` | `Eigen::VectorXf` | `const Eigen::MatrixXf& A, const Eigen::VectorXf& b` | Linear solve: `Ax = b` |
| `cuda_inverse` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& A` | Matrix inverse |

---

### Complex Number Operations

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_complex_mul_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, size_t n` | Complex multiply |
| `cuda_complex_conj_mul_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, size_t n` | Complex conjugate multiply |
| `cuda_complex_dot_f32` | `void` | `float* out_re, float* out_im, const float* a_re, const float* a_im, const float* b_re, const float* b_im, size_t n` | Complex dot product |
| `cuda_complex_magnitude_f32` | `void` | `float* out, const float* re, const float* im, size_t n` | Magnitude |
| `cuda_complex_phase_f32` | `void` | `float* out, const float* re, const float* im, size_t n` | Phase angle |
| `cuda_complex_exp_f32` | `void` | `float* out_re, float* out_im, const float* phase, size_t n` | Complex exponential |

---

### FFT Operations (cuFFT)

**CudaFFTPlan Class**:
```cpp
class CudaFFTPlan {
    bool create_1d(size_t n, bool inverse = false);
    bool create_1d_batch(size_t n, size_t batch, bool inverse = false);
    bool create_2d(size_t nx, size_t ny, bool inverse = false);
    void execute(float* inout);
    void execute(const float* in, float* out);
    void destroy();
};
```

**One-shot FFT Functions**:

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_fft_1d_f32` | `void` | `float* inout, size_t n, bool inverse` | 1D FFT |
| `cuda_fft_1d_batch_f32` | `void` | `float* inout, size_t n, size_t batch, bool inverse` | Batched 1D FFT |
| `cuda_fft_2d_f32` | `void` | `float* inout, size_t nx, size_t ny, bool inverse` | 2D FFT |

**Eigen Wrappers**:

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_fft` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& x` | Forward FFT |
| `cuda_ifft` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& x` | Inverse FFT |
| `cuda_fft2` | `Eigen::MatrixXcf` | `const Eigen::MatrixXcf& x` | 2D FFT |
| `cuda_ifft2` | `Eigen::MatrixXcf` | `const Eigen::MatrixXcf& x` | 2D inverse FFT |
| `cuda_rfft` | `Eigen::VectorXcf` | `const Eigen::VectorXf& x` | Real-to-complex FFT |
| `cuda_irfft` | `Eigen::VectorXf` | `const Eigen::VectorXcf& x, size_t n` | Complex-to-real inverse FFT |

---

### Convolution

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_conv1d_f32` | `void` | `float* out, const float* signal, const float* kernel, size_t signal_len, size_t kernel_len` | 1D convolution |
| `cuda_conv2d_f32` | `void` | `float* out, const float* image, const float* kernel, int img_h, int img_w, int kern_h, int kern_w` | 2D convolution |
| `cuda_fftconv1d_f32` | `void` | `float* out, const float* signal, const float* kernel, size_t signal_len, size_t kernel_len` | FFT-based 1D convolution |
| `cuda_conv1d` | `Eigen::VectorXf` | `const Eigen::VectorXf& signal, const Eigen::VectorXf& kernel` | Eigen 1D convolution |
| `cuda_conv2d` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& image, const Eigen::MatrixXf& kernel` | Eigen 2D convolution |

---

### Radar Signal Processing (GPU)

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_caf` | `Eigen::MatrixXf` | `const Eigen::VectorXcf& ref, const Eigen::VectorXcf& surv, size_t n_doppler, float doppler_start, float doppler_step, float sample_rate, size_t n_range` | Cross-Ambiguity Function |
| `cuda_cfar_2d` | `Eigen::MatrixXi` | `const Eigen::MatrixXf& power_map, int guard_range, int guard_doppler, int ref_range, int ref_doppler, float pfa_factor` | 2D CFAR detector |
| `cuda_cfar_ca` | `Eigen::VectorXi` | `const Eigen::VectorXf& power, int guard_cells, int ref_cells, float pfa_factor` | 1D CA-CFAR |
| `cuda_doppler_process` | `Eigen::MatrixXcf` | `const Eigen::MatrixXcf& pulse_data, size_t fft_size, int window_type` | Doppler processing |
| `cuda_bartlett_spectrum` | `Eigen::VectorXf` | `const Eigen::VectorXcf& array_data, float d_lambda, int n_angles` | Bartlett beamformer |
| `cuda_steering_vectors_ula` | `Eigen::MatrixXcf` | `int n_elements, float d_lambda, const Eigen::VectorXf& angles` | ULA steering vectors |
| `cuda_nlms_filter` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& surv, const Eigen::VectorXcf& ref, int filter_len, float mu, float eps` | NLMS adaptive filter |
| `cuda_projection_clutter` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& surv, const Eigen::MatrixXcf& clutter_subspace` | Projection clutter cancellation |

---

### Window Functions (GPU)

```cpp
enum class WindowType {
    RECTANGULAR, HAMMING, HANNING, BLACKMAN,
    BLACKMAN_HARRIS, KAISER, GAUSSIAN, TUKEY
};
```

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cuda_generate_window` | `Eigen::VectorXf` | `size_t n, WindowType type, float param` | Generate window on GPU |
| `cuda_apply_window` | `void` | `Eigen::VectorXf& data, const Eigen::VectorXf& window` | Apply window (real) |
| `cuda_apply_window` | `void` | `Eigen::VectorXcf& data, const Eigen::VectorXf& window` | Apply window (complex) |

---

### Multi-GPU Support

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `set_device` | `void` | `int device_id` | Set active CUDA device |
| `get_device` | `int` | - | Get current device |
| `enable_peer_access` | `bool` | `int device_from, int device_to` | Enable P2P access |
| `parallel_for_devices` | `void` | `const std::vector<int>& devices, Func&& func` | Distribute workload |

---

### Performance Profiling

**CudaTimer**:
```cpp
class CudaTimer {
    void start();
    void stop();
    float elapsed_ms() const;
};
```

**Bandwidth Measurement**:
```cpp
struct BandwidthStats {
    float host_to_device_gbps;
    float device_to_host_gbps;
    float device_to_device_gbps;
};

BandwidthStats measure_bandwidth(size_t bytes = 256 * 1024 * 1024);
```

---

### Error Handling

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `get_last_error` | `std::string` | - | Get last CUDA error message |
| `check_cuda_error` | `bool` | `const char* operation` | Check and report errors |

---

### Shorthand Aliases

For API compatibility with NEON backend:

| Shorthand | Full Function |
|-----------|---------------|
| `cuda_add(a, b)` | `cuda_vec_add(a, b)` |
| `cuda_mul(a, b)` | `cuda_vec_mul(a, b)` |
| `cuda_scale(a, s)` | `cuda_vec_scale(a, s)` |
| `cuda_dot(a, b)` | `cuda_vec_dot(a, b)` |
| `cuda_sum(a)` | `cuda_reduce_sum(a)` |
| `cuda_max(a)` | `cuda_reduce_max(a)` |
| `cuda_min(a)` | `cuda_reduce_min(a)` |
| `cuda_gemm(A, B)` | `cuda_mat_mul(A, B)` |
| `cuda_gemv(A, x)` | `cuda_mat_vec_mul(A, x)` |
| `cuda_transpose(A)` | `cuda_mat_transpose(A)` |
| `cuda_complex_abs(a)` | `cuda_complex_magnitude(a)` |
| `cuda_complex_arg(a)` | `cuda_complex_phase(a)` |

---

## Vulkan Backend (Cross-platform GPU)

**Header**: `#include <optmath/vulkan_backend.hpp>`
**Namespace**: `optmath::vulkan`
**Target**: Any Vulkan 1.2+ GPU (NVIDIA, AMD, Intel, Raspberry Pi 5 VideoCore VII)

### Availability

```cpp
bool is_available();
```

### Context Management

```cpp
class VulkanContext {
    static VulkanContext& get();
    bool init();
    void cleanup();

#ifdef OPTMATH_USE_VULKAN
    VkDevice device;
    VkQueue computeQueue;
    VkCommandPool commandPool;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
#endif
};
```

---

### Vector Operations

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `vulkan_vec_add` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector addition |
| `vulkan_vec_sub` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Vector subtraction |
| `vulkan_vec_mul` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise multiply |
| `vulkan_vec_div` | `Eigen::VectorXf` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Element-wise divide |
| `vulkan_vec_dot` | `float` | `const Eigen::VectorXf& a, const Eigen::VectorXf& b` | Dot product |
| `vulkan_vec_norm` | `float` | `const Eigen::VectorXf& a` | L2 norm |

---

### Matrix Operations

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `vulkan_mat_add` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Matrix addition |
| `vulkan_mat_sub` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Matrix subtraction |
| `vulkan_mat_mul` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Matrix multiplication (16x16 tiled) |
| `vulkan_mat_transpose` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a` | Transpose |
| `vulkan_mat_scale` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, float scalar` | Scalar multiply |
| `vulkan_mat_vec_mul` | `Eigen::VectorXf` | `const Eigen::MatrixXf& a, const Eigen::VectorXf& v` | Matrix-vector multiply |
| `vulkan_mat_outer_product` | `Eigen::MatrixXf` | `const Eigen::VectorXf& u, const Eigen::VectorXf& v` | Outer product |
| `vulkan_mat_elementwise_mul` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& a, const Eigen::MatrixXf& b` | Element-wise multiply |

---

### DSP Operations

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `vulkan_convolution_1d` | `Eigen::VectorXf` | `const Eigen::VectorXf& x, const Eigen::VectorXf& k` | 1D convolution |
| `vulkan_convolution_2d` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& x, const Eigen::MatrixXf& k` | 2D convolution |
| `vulkan_correlation_1d` | `Eigen::VectorXf` | `const Eigen::VectorXf& x, const Eigen::VectorXf& k` | 1D correlation |
| `vulkan_correlation_2d` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& x, const Eigen::MatrixXf& k` | 2D correlation |

---

### Reductions & Scan

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `vulkan_reduce_sum` | `float` | `const Eigen::VectorXf& a` | Sum all elements |
| `vulkan_reduce_max` | `float` | `const Eigen::VectorXf& a` | Maximum element |
| `vulkan_reduce_min` | `float` | `const Eigen::VectorXf& a` | Minimum element |
| `vulkan_scan_prefix_sum` | `Eigen::VectorXf` | `const Eigen::VectorXf& a` | Parallel prefix sum |

---

### FFT Operations

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `vulkan_fft_radix2` | `void` | `Eigen::VectorXf& data, bool inverse` | Radix-2 FFT (in-place, interleaved) |
| `vulkan_fft_radix4` | `void` | `Eigen::VectorXf& data, bool inverse` | Radix-4 FFT (in-place, interleaved) |

**Note**: Data format is interleaved complex: `[re0, im0, re1, im1, ...]`. Size must be `2 * N` where N is a power of 2.

---

### Vulkan Compute Shaders

37 GLSL compute shaders compiled to SPIR-V:

| Shader | Purpose |
|--------|---------|
| `vec_add.comp.glsl` | Vector addition |
| `vec_sub.comp.glsl` | Vector subtraction |
| `vec_mul.comp.glsl` | Vector multiplication |
| `vec_div.comp.glsl` | Vector division |
| `vec_dot.comp.glsl` | Dot product |
| `vec_norm.comp.glsl` | L2 norm |
| `mat_add.comp.glsl` | Matrix addition |
| `mat_sub.comp.glsl` | Matrix subtraction |
| `mat_mul.comp.glsl` | Matrix multiplication |
| `mat_mul_tiled.comp.glsl` | Tiled GEMM (16x16 shared memory) |
| `mat_transpose.comp.glsl` | Matrix transpose |
| `mat_scale.comp.glsl` | Scalar multiply |
| `mat_vec_mul.comp.glsl` | Matrix-vector multiply |
| `mat_outer_product.comp.glsl` | Outer product |
| `mat_elementwise_mul.comp.glsl` | Element-wise multiply |
| `reduce_sum.comp.glsl` | Sum reduction |
| `reduce_max.comp.glsl` | Max reduction |
| `reduce_min.comp.glsl` | Min reduction |
| `reduce_complete.comp.glsl` | Complete reduction |
| `scan_local.comp.glsl` | Local prefix scan |
| `scan_block_sums.comp.glsl` | Block sum scan |
| `scan_add_offsets.comp.glsl` | Add scan offsets |
| `scan_prefix_sum.comp.glsl` | Full prefix sum |
| `convolution_1d.comp.glsl` | 1D convolution |
| `convolution_1d_optimized.comp.glsl` | Optimized 1D convolution |
| `convolution_2d.comp.glsl` | 2D convolution |
| `convolution_2d_optimized.comp.glsl` | Optimized 2D convolution |
| `correlation_1d.comp.glsl` | 1D correlation |
| `correlation_2d.comp.glsl` | 2D correlation |
| `fft_radix2.comp.glsl` | Radix-2 FFT |
| `fft_radix2_optimized.comp.glsl` | Optimized radix-2 FFT |
| `fft_radix4.comp.glsl` | Radix-4 FFT |
| `ifft_radix2.comp.glsl` | Inverse radix-2 FFT |
| `ifft_radix4.comp.glsl` | Inverse radix-4 FFT |
| `caf_doppler_shift.comp.glsl` | CAF Doppler shift |
| `caf_xcorr.comp.glsl` | CAF cross-correlation |
| `cfar_2d.comp.glsl` | 2D CFAR detection |

---

## Radar Kernels (Signal Processing)

**Header**: `#include <optmath/radar_kernels.hpp>`
**Namespace**: `optmath::radar`
**Target**: Passive radar, SDR signal processing

### Window Types

```cpp
enum class WindowType {
    RECTANGULAR,      // No window (boxcar)
    HAMMING,          // Hamming window
    HANNING,          // Hann window
    BLACKMAN,         // Blackman window
    BLACKMAN_HARRIS,  // Blackman-Harris window
    KAISER            // Kaiser window (param = beta)
};
```

---

### Window Functions

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `generate_window_f32` | `void` | `float* window, std::size_t n, WindowType type, float beta` | Generate window coefficients |
| `apply_window_f32` | `void` | `float* data, const float* window, std::size_t n` | Apply window (real, in-place) |
| `apply_window_complex_f32` | `void` | `float* data_re, float* data_im, const float* window, std::size_t n` | Apply window (complex) |
| `generate_window` | `Eigen::VectorXf` | `std::size_t n, WindowType type, float beta` | Eigen window generator |
| `apply_window` | `void` | `Eigen::VectorXf& data, const Eigen::VectorXf& window` | Eigen window (real) |
| `apply_window` | `void` | `Eigen::VectorXcf& data, const Eigen::VectorXf& window` | Eigen window (complex) |

---

### Cross-Correlation

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `xcorr_f32` | `void` | `float* out, const float* x, std::size_t nx, const float* y, std::size_t ny` | Real cross-correlation (size: nx+ny-1) |
| `xcorr_complex_f32` | `void` | `float* out_re, float* out_im, const float* x_re, const float* x_im, std::size_t nx, const float* y_re, const float* y_im, std::size_t ny` | Complex cross-correlation |
| `xcorr` | `Eigen::VectorXf` | `const Eigen::VectorXf& x, const Eigen::VectorXf& y` | Real cross-correlation |
| `xcorr` | `Eigen::VectorXcf` | `const Eigen::VectorXcf& x, const Eigen::VectorXcf& y` | Complex cross-correlation |

---

### Cross-Ambiguity Function (CAF)

The CAF is the core passive radar processing operation, measuring correlation between reference and surveillance signals across multiple Doppler shifts and range delays.

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `caf_f32` | `void` | `float* out_mag, const float* ref_re, const float* ref_im, const float* surv_re, const float* surv_im, std::size_t n_samples, std::size_t n_doppler_bins, float doppler_start, float doppler_step, float sample_rate, std::size_t n_range_bins` | Direct CAF computation |
| `caf_fft_f32` | `void` | `float* out_mag, const float* ref_re, const float* ref_im, const float* surv_re, const float* surv_im, std::size_t n_samples, std::size_t n_doppler_bins, float doppler_start, float doppler_step, float sample_rate, std::size_t n_range_bins` | FFT-based CAF (faster for large arrays) |
| `caf` | `Eigen::MatrixXf` | `const Eigen::VectorXcf& ref, const Eigen::VectorXcf& surv, std::size_t n_doppler_bins, float doppler_start, float doppler_step, float sample_rate, std::size_t n_range_bins` | Eigen CAF wrapper |

**Parameters**:
- `ref`: Reference signal from transmitter (FM broadcast, DVB-T, etc.)
- `surv`: Surveillance signal from receiver
- `n_doppler_bins`: Number of Doppler frequency bins to compute
- `doppler_start`: Starting Doppler frequency (Hz)
- `doppler_step`: Doppler bin spacing (Hz)
- `sample_rate`: Sample rate of signals (Hz)
- `n_range_bins`: Number of range (delay) bins

**Output**: Range-Doppler magnitude matrix [n_doppler_bins x n_range_bins]

---

### CFAR Detection

Constant False Alarm Rate detection maintains constant Pfa in varying clutter environments.

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `cfar_ca_f32` | `void` | `std::uint8_t* detections, float* threshold, const float* input, std::size_t n, std::size_t guard_cells, std::size_t reference_cells, float pfa_factor` | 1D Cell-Averaging CFAR |
| `cfar_2d_f32` | `void` | `std::uint8_t* detections, const float* input, std::size_t n_doppler, std::size_t n_range, std::size_t guard_range, std::size_t guard_doppler, std::size_t ref_range, std::size_t ref_doppler, float pfa_factor` | 2D CFAR for range-Doppler |
| `cfar_os_f32` | `void` | `std::uint8_t* detections, float* threshold, const float* input, std::size_t n, std::size_t guard_cells, std::size_t reference_cells, std::size_t k_select, float pfa_factor` | Ordered-Statistic CFAR (robust to clutter edges) |
| `cfar_ca` | `Eigen::Matrix<uint8_t,Dynamic,1>` | `const Eigen::VectorXf& input, std::size_t guard_cells, std::size_t reference_cells, float pfa_factor` | Eigen 1D CFAR |
| `cfar_2d` | `Eigen::Matrix<uint8_t,Dynamic,Dynamic>` | `const Eigen::MatrixXf& input, std::size_t guard_range, std::size_t guard_doppler, std::size_t ref_range, std::size_t ref_doppler, float pfa_factor` | Eigen 2D CFAR |

**CFAR Cell Structure**:
```
[ref cells] [guard cells] [CUT] [guard cells] [ref cells]
```

---

### Clutter Filtering

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `nlms_filter_f32` | `void` | `float* output, float* weights, const float* input, const float* reference, std::size_t n, std::size_t filter_length, float mu, float eps` | Normalized LMS adaptive filter |
| `projection_clutter_f32` | `void` | `float* output, const float* input, const float* clutter_subspace, std::size_t n, std::size_t subspace_dim` | Projection clutter cancellation |
| `nlms_filter` | `Eigen::VectorXf` | `const Eigen::VectorXf& input, const Eigen::VectorXf& reference, std::size_t filter_length, float mu, float eps` | Eigen NLMS filter |
| `projection_clutter` | `Eigen::VectorXf` | `const Eigen::VectorXf& input, const Eigen::MatrixXf& clutter_subspace` | Eigen projection cancellation |

**NLMS Parameters**:
- `filter_length`: Number of taps (typically 32-128)
- `mu`: Adaptation step size (0 < mu < 2, typically 0.1)
- `eps`: Regularization constant (1e-6)

---

### Doppler Processing

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `doppler_fft_f32` | `void` | `float* output_re, float* output_im, const float* input_re, const float* input_im, std::size_t n_pulses, std::size_t n_range, std::size_t fft_size` | Doppler FFT across pulses |
| `mti_filter_f32` | `void` | `float* output, const float* input, std::size_t n_pulses, std::size_t n_range, const float* coeffs, std::size_t n_coeffs` | Moving Target Indicator filter |
| `doppler_fft` | `Eigen::MatrixXcf` | `const Eigen::MatrixXcf& input, std::size_t fft_size` | Eigen Doppler FFT |
| `mti_filter` | `Eigen::MatrixXf` | `const Eigen::MatrixXf& input, const Eigen::VectorXf& coeffs` | Eigen MTI filter |

**Common MTI Coefficients**:
- 2-pulse canceller: `[1, -1]`
- 3-pulse canceller: `[1, -2, 1]`

---

### Beamforming

| Function | Return Type | Parameters | Description |
|----------|-------------|------------|-------------|
| `beamform_delay_sum_f32` | `void` | `float* output, const float* inputs, const int* delays, const float* weights, std::size_t n_channels, std::size_t n_samples` | Delay-and-sum beamformer |
| `beamform_phase_f32` | `void` | `float* output_re, float* output_im, const float* inputs_re, const float* inputs_im, const float* phases, const float* weights, std::size_t n_channels, std::size_t n_samples` | Phase-shift beamformer |
| `steering_vector_ula_f32` | `void` | `float* steering_re, float* steering_im, std::size_t n_elements, float d_lambda, float theta_rad` | ULA steering vector |
| `beamform_delay_sum` | `Eigen::VectorXf` | `const Eigen::MatrixXf& inputs, const Eigen::VectorXi& delays, const Eigen::VectorXf& weights` | Eigen delay-sum beamformer |
| `beamform_phase` | `Eigen::VectorXcf` | `const Eigen::MatrixXcf& inputs, const Eigen::VectorXf& phases, const Eigen::VectorXf& weights` | Eigen phase beamformer |
| `steering_vector_ula` | `Eigen::VectorXcf` | `std::size_t n_elements, float d_lambda, float theta_rad` | Eigen ULA steering vector |

**Steering Vector Formula** (ULA):
```
a(θ)[n] = exp(j * 2π * d/λ * n * sin(θ))
```

---

## Quick Reference Tables

### Function Count by Backend

| Backend | Low-Level | Eigen Wrappers | Total |
|---------|-----------|----------------|-------|
| NEON | 48 | 56 | 104 |
| CUDA | 98 | 144 | 242 |
| Vulkan | 0 | 23 | 23 |
| Radar | 24 | 24 | 48 |
| **Total** | 170 | 247 | **417** |

### Performance Comparison

| Operation | Size | NEON (Pi5) | CUDA (RTX 4090) | Vulkan (Pi5) |
|-----------|------|------------|-----------------|--------------|
| Dot Product | 4096 | 0.8 μs | 0.02 μs | 0.5 μs |
| GEMM | 256x256 | 1.2 ms | 0.008 ms | 2.5 ms |
| GEMM | 1024x1024 | 45 ms | 0.08 ms | N/A |
| FFT | 4096 | 0.4 ms | 0.01 ms | 0.3 ms |
| FFT | 65536 | 8 ms | 0.12 ms | 5 ms |
| CAF | 4096x64 | 5.2 ms | 0.3 ms | 2.1 ms |
| Exp | 1M elements | 1.2 ms | 0.03 ms | N/A |

### Supported Architectures

| Architecture | Backend | Compute Version | Key Features |
|--------------|---------|-----------------|--------------|
| ARM Cortex-A76 | NEON | ARMv8-A | 128-bit SIMD, 2.4 GHz |
| VideoCore VII | Vulkan | 1.2 | 12 QPU cores, 1 GFLOPS |
| NVIDIA Turing | CUDA | 7.5 | Tensor Cores Gen 1 |
| NVIDIA Ampere | CUDA | 8.0/8.6 | Tensor Cores Gen 3, TF32 |
| NVIDIA Ada | CUDA | 8.9 | Tensor Cores Gen 4, FP8 |
| NVIDIA Hopper | CUDA | 9.0 | Transformer Engine |
| NVIDIA Blackwell | CUDA | 10.0 | Tensor Cores Gen 5, FP8 |
| AMD RDNA2/3 | Vulkan | 1.3 | Compute units |
| Intel Arc | Vulkan | 1.3 | Xe-HPG cores |

---

## License

MIT License - Copyright (c) 2026 Dr Robert W McGwier, PhD
