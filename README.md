# OptMathKernels

**High-Performance Numerical Library for ARM SBCs and NVIDIA GPUs**

OptMathKernels is a C++20 numerical library optimized for **Raspberry Pi 5**, **Orange Pi 6 Plus**, and **NVIDIA CUDA GPUs**. It seamlessly bridges **Eigen** (CPU), **ARM NEON** (SIMD), **ARM SVE2** (Scalable Vectors), **Vulkan** (Compute Shaders), and **CUDA** (NVIDIA GPUs) into a single, easy-to-use API.

Designed to accelerate math and signal processing tasks by leveraging:
- **Raspberry Pi 5**: Cortex-A76 NEON and VideoCore VII GPU
- **Orange Pi 6 Plus**: Cortex-A720 SVE2/FCMA/I8MM and Mali-G720-Immortalis GPU
- **NVIDIA GPUs**: cuBLAS, cuFFT, cuSOLVER, and Tensor Cores (Volta+)

While remaining compatible with standard Linux x86/ARM environments.

---

## Table of Contents

- [Key Applications](#key-applications)
- [Features](#features)
- [Hardware Support](#hardware-support)
  - [Orange Pi 6 Plus (CIX P1)](#orange-pi-6-plus-cix-p1)
  - [Raspberry Pi 5](#raspberry-pi-5)
  - [x86_64 (Intel/AMD)](#x86_64-intelamd)
  - [NVIDIA CUDA/RTX](#nvidia-cudartx)
  - [Vulkan GPU Compute](#vulkan-gpu-compute)
- [Installation](#installation)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [License](#license)
- [Recent Changes](#recent-changes)

---

## Key Applications

### Passive Radar Signal Processing

OptMathKernels powers the [PassiveRadar_Kraken](https://github.com/n4hy/PassiveRadar_Kraken_NvidiaCUDA,git)
project, providing hardware-accelerated kernels for:

**ARM NEON (Raspberry Pi 5):**
| Operation | Speedup | Application |
|-----------|---------|-------------|
| Complex multiply | 4-8x | CAF Doppler shifting |
| Dot product | 4-6x | NLMS adaptive filter |
| GEMM (blocked) | 3-5x | Beamforming, covariance |
| Transcendentals | 10-50x | Phase computation |
| FFT (Vulkan) | 10x | Large batch processing |

**NVIDIA CUDA (Workstation/Server):**
| Operation | Speedup | GPU Features |
|-----------|---------|--------------|
| GEMM (1024x1024) | 50-100x | Tensor Cores (Ampere+) |
| FFT (64K points) | 20-50x | cuFFT optimized |
| CAF (full map) | 30-80x | Batched FFT + complex ops |
| CFAR 2D | 10-30x | Parallel threshold compute |
| Beamforming | 20-50x | cuBLAS matrix-vector |
| Transcendentals | 50-200x | CUDA fast math intrinsics |

### General Numerical Computing
- Machine learning inference (activation functions, matrix ops)
- Digital signal processing (filtering, FFT, convolution)
- Scientific computing (linear algebra, statistics)
- Real-time audio/video processing

---

## Features

### Core Capabilities

- **NEON Acceleration**: Hand-tuned ARMv8 NEON intrinsics for SIMD acceleration on 64-bit ARM (aarch64). Includes optimized matrix multiplication, convolution, and vector math.
- **Vulkan Compute**: Massive parallel offloading to the GPU (VideoCore VII on Pi 5). Supports large vector operations, matrix math, FFT (Radix-2/4), and reductions.
- **Eigen Integration**: Fully compatible with `Eigen::VectorXf`, `Eigen::MatrixXf`, and `Eigen::VectorXcf`. Pass your existing data structures directly to accelerated kernels.
- **Easy Integration**: Standard CMake package that installs to `/usr/local` and works with `find_package(OptMathKernels)`.

### Radar Signal Processing (`optmath::radar`)

- **Cross-Ambiguity Function (CAF)**: Core passive radar operation for range-Doppler detection
- **CFAR Detection**: 1D/2D Cell-Averaging and Ordered Statistic CFAR detectors
- **Clutter Filtering**: NLMS adaptive filter and projection-based clutter cancellation
- **Doppler Processing**: FFT-based Doppler processing and MTI filters
- **Beamforming**: Delay-and-sum and phase-shift beamformers with steering vector generation
- **Window Functions**: Hamming, Hanning, Blackman, Blackman-Harris, Kaiser

### Optimized Kernels (`optmath::neon`)

- **Vectorized Transcendentals**: Fast NEON-accelerated exp, sin, cos, sigmoid, tanh (~10-50x faster than scalar)
- **Complex Operations**: Vectorized complex multiply, conjugate multiply, magnitude, phase
- **Cache-Blocked GEMM**: 8x8 microkernel with runtime cache blocking (MC/KC/NC auto-tuned for detected L3 cache size)
- **Reductions**: Sum, max, min, dot product with horizontal NEON adds

### SVE2 Acceleration (`optmath::sve2`)

- **Predicated Vector Operations**: All loops use `svwhilelt`/`svptest` predication - zero tail-handling code
- **FCMA Complex Multiply**: 2-instruction complex multiply via `svcmla` (vs 4 NEON instructions)
- **I8MM GEMM**: Int8 matrix multiply using `svmmla_s32` with float32 quantize/dequantize
- **SVE2 Cache-Blocked GEMM**: Tuned for Cortex-A720 12MB L3 (MC=256, KC=512, NC=1024)
- **SVE2 Transcendentals**: Same polynomial approximations as NEON but with predicated loops
- **SVE2 Radar DSP**: FCMA-accelerated CAF, cross-correlation, and beamforming
- **SVE2 Complex Operations**: Split and interleaved formats, dot product, magnitude, phase

### Platform Detection (`optmath::platform`)

- **CPU Topology Detection**: Identifies big.LITTLE core clusters via `/proc/cpuinfo` and sysfs
- **Thread Affinity**: Pin threads to performance (A720) or efficiency (A520) cores
- **SVE Vector Length**: Runtime detection via `prctl(PR_SVE_GET_VL)`
- **Cache-Aware GEMM Tuning**: Auto-selects blocking parameters based on detected L3 cache size

### DSP Kernels (`optmath::neon`)

- **Polyphase Resampler**: Rational L/M rate conversion with NEON-optimized FIR per phase, streaming and one-shot APIs
- **Biquad IIR Filter**: Direct Form II Transposed with cascade support, design helpers for lowpass/highpass/bandpass/notch
- **2D Convolution**: NEON-vectorized general NxM convolution, separable kernels, unrolled 3x3 and 5x5 specializations

### Dense Linear Algebra (`optmath::neon`)

- **Triangular Solve (TRSV/TRSM)**: Column-oriented forward/backward substitution with NEON-vectorized AXPY updates, unit-diagonal and transpose variants, multi-RHS support
- **Cholesky Decomposition**: Unblocked right-looking A = L*L^T with NEON dot products, SPD validation with error reporting
- **LU Decomposition**: Partial pivoting with NEON-vectorized column scaling and rank-1 trailing updates
- **QR Decomposition**: Householder reflections with NEON-vectorized reflector application, explicit Q extraction
- **Solvers**: General solve (LU), SPD solve (Cholesky), matrix inverse (LU + TRSM)

### GPU Acceleration (`optmath::vulkan`)

- **Tiled GPU Matrix Multiply**: 16x16 shared memory tiles (default), 32x32 tiles for Mali-G720
- **FFT**: Radix-2/4 FFT with butterfly operations in compute shaders
- **Convolution**: 1D and 2D convolution with separable kernel optimization
- **Vector Operations**: Add, multiply, dot product, reductions
- **Mali-G720 Optimized**: Auto-selects 1024-thread shaders with subgroup arithmetic on Mali GPUs

### NVIDIA CUDA Acceleration (`optmath::cuda`)

- **cuBLAS Integration**: Level 1, 2, and 3 BLAS operations with Tensor Core support
- **cuFFT**: High-performance FFT up to 64M points, batched transforms
- **cuSOLVER**: Cholesky, LU, QR, SVD, eigenvalue decomposition
- **Tensor Cores**: Automatic acceleration on Volta+ (SM 7.0+) and Ampere+ (SM 8.0+)
- **Mixed Precision**: FP32, TF32, FP16 compute modes
- **Multi-GPU**: Device enumeration, peer-to-peer access, workload distribution
- **Unified Memory**: Simplified CPU-GPU data management
- **Radar Processing**: Full GPU-accelerated CAF, CFAR, beamforming, NLMS filter
- **Optimized CAF**: Zero CPU-GPU transfers inside Doppler loop for maximum throughput

---

## Hardware Support

### Orange Pi 6 Plus (CIX P1)

**Target Hardware**: Orange Pi 6 Plus with CIX P1 CD8160 SoC (ARMv9)

| Feature | Specification |
|---------|---------------|
| **CPU** | 8x Cortex-A720 + 4x Cortex-A520 (big.LITTLE) |
| **SIMD** | NEON (128-bit) + SVE2 (128-bit VL) + FCMA + I8MM + BF16 |
| **GPU** | Mali-G720-Immortalis (Vulkan 1.3) |
| **Memory** | 8GB/16GB LPDDR5 |
| **L1 Cache** | 64KB per core |
| **L2 Cache** | 512KB per core |
| **L3 Cache** | 12MB shared |

**SVE2 Optimizations**:
- Predicated loops eliminate all scalar tail handling (vs 28+ NEON tail loops)
- FCMA complex multiply: 2 instructions vs 4 NEON instructions per complex multiply
- I8MM: Hardware int8 matrix multiply with `svmmla_s32`
- Cache blocking tuned for 12MB L3: MC=256, KC=512, NC=1024

**Mali-G720-Immortalis GPU (Vulkan)**:
| Feature | Specification |
|---------|---------------|
| **Vulkan** | 1.3 |
| **Shared Memory** | 32KB per workgroup |
| **Max Workgroup Size** | 1024 |
| **Subgroup Size** | 16 |
| **Subgroup Ops** | Arithmetic (subgroupAdd, etc.) |

- 32x32 tiled GEMM shader (1024 threads, 8KB shared memory)
- 1024-thread reduction with subgroup-level arithmetic

**Build for Orange Pi 6 Plus**:
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_NEON=ON \
      -DENABLE_SVE2=ON \
      -DENABLE_VULKAN=ON \
      -DENABLE_CUDA=OFF \
      -DBUILD_TESTS=ON ..
make -j$(nproc)
```

---

### Raspberry Pi 5

**Target Hardware**: Raspberry Pi 5 with Broadcom BCM2712 SoC (Cortex-A76)

| Feature | Specification |
|---------|---------------|
| **CPU** | Quad-core ARM Cortex-A76 @ 2.4 GHz |
| **SIMD** | NEON (128-bit Advanced SIMD) |
| **GPU** | VideoCore VII (Vulkan 1.2) |
| **Memory** | 4GB/8GB LPDDR4X-4267 |
| **L1 Cache** | 64KB per core |
| **L2 Cache** | 512KB per core |
| **L3 Cache** | 2MB shared |

**NEON Optimizations**:
- Complex multiply-accumulate (4 complex float32 per cycle)
- Vectorized dot products (8 float32 per instruction)
- SIMD FFT butterflies with `vfmaq_f32`
- Parallel magnitude/phase computation
- Cache-blocked GEMM tuned for A76 cache hierarchy

**Cache Blocking Parameters** (auto-tuned per platform):
```cpp
// Pi 5 (2MB L3):  MC=128, KC=256, NC=512
// Orange Pi 6+ (12MB L3): MC=256, KC=512, NC=1024
// 8x8 microkernel with 4x4 register tiles (32 NEON registers)
neon_gemm_blocked_f32(C, A, B, M, N, K, lda, ldb, ldc);
```

**Performance on Pi 5**:
| Operation | NEON Time | Scalar Time | Speedup |
|-----------|-----------|-------------|---------|
| Complex dot 4096 | 0.8 μs | 12.4 μs | 15.5x |
| GEMM 256x256 | 1.2 ms | 18 ms | 15x |
| CAF 4096x64 | 5.2 ms | 82 ms | 15.8x |
| CFAR 2D 256x64 | 0.9 ms | 11.2 ms | 12.4x |
| Exp 1M elements | 1.2 ms | 54 ms | 45x |
| Sin 1M elements | 1.5 ms | 48 ms | 32x |

---

### x86_64 (Intel/AMD)

**Supported Processors**: Any x86_64 CPU with SSE4.2 or AVX2

| Feature | Minimum | Recommended |
|---------|---------|-------------|
| **CPU** | Intel Core i3 / AMD Ryzen 3 | Intel Core i7 / AMD Ryzen 7 |
| **SIMD** | SSE4.2 | AVX2/AVX-512 |
| **Memory** | 8GB DDR4 | 16GB+ DDR4/DDR5 |

**Eigen3 Auto-Vectorization**: The OptMathKernels library uses Eigen3's expression templates which automatically vectorize for AVX/AVX2/AVX-512 when available on x86_64.

```bash
# Check for AVX2 support
cat /proc/cpuinfo | grep avx2
```

---

### NVIDIA CUDA/RTX

**Supported GPUs**: NVIDIA GPUs with Compute Capability 7.0+ (Turing and later)

| GPU Generation | Architecture | Compute Capability | CUDA Cores | Tensor Cores | Memory |
|----------------|--------------|-------------------|------------|--------------|--------|
| **RTX 2000** | Turing | 7.5 | 2304-4608 | 288-576 | 8-11 GB |
| **RTX 3000** | Ampere | 8.6 | 3584-10496 | 112-328 | 8-24 GB |
| **RTX 4000** | Ada Lovelace | 8.9 | 5888-16384 | 184-512 | 8-24 GB |
| **RTX 5000** | Blackwell | 10.0+ | TBD | TBD | 16-32 GB |
| **Jetson Orin** | Ampere | 8.7 | 1024-2048 | 32-64 | 8-64 GB |
| **Tesla V100** | Volta | 7.0 | 5120 | 640 | 16-32 GB |
| **A100** | Ampere | 8.0 | 6912 | 432 | 40-80 GB |
| **H100** | Hopper | 9.0 | 14592 | 456 | 80 GB |

**Tensor Core Generations**:
| Generation | Architecture | Precision | Peak TFLOPS |
|------------|--------------|-----------|-------------|
| Gen 1 | Turing | FP16 | 110 |
| Gen 3 | Ampere | TF32, FP16, BF16 | 312 |
| Gen 4 | Ada | TF32, FP16, BF16, FP8 | 660 |
| Gen 5 | Blackwell | TF32, FP16, BF16, FP8, FP4 | TBD |

**CUDA Performance (RTX 4090)**:
| Operation | CUDA Time | CPU Time | Speedup |
|-----------|-----------|----------|---------|
| Complex FFT 65536 | 0.12 ms | 8.4 ms | 70x |
| GEMM 1024x1024 | 0.08 ms | 45 ms | 562x |
| 2D Convolution 512x512 | 0.15 ms | 120 ms | 800x |
| CAF 16384x256 | 1.2 ms | 450 ms | 375x |
| Exp 1M elements | 0.03 ms | 54 ms | 1800x |

---

### Vulkan GPU Compute

**Supported GPUs**: Any Vulkan 1.2+ capable GPU

| Vendor | GPUs | Vulkan Version | Driver |
|--------|------|----------------|--------|
| **NVIDIA** | GTX 900+, RTX series | 1.3 | Proprietary |
| **AMD** | RX 400+, RDNA/RDNA2/RDNA3 | 1.3 | Mesa RADV |
| **Intel** | UHD 600+, Arc | 1.3 | Mesa ANV |
| **Raspberry Pi 5** | VideoCore VII | 1.2 | Mesa V3D |
| **Orange Pi 6 Plus** | Mali-G720-Immortalis | 1.3 | Mali (panfrost/panvk) |
| **Qualcomm** | Adreno 6xx+ | 1.1 | Proprietary |

**VideoCore VII (Raspberry Pi 5)**:
| Feature | Specification |
|---------|---------------|
| **QPU Cores** | 12 |
| **Clock** | 800 MHz |
| **Compute** | ~1 GFLOPS FP32 |
| **Shared Memory** | 4KB per workgroup |
| **Max Workgroup Size** | 256 |

**Vulkan Compute Shaders**: 39 GLSL compute shaders compiled to SPIR-V:
- Vector operations (add, sub, mul, div, dot, norm)
- Matrix operations (add, mul, transpose, scale)
- Reductions (sum, max, min, prefix scan)
- Convolution (1D, 2D, optimized variants)
- FFT (radix-2, radix-4, forward/inverse)
- Radar (CAF Doppler shift, CFAR 2D)

---

## Installation

### Prerequisites

#### Raspberry Pi 5 / Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libeigen3-dev \
    libvulkan-dev \
    mesa-vulkan-drivers \
    vulkan-tools \
    glslang-tools \
    glslc
```

#### NVIDIA CUDA Support (Optional)

```bash
# Ubuntu/Debian with NVIDIA driver already installed
sudo apt install -y nvidia-cuda-toolkit

# Or download from NVIDIA (recommended for latest version)
# https://developer.nvidia.com/cuda-downloads

# Ubuntu 22.04/24.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

### Build & Install

#### 1. Clone the Repository
```bash
git clone https://github.com/n4hy/OptimizedKernelsForRaspberryPi5_NvidiaCUDA.git
cd OptimizedKernelsForRaspberryPi5_NvidiaCUDA
```

#### 2. Configure and Build

**Raspberry Pi 5 (NEON + Vulkan):**
```bash
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DENABLE_NEON=ON \
      -DENABLE_VULKAN=ON \
      -DENABLE_CUDA=OFF \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

make -j$(nproc)
```

**NVIDIA Workstation (CUDA + optional Vulkan):**
```bash
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DENABLE_NEON=OFF \
      -DENABLE_VULKAN=ON \
      -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="75;86;89;90" \
      -DBUILD_TESTS=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

make -j$(nproc)
```

**CMake Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_NEON` | ON (ARM) | Enable ARM NEON SIMD |
| `ENABLE_SVE2` | ON | Enable ARM SVE2 (ARMv9) |
| `ENABLE_FCMA` | ON | Enable FCMA complex multiply |
| `ENABLE_I8MM` | ON | Enable I8MM int8 matrix multiply |
| `ENABLE_VULKAN` | ON | Enable Vulkan compute |
| `ENABLE_CUDA` | OFF | Enable NVIDIA CUDA |
| `CMAKE_CUDA_ARCHITECTURES` | 75;86;89 | CUDA compute capabilities |
| `BUILD_TESTS` | ON | Build GoogleTest tests |
| `BUILD_BENCHMARKS` | OFF | Build Google Benchmark |
| `CMAKE_POSITION_INDEPENDENT_CODE` | OFF | Enable -fPIC (required for shared libs) |

#### 3. Run Tests
```bash
ctest --output-on-failure
```

#### 4. Install
```bash
sudo make install
sudo ldconfig
```

#### 5. Verify Installation
```bash
ls /usr/local/lib/libOptMathKernels*
ls /usr/local/include/optmath/
ls /usr/local/lib/cmake/OptMathKernels/
```

---

## API Reference

For complete API documentation of all **490+ functions**, see:

**[FunctionsIncluded.md](FunctionsIncluded.md)** - Complete API Reference

### Quick Reference by Backend

| Backend | Functions | Description |
|---------|-----------|-------------|
| **NEON** | 104 | ARM SIMD operations for Raspberry Pi 5 / Orange Pi 6 Plus |
| **SVE2** | 46 | ARM SVE2/FCMA/I8MM for ARMv9 (Orange Pi 6 Plus) |
| **Platform** | 10 | CPU topology detection, thread affinity, cache tuning |
| **CUDA** | 242 | NVIDIA GPU kernels (cuBLAS, cuFFT, cuSOLVER) |
| **Vulkan** | 23 | Cross-platform GPU compute shaders |
| **Radar** | 48 | Passive radar signal processing |

### Headers

```cpp
#include <optmath/neon_kernels.hpp>    // ARM NEON operations
#include <optmath/sve2_kernels.hpp>    // ARM SVE2/FCMA/I8MM operations
#include <optmath/platform.hpp>        // CPU detection, thread affinity
#include <optmath/vulkan_backend.hpp>  // Vulkan GPU compute
#include <optmath/cuda_backend.hpp>    // NVIDIA CUDA operations
#include <optmath/radar_kernels.hpp>   // Radar signal processing
```

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyApp)

find_package(OptMathKernels REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE
    OptMathKernels::OptMathKernels
    Eigen3::Eigen
)
```

---

## Usage Examples

### NEON Vector Operations

```cpp
#include <optmath/neon_kernels.hpp>

if (optmath::neon::is_available()) {
    Eigen::VectorXf a = Eigen::VectorXf::Random(1024);
    Eigen::VectorXf b = Eigen::VectorXf::Random(1024);

    // Basic operations
    Eigen::VectorXf c = optmath::neon::neon_add(a, b);
    Eigen::VectorXf d = optmath::neon::neon_mul(a, b);
    float dot = optmath::neon::neon_dot(a, b);
    float norm = optmath::neon::neon_norm(a);

    // Reductions
    float sum = optmath::neon::neon_reduce_sum(a);
    float max = optmath::neon::neon_reduce_max(a);
}
```

### NEON Matrix Operations

```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(256, 256);
Eigen::MatrixXf B = Eigen::MatrixXf::Random(256, 256);

// Basic GEMM (4x4 tiled)
Eigen::MatrixXf C = optmath::neon::neon_gemm(A, B);

// Optimized blocked GEMM (8x8 microkernel, cache blocking)
Eigen::MatrixXf D = optmath::neon::neon_gemm_blocked(A, B);

// Other operations
Eigen::MatrixXf At = optmath::neon::neon_mat_transpose(A);
Eigen::MatrixXf As = optmath::neon::neon_mat_scale(A, 2.5f);
```

### NEON Vectorized Transcendentals

```cpp
std::vector<float> input(N), output(N);

// Fast exp approximation (~12% relative error at extremes, better near zero)
optmath::neon::neon_fast_exp_f32(output.data(), input.data(), N);

// Fast sin/cos approximation (~1e-5 accuracy)
optmath::neon::neon_fast_sin_f32(output.data(), input.data(), N);
optmath::neon::neon_fast_cos_f32(output.data(), input.data(), N);

// Fast activation functions (~3% sigmoid, ~6% tanh)
optmath::neon::neon_fast_sigmoid_f32(output.data(), input.data(), N);
optmath::neon::neon_fast_tanh_f32(output.data(), input.data(), N);
```

### NEON Complex Operations

```cpp
Eigen::VectorXcf a = Eigen::VectorXcf::Random(1024);
Eigen::VectorXcf b = Eigen::VectorXcf::Random(1024);

// Element-wise complex operations
Eigen::VectorXcf c = optmath::neon::neon_complex_mul(a, b);
Eigen::VectorXcf d = optmath::neon::neon_complex_conj_mul(a, b);  // a * conj(b)

// Complex dot product: sum(a * conj(b))
std::complex<float> dot = optmath::neon::neon_complex_dot(a, b);

// Magnitude and phase
Eigen::VectorXf mag = optmath::neon::neon_complex_magnitude(a);
Eigen::VectorXf phase = optmath::neon::neon_complex_phase(a);
```

### NEON DSP: Polyphase Resampler

```cpp
#include <optmath/neon_kernels.hpp>
using namespace optmath::neon;

// Rational rate conversion by L/M (e.g., 48kHz -> 44.1kHz ≈ 147/160)
std::size_t L = 147, M = 160;

// Design a lowpass prototype filter (cutoff at min(pi/L, pi/M))
std::vector<float> filter = /* your FIR lowpass filter */;

// One-shot resampling
std::vector<float> input(10000), output(input.size() * L / M + 100);
std::size_t out_len = 0;
neon_resample_oneshot_f32(output.data(), &out_len,
                           input.data(), input.size(),
                           filter.data(), filter.size(), L, M);

// Streaming resampling (maintains state between calls)
PolyphaseResamplerState state;
neon_resample_init(state, filter.data(), filter.size(), L, M);
std::size_t n = neon_resample_f32(output.data(), input.data(), input.size(), state);

// Eigen wrapper
Eigen::VectorXf in_vec = Eigen::VectorXf::Random(10000);
Eigen::VectorXf filt_vec = Eigen::Map<Eigen::VectorXf>(filter.data(), filter.size());
Eigen::VectorXf result = neon_resample(in_vec, filt_vec, L, M);
```

### NEON DSP: Biquad IIR Filter

```cpp
#include <optmath/neon_kernels.hpp>
using namespace optmath::neon;

// Design a lowpass filter (fc=1000Hz, fs=48000Hz, Q=0.707)
BiquadCoeffs lp = neon_biquad_lowpass(1000.0f, 48000.0f);

// Process audio samples
std::vector<float> input(4096), output(4096);
BiquadState state;
neon_biquad_f32(output.data(), input.data(), input.size(), lp, state);

// Cascade for higher-order filtering (4th order = 2 biquad sections)
BiquadCoeffs cascade[2] = {
    neon_biquad_lowpass(1000.0f, 48000.0f),
    neon_biquad_lowpass(1000.0f, 48000.0f)
};
BiquadState states[2] = {};
neon_biquad_cascade_f32(output.data(), input.data(), input.size(),
                         cascade, states, 2);

// Other filter types
BiquadCoeffs hp = neon_biquad_highpass(500.0f, 48000.0f);
BiquadCoeffs bp = neon_biquad_bandpass(1000.0f, 48000.0f, 5.0f);
BiquadCoeffs notch = neon_biquad_notch(60.0f, 48000.0f, 30.0f);  // 60Hz hum removal
```

### NEON DSP: 2D Convolution

```cpp
#include <optmath/neon_kernels.hpp>
using namespace optmath::neon;

// Input image in row-major layout
std::size_t rows = 480, cols = 640;
std::vector<float> image(rows * cols);

// 3x3 Sobel edge detection (horizontal)
float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
std::vector<float> edges((rows - 2) * (cols - 2));
neon_conv2d_3x3_f32(edges.data(), image.data(), rows, cols, sobel_x);

// 5x5 Gaussian blur
float gauss5x5[25] = { /* your kernel */ };
std::vector<float> blurred((rows - 4) * (cols - 4));
neon_conv2d_5x5_f32(blurred.data(), image.data(), rows, cols, gauss5x5);

// Separable convolution (faster for separable kernels)
float row_k[] = {0.25f, 0.5f, 0.25f};
float col_k[] = {0.25f, 0.5f, 0.25f};
neon_conv2d_separable_f32(blurred.data(), image.data(), rows, cols,
                           row_k, 3, col_k, 3);

// Eigen wrapper
Eigen::MatrixXf img = Eigen::MatrixXf::Random(64, 64);
Eigen::MatrixXf kern(3, 3);
kern << 1, 2, 1, 2, 4, 2, 1, 2, 1;
kern /= 16.0f;
Eigen::MatrixXf result = neon_conv2d(img, kern);
```

### NEON Dense Linear Algebra

```cpp
#include <optmath/neon_kernels.hpp>
using namespace optmath::neon;

// Cholesky decomposition (A = L * L^T)
Eigen::MatrixXf A = /* symmetric positive definite matrix */;
Eigen::MatrixXf L = neon_cholesky(A);  // returns lower triangular L

// LU decomposition with partial pivoting
Eigen::MatrixXf M = Eigen::MatrixXf::Random(64, 64);
auto [LU, piv] = neon_lu(M);

// QR decomposition (Householder)
auto [Q, R] = neon_qr(M);  // Q orthogonal, R upper triangular

// Solve A*x = b (general via LU)
Eigen::VectorXf b = Eigen::VectorXf::Random(64);
Eigen::VectorXf x = neon_solve(M, b);

// Solve SPD system via Cholesky (faster for symmetric positive definite)
Eigen::VectorXf x_spd = neon_solve_spd(A, b);

// Triangular solve
Eigen::VectorXf y = neon_trsv_lower(L, b);  // solve L*y = b
Eigen::VectorXf z = neon_trsv_upper(R, b);  // solve R*z = b

// Matrix inverse
Eigen::MatrixXf Minv = neon_inverse(M);
```

### Radar: Cross-Ambiguity Function (CAF)

```cpp
#include <optmath/radar_kernels.hpp>
using namespace optmath::radar;

// Reference and surveillance signals
Eigen::VectorXcf ref = /* transmitter signal */;
Eigen::VectorXcf surv = /* receiver signal */;

// CAF parameters
size_t n_doppler_bins = 101;
size_t n_range_bins = 500;
float doppler_start = -500.0f;  // Hz
float doppler_step = 10.0f;      // Hz
float sample_rate = 1e6f;        // Hz

// Compute CAF (output: n_doppler x n_range magnitude matrix)
Eigen::MatrixXf caf_mag = caf(ref, surv,
                               n_doppler_bins, doppler_start, doppler_step,
                               sample_rate, n_range_bins);

// Find peak (target detection)
Eigen::Index doppler_bin, range_bin;
float peak_mag = caf_mag.maxCoeff(&doppler_bin, &range_bin);
```

### Radar: CFAR Detection

```cpp
// 1D CA-CFAR
Eigen::VectorXf power_data = /* input power */;
size_t guard_cells = 4;
size_t reference_cells = 16;
float pfa_factor = 10.0f;

auto detections = cfar_ca(power_data, guard_cells, reference_cells, pfa_factor);

// 2D CFAR for range-Doppler maps
Eigen::MatrixXf range_doppler_map = /* CAF output */;
auto det_2d = cfar_2d(range_doppler_map,
                      guard_range, guard_doppler,
                      ref_range, ref_doppler,
                      pfa_factor);
```

### CUDA Operations

```cpp
#include <optmath/cuda_backend.hpp>

if (optmath::cuda::is_available()) {
    optmath::cuda::init();

    Eigen::VectorXf a = Eigen::VectorXf::Random(1000000);
    Eigen::VectorXf b = Eigen::VectorXf::Random(1000000);

    // Vector operations
    Eigen::VectorXf c = optmath::cuda::cuda_add(a, b);
    float dot = optmath::cuda::cuda_dot(a, b);

    // Matrix operations (Tensor Core accelerated)
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(1024, 1024);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(1024, 1024);
    Eigen::MatrixXf C = optmath::cuda::cuda_gemm(A, B);

    // FFT
    Eigen::VectorXcf signal = Eigen::VectorXcf::Random(65536);
    Eigen::VectorXcf spectrum = optmath::cuda::cuda_fft(signal);

    // Transcendentals
    Eigen::VectorXf exp_x = optmath::cuda::cuda_exp(a);
    Eigen::VectorXf sigmoid_x = optmath::cuda::cuda_sigmoid(a);
}
```

### Vulkan Operations

```cpp
#include <optmath/vulkan_backend.hpp>

if (optmath::vulkan::is_available()) {
    Eigen::VectorXf a = Eigen::VectorXf::Random(1000000);
    Eigen::VectorXf b = Eigen::VectorXf::Random(1000000);

    // Vector operations
    Eigen::VectorXf c = optmath::vulkan::vulkan_vec_add(a, b);
    float dot = optmath::vulkan::vulkan_vec_dot(a, b);

    // Matrix operations
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(512, 512);
    Eigen::MatrixXf C = optmath::vulkan::vulkan_mat_mul(A, B);

    // Convolution
    Eigen::VectorXf signal = Eigen::VectorXf::Random(10000);
    Eigen::VectorXf kernel = Eigen::VectorXf::Random(64);
    Eigen::VectorXf result = optmath::vulkan::vulkan_convolution_1d(signal, kernel);
}
```

---

## Benchmarking

```bash
cmake -DBUILD_BENCHMARKS=ON ..
make -j$(nproc)

# Run individual benchmarks
./benchmarks/bench_neon_transcendentals  # exp, sin, cos, sigmoid, tanh
./benchmarks/bench_neon_gemm             # Matrix multiplication
./benchmarks/bench_neon_fft              # FIR, cross-correlation, complex ops
./benchmarks/bench_vulkan_matmul         # GPU vector/matrix operations
./benchmarks/bench_radar_caf             # CAF, CFAR, beamforming
```

**Example output (Raspberry Pi 5)**:
```
BM_NEON_Exp_Approx/1048576    1.2 ms    1.2 ms    580  FLOPS=13.1G/s
BM_Std_Exp/1048576           45.2 ms   45.1 ms     15  FLOPS=23.2M/s
BM_NEON_ComplexMul/65536      0.1 ms    0.1 ms   6200  Elements/s=655M
BM_CAF_NEON/4096x64x256       4.8 ms    4.7 ms    148  CAF/s=212
```

---

## Troubleshooting

### Build Issues

**"Vulkan not found" during CMake**:
```bash
# Install Vulkan SDK
sudo apt install -y libvulkan-dev vulkan-tools glslang-tools

# Verify Vulkan
vulkaninfo | head -20
```

**"Could NOT find GTest"**:
```bash
# GTest is fetched automatically via FetchContent
# Clear build directory and re-run CMake
rm -rf build && mkdir build && cd build
cmake ..
```

**"CUDA not found"**:
```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Add to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**"-fPIC" linking error when building shared libraries**:
```bash
# Rebuild with position-independent code
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make -j$(nproc)
sudo make install
```

### Runtime Issues

**"failed to open file: vec_add.comp.spv"**:
```bash
# Run sudo make install to place shaders in /usr/local/share/...
sudo make install

# Or set shader path manually
export OPTMATH_KERNELS_PATH=/usr/local/share/optmathkernels/shaders/
```

**NEON tests skipped**:
```
# NEON is only enabled on ARM platforms
# On x86, tests will skip with "NEON not available"
```

**Eigen3 not found when using installed package**:
```bash
sudo apt install libeigen3-dev
```

### Performance Issues

**High CPU usage on Pi 5**:
- Use NEON for operations that fit in L2 cache (<512KB)
- Use Vulkan for large arrays where GPU parallelism helps
- Enable cache blocking for large GEMM operations

**GPU not being used**:
```bash
# Check Vulkan
vulkaninfo | grep "GPU"

# Check CUDA
nvidia-smi
```

---

## File Structure

```
OptMathKernels/
├── include/optmath/
│   ├── neon_kernels.hpp      # NEON API declarations (124 functions)
│   ├── sve2_kernels.hpp      # SVE2/FCMA/I8MM API declarations (46 functions)
│   ├── platform.hpp          # Platform detection, thread affinity (10 functions)
│   ├── vulkan_backend.hpp    # Vulkan API declarations (23 functions)
│   ├── cuda_backend.hpp      # CUDA API declarations (242 functions)
│   └── radar_kernels.hpp     # Radar processing API (48 functions)
├── src/
│   ├── neon/
│   │   ├── neon_kernels.cpp        # Core NEON + transcendentals
│   │   ├── neon_complex.cpp        # Complex number operations
│   │   ├── neon_gemm_optimized.cpp # Cache-blocked GEMM (runtime-tuned)
│   │   ├── neon_radar.cpp          # Radar signal processing
│   │   ├── neon_resample.cpp       # Polyphase resampler
│   │   ├── neon_iir.cpp            # Biquad IIR filter
│   │   ├── neon_conv2d.cpp         # 2D convolution
│   │   └── neon_linalg.cpp         # Dense linear algebra (Cholesky, LU, QR, solve, inverse)
│   ├── sve2/
│   │   ├── sve2_kernels.cpp        # SVE2 vector ops, transcendentals, GEMM, I8MM
│   │   ├── sve2_complex.cpp        # SVE2/FCMA complex operations
│   │   └── sve2_radar.cpp          # SVE2 radar DSP (CAF, xcorr, beamform)
│   ├── platform/
│   │   └── platform.cpp            # CPU topology, thread affinity, cache detection
│   ├── vulkan/
│   │   ├── vulkan_backend.cpp      # Vulkan context & dispatch (Mali-G720 auto-detect)
│   │   └── shaders/                # 39 GLSL compute shaders
│   │       ├── vec_add.comp.glsl
│   │       ├── mat_mul_tiled.comp.glsl
│   │       ├── mat_mul_tiled_mali.comp.glsl  # 32x32 tiles for Mali-G720
│   │       ├── reduce_sum_mali.comp.glsl     # 1024-thread subgroup reduction
│   │       ├── fft_radix2.comp.glsl
│   │       ├── caf_doppler_shift.comp.glsl
│   │       ├── cfar_2d.comp.glsl
│   │       └── ... (32 more shaders)
│   └── cuda/
│       ├── cuda_backend.cpp        # Context, memory management
│       ├── cuda_kernels.cu         # Vector ops, transcendentals
│       ├── cuda_complex.cu         # Complex ops, FFT
│       └── cuda_radar.cu           # CAF, CFAR, beamforming
├── tests/
│   ├── test_neon_kernels.cpp       # NEON unit tests
│   ├── test_neon_complex.cpp       # Complex operation tests
│   ├── test_neon_transcendentals.cpp
│   ├── test_neon_resample.cpp      # Polyphase resampler tests
│   ├── test_neon_iir.cpp           # Biquad IIR filter tests
│   ├── test_neon_conv2d.cpp        # 2D convolution tests
│   ├── test_neon_linalg.cpp        # Dense linear algebra tests (21 tests)
│   ├── test_sve2_kernels.cpp       # SVE2 unit tests (18 tests)
│   ├── test_platform.cpp           # Platform detection tests (9 tests)
│   ├── test_vulkan_vector.cpp      # Vulkan vector tests
│   ├── test_vulkan_matrix.cpp      # Vulkan matrix tests
│   ├── test_vulkan_dsp.cpp         # Vulkan DSP tests
│   ├── test_cuda_kernels.cpp       # CUDA unit tests
│   ├── test_cuda_radar.cpp         # CUDA radar tests
│   ├── test_radar_caf.cpp          # CAF computation tests
│   └── test_radar_cfar.cpp         # CFAR detection tests
├── benchmarks/
│   ├── bench_neon_transcendentals.cpp
│   ├── bench_neon_gemm.cpp
│   ├── bench_vulkan_matmul.cpp
│   └── bench_radar_caf.cpp
├── examples/
│   ├── demo.cpp                    # NEON/Vulkan demo
│   └── cuda_demo.cpp               # CUDA demo
├── cmake/
│   └── OptMathKernelsConfig.cmake.in  # CMake package config
├── FunctionsIncluded.md            # Complete API reference (417+ functions)
└── README.md                       # This file
```

---

## Related Projects

- **[PassiveRadar_Kraken](https://github.com/n4hy/PassiveRadar_Kraken)**: Complete passive bistatic radar system using OptMathKernels for acceleration. Includes GNU Radio blocks, multi-target tracking, and real-time displays.

---

## Recent Changes

### v0.5.0 - Orange Pi 6 Plus: SVE2, FCMA, I8MM, Mali-G720 (March 2026)

**New Platform: Orange Pi 6 Plus (CIX P1 CD8160)**

Full hardware-specific optimization for the CIX P1 SoC's ARMv9 big.LITTLE cores and Mali-G720-Immortalis GPU.

**SVE2 Acceleration** (`sve2_kernels.cpp`, `sve2_complex.cpp`, `sve2_radar.cpp`):
- Predicated vector operations: `svwhilelt`/`svptest` loops eliminate all scalar tail handling
- FCMA complex multiply: `svcmla_f32_z` for 2-instruction complex multiply (vs 4 NEON)
- I8MM GEMM: `svmmla_s32` hardware int8 matrix multiply with float32 quantization
- SVE2 cache-blocked GEMM tuned for A720 12MB L3 (MC=256, KC=512, NC=1024)
- Complete SVE2 transcendentals: exp, sin, cos, sigmoid, tanh
- SVE2 radar DSP: CAF with FCMA inner loop, cross-correlation, beamforming

**Platform Detection** (`platform.cpp`):
- CPU topology detection via `/proc/cpuinfo` and sysfs (A720/A520 part IDs)
- Thread affinity: pin to performance or efficiency cores via `sched_setaffinity`
- SVE vector length detection via `prctl(PR_SVE_GET_VL)`
- Runtime GEMM cache blocking parameter selection based on L3 cache size

**NEON GEMM Runtime Tuning** (`neon_gemm_optimized.cpp`):
- Cache blocking parameters now auto-selected at runtime based on detected hardware
- 12MB L3 (A720): MC=256, KC=512, NC=1024; 2MB L3 (A76): MC=128, KC=256, NC=512

**Mali-G720 Vulkan Shaders** (`vulkan_backend.cpp`):
- Auto-detects Mali-G720 via vendor ID and device name
- 32x32 tiled GEMM shader (1024 threads, 8KB shared memory)
- 1024-thread reduction with `subgroupAdd()` subgroup arithmetic
- Transparent fallback to standard shaders on non-Mali GPUs

**New Tests:**
| Test Suite | Tests | Description |
|------------|-------|-------------|
| `test_sve2_kernels` | 18 | SVE2 correctness, edge cases, stress tests |
| `test_platform` | 9 | CPU topology, affinity, cache detection |

---

### v0.4.0 - Dense Linear Algebra: Cholesky, LU, QR, Solve, Inverse (February 2026)

**New Dense Linear Algebra Kernels** (`neon_linalg.cpp`):

- **Triangular Solve (TRSV/TRSM)**:
  - Forward/backward substitution with NEON-vectorized AXPY column updates (`vld1q_f32`/`vmlaq_f32`)
  - Unit-diagonal and transpose variants for composing with LU and Cholesky
  - Multi-RHS TRSM for matrix inverse computation

- **Cholesky Decomposition** (`neon_cholesky_f32`):
  - Unblocked right-looking algorithm with NEON dot products
  - SPD validation: returns 1-based error index if matrix is not positive definite
  - Clean lower-triangular output (upper triangle zeroed)

- **LU Decomposition** (`neon_lu_f32`):
  - Partial pivoting with argmax pivot search
  - NEON-vectorized column scaling and rank-1 trailing submatrix updates
  - Row permutation vector output for solver composition

- **QR Decomposition** (`neon_qr_f32`):
  - Householder reflections with stable sign choice to avoid cancellation
  - NEON-vectorized reflector application to trailing columns
  - Explicit Q extraction via reverse-order reflector accumulation

- **Solvers**:
  - `neon_solve_f32`: General A*x=b via LU + TRSV
  - `neon_solve_spd_f32`: SPD A*x=b via Cholesky + TRSV
  - `neon_inverse_f32`: A^{-1} via LU + TRSM on identity columns

- **Eigen Wrappers**: `neon_cholesky`, `neon_lu`, `neon_qr`, `neon_trsv_lower`, `neon_trsv_upper`, `neon_solve`, `neon_solve_spd`, `neon_inverse`

**New Tests:**
| Test Suite | Tests | Status |
|------------|-------|--------|
| `test_neon_linalg` | 21 | Passed |

**Total: 88 individual test cases passed (21 new + 67 existing)**

### v0.3.0 - DSP Kernels: Resampler, IIR, 2D Convolution (February 2026)

**New DSP Kernels:**

- **Polyphase Resampler** (`neon_resample.cpp`):
  - Rational L/M sample rate conversion using polyphase decomposition
  - NEON-optimized FIR filtering per phase (reuses `neon_dot_f32`)
  - Phase accumulator algorithm with delay line state management
  - Both streaming (`neon_resample_f32`) and one-shot (`neon_resample_oneshot_f32`) APIs

- **Biquad IIR Filter** (`neon_iir.cpp`):
  - Direct Form II Transposed for optimal numerical stability
  - Single section (`neon_biquad_f32`) and cascade (`neon_biquad_cascade_f32`)
  - In-place processing supported (out and in may alias)
  - Design helpers using Audio EQ Cookbook formulas: lowpass, highpass, bandpass, notch

- **2D Convolution** (`neon_conv2d.cpp`):
  - General NxM convolution with NEON vectorization over output columns (4 at a time)
  - Separable 2D convolution: row pass reuses `neon_fir_f32`, NEON column pass
  - Fully unrolled 3x3 kernel (9 broadcast coefficients, 9 multiply-accumulates)
  - Unrolled 5x5 kernel specialization
  - Row-major layout for natural image/matrix processing

**New Tests:**
| Test Suite | Tests | Status |
|------------|-------|--------|
| `test_neon_resample` | 7 | Passed |
| `test_neon_iir` | 10 | Passed |
| `test_neon_conv2d` | 9 | Passed |

**Total: 67 individual test cases passed (26 new + 41 existing)**

### v0.2.2 - Cross-Platform Verification (January 2026)

**Cross-Platform Testing:**
- Verified full compatibility between AMD64 (x86_64) and ARM64 (aarch64) platforms
- All 10 test suites pass on Raspberry Pi 5 (ARM Cortex-A76)
- Tested with GCC 14.2.0, Eigen 3.4.0, Vulkan 1.4.309

**Test Results (Raspberry Pi 5):**
| Test Suite | Tests | Status |
|------------|-------|--------|
| `test_basic` | 1 | Passed |
| `test_neon_kernels` | 4 | Passed |
| `test_neon_complex` | 7 | Passed |
| `test_neon_transcendentals` | 10 | Passed |
| `test_vulkan_vector` | 1 | Passed |
| `test_vulkan_matrix` | 1 | Passed |
| `test_vulkan_dsp` | 1 | Passed |
| `test_vulkan_advanced` | 1 | Passed |
| `test_radar_caf` | 6 | Passed |
| `test_radar_cfar` | 9 | Passed |

**Total: 41 individual test cases passed**

### v0.2.1 - Performance and Bug Fixes

**CUDA Backend:**
- **Optimized `cuda_caf()`**: Eliminated all CPU-GPU memory transfers inside the Doppler processing loop
  - Added `kernel_interleave_complex_f32` for GPU-side complex array interleaving
  - Added `kernel_complex_conj_mul_interleaved_f32` for frequency-domain multiplication
  - Added `kernel_magnitude_interleaved_f32` for magnitude extraction
  - Result: ~6 fewer `cudaMemcpy` calls per Doppler bin, data stays on GPU until final output
- **Fixed `cuda_vec_sum_f32()`**: Removed dead code (unused buffer allocation and kernel call)

**NEON Backend:**
- **Fixed `neon_fast_tanh_f32()`**: Removed dead code line that was immediately overwritten

**Testing:**
- All 10 test suites pass (NEON, Vulkan, CUDA, Radar)

---

## License

MIT License - See LICENSE file for details.

---

## Author

**N4HY - Bob McGwier**
Dr Robert W McGwier, PhD

All unit tests,  much of the debugging, and all documentation except the earliest pieces are all Claude code (AMAZING).
