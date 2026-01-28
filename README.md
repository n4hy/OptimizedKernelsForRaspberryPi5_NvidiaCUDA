# OptMathKernels

**High-Performance Numerical Library for Raspberry Pi 5 and NVIDIA GPUs**

OptMathKernels is a C++20 numerical library optimized for **Raspberry Pi 5** and **NVIDIA CUDA GPUs**. It seamlessly bridges **Eigen** (CPU), **ARM NEON** (SIMD), **Vulkan** (Compute Shaders), and **CUDA** (NVIDIA GPUs) into a single, easy-to-use API.

Designed to accelerate math and signal processing tasks by leveraging:
- **Raspberry Pi 5**: Cortex-A76 NEON and VideoCore VII GPU
- **NVIDIA GPUs**: cuBLAS, cuFFT, cuSOLVER, and Tensor Cores (Volta+)

While remaining compatible with standard Linux x86/ARM environments.

---

## Table of Contents

- [Key Applications](#key-applications)
- [Features](#features)
- [Hardware Support](#hardware-support)
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
- **Cache-Blocked GEMM**: 8x8 microkernel with MC=128, KC=256 blocking for Cortex-A76
- **Reductions**: Sum, max, min, dot product with horizontal NEON adds

### GPU Acceleration (`optmath::vulkan`)

- **Tiled GPU Matrix Multiply**: 16x16 shared memory tiles for efficient GPU GEMM
- **FFT**: Radix-2/4 FFT with butterfly operations in compute shaders
- **Convolution**: 1D and 2D convolution with separable kernel optimization
- **Vector Operations**: Add, multiply, dot product, reductions

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

**Cache Blocking Parameters** (optimized for Cortex-A76):
```cpp
// MC=128, KC=256, NC=512 chosen to maximize L2 utilization
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
| **Qualcomm** | Adreno 6xx+ | 1.1 | Proprietary |

**VideoCore VII (Raspberry Pi 5)**:
| Feature | Specification |
|---------|---------------|
| **QPU Cores** | 12 |
| **Clock** | 800 MHz |
| **Compute** | ~1 GFLOPS FP32 |
| **Shared Memory** | 4KB per workgroup |
| **Max Workgroup Size** | 256 |

**Vulkan Compute Shaders**: 37 GLSL compute shaders compiled to SPIR-V:
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

For complete API documentation of all **396+ functions**, see:

**[FunctionsIncluded.md](FunctionsIncluded.md)** - Complete API Reference

### Quick Reference by Backend

| Backend | Functions | Description |
|---------|-----------|-------------|
| **NEON** | 83 | ARM SIMD operations for Raspberry Pi 5 |
| **CUDA** | 242 | NVIDIA GPU kernels (cuBLAS, cuFFT, cuSOLVER) |
| **Vulkan** | 23 | Cross-platform GPU compute shaders |
| **Radar** | 48 | Passive radar signal processing |

### Headers

```cpp
#include <optmath/neon_kernels.hpp>    // ARM NEON operations
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

// Fast exp approximation (~1e-6 relative error)
optmath::neon::neon_exp_f32_approx(output.data(), input.data(), N);

// Fast sin/cos approximation
optmath::neon::neon_sin_f32_approx(output.data(), input.data(), N);
optmath::neon::neon_cos_f32_approx(output.data(), input.data(), N);

// Fast activation functions
optmath::neon::neon_sigmoid_f32_fast(output.data(), input.data(), N);
optmath::neon::neon_tanh_f32_fast(output.data(), input.data(), N);
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
│   ├── neon_kernels.hpp      # NEON API declarations (83 functions)
│   ├── vulkan_backend.hpp    # Vulkan API declarations (23 functions)
│   ├── cuda_backend.hpp      # CUDA API declarations (242 functions)
│   └── radar_kernels.hpp     # Radar processing API (48 functions)
├── src/
│   ├── neon/
│   │   ├── neon_kernels.cpp        # Core NEON + transcendentals
│   │   ├── neon_complex.cpp        # Complex number operations
│   │   ├── neon_gemm_optimized.cpp # Cache-blocked GEMM
│   │   └── neon_radar.cpp          # Radar signal processing
│   ├── vulkan/
│   │   ├── vulkan_backend.cpp      # Vulkan context & dispatch
│   │   └── shaders/                # 37 GLSL compute shaders
│   │       ├── vec_add.comp.glsl
│   │       ├── mat_mul_tiled.comp.glsl
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
├── FunctionsIncluded.md            # Complete API reference (396+ functions)
└── README.md                       # This file
```

---

## Related Projects

- **[PassiveRadar_Kraken](https://github.com/n4hy/PassiveRadar_Kraken)**: Complete passive bistatic radar system using OptMathKernels for acceleration. Includes GNU Radio blocks, multi-target tracking, and real-time displays.

---

## Recent Changes

### v0.2.1 - Performance and Bug Fixes

**CUDA Backend:**
- **Optimized `cuda_caf()`**: Eliminated all CPU-GPU memory transfers inside the Doppler processing loop
  - Added `kernel_interleave_complex_f32` for GPU-side complex array interleaving
  - Added `kernel_complex_conj_mul_interleaved_f32` for frequency-domain multiplication
  - Added `kernel_magnitude_interleaved_f32` for magnitude extraction
  - Result: ~6 fewer `cudaMemcpy` calls per Doppler bin, data stays on GPU until final output
- **Fixed `cuda_vec_sum_f32()`**: Removed dead code (unused buffer allocation and kernel call)

**NEON Backend:**
- **Fixed `neon_tanh_f32_fast()`**: Removed dead code line that was immediately overwritten

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
