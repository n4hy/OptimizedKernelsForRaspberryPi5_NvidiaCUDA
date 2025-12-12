# OptMathKernels

**OptMathKernels** is a high-performance C++20 numerical library optimized for **Raspberry Pi 5**. It seamlessly bridges **Eigen** (CPU), **ARM NEON** (SIMD), and **Vulkan** (Compute Shaders) into a single, easy-to-use API.

It is designed to accelerate math and signal processing tasks by leveraging the specialized hardware of the Raspberry Pi 5 (Cortex-A76 NEON and VideoCore VII GPU), while remaining compatible with standard Linux x86/ARM environments.

---

## 🚀 Features

*   **NEON Acceleration**: Hand-tuned ARMv8 NEON intrinsics for SIMD acceleration on 64-bit ARM (aarch64). Includes optimized matrix multiplication, convolution, and vector math.
*   **Vulkan Compute**: Massive parallel offloading to the GPU (VideoCore VII on Pi 5). Supports large vector operations, matrix math, FFT (Radix-2/4), and reductions.
*   **Eigen Integration**: Fully compatible with `Eigen::VectorXf` and `Eigen::MatrixXf`. Pass your existing data structures directly to accelerated kernels.
*   **Easy Integration**: Standard CMake package that installs to `/usr/local` and works with `find_package(OptMathKernels)`.

---

## 📦 Prerequisites

Before building, ensure you have the necessary dependencies installed.

### Raspberry Pi 5 / Ubuntu / Debian

Run the following command to install build tools, CMake, and Vulkan development headers:

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

> **Note:** GoogleTest is automatically fetched during the build (via CMake `FetchContent`), so `libgtest-dev` is not strictly required but recommended for caching.

---

## 🛠️ Build & Install

We provide a robust build process using CMake.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/OptMathKernels.git
cd OptMathKernels
```

### 2. Configure and Build
```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
# - ENABLE_NEON=ON:  Enables ARM NEON optimizations (auto-detected on ARM)
# - ENABLE_VULKAN=ON: Enables GPU Compute kernels
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_NEON=ON \
      -DENABLE_VULKAN=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

# Build using all available cores
make -j$(nproc)
```

### 3. Run Tests (Verification)
Verify that everything is working correctly on your hardware.
```bash
ctest --output-on-failure
```
You should see all tests pass (`test_basic`, `test_vulkan_vector`, `test_neon_kernels`, etc.).

### 4. Install
Install the library and headers to the system (default `/usr/local`).
```bash
sudo make install
```

---

## 📖 Usage Guide

### CMake Integration
In your project's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyApp)

find_package(OptMathKernels REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE OptMathKernels::OptMathKernels)
```

### C++ API Example

#### 1. Vulkan Backend (GPU)
For large datasets (>100k elements) or heavy compute (FFT, Convolution).

```cpp
#include <optmath/vulkan_backend.hpp>
#include <Eigen/Dense>
#include <iostream>

int main() {
    // 1. Initialize Check
    if (!optmath::vulkan::is_available()) {
        std::cerr << "Vulkan not available!" << std::endl;
        return 1;
    }

    // 2. Data Prep
    int N = 1000000;
    Eigen::VectorXf a = Eigen::VectorXf::Random(N);
    Eigen::VectorXf b = Eigen::VectorXf::Random(N);

    // 3. GPU Compute
    // Vector Addition
    Eigen::VectorXf c = optmath::vulkan::vulkan_vec_add(a, b);

    // FFT (In-place)
    // Input size must be power of 2.
    // Data treated as interleaved complex (Real, Imag, Real, Imag...)
    Eigen::VectorXf fft_data(2048);
    fft_data.setRandom();
    optmath::vulkan::vulkan_fft_radix2(fft_data, false); // Forward

    std::cout << "Computed on GPU!" << std::endl;
    return 0;
}
```

#### 2. NEON Backend (CPU SIMD)
For low-latency operations or smaller matrices where GPU transfer overhead is too high.

```cpp
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>

int main() {
    if (optmath::neon::is_available()) {
        int M = 64, K = 64, N = 64;
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);

        // Accelerated Matrix Multiplication (GEMM)
        Eigen::MatrixXf C = optmath::neon::neon_gemm(A, B);

        // Accelerated Vector Norm
        Eigen::VectorXf v = Eigen::VectorXf::Random(1024);
        float n = optmath::neon::neon_norm(v);
    }
    return 0;
}
```

---

## 🔐 Environment Configuration

### Shader Location (`OPTMATH_KERNELS_PATH`)

OptMathKernels compiles shaders into SPIR-V (`.spv`) files. By default, the library looks for them in:
1.  The current working directory.
2.  `../src/` relative path (useful during development).
3.  `/usr/local/share/optmathkernels/shaders/` (standard install path).

If you are running your application from a non-standard location and getting "Shader file not found" errors, set the `OPTMATH_KERNELS_PATH` environment variable:

```bash
export OPTMATH_KERNELS_PATH=/path/to/installation/share/optmathkernels/shaders/
./my_app
```

---

## ⚡ Raspberry Pi 5 Optimization Tips

1.  **Vulkan Driver**: Ensure the `v3d` kernel module is loaded.
    ```bash
    lsmod | grep v3d
    ```
    If missing, add `dtoverlay=vc4-kms-v3d` to `/boot/firmware/config.txt` and reboot.

2.  **Performance**:
    *   **NEON**: Faster for operations that fit in CPU cache (L2/L3) or are memory-bandwidth bound on small arrays.
    *   **Vulkan**: Faster for heavy arithmetic (Convolution, FFT, huge Matrix Mul) where the GPU's parallelism outweighs data transfer costs.

---

## 🧪 Troubleshooting

*   **"Vulkan not found" during CMake**:
    *   Ensure `libvulkan-dev` is installed.
    *   Check `vulkaninfo` runs correctly.

*   **"Could NOT find GTest"**:
    *   The build script now fetches GTest automatically. If you see issues, try clearing the build directory: `rm -rf build` and re-running CMake.

*   **Runtime "failed to open file: vec_add.comp.spv"**:
    *   This means the library cannot find the compiled shaders.
    *   **Fix**: Run `sudo make install` to place them in `/usr/local/share/...`.
    *   **Fix**: Or set `export OPTMATH_KERNELS_PATH=/path/to/your/build/src/` before running.
