# OptMathKernels

**TL;DR**: A high-performance C++20 numerical library optimized for Raspberry Pi 5. It seamlessly bridges **Eigen** (CPU), **ARM NEON** (SIMD), and **Vulkan** (Compute Shaders) into a single, easy-to-use API.

---

## 🚀 Features

*   **NEON**: Hand-tuned ARMv8 NEON intrinsics for SIMD acceleration on 64-bit ARM (aarch64).
*   **Vulkan**: Compute shader backend for offloading massive parallel tasks to the GPU (VideoCore VII on Pi 5).
*   **Eigen Integration**: Seamlessly switch between CPU (Eigen), NEON, and Vulkan backends using Eigen types (`Eigen::VectorXf`, etc.).
*   **Easy Integration**: Standard CMake package that installs to `/usr/local` (or custom prefix) and integrates via `find_package`.

---

## 🛠️ Builder Guide

### Prerequisites
*   **C++ Compiler**: GCC 10+ or Clang 11+ (C++20 support required).
*   **CMake**: 3.18 or newer.
*   **Vulkan SDK**: `libvulkan-dev`, `glslang-tools` (for shader compilation).
*   **Eigen3**: Will be automatically fetched if not found on the system.

### Build & Install

```bash
# 1. Clone
git clone <repo_url>
cd OptMathKernels

# 2. Configure
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_NEON=ON \
      -DENABLE_VULKAN=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

# 3. Build
cmake --build . -j$(nproc)

# 4. Test (Optional)
ctest --output-on-failure

# 5. Install
sudo cmake --install .
```

### CMake Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_NEON` | `ON` | Enable ARM NEON intrinsics. Automatically simulated/disabled on non-ARM. |
| `ENABLE_VULKAN` | `ON` | Enable Vulkan backend. Requires Vulkan headers and `glslangValidator`. |
| `BUILD_EXAMPLES` | `ON` | Build example executable (`demo`). |
| `BUILD_TESTS` | `ON` | Build unit tests. |

---

## 📖 User Guide

### Integration
Once installed, use `find_package` in your own `CMakeLists.txt`:

```cmake
find_package(OptMathKernels REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE OptMathKernels::OptMathKernels)
```

### API Reference

#### 1. NEON (ARM SIMD)
Header: `#include <optmath/neon_kernels.hpp>`

```cpp
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>

// Check runtime availability
if (optmath::neon::is_available()) {
    Eigen::VectorXf a(10), b(10);
    // ... fill a, b ...

    // Hardware accelerated operations
    Eigen::VectorXf sum = optmath::neon::neon_add(a, b);
    float dot = optmath::neon::neon_dot(a, b);

    // In-place activations
    optmath::neon::neon_relu(a);
}
```

#### 2. Vulkan (GPU Compute)
Header: `#include <optmath/vulkan_backend.hpp>`

```cpp
#include <optmath/vulkan_backend.hpp>
#include <Eigen/Dense>

if (optmath::vulkan::is_available()) {
    Eigen::VectorXf a(1000), b(1000);
    // ... fill a, b ...

    // Offload to GPU
    Eigen::VectorXf sum = optmath::vulkan::vulkan_vec_add(a, b);
    float dot = optmath::vulkan::vulkan_vec_dot(a, b);

    // 1D Convolution
    Eigen::VectorXf x(1000), h(50);
    Eigen::VectorXf y = optmath::vulkan::vulkan_conv1d(x, h);
}
```

---

## ⚠️ Raspberry Pi 5 Notes

*   **Vulkan**: Ensure your Pi is running a recent OS (Bookworm 64-bit recommended) with Vulkan drivers installed (`mesa-vulkan-drivers`).
*   **Overclocking**: While this library is optimized for performance, ensure adequate cooling if running heavy compute loops on the Pi 5.
*   **Performance**: Small workloads are generally faster on CPU/NEON due to GPU dispatch overhead. Vulkan shines with large vectors (N > 100k).
