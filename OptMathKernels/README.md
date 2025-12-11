# OptMathKernels

**TL;DR**: A high-performance C++20 numerical library optimized for Raspberry Pi 5. It seamlessly bridges **Eigen** (CPU), **ARM NEON** (SIMD), and **Vulkan** (Compute Shaders) into a single, easy-to-use API.

---

## 🚀 Features

*   **Core**: Wraps [Eigen3](https://eigen.tuxfamily.org/) for robust linear algebra on the CPU.
*   **NEON**: Hand-tuned ARMv8 NEON intrinsics for SIMD acceleration on 64-bit ARM (aarch64).
*   **Vulkan**: Compute shader backend for offloading massive parallel tasks to the GPU (VideoCore VII on Pi 5).
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
| `BUILD_EXAMPLES` | `ON` | Build example executable (`example_main`). |
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

#### 1. Core (Eigen Wrapper)
Header: `#include <optmath/core.hpp>`

```cpp
#include <optmath/core.hpp>

optmath::Core core;

// Verify Eigen is linked and working
bool ok = core.verify_eigen();

// SAXPY: Result = a * x + y
// Uses Eigen::Map for zero-copy efficiency where possible.
std::vector<float> x = {1, 2, 3};
std::vector<float> y = {4, 5, 6};
std::vector<float> res = core.saxpy(2.0f, x, y);
```

#### 2. NEON (ARM SIMD)
Header: `#include <optmath/neon_kernels.hpp>`

```cpp
#include <optmath/neon_kernels.hpp>

// Check runtime availability (or compilation support)
if (optmath::neon::is_available()) {
    // Hardware accelerated vector addition
    // Falls back to empty vector if NEON is disabled at compile time.
    std::vector<float> res = optmath::neon::add_vectors(x, y);
}
```

#### 3. Vulkan (GPU Compute)
Header: `#include <optmath/vulkan_backend.hpp>`

```cpp
#include <optmath/vulkan_backend.hpp>

if (optmath::vulkan::is_available()) {
    // Initialize context (selects physical device, etc.)
    if (optmath::vulkan::init()) {
        // Run compute shader: adds scalar to every element
        float scalar = 5.0f;
        std::vector<float> res = optmath::vulkan::compute_add_scalar(x, scalar);
    }
}
```

---

## ⚠️ Raspberry Pi 5 Notes

*   **Vulkan**: Ensure your Pi is running a recent OS (Bookworm 64-bit recommended) with Vulkan drivers installed (`mesa-vulkan-drivers`).
*   **Overclocking**: While this library is optimized for performance, ensure adequate cooling if running heavy compute loops on the Pi 5.
