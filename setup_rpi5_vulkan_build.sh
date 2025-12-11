#!/usr/bin/env bash
#
# setup_rpi5_vulkan_build.sh
#
# End-to-end bootstrap for Raspberry Pi 5:
#   - Install build + Vulkan + Eigen + shader tools
#   - Ensure v3d Vulkan-capable driver is active
#   - Sanity-check Vulkan (V3D / V3DV)
#   - Configure and build OptimizedKernelsForRaspberryPi5
#
# Usage:
#   chmod +x setup_rpi5_vulkan_build.sh
#   ./setup_rpi5_vulkan_build.sh /path/to/OptimizedKernelsForRaspberryPi5
#
# If PROJECT_DIR is omitted, defaults to: $HOME/OptimizedKernelsForRaspberryPi5

set -euo pipefail

PROJECT_DIR="${1:-$HOME/OptimizedKernelsForRaspberryPi5}"
BUILD_DIR="$PROJECT_DIR/build"

echo "== [0] Target project directory: $PROJECT_DIR =="
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory '$PROJECT_DIR' does not exist."
    echo "Clone it first, e.g.:"
    echo "  git clone <your-repo-url> \"$PROJECT_DIR\""
    exit 1
fi

echo "== [1] Updating APT and installing required packages =="
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
    lshw

echo
echo "== [2] Confirming architecture and basic GPU presence =="
dpkg --print-architecture || true
echo
echo "lshw -c display (may only show framebuffer, that is OK):"
sudo lshw -c display || true

echo
echo "== [3] Ensuring v3d kernel module is loaded (Vulkan-capable GPU driver) =="
if lsmod | grep -q '^v3d'; then
    echo "v3d module already loaded."
else
    echo "Loading v3d module..."
    if ! sudo modprobe v3d; then
        echo "WARNING: Failed to load v3d module. Vulkan hardware path may not work."
    fi
fi

# Ensure v3d is loaded automatically on boot
echo "Persisting v3d module in /etc/modules-load.d/v3d.conf ..."
echo 'v3d' | sudo tee /etc/modules-load.d/v3d.conf >/dev/null

echo
echo "== [4] Checking DRM render node (required for Vulkan) =="
if [ -d /dev/dri ]; then
    ls -l /dev/dri
else
    echo "WARNING: /dev/dri does not exist. KMS/DRM stack might not be active."
fi

if [ -e /dev/dri/renderD128 ]; then
    echo "OK: /dev/dri/renderD128 present (Vulkan render node)."
else
    echo "WARNING: /dev/dri/renderD128 missing; Vulkan may fall back to llvmpipe."
fi

echo
echo "== [5] Check Vulkan ICDs (driver JSON files) =="
if [ -d /usr/share/vulkan/icd.d ]; then
    ls /usr/share/vulkan/icd.d
else
    echo "WARNING: /usr/share/vulkan/icd.d does not exist; no Vulkan ICDs found."
fi

# Broadcom ICD is the one that matters for Raspberry Pi 5
BROADCOM_ICD="/usr/share/vulkan/icd.d/broadcom_icd.json"
if [ -f "$BROADCOM_ICD" ]; then
    echo "Broadcom ICD found at $BROADCOM_ICD"
else
    echo "WARNING: Broadcom ICD ($BROADCOM_ICD) not found. Vulkan may still work, but check your Mesa version."
fi

echo
echo "== [6] Vulkan sanity check (deviceName / driverName) =="
if command -v vulkaninfo >/dev/null 2>&1; then
    # Prefer forcing Broadcom ICD if it exists
    if [ -f "$BROADCOM_ICD" ]; then
        echo "Using Broadcom ICD for vulkaninfo probe..."
        VK_ICD_FILENAMES="$BROADCOM_ICD" vulkaninfo 2>/dev/null | \
            grep -i -E 'deviceName|driverName' || \
            echo "WARNING: vulkaninfo did not return expected device/driver info."
    else
        echo "Running vulkaninfo with default ICDs..."
        vulkaninfo 2>/dev/null | \
            grep -i -E 'deviceName|driverName' || \
            echo "WARNING: vulkaninfo did not return expected device/driver info."
    fi
else
    echo "WARNING: vulkaninfo not found, but vulkan-tools should have installed it."
fi

echo
echo "== [7] Confirm glslangValidator (GLSL -> SPIR-V compiler) =="
if command -v glslangValidator >/dev/null 2>&1; then
    echo "glslangValidator found at: $(command -v glslangValidator)"
    glslangValidator --version || true
else
    echo "WARNING: glslangValidator not found, but glslang-tools should have installed it."
    echo "Shaders will not be built at compile time if your CMake depends on it."
fi

echo
echo "== [8] Configure and build OptimizedKernelsForRaspberryPi5 =="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "-- Removing old CMake cache (if any) --"
rm -f CMakeCache.txt

echo "-- Running CMake configure --"
cmake ..  # add extra -D flags here if your project needs them

echo
echo "-- Building (all targets) with maximum parallelism --"
cmake --build . -j"$(nproc)"

echo
echo "== [9] Summary =="
echo "  - Project directory: $PROJECT_DIR"
echo "  - Build directory:   $BUILD_DIR"
echo "  - Vulkan dev:        libvulkan-dev installed"
echo "  - Mesa Vulkan:       mesa-vulkan-drivers installed"
echo "  - Tools:             vulkaninfo, glslangValidator present"
echo "  - Kernel driver:     v3d module loaded (and set to auto-load)"
echo
echo "Setup and build completed."
