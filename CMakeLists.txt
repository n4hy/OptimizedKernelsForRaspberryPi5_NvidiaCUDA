cmake_minimum_required(VERSION 3.18)

project(OptMathKernels
    VERSION 0.6.3
    LANGUAGES CXX
)

# Default to a Release build when the user doesn't specify one. With single-
# config generators (Make/Ninja — the norm on the Pi), an unset build type
# means empty flags => -O0, silently compiling every NEON/Vulkan-host kernel
# unoptimized. Guard so a plain `cmake ..` still gets -O3.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug/Release/RelWithDebInfo/MinSizeRel)" FORCE)
    message(STATUS "No CMAKE_BUILD_TYPE specified; defaulting to Release")
endif()

# Probe for link-time / interprocedural optimization support. Applied (scoped to
# the OptMathKernels library, Release only) in src/CMakeLists.txt so the NEON
# kernels — each in its own translation unit — get cross-TU inlining.
include(CheckIPOSupported)
check_ipo_supported(RESULT OPTMATH_IPO_SUPPORTED OUTPUT OPTMATH_IPO_ERROR)
if(OPTMATH_IPO_SUPPORTED)
    message(STATUS "Interprocedural optimization (LTO) available")
else()
    message(STATUS "Interprocedural optimization (LTO) not available: ${OPTMATH_IPO_ERROR}")
endif()

# Check for CUDA support early (before setting standards)
include(CheckLanguage)
check_language(CUDA)

# Enforce C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build all libraries with position-independent code (-fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Installation directories
include(GNUInstallDirs)

# Options
# CUDA defaults OFF on ARM (no NVIDIA GPU on a Pi 5 / Orange Pi). The lightweight
# check_language(CUDA) probe above always runs; defaulting OFF skips the heavier
# enable_language(CUDA) + find_package(CUDAToolkit) below. Override -DENABLE_CUDA=ON.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(OPTMATH_CUDA_DEFAULT OFF)
else()
    set(OPTMATH_CUDA_DEFAULT ON)
endif()

option(ENABLE_NEON      "Enable NEON-optimized kernels"     ON)
option(ENABLE_SVE2      "Enable SVE2-optimized kernels"     ON)
option(ENABLE_FCMA      "Enable FCMA complex multiply"      ON)
option(ENABLE_I8MM      "Enable I8MM int8 matrix multiply"  ON)
option(ENABLE_VULKAN    "Enable Vulkan-based kernels"       ON)
option(ENABLE_CUDA      "Enable NVIDIA CUDA-based kernels"  ${OPTMATH_CUDA_DEFAULT})
option(BUILD_EXAMPLES   "Build examples"                    ON)
option(BUILD_TESTS      "Build tests"                       ON)
option(BUILD_BENCHMARKS "Build benchmarks"                  OFF)

# --- Dependencies ---

# 1. Eigen3
# First try to find it on the system
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally. Fetching...")
    include(FetchContent)

    # Disable Eigen's own tests and docs to save build time and file count
    # We force these variables to ensure they propagate to the sub-build
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable Eigen pkgconfig" FORCE)
    set(BUILD_TESTING_SAVE ${BUILD_TESTING})
    set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)

    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
    )
    FetchContent_MakeAvailable(Eigen3)

    # Restore BUILD_TESTING
    set(BUILD_TESTING ${BUILD_TESTING_SAVE} CACHE BOOL "" FORCE)

    # After fetch, verify target exists
    if(NOT TARGET Eigen3::Eigen)
        # Eigen's CMakeLists usually defines 'eigen', but we want consistent namespacing
        if(TARGET eigen)
            add_library(Eigen3::Eigen ALIAS eigen)
        else()
             # Fallback if the fetch didn't define targets as expected
             message(STATUS "Manually defining Eigen3::Eigen target from fetch...")
             add_library(Eigen3::Eigen INTERFACE IMPORTED)
             target_include_directories(Eigen3::Eigen INTERFACE ${eigen3_SOURCE_DIR})
        endif()
    endif()
else()
    message(STATUS "Found Eigen3: ${Eigen3_VERSION}")
endif()

# 2. Vulkan
if(ENABLE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found: ${Vulkan_VERSION}")
    else()
        message(WARNING "Vulkan not found. Disabling Vulkan support.")
        set(ENABLE_VULKAN OFF)
    endif()
endif()

# 3. OpenMP (multi-core CPU parallelism for the NEON kernels)
# The Pi 5 has 4 Cortex-A76 cores; the hot GEMM/CAF/CFAR/conv kernels are
# otherwise single-threaded. Linked PUBLIC in src/CMakeLists.txt so consumers
# pick up libgomp. Pragmas are guarded with #ifdef _OPENMP so a no-OpenMP
# build still compiles serially.
option(ENABLE_OPENMP "Enable OpenMP multi-threading of CPU kernels" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION} (multi-core CPU kernels enabled)")
    else()
        message(STATUS "OpenMP not found; CPU kernels will run single-threaded")
    endif()
endif()

# 4. CUDA (NVIDIA GPU support)
if(ENABLE_CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
            message(STATUS "  CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")

            # Set CUDA standard
            set(CMAKE_CUDA_STANDARD 17)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)

            # GPU Architecture Configuration
            # By default, build for multiple architectures for broad compatibility.
            # Use -DOPTMATH_CUDA_NATIVE=ON to build only for the detected GPU (faster compile).
            # Use -DCMAKE_CUDA_ARCHITECTURES="75;86" to specify exact architectures.
            option(OPTMATH_CUDA_NATIVE "Build only for detected GPU architecture (faster compile)" OFF)

            if(OPTMATH_CUDA_NATIVE)
                # Use CMake's auto-detected architecture (single GPU)
                message(STATUS "  CUDA Native mode - Building for detected GPU: SM ${CMAKE_CUDA_ARCHITECTURES}")
            else()
                # Build for multiple architectures based on CUDA toolkit version
                if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                    # CUDA 13+ dropped support for SM < 75 (Maxwell, Pascal, Volta removed)
                    # Supported: Turing (75), Ampere (80,86,87,89), Hopper (90,90a), Blackwell (100,103)
                    # SM 100: RTX 5090/5080 (GB202/GB203), SM 103: RTX 5070/5060 (GB205/GB206)
                    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;100;103" CACHE STRING "CUDA architectures" FORCE)
                    message(STATUS "  CUDA 13+ detected - Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
                    message(STATUS "  Note: CUDA 13 minimum is SM 7.5 (Turing). For older GPUs, use CUDA 12.x")
                elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
                    # CUDA 12.x supports SM 50+ (Maxwell through Hopper/Blackwell with 12.8+)
                    set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89" CACHE STRING "CUDA architectures" FORCE)
                    message(STATUS "  CUDA 12.x detected - Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
                elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
                    # CUDA 11.x supports SM 35+ (Kepler through Ampere)
                    set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86" CACHE STRING "CUDA architectures" FORCE)
                    message(STATUS "  CUDA 11.x detected - Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
                else()
                    # Older CUDA - use conservative defaults
                    set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75" CACHE STRING "CUDA architectures" FORCE)
                    message(STATUS "  CUDA <11.0 detected - Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
                endif()
            endif()

            # CUDA Toolkit version detection for architecture-specific features
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                message(STATUS "  CUDA 13+ - Full Blackwell (SM 100) native support")
                add_compile_definitions(OPTMATH_CUDA_13_PLUS)
            endif()

            # Architecture-specific optimizations (newest to oldest)
            if("103" IN_LIST CMAKE_CUDA_ARCHITECTURES OR "100" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Blackwell (SM 100/103) enabled - RTX 5090/5080/5070 optimizations")
                add_compile_definitions(OPTMATH_CUDA_BLACKWELL)
            endif()
            if("90" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Hopper (SM 90) enabled - H100 optimizations")
                add_compile_definitions(OPTMATH_CUDA_HOPPER)
            endif()
            if("89" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Ada Lovelace (SM 89) enabled - RTX 4090/4080/4070 support")
                add_compile_definitions(OPTMATH_CUDA_ADA)
            endif()
            if("86" IN_LIST CMAKE_CUDA_ARCHITECTURES OR "80" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Ampere (SM 80/86) enabled - RTX 3090/3080/A100 support")
                add_compile_definitions(OPTMATH_CUDA_AMPERE)
            endif()
            if("75" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Turing (SM 75) enabled - RTX 2080/2070/2060 support")
                add_compile_definitions(OPTMATH_CUDA_TURING)
            endif()
            if("70" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Volta (SM 70) enabled - Tesla V100/Titan V support")
                add_compile_definitions(OPTMATH_CUDA_VOLTA)
            endif()
            if("60" IN_LIST CMAKE_CUDA_ARCHITECTURES OR "61" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Pascal (SM 60/61) enabled - GTX 1080/1070/P100 support")
                add_compile_definitions(OPTMATH_CUDA_PASCAL)
            endif()
            if("50" IN_LIST CMAKE_CUDA_ARCHITECTURES OR "52" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Maxwell (SM 50/52) enabled - GTX 980/970/Titan X support")
                add_compile_definitions(OPTMATH_CUDA_MAXWELL)
            endif()

            # Add CUDA compile definitions
            add_compile_definitions(OPTMATH_USE_CUDA)
        else()
            message(WARNING "CUDA Toolkit not found. Disabling CUDA support.")
            set(ENABLE_CUDA OFF)
        endif()
    else()
        message(WARNING "CUDA compiler not found. Disabling CUDA support.")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# --- ARM / Pi 5 Detection ---
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ON_ARM TRUE)
    message(STATUS "Detected ARM platform")
else()
    set(ON_ARM FALSE)
    message(STATUS "Detected non-ARM platform")
endif()

if (ENABLE_NEON AND ON_ARM)
    message(STATUS "Building with NEON optimizations for ARM")
    add_compile_definitions(OPTMATH_USE_NEON)

    # Tune codegen for the host ARM core. Without this the compiler schedules
    # for a generic ARMv8-A pipeline and cannot use A76 features (FEAT_FP16,
    # dotprod) or its dual-FMA scheduling model. On a native SBC build,
    # -mcpu=native selects the exact core (Cortex-A76 on the Pi 5, Cortex-A720
    # on the Orange Pi 6); fall back to an explicit A76 target, then to a
    # generic ARMv8.2 baseline with the features the Pi 5 exposes.
    include(CheckCXXCompilerFlag)
    set(OPTMATH_ARCH_FLAG "")
    check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
    if(HAS_MCPU_NATIVE)
        set(OPTMATH_ARCH_FLAG "-mcpu=native")
    else()
        check_cxx_compiler_flag("-mcpu=cortex-a76" HAS_MCPU_A76)
        if(HAS_MCPU_A76)
            set(OPTMATH_ARCH_FLAG "-mcpu=cortex-a76")
        else()
            check_cxx_compiler_flag("-march=armv8.2-a+fp16+dotprod" HAS_ARMV82_FEATURES)
            if(HAS_ARMV82_FEATURES)
                set(OPTMATH_ARCH_FLAG "-march=armv8.2-a+fp16+dotprod")
            endif()
        endif()
    endif()
    if(OPTMATH_ARCH_FLAG)
        message(STATUS "  ARM CPU tuning: ${OPTMATH_ARCH_FLAG}")
        # C++ only — SVE2 TUs carry their own -march, and CUDA (nvcc) would
        # reject a bare -mcpu flag.
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${OPTMATH_ARCH_FLAG}>)
    endif()

    # Half-precision (FP16) kernels: only where the target has FEAT_FP16 vector
    # arithmetic (the Cortex-A76 does; a generic ARMv8-A core does not). Probe
    # with the arch flag we just selected.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "${OPTMATH_ARCH_FLAG}")
    check_cxx_source_compiles(
        "#include <arm_neon.h>
         int main(){ float16x8_t a = vdupq_n_f16(0); a = vfmaq_f16(a, a, a); return (int)vgetq_lane_f16(a, 0); }"
        OPTMATH_HAS_FP16)
    unset(CMAKE_REQUIRED_FLAGS)
    if(OPTMATH_HAS_FP16)
        message(STATUS "  FP16 half-precision kernels enabled")
        add_compile_definitions(OPTMATH_USE_FP16)
    endif()

    # Int8 dot-product (SDOT/UDOT, FEAT_DotProd / "asimddp"). The Cortex-A76 has
    # it; a generic ARMv8-A core does not. Probe with the selected arch flag.
    set(CMAKE_REQUIRED_FLAGS "${OPTMATH_ARCH_FLAG}")
    check_cxx_source_compiles(
        "#include <arm_neon.h>
         int main(){ int32x4_t a = vdupq_n_s32(0); int8x16_t b = vdupq_n_s8(1);
                     a = vdotq_s32(a, b, b); return (int)vgetq_lane_s32(a, 0); }"
        OPTMATH_HAS_DOTPROD)
    unset(CMAKE_REQUIRED_FLAGS)
    if(OPTMATH_HAS_DOTPROD)
        message(STATUS "  Int8 dot-product (SDOT) kernels enabled")
        add_compile_definitions(OPTMATH_USE_DOTPROD)
    endif()
elseif(ENABLE_NEON AND NOT ON_ARM)
    message(STATUS "NEON enabled but not on ARM. NEON optimizations will be skipped/simulated.")
endif()

# --- SVE2 / FCMA / I8MM Detection ---
# HAS_SVE2 (consumed by src/CMakeLists.txt to decide whether to compile the
# sve2/*.cpp sources) must reflect whether SVE2 will actually run — NOT merely
# whether the compiler accepts -march=armv9-a+sve2. A recent GCC/Clang accepts
# that flag on a Pi 5, but the Cortex-A76 is ARMv8.2-A with no SVE2: compiling
# those TUs at ARMv9 baseline lets non-intrinsic instructions outside the
# hwcap-guarded paths fault (SIGILL). So we additionally require the build host
# to actually report SVE2 (native build), with an override for cross-compiles.
option(OPTMATH_FORCE_SVE2 "Force-enable SVE2 sources (cross-compiling to an SVE2 target)" OFF)
if (ENABLE_SVE2 AND ON_ARM)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=armv9-a+sve2" COMPILER_ACCEPTS_SVE2)

    set(CPU_HAS_SVE2 FALSE)
    if(EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" _optmath_cpuinfo)
        string(FIND "${_optmath_cpuinfo}" "sve2" _optmath_sve2_pos)
        if(NOT _optmath_sve2_pos EQUAL -1)
            set(CPU_HAS_SVE2 TRUE)
        endif()
    endif()

    if(COMPILER_ACCEPTS_SVE2 AND (CPU_HAS_SVE2 OR OPTMATH_FORCE_SVE2))
        set(HAS_SVE2 TRUE)
    else()
        set(HAS_SVE2 FALSE)
    endif()

    if(HAS_SVE2)
        message(STATUS "Building with SVE2 optimizations for ARM")
        add_compile_definitions(OPTMATH_USE_SVE2)

        if(ENABLE_FCMA)
            message(STATUS "  FCMA complex multiply enabled")
            add_compile_definitions(OPTMATH_USE_FCMA)
        endif()

        if(ENABLE_I8MM)
            check_cxx_compiler_flag("-march=armv9-a+sve2+i8mm" HAS_I8MM)
            if(HAS_I8MM)
                message(STATUS "  I8MM int8 matrix multiply enabled")
                add_compile_definitions(OPTMATH_USE_I8MM)
            else()
                message(STATUS "  I8MM not supported by compiler")
            endif()
        endif()
    elseif(NOT COMPILER_ACCEPTS_SVE2)
        message(STATUS "SVE2 not supported by compiler. SVE2 optimizations disabled.")
    else()
        message(STATUS "SVE2 accepted by compiler but host CPU lacks SVE2 "
                       "(e.g. Cortex-A76 / Pi 5); SVE2 sources skipped. "
                       "Use -DOPTMATH_FORCE_SVE2=ON to override for cross-builds.")
    endif()
elseif(ENABLE_SVE2 AND NOT ON_ARM)
    message(STATUS "SVE2 enabled but not on ARM. SVE2 optimizations will be skipped.")
endif()

if (ENABLE_VULKAN)
    add_compile_definitions(OPTMATH_USE_VULKAN)
endif()

# CUDA definitions already added in detection section above

# --- Conservative codegen flags (keep IEEE semantics) ---
# -fno-math-errno: the hot radar loops call std::cos/std::sin/std::sqrt per
#   element (neon_radar.cpp). Treating libm as errno-setting blocks inlining and
#   the A76 fsqrt/reciprocal paths; dropping errno lets them vectorize.
# -fno-semantic-interposition: under the library's -fPIC, intra-library calls to
#   non-static functions (e.g. neon::neon_fast_cos_f32 from caf_f32) otherwise
#   route through the GOT and cannot inline.
# Neither relaxes NaN/Inf/signed-zero/reassociation (unlike -ffast-math), so the
# numerically sensitive filter paths keep full IEEE behavior.
include(CheckCXXCompilerFlag)
set(OPTMATH_CODEGEN_FLAGS "")
foreach(_optmath_flag -fno-math-errno -fno-semantic-interposition)
    string(MAKE_C_IDENTIFIER "HAS_${_optmath_flag}" _optmath_flagvar)
    check_cxx_compiler_flag("${_optmath_flag}" ${_optmath_flagvar})
    if(${_optmath_flagvar})
        list(APPEND OPTMATH_CODEGEN_FLAGS ${_optmath_flag})
    endif()
endforeach()
if(OPTMATH_CODEGEN_FLAGS)
    message(STATUS "Conservative codegen flags: ${OPTMATH_CODEGEN_FLAGS}")
    # Wrap each flag in its own genex: a list inside a single $<...:a;b> breaks
    # because the closing '>' attaches to the last list element.
    foreach(_optmath_cg ${OPTMATH_CODEGEN_FLAGS})
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${_optmath_cg}>)
    endforeach()
endif()

# --- Subdirectories ---
add_subdirectory(src)

# Export OPTMATH_USE_SVE2 to CONSUMERS (INTERFACE) when the SVE2 sources were
# actually compiled. add_compile_definitions() above only affects this project's
# own TUs; without this a downstream that dispatches to optmath::sve2::* (e.g.
# FilterMath.h) could not tell whether the symbols exist. Guarded by HAS_SVE2, so
# on a Cortex-A76 / Pi 5 (no SVE2) it is a no-op and consumers fall back to NEON.
if(HAS_SVE2)
    target_compile_definitions(OptMathKernels INTERFACE OPTMATH_USE_SVE2)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
