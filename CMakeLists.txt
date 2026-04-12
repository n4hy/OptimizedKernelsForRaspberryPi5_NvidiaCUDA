cmake_minimum_required(VERSION 3.18)

project(OptMathKernels
    VERSION 0.5.13
    LANGUAGES CXX
)

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
option(ENABLE_NEON      "Enable NEON-optimized kernels"     ON)
option(ENABLE_SVE2      "Enable SVE2-optimized kernels"     ON)
option(ENABLE_FCMA      "Enable FCMA complex multiply"      ON)
option(ENABLE_I8MM      "Enable I8MM int8 matrix multiply"  ON)
option(ENABLE_VULKAN    "Enable Vulkan-based kernels"       ON)
option(ENABLE_CUDA      "Enable NVIDIA CUDA-based kernels"  ON)
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

# 3. CUDA (NVIDIA GPU support)
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
elseif(ENABLE_NEON AND NOT ON_ARM)
    message(STATUS "NEON enabled but not on ARM. NEON optimizations will be skipped/simulated.")
endif()

# --- SVE2 / FCMA / I8MM Detection ---
if (ENABLE_SVE2 AND ON_ARM)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=armv9-a+sve2" HAS_SVE2)
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
    else()
        message(STATUS "SVE2 not supported by compiler. SVE2 optimizations disabled.")
    endif()
elseif(ENABLE_SVE2 AND NOT ON_ARM)
    message(STATUS "SVE2 enabled but not on ARM. SVE2 optimizations will be skipped.")
endif()

if (ENABLE_VULKAN)
    add_compile_definitions(OPTMATH_USE_VULKAN)
endif()

# CUDA definitions already added in detection section above

# --- Subdirectories ---
add_subdirectory(src)

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
