cmake_minimum_required(VERSION 3.18)

project(OptMathKernels
    VERSION 0.4.0
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

            # Detect GPU architecture
            # Default to common architectures if not specified
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                # RTX 2000 (Turing 75), RTX 3000 (Ampere 80,86), RTX 4000 (Ada 89), RTX 5000 (Blackwell 100)
                set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;100")
                message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
            endif()

            # Architecture-specific optimizations
            if("100" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Blackwell (SM 100) enabled - RTX 5090/5080 optimizations active")
                add_compile_definitions(OPTMATH_CUDA_BLACKWELL)
            endif()
            if("89" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Ada Lovelace (SM 89) enabled - RTX 4090/4080/4070 support")
                add_compile_definitions(OPTMATH_CUDA_ADA)
            endif()
            if("86" IN_LIST CMAKE_CUDA_ARCHITECTURES OR "80" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Ampere (SM 80/86) enabled - RTX 3090/3080/3070 support")
                add_compile_definitions(OPTMATH_CUDA_AMPERE)
            endif()
            if("75" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                message(STATUS "  Turing (SM 75) enabled - RTX 2080/2070/2060 support")
                add_compile_definitions(OPTMATH_CUDA_TURING)
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
