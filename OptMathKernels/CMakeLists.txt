cmake_minimum_required(VERSION 3.18)

project(OptMathKernels
    VERSION 0.1.0
    LANGUAGES CXX
)

# Enforce C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Installation directories
include(GNUInstallDirs)

# Options
option(ENABLE_NEON    "Enable NEON-optimized kernels" ON)
option(ENABLE_VULKAN  "Enable Vulkan-based kernels"   ON)
option(BUILD_EXAMPLES "Build examples"                ON)
option(BUILD_TESTS    "Build tests"                   ON)

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

if (ENABLE_VULKAN)
    add_compile_definitions(OPTMATH_USE_VULKAN)
endif()

# --- Subdirectories ---
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
