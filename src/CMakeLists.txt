# Sources
set(SOURCES
    neon/neon_kernels.cpp
    neon/neon_complex.cpp
    neon/neon_gemm_optimized.cpp
    neon/neon_radar.cpp
    neon/neon_resample.cpp
    neon/neon_iir.cpp
    neon/neon_conv2d.cpp
    neon/neon_linalg.cpp
    vulkan/vulkan_backend.cpp
)

# CUDA sources (added conditionally)
if(ENABLE_CUDA)
    set(CUDA_SOURCES
        cuda/cuda_backend.cpp
        cuda/cuda_kernels.cu
        cuda/cuda_complex.cu
        cuda/cuda_radar.cu
    )
    list(APPEND SOURCES ${CUDA_SOURCES})

    # Set CUDA-specific properties
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

    # Suppress NVCC warning 20014 about Eigen host functions in host/device context
    # This is a known Eigen + CUDA template instantiation compatibility issue
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20014")
endif()

if(ENABLE_VULKAN)
    # Shader compilation logic
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(GLSLANG_VALIDATOR)
        # Define shaders to compile
        set(SHADERS
            # Vector operations
            vec_add.comp.glsl
            vec_sub.comp.glsl
            vec_mul.comp.glsl
            vec_div.comp.glsl
            vec_dot.comp.glsl
            vec_norm.comp.glsl
            # Matrix operations
            mat_add.comp.glsl
            mat_sub.comp.glsl
            mat_mul.comp.glsl
            mat_mul_tiled.comp.glsl
            mat_transpose.comp.glsl
            mat_scale.comp.glsl
            mat_vec_mul.comp.glsl
            mat_outer_product.comp.glsl
            mat_elementwise_mul.comp.glsl
            # Convolution and correlation
            convolution_1d.comp.glsl
            convolution_2d.comp.glsl
            convolution_1d_optimized.comp.glsl
            convolution_2d_optimized.comp.glsl
            correlation_1d.comp.glsl
            correlation_2d.comp.glsl
            # Reductions
            reduce_sum.comp.glsl
            reduce_max.comp.glsl
            reduce_min.comp.glsl
            reduce_complete.comp.glsl
            # Scan / Prefix sum
            scan_prefix_sum.comp.glsl
            scan_local.comp.glsl
            scan_block_sums.comp.glsl
            scan_add_offsets.comp.glsl
            # FFT
            fft_radix2.comp.glsl
            fft_radix2_optimized.comp.glsl
            ifft_radix2.comp.glsl
            fft_radix4.comp.glsl
            ifft_radix4.comp.glsl
            # Radar processing
            caf_doppler_shift.comp.glsl
            caf_xcorr.comp.glsl
            cfar_2d.comp.glsl
        )

        set(SPV_OUTPUTS "")

        # Create the directory for the output file
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        foreach(SHADER ${SHADERS})
             set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/${SHADER})
             set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER})
             string(REPLACE ".glsl" ".spv" OUTPUT_FILE ${OUTPUT_FILE}) # e.g. vec_add.comp.spv

             list(APPEND SPV_OUTPUTS ${OUTPUT_FILE})

             add_custom_command(
                OUTPUT ${OUTPUT_FILE}
                COMMAND ${GLSLANG_VALIDATOR} -V ${SOURCE_FILE} -o ${OUTPUT_FILE}
                DEPENDS ${SOURCE_FILE}
                COMMENT "Compiling ${SHADER} to SPIR-V"
            )
        endforeach()

        # We create a custom target to ensure shaders are built.
        add_custom_target(optmath_shaders ALL DEPENDS ${SPV_OUTPUTS})

        # Install shaders to a known location (datadir/optmathkernels/shaders)
        # And also copy them to the binary directory of the example app for easy finding
        install(FILES ${SPV_OUTPUTS} DESTINATION ${CMAKE_INSTALL_DATADIR}/optmathkernels/shaders)

    else()
        message(WARNING "glslangValidator not found. Shaders will not be compiled.")
    endif()
endif()

add_library(OptMathKernels ${SOURCES})
add_library(OptMathKernels::OptMathKernels ALIAS OptMathKernels)

# Include directories
target_include_directories(OptMathKernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dependencies
target_link_libraries(OptMathKernels PUBLIC Eigen3::Eigen)

if(ENABLE_VULKAN)
    target_link_libraries(OptMathKernels PRIVATE Vulkan::Vulkan)
    if(GLSLANG_VALIDATOR)
        add_dependencies(OptMathKernels optmath_shaders)
    endif()
endif()

# CUDA libraries
if(ENABLE_CUDA)
    target_link_libraries(OptMathKernels PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
        CUDA::cusolver
    )

    # Set CUDA properties on the library
    set_target_properties(OptMathKernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Enable CUDA fast math optimizations (optional, improves performance)
    target_compile_options(OptMathKernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    # Blackwell (RTX 5090) specific optimizations
    if("100" IN_LIST CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(OptMathKernels PRIVATE
            # Maximum register usage for Blackwell's larger register file
            $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255>
            # Enable all PTX optimizations
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas --allow-expensive-optimizations=true>
        )
    endif()
endif()

# Installation
install(TARGETS OptMathKernels
    EXPORT  OptMathKernelsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ../include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export configuration
install(EXPORT OptMathKernelsTargets
    FILE OptMathKernelsTargets.cmake
    NAMESPACE OptMathKernels::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptMathKernels
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OptMathKernelsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OptMathKernelsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptMathKernels
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OptMathKernelsConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptMathKernels
)
