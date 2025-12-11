# Sources
set(SOURCES
    core/core.cpp
    neon/neon_kernels.cpp
    vulkan/vulkan_backend.cpp
)

if(ENABLE_VULKAN)
    # Shader compilation logic
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(GLSLANG_VALIDATOR)
        set(SHADER_SOURCE vulkan/shaders/simple.comp.glsl)
        set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/simple.comp.spv)

        # Create the directory for the output file
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE} -o ${SHADER_SPV}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
            COMMENT "Compiling ${SHADER_SOURCE} to SPIR-V"
        )

        # We create a custom target to ensure shaders are built.
        add_custom_target(optmath_shaders ALL DEPENDS ${SHADER_SPV})

        # In a real app, you might want to embed this SPV into the binary or install it.
        # Here we will install it to share/optmathkernels/shaders
        install(FILES ${SHADER_SPV} DESTINATION ${CMAKE_INSTALL_DATADIR}/optmathkernels/shaders)

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
