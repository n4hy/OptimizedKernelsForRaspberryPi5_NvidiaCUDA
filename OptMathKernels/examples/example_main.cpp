#include <iostream>
#include <vector>
#include "optmath/core.hpp"
#include "optmath/neon_kernels.hpp"
#include "optmath/vulkan_backend.hpp"

void print_vector(const std::string& name, const std::vector<float>& v) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < std::min(v.size(), size_t(5)); ++i) {
        std::cout << v[i] << " ";
    }
    if (v.size() > 5) std::cout << "... ";
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "OptMathKernels Example\n";
    std::cout << "======================\n";

    // 1. Core (Eigen)
    optmath::Core core;
    if (core.verify_eigen()) {
        std::cout << "[Core] Eigen is working.\n";
    } else {
        std::cerr << "[Core] Eigen verification failed!\n";
    }

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f, 40.0f};
    float a = 2.0f;

    try {
        auto result = core.saxpy(a, x, y);
        print_vector("Core SAXPY (2*x + y)", result);
    } catch (const std::exception& e) {
        std::cerr << "[Core] Error: " << e.what() << std::endl;
    }

    // 2. NEON
    if (optmath::neon::is_available()) {
        std::cout << "\n[NEON] Available.\n";
        auto neon_res = optmath::neon::add_vectors(x, y);
        print_vector("NEON Add", neon_res);
    } else {
        std::cout << "\n[NEON] Not compiled in or unavailable.\n";
    }

    // 3. Vulkan
    if (optmath::vulkan::is_available()) {
        std::cout << "\n[Vulkan] Available.\n";
        if (optmath::vulkan::init()) {
            auto vk_res = optmath::vulkan::compute_add_scalar(x, 5.0f);
            print_vector("Vulkan Add Scalar (+5)", vk_res);
        } else {
            std::cerr << "[Vulkan] Initialization failed.\n";
        }
    } else {
        std::cout << "\n[Vulkan] Not compiled in or unavailable.\n";
    }

    return 0;
}
