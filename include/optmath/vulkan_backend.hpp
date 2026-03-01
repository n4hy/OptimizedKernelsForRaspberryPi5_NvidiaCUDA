#pragma once

#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>

#ifdef OPTMATH_USE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace optmath {
namespace vulkan {

    /**
     * @brief Checks if Vulkan support is compiled in and available at runtime.
     */
    bool is_available();

    /**
     * @brief Singleton or Context manager for Vulkan.
     */
    class VulkanContext {
    public:
        static VulkanContext& get();

        bool init();
        void cleanup();

        // Very basic accessors for the demo
        // In real code, these would be encapsulated
#ifdef OPTMATH_USE_VULKAN
        VkDevice device = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        uint32_t computeQueueFamilyIndex = 0;

        // Helper to find memory type
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
#endif
        // GPU detection flags
        bool isMaliGpu = false;
        bool isMaliG720 = false;

    private:
        VulkanContext() = default;
        ~VulkanContext() { cleanup(); }
        bool initialized = false;
#ifdef OPTMATH_USE_VULKAN
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
#endif
    };

    // --- Eigen Wrappers ---

    Eigen::VectorXf vulkan_vec_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf vulkan_vec_sub(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf vulkan_vec_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf vulkan_vec_div(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    float           vulkan_vec_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    float           vulkan_vec_norm(const Eigen::VectorXf& a);

    Eigen::MatrixXf vulkan_mat_add(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b);
    Eigen::MatrixXf vulkan_mat_sub(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b);
    Eigen::MatrixXf vulkan_mat_mul(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b);
    Eigen::MatrixXf vulkan_mat_transpose(const Eigen::MatrixXf& a);
    Eigen::MatrixXf vulkan_mat_scale(const Eigen::MatrixXf& a, float scalar);

    Eigen::VectorXf vulkan_mat_vec_mul(const Eigen::MatrixXf& a, const Eigen::VectorXf& v);
    Eigen::MatrixXf vulkan_mat_outer_product(const Eigen::VectorXf& u, const Eigen::VectorXf& v);
    Eigen::MatrixXf vulkan_mat_elementwise_mul(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b);

    // DSP
    Eigen::VectorXf vulkan_convolution_1d(const Eigen::VectorXf& x, const Eigen::VectorXf& k);
    Eigen::MatrixXf vulkan_convolution_2d(const Eigen::MatrixXf& x, const Eigen::MatrixXf& k);
    Eigen::VectorXf vulkan_correlation_1d(const Eigen::VectorXf& x, const Eigen::VectorXf& k);
    Eigen::MatrixXf vulkan_correlation_2d(const Eigen::MatrixXf& x, const Eigen::MatrixXf& k);

    // Reductions & Scan
    float vulkan_reduce_sum(const Eigen::VectorXf& a);
    float vulkan_reduce_max(const Eigen::VectorXf& a);
    float vulkan_reduce_min(const Eigen::VectorXf& a);
    Eigen::VectorXf vulkan_scan_prefix_sum(const Eigen::VectorXf& a);

    // FFT
    // Input/Output: Interleaved Complex (Real, Imag, Real, Imag...). Size must be 2*N.
    void vulkan_fft_radix2(Eigen::VectorXf& data, bool inverse);
    void vulkan_fft_radix4(Eigen::VectorXf& data, bool inverse);

    // Deprecated alias
    inline Eigen::VectorXf vulkan_conv1d(const Eigen::VectorXf& x, const Eigen::VectorXf& k) { return vulkan_convolution_1d(x, k); }

}
}
