#pragma once
#include <vector>
#include <string>

namespace optmath {
namespace vulkan {

    /**
     * @brief Checks if Vulkan backend is available.
     */
    bool is_available();

    /**
     * @brief Initialize Vulkan context (stub/example).
     */
    bool init();

    /**
     * @brief Run a simple compute shader that adds a scalar to a vector.
     */
    std::vector<float> compute_add_scalar(const std::vector<float>& input, float scalar);

}
}
