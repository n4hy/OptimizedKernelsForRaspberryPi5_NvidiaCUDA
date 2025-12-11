#pragma once
#include <vector>

namespace optmath {
namespace neon {

    /**
     * @brief Checks if NEON acceleration was compiled in.
     */
    bool is_available();

    /**
     * @brief Performs vector addition using NEON intrinsics if available.
     * @return Result vector, or empty vector if NEON is disabled.
     */
    std::vector<float> add_vectors(const std::vector<float>& a, const std::vector<float>& b);

}
}
