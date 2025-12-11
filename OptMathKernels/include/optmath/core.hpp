#pragma once

#include <vector>
#include <Eigen/Dense>

namespace optmath {

class Core {
public:
    Core();
    ~Core();

    /**
     * @brief Computes v = a * x + y (SAXPY operation)
     * @param a Scalar multiplier
     * @param x Input vector
     * @param y Input vector
     * @return Resulting vector
     */
    std::vector<float> saxpy(float a, const std::vector<float>& x, const std::vector<float>& y);

    /**
     * @brief Simple check to verify Eigen linking and math.
     */
    bool verify_eigen();
};

} // namespace optmath
