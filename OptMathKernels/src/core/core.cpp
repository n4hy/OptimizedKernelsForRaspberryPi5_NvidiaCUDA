#include "optmath/core.hpp"
#include <iostream>
#include <stdexcept>

namespace optmath {

Core::Core() {}
Core::~Core() {}

std::vector<float> Core::saxpy(float a, const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Vector sizes do not match");
    }

    size_t n = x.size();

    // Map std::vector to Eigen::Map for zero-copy wrapper
    Eigen::Map<const Eigen::VectorXf> eigen_x(x.data(), n);
    Eigen::Map<const Eigen::VectorXf> eigen_y(y.data(), n);

    // Perform operation
    Eigen::VectorXf result = a * eigen_x + eigen_y;

    // Copy back to std::vector
    std::vector<float> out(result.data(), result.data() + result.size());
    return out;
}

bool Core::verify_eigen() {
    Eigen::Matrix2f m;
    m << 1, 2, 3, 4;
    return (m(0,0) == 1 && m(1,1) == 4);
}

} // namespace optmath
