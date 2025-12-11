#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "optmath/core.hpp"

// Simple test runner
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cerr << "Assertion failed: " << #a << " != " << #b << " (" << (a) << " != " << (b) << ")\n"; \
        return 1; \
    }

#define ASSERT_NEAR(a, b, eps) \
    if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "Assertion failed: " << #a << " approx " << #b << "\n"; \
        return 1; \
    }

int main() {
    optmath::Core core;
    if (!core.verify_eigen()) return 1;

    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {4.0f, 5.0f, 6.0f};
    float a = 2.0f;

    // 2*[1,2,3] + [4,5,6] = [2,4,6] + [4,5,6] = [6,9,12]
    auto res = core.saxpy(a, x, y);

    ASSERT_EQ(res.size(), 3);
    ASSERT_NEAR(res[0], 6.0f, 1e-5f);
    ASSERT_NEAR(res[1], 9.0f, 1e-5f);
    ASSERT_NEAR(res[2], 12.0f, 1e-5f);

    std::cout << "Core tests passed.\n";
    return 0;
}
