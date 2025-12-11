#include "optmath/neon_kernels.hpp"

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

bool is_available() {
#ifdef OPTMATH_USE_NEON
    return true;
#else
    return false;
#endif
}

std::vector<float> add_vectors(const std::vector<float>& a, const std::vector<float>& b) {
#ifdef OPTMATH_USE_NEON
    if (a.size() != b.size()) return {};
    size_t n = a.size();
    std::vector<float> result(n);

    size_t i = 0;
    // Process 4 floats at a time (128-bit register)
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }

    // Handle cleanup for remaining elements
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
#else
    // Return empty to indicate implementation not active
    return {};
#endif
}

}
}
