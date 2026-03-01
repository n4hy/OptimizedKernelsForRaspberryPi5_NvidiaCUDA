#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>

namespace optmath {
namespace platform {

// =========================================================================
// CPU Information
// =========================================================================

struct CoreInfo {
    int cpu_id;
    uint32_t part_id;       // ARM CPU part (0xd81=A720, 0xd80=A520, 0xd0b=A76)
    uint32_t max_freq_khz;
    uint32_t capacity;      // Linux scheduler capacity (0-1024)
    int cluster_id;         // Frequency cluster
};

struct CpuInfo {
    int total_cores;
    std::vector<CoreInfo> cores;
    std::vector<int> performance_cores;  // CPU IDs sorted by capacity descending
    std::vector<int> efficiency_cores;   // CPU IDs for low-capacity cores
    std::size_t l3_cache_bytes;
    int sve_vector_length_bytes;         // SVE VL in bytes (0 if no SVE)
    bool has_sve2;
    bool has_fcma;
    bool has_i8mm;
    bool has_bf16;
    std::string model_name;
};

/**
 * @brief Detect CPU information from sysfs and /proc/cpuinfo.
 * Cached after first call.
 */
const CpuInfo& detect_cpu_info();

/**
 * @brief Get CPU IDs of performance (big) cores.
 * On CIX P1: A720 cores (0xd81). On Pi 5: all Cortex-A76 cores.
 */
std::vector<int> get_performance_cores();

/**
 * @brief Get CPU IDs of efficiency (LITTLE) cores.
 * On CIX P1: A520 cores (0xd80). On Pi 5: empty (no LITTLE cores).
 */
std::vector<int> get_efficiency_cores();

/**
 * @brief Pin calling thread to performance cores.
 * @return 0 on success, -1 on failure.
 */
int pin_thread_to_performance_cores();

/**
 * @brief Pin calling thread to a specific CPU core.
 * @return 0 on success, -1 on failure.
 */
int pin_thread_to_core(int cpu_id);

/**
 * @brief Get SVE vector length in bytes via prctl(PR_SVE_GET_VL).
 * @return VL in bytes, or 0 if SVE is not available.
 */
int get_sve_vector_length();

/**
 * @brief Get L3 cache size in bytes from sysfs.
 * @return L3 size in bytes, or 0 if not detectable.
 */
std::size_t get_l3_cache_size();

/**
 * @brief Get optimal GEMM cache blocking MC parameter for detected hardware.
 * Returns 256 for L3 >= 8MB (A720), 128 for smaller (A76).
 */
std::size_t get_gemm_mc();

/**
 * @brief Get optimal GEMM cache blocking KC parameter.
 * Returns 512 for L3 >= 8MB, 256 otherwise.
 */
std::size_t get_gemm_kc();

/**
 * @brief Get optimal GEMM cache blocking NC parameter.
 * Returns 1024 for L3 >= 8MB, 512 otherwise.
 */
std::size_t get_gemm_nc();

} // namespace platform
} // namespace optmath
