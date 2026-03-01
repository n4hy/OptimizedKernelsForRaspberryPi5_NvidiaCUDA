#include "optmath/platform.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <mutex>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__aarch64__) && defined(__linux__)
#include <sys/prctl.h>
#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif
#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffff
#endif
#endif

namespace optmath {
namespace platform {

// =========================================================================
// File reading helpers
// =========================================================================

static std::string read_sysfs_string(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::string val;
    std::getline(f, val);
    return val;
}

static uint32_t read_sysfs_uint(const std::string& path) {
    std::string s = read_sysfs_string(path);
    if (s.empty()) return 0;
    try {
        return static_cast<uint32_t>(std::stoul(s));
    } catch (...) {
        return 0;
    }
}

static std::size_t parse_cache_size(const std::string& s) {
    if (s.empty()) return 0;
    std::size_t val = 0;
    try {
        val = std::stoul(s);
    } catch (...) {
        return 0;
    }
    // Handle K/M suffixes
    if (s.back() == 'K' || s.back() == 'k') val *= 1024;
    else if (s.back() == 'M' || s.back() == 'm') val *= 1024 * 1024;
    return val;
}

// =========================================================================
// CPU Info Detection
// =========================================================================

static CpuInfo build_cpu_info() {
    CpuInfo info{};

#ifdef __linux__
    // Count online CPUs
    int n_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    info.total_cores = n_cpus;

    // Parse /proc/cpuinfo for part IDs and model name
    std::vector<uint32_t> part_ids(n_cpus, 0);
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        int current_cpu = -1;
        while (std::getline(f, line)) {
            if (line.find("processor") == 0) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    current_cpu = std::stoi(line.substr(pos + 1));
                }
            } else if (line.find("CPU part") == 0 && current_cpu >= 0 && current_cpu < n_cpus) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    std::string hex = line.substr(pos + 1);
                    // Trim whitespace
                    hex.erase(0, hex.find_first_not_of(" \t"));
                    part_ids[current_cpu] = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
                }
            } else if (line.find("model name") == 0 && info.model_name.empty()) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    info.model_name = line.substr(pos + 1);
                    // Trim leading whitespace
                    info.model_name.erase(0, info.model_name.find_first_not_of(" \t"));
                }
            } else if (line.find("Features") == 0 && !info.has_sve2) {
                info.has_sve2 = (line.find(" sve2 ") != std::string::npos ||
                                 line.find(" sve2\n") != std::string::npos ||
                                 line.find("\tsve2 ") != std::string::npos);
                info.has_fcma = (line.find(" fcma ") != std::string::npos ||
                                 line.find(" fcma\n") != std::string::npos ||
                                 line.find("\tfcma ") != std::string::npos);
                info.has_i8mm = (line.find(" i8mm ") != std::string::npos ||
                                 line.find(" i8mm\n") != std::string::npos ||
                                 line.find("\ti8mm ") != std::string::npos);
                info.has_bf16 = (line.find(" bf16 ") != std::string::npos ||
                                 line.find(" bf16\n") != std::string::npos ||
                                 line.find("\tbf16 ") != std::string::npos);
            }
        }
    }

    // Build per-core info from sysfs
    info.cores.resize(n_cpus);
    for (int i = 0; i < n_cpus; ++i) {
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        info.cores[i].cpu_id = i;
        info.cores[i].part_id = part_ids[i];
        info.cores[i].max_freq_khz = read_sysfs_uint(base + "/cpufreq/cpuinfo_max_freq");
        info.cores[i].capacity = read_sysfs_uint(base + "/cpu_capacity");
        info.cores[i].cluster_id = -1; // Will be assigned below
    }

    // Assign cluster IDs based on max_freq
    std::vector<uint32_t> unique_freqs;
    for (auto& c : info.cores) {
        if (std::find(unique_freqs.begin(), unique_freqs.end(), c.max_freq_khz) == unique_freqs.end()) {
            unique_freqs.push_back(c.max_freq_khz);
        }
    }
    std::sort(unique_freqs.begin(), unique_freqs.end());
    for (auto& c : info.cores) {
        for (int j = 0; j < static_cast<int>(unique_freqs.size()); ++j) {
            if (c.max_freq_khz == unique_freqs[j]) {
                c.cluster_id = j;
                break;
            }
        }
    }

    // Classify performance vs efficiency cores
    // A520 (0xd80) = efficiency, everything else = performance
    // Fallback: capacity < 512 = efficiency
    for (int i = 0; i < n_cpus; ++i) {
        bool is_efficiency = false;
        if (info.cores[i].part_id == 0xd80) { // A520
            is_efficiency = true;
        } else if (info.cores[i].part_id == 0 && info.cores[i].capacity > 0 && info.cores[i].capacity < 512) {
            is_efficiency = true;
        }

        if (is_efficiency) {
            info.efficiency_cores.push_back(i);
        } else {
            info.performance_cores.push_back(i);
        }
    }

    // Sort performance cores by capacity descending (fastest first)
    std::sort(info.performance_cores.begin(), info.performance_cores.end(),
              [&](int a, int b) {
                  return info.cores[a].capacity > info.cores[b].capacity;
              });

    // Detect L3 cache size
    info.l3_cache_bytes = 0;
    for (int idx = 0; idx < 10; ++idx) {
        std::string cache_base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx);
        std::string level_str = read_sysfs_string(cache_base + "/level");
        if (level_str.empty()) break;
        int level = 0;
        try { level = std::stoi(level_str); } catch (...) {}
        if (level == 3) {
            std::string size_str = read_sysfs_string(cache_base + "/size");
            info.l3_cache_bytes = parse_cache_size(size_str);
            break;
        }
    }

    // If sysfs doesn't report L3, try heuristic based on CPU part
    if (info.l3_cache_bytes == 0) {
        // Known L3 sizes by CPU part
        for (auto& c : info.cores) {
            if (c.part_id == 0xd81) { // A720 - CIX P1 has 12MB shared L3
                info.l3_cache_bytes = 12 * 1024 * 1024;
                break;
            } else if (c.part_id == 0xd0b) { // A76 - Pi5 has 2MB L3
                info.l3_cache_bytes = 2 * 1024 * 1024;
                break;
            }
        }
    }

    // Detect SVE vector length
    info.sve_vector_length_bytes = 0;
#if defined(__aarch64__)
    {
        int vl = prctl(PR_SVE_GET_VL, 0, 0, 0, 0);
        if (vl > 0) {
            info.sve_vector_length_bytes = vl & PR_SVE_VL_LEN_MASK;
        }
    }
#endif

#else // !__linux__
    info.total_cores = 1;
    info.l3_cache_bytes = 0;
    info.sve_vector_length_bytes = 0;
    info.has_sve2 = false;
    info.has_fcma = false;
    info.has_i8mm = false;
    info.has_bf16 = false;
#endif

    return info;
}

static std::once_flag g_cpu_info_once;
static CpuInfo g_cpu_info;

const CpuInfo& detect_cpu_info() {
    std::call_once(g_cpu_info_once, []() {
        g_cpu_info = build_cpu_info();
    });
    return g_cpu_info;
}

std::vector<int> get_performance_cores() {
    return detect_cpu_info().performance_cores;
}

std::vector<int> get_efficiency_cores() {
    return detect_cpu_info().efficiency_cores;
}

int pin_thread_to_performance_cores() {
#ifdef __linux__
    auto cores = get_performance_cores();
    if (cores.empty()) return -1;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cores) {
        CPU_SET(cpu, &cpuset);
    }
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#else
    return -1;
#endif
}

int pin_thread_to_core(int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#else
    (void)cpu_id;
    return -1;
#endif
}

int get_sve_vector_length() {
#if defined(__aarch64__) && defined(__linux__)
    int vl = prctl(PR_SVE_GET_VL, 0, 0, 0, 0);
    if (vl > 0) return vl & PR_SVE_VL_LEN_MASK;
#endif
    return 0;
}

std::size_t get_l3_cache_size() {
    return detect_cpu_info().l3_cache_bytes;
}

std::size_t get_gemm_mc() {
    std::size_t l3 = get_l3_cache_size();
    return (l3 >= 8 * 1024 * 1024) ? 256 : 128;
}

std::size_t get_gemm_kc() {
    std::size_t l3 = get_l3_cache_size();
    return (l3 >= 8 * 1024 * 1024) ? 512 : 256;
}

std::size_t get_gemm_nc() {
    std::size_t l3 = get_l3_cache_size();
    return (l3 >= 8 * 1024 * 1024) ? 1024 : 512;
}

} // namespace platform
} // namespace optmath
