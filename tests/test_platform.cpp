#include <gtest/gtest.h>
#include <optmath/platform.hpp>
#include <algorithm>
#include <sched.h>

using namespace optmath::platform;

// ---------------------------------------------------------------------------
// Test 1: detect_cpu_info() returns correct basic info
// ---------------------------------------------------------------------------
TEST(PlatformTest, DetectCPUInfo) {
    const CpuInfo& info = detect_cpu_info();

    EXPECT_EQ(info.total_cores, 12);
    EXPECT_EQ(static_cast<int>(info.cores.size()), 12);
    EXPECT_FALSE(info.model_name.empty());
}

// ---------------------------------------------------------------------------
// Test 2: Performance core list matches expected A720 cores
// ---------------------------------------------------------------------------
TEST(PlatformTest, PerformanceCores) {
    std::vector<int> perf = get_performance_cores();

    EXPECT_EQ(static_cast<int>(perf.size()), 8);

    // Expected performance cores: 0, 1, 6, 7, 8, 9, 10, 11
    const int expected[] = {0, 1, 6, 7, 8, 9, 10, 11};
    for (int cpu : expected) {
        EXPECT_NE(std::find(perf.begin(), perf.end(), cpu), perf.end())
            << "Performance cores should contain CPU " << cpu;
    }
}

// ---------------------------------------------------------------------------
// Test 3: Efficiency core list matches expected A520 cores
// ---------------------------------------------------------------------------
TEST(PlatformTest, EfficiencyCores) {
    std::vector<int> eff = get_efficiency_cores();

    EXPECT_EQ(static_cast<int>(eff.size()), 4);

    // Expected efficiency cores: 2, 3, 4, 5
    const int expected[] = {2, 3, 4, 5};
    for (int cpu : expected) {
        EXPECT_NE(std::find(eff.begin(), eff.end(), cpu), eff.end())
            << "Efficiency cores should contain CPU " << cpu;
    }
}

// ---------------------------------------------------------------------------
// Test 4: Pinning to performance cores succeeds and is verifiable
// ---------------------------------------------------------------------------
TEST(PlatformTest, PinToPerformance) {
    int ret = pin_thread_to_performance_cores();
    EXPECT_EQ(ret, 0) << "pin_thread_to_performance_cores() should return 0 on success";

    // Verify the affinity mask only contains performance cores
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int rc = sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ASSERT_EQ(rc, 0) << "sched_getaffinity failed";

    std::vector<int> perf = get_performance_cores();
    bool pinned_to_perf = false;
    for (int cpu : perf) {
        if (CPU_ISSET(cpu, &cpuset)) {
            pinned_to_perf = true;
            break;
        }
    }
    EXPECT_TRUE(pinned_to_perf)
        << "Thread should be pinned to at least one performance core";
}

// ---------------------------------------------------------------------------
// Test 5: SVE vector length is 16 bytes (128-bit)
// ---------------------------------------------------------------------------
TEST(PlatformTest, SVEVectorLength) {
    int vl = get_sve_vector_length();
    EXPECT_EQ(vl, 16) << "SVE vector length should be 16 bytes on CIX P1";
}

// ---------------------------------------------------------------------------
// Test 6: L3 cache size is at least 8 MB
// ---------------------------------------------------------------------------
TEST(PlatformTest, L3CacheSize) {
    std::size_t l3 = get_l3_cache_size();
    EXPECT_GE(l3, static_cast<std::size_t>(8388608))
        << "L3 cache should be >= 8 MB; got " << l3 << " bytes";
}

// ---------------------------------------------------------------------------
// Test 7: GEMM blocking parameters for large-L3 hardware
// ---------------------------------------------------------------------------
TEST(PlatformTest, GEMMBlockingParams) {
    EXPECT_EQ(get_gemm_mc(), static_cast<std::size_t>(256));
    EXPECT_EQ(get_gemm_kc(), static_cast<std::size_t>(512));
    EXPECT_EQ(get_gemm_nc(), static_cast<std::size_t>(1024));
}

// ---------------------------------------------------------------------------
// Test 8: Core part IDs distinguish A720 from A520
// ---------------------------------------------------------------------------
TEST(PlatformTest, CorePartIDs) {
    const CpuInfo& info = detect_cpu_info();

    // CPU 0 should be A720 (part_id 0xd81)
    EXPECT_EQ(info.cores[0].part_id, static_cast<uint32_t>(0xd81))
        << "CPU 0 should be Cortex-A720 (0xd81)";

    // CPU 2 should be A520 (part_id 0xd80)
    EXPECT_EQ(info.cores[2].part_id, static_cast<uint32_t>(0xd80))
        << "CPU 2 should be Cortex-A520 (0xd80)";
}

// ---------------------------------------------------------------------------
// Test 9: Feature flags for SVE2, FCMA, I8MM, BF16
// ---------------------------------------------------------------------------
TEST(PlatformTest, FeatureFlags) {
    const CpuInfo& info = detect_cpu_info();

    EXPECT_TRUE(info.has_sve2) << "CIX P1 should support SVE2";
    EXPECT_TRUE(info.has_fcma) << "CIX P1 should support FCMA";
    EXPECT_TRUE(info.has_i8mm) << "CIX P1 should support I8MM";
    EXPECT_TRUE(info.has_bf16) << "CIX P1 should support BF16";
}
