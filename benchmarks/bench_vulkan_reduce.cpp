// A/B benchmark: barrier-tree vs subgroup-shuffle sum reduction on the GPU.
// On the Pi 5 (Broadcom V3D) subgroup ARITHMETIC ops are unavailable, so the
// subgroup path is built from SHUFFLE. This measures whether that is actually
// faster than the shared-memory barrier tree, and validates correctness first.
#include "bench_common.hpp"
#include <optmath/vulkan_backend.hpp>

#include <cmath>
#include <cstdio>

using namespace optmath::vulkan;

namespace {
// Verify the chosen backend matches a CPU reference before timing, so a broken
// or unsupported subgroup path fails loudly instead of benchmarking garbage.
bool validate(ReduceBackend backend, size_t n, std::string& err) {
    Eigen::VectorXf a = bench::random_vector_f32(n);
    double ref = 0.0;
    for (int i = 0; i < a.size(); ++i) ref += a[i];

    set_reduce_backend(backend);
    float got = vulkan_reduce_sum(a);
    set_reduce_backend(ReduceBackend::Auto);

    // GPU reduction order differs from the naive CPU sum, so allow a relative
    // tolerance that grows slowly with n.
    double tol = std::abs(ref) * 1e-4 + 1e-2 * std::sqrt((double)n);
    if (std::abs((double)got - ref) > tol) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "n=%zu got=%.4f ref=%.4f (tol=%.4f)",
                      n, (double)got, ref, tol);
        err = buf;
        return false;
    }
    return true;
}

void run(benchmark::State& state, ReduceBackend backend) {
    if (!is_available()) { state.SkipWithError("Vulkan not available"); return; }
    if (backend == ReduceBackend::Subgroup && !subgroup_reduce_available()) {
        state.SkipWithError("subgroup reduction not supported on this device");
        return;
    }
    size_t n = state.range(0);
    std::string err;
    if (!validate(backend, n, err)) {
        state.SkipWithError(("incorrect result: " + err).c_str());
        return;
    }

    Eigen::VectorXf a = bench::random_vector_f32(n);
    set_reduce_backend(backend);
    volatile float sink = 0.0f;
    for (auto _ : state) {
        sink = vulkan_reduce_sum(a);
        benchmark::DoNotOptimize(sink);
    }
    set_reduce_backend(ReduceBackend::Auto);
    bench::set_flops(state, static_cast<double>(n));
}
}  // namespace

static void BM_ReduceSum_BarrierTree(benchmark::State& s) { run(s, ReduceBackend::BarrierTree); }
static void BM_ReduceSum_Subgroup(benchmark::State& s)    { run(s, ReduceBackend::Subgroup); }

BENCHMARK(BM_ReduceSum_BarrierTree)->RangeMultiplier(8)->Range(1024, 4194304)->UseRealTime();
BENCHMARK(BM_ReduceSum_Subgroup)->RangeMultiplier(8)->Range(1024, 4194304)->UseRealTime();

BENCHMARK_MAIN();
