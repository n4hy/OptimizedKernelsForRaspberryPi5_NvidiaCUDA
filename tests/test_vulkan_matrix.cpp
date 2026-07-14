#include <gtest/gtest.h>
#include <optmath/vulkan_backend.hpp>
#include <Eigen/Dense>

// Helper to check approximate equality
static void expect_approx_equal(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, float tol = 1e-4) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a(i), b(i), tol) << "at index " << i;
    }
}

TEST(VulkanMatrixTest, MatrixOperations) {
    if (!optmath::vulkan::is_available()) {
        GTEST_SKIP() << "Vulkan not available, skipping test.";
    }

    int M = 64;
    int N = 64;
    int K = 64;

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, N);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(M, N);

    // Add
    {
        Eigen::MatrixXf expected = A + B;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_add(A, B);
        expect_approx_equal(result, expected);
    }

    // Sub
    {
        Eigen::MatrixXf expected = A - B;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_sub(A, B);
        expect_approx_equal(result, expected);
    }

    // Scale
    {
        float scalar = 2.5f;
        Eigen::MatrixXf expected = A * scalar;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_scale(A, scalar);
        expect_approx_equal(result, expected);
    }

    // Transpose
    {
        Eigen::MatrixXf expected = A.transpose();
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_transpose(A);
        expect_approx_equal(result, expected);
    }

    // Mul
    {
        Eigen::MatrixXf MatA = Eigen::MatrixXf::Random(M, K);
        Eigen::MatrixXf MatB = Eigen::MatrixXf::Random(K, N);

        Eigen::MatrixXf expected = MatA * MatB;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_mul(MatA, MatB);
        // Matrix mul accumulates more error
        expect_approx_equal(result, expected, 1e-2);
    }
}

// Both GEMM backends must agree. Auto keeps GEMM on the CPU on V3D (the shader
// measures 24-51x slower there); Gpu forces the shader. Same math either way, so
// this also guards the forced-GPU path from bit-rotting once Auto stops using it.
TEST(VulkanMatrix, MatMulBackendsAgree) {
    using namespace optmath::vulkan;
    if (!is_available()) GTEST_SKIP() << "Vulkan not available";

    const MatMulBackend saved = get_matmul_backend();
    // 256^3 is at OPTMATH_VK_MATMUL_MIN, i.e. where the old code first offloaded.
    for (int N : {64, 256}) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);

        set_matmul_backend(MatMulBackend::Cpu);
        Eigen::MatrixXf on_cpu = vulkan_mat_mul(A, B);
        set_matmul_backend(MatMulBackend::Gpu);
        Eigen::MatrixXf on_gpu = vulkan_mat_mul(A, B);
        set_matmul_backend(saved);

        expect_approx_equal(on_cpu, A * B, 1e-2);
        expect_approx_equal(on_gpu, A * B, 1e-2);
    }
    EXPECT_EQ(get_matmul_backend(), saved);
}

// The Pi 5's V3D must not be chosen for GEMM by default: that was a 24-51x
// regression for every caller of vulkan_mat_mul at N>=256.
TEST(VulkanMatrix, AutoDoesNotOffloadGemmOnV3D) {
    using namespace optmath::vulkan;
    if (!is_available()) GTEST_SKIP() << "Vulkan not available";
    ASSERT_EQ(get_matmul_backend(), MatMulBackend::Auto) << "default must be Auto";
#ifdef OPTMATH_USE_VULKAN
    if (VulkanContext::get().isBroadcomGpu)
        EXPECT_FALSE(matmul_gpu_preferred()) << "V3D must stay on the CPU for GEMM";
#endif
    set_matmul_backend(MatMulBackend::Cpu);
    EXPECT_FALSE(matmul_gpu_preferred());
    set_matmul_backend(MatMulBackend::Auto);
}
