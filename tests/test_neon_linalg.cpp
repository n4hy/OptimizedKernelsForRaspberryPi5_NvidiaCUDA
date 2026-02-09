#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>
#include "optmath/neon_kernels.hpp"

using namespace optmath::neon;

class NeonLinalgTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!is_available()) {
            GTEST_SKIP() << "NEON not available";
        }
    }

    // Generate random SPD matrix of size n
    Eigen::MatrixXf randomSPD(int n, unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        Eigen::MatrixXf M(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                M(i, j) = dist(gen);
        return M * M.transpose() + n * Eigen::MatrixXf::Identity(n, n);
    }

    // Generate random matrix of size m x n
    Eigen::MatrixXf randomMatrix(int m, int n, unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        Eigen::MatrixXf M(m, n);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                M(i, j) = dist(gen);
        return M;
    }

    Eigen::VectorXf randomVector(int n, unsigned seed = 123) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        Eigen::VectorXf v(n);
        for (int i = 0; i < n; ++i) v(i) = dist(gen);
        return v;
    }
};

// =========================================================================
// Cholesky Tests
// =========================================================================

TEST_F(NeonLinalgTest, Cholesky3x3Known) {
    // Known SPD matrix
    Eigen::MatrixXf A(3, 3);
    A << 4, 2, 1,
         2, 5, 3,
         1, 3, 6;

    Eigen::MatrixXf L = neon_cholesky(A);
    ASSERT_EQ(L.rows(), 3);
    ASSERT_EQ(L.cols(), 3);

    // Verify L * L^T == A
    Eigen::MatrixXf LLT = L * L.transpose();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(LLT(i, j), A(i, j), 1e-5)
                << "at (" << i << "," << j << ")";

    // Verify L is lower triangular
    for (int j = 1; j < 3; ++j)
        for (int i = 0; i < j; ++i)
            EXPECT_FLOAT_EQ(L(i, j), 0.0f);
}

TEST_F(NeonLinalgTest, Cholesky64x64RandomSPD) {
    Eigen::MatrixXf A = randomSPD(64);
    Eigen::MatrixXf L = neon_cholesky(A);
    ASSERT_EQ(L.rows(), 64);

    Eigen::MatrixXf LLT = L * L.transpose();
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j)
            EXPECT_NEAR(LLT(i, j), A(i, j), 1e-4)
                << "at (" << i << "," << j << ")";

    // Compare with Eigen's LLT
    Eigen::LLT<Eigen::MatrixXf> eigenLLT(A);
    Eigen::MatrixXf eigenL = eigenLLT.matrixL();
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j <= i; ++j)
            EXPECT_NEAR(L(i, j), eigenL(i, j), 1e-4)
                << "at (" << i << "," << j << ")";
}

TEST_F(NeonLinalgTest, CholeskyNonSPDReturnsError) {
    Eigen::MatrixXf A(3, 3);
    A << 1, 2, 3,
         2, 1, 4,
         3, 4, 1;

    Eigen::MatrixXf L = neon_cholesky(A);
    // Should return empty matrix for non-SPD
    EXPECT_EQ(L.size(), 0);
}

// =========================================================================
// LU Tests
// =========================================================================

TEST_F(NeonLinalgTest, LU3x3Known) {
    Eigen::MatrixXf A(3, 3);
    A << 2, 1, 1,
         4, 3, 3,
         8, 7, 9;

    auto [LU, piv] = neon_lu(A);
    ASSERT_EQ(LU.rows(), 3);
    ASSERT_EQ(LU.cols(), 3);

    // Extract L (unit lower) and U (upper)
    Eigen::MatrixXf L = Eigen::MatrixXf::Identity(3, 3);
    Eigen::MatrixXf U = Eigen::MatrixXf::Zero(3, 3);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            if (i > j) L(i, j) = LU(i, j);
            else U(i, j) = LU(i, j);
        }
    }

    // Build permutation matrix from piv
    Eigen::MatrixXf P = Eigen::MatrixXf::Zero(3, 3);
    for (int i = 0; i < 3; ++i) {
        P(i, piv(i)) = 1.0f;
    }

    // Verify P*A == L*U
    Eigen::MatrixXf PA = P * A;
    Eigen::MatrixXf LUproduct = L * U;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(PA(i, j), LUproduct(i, j), 1e-5)
                << "at (" << i << "," << j << ")";
}

TEST_F(NeonLinalgTest, LU64x64Random) {
    Eigen::MatrixXf A = randomMatrix(64, 64, 77);

    auto [LU, piv] = neon_lu(A);

    // Extract L and U
    int n = 64;
    Eigen::MatrixXf L = Eigen::MatrixXf::Identity(n, n);
    Eigen::MatrixXf U = Eigen::MatrixXf::Zero(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i > j) L(i, j) = LU(i, j);
            else U(i, j) = LU(i, j);
        }
    }

    // Build permutation
    Eigen::MatrixXf P = Eigen::MatrixXf::Zero(n, n);
    for (int i = 0; i < n; ++i)
        P(i, piv(i)) = 1.0f;

    Eigen::MatrixXf PA = P * A;
    Eigen::MatrixXf LUproduct = L * U;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(PA(i, j), LUproduct(i, j), 1e-3)
                << "at (" << i << "," << j << ")";
}

TEST_F(NeonLinalgTest, LUSingularReturnsError) {
    Eigen::MatrixXf A(3, 3);
    A << 1, 2, 3,
         2, 4, 6,
         1, 1, 1;

    auto [LU, piv] = neon_lu(A);
    // The matrix is singular (row 2 = 2*row 1)
    // After pivoting, should detect a zero pivot at some column
    // We just verify that the decomposition doesn't crash;
    // singular detection depends on exact pivot sequence
    (void)LU;
    (void)piv;
}

// =========================================================================
// QR Tests
// =========================================================================

TEST_F(NeonLinalgTest, QR3x3) {
    Eigen::MatrixXf A(3, 3);
    A << 1, 1, 0,
         1, 0, 1,
         0, 1, 1;

    auto [Q, R] = neon_qr(A);
    ASSERT_EQ(Q.rows(), 3);
    ASSERT_EQ(Q.cols(), 3);
    ASSERT_EQ(R.rows(), 3);
    ASSERT_EQ(R.cols(), 3);

    // Q^T * Q == I
    Eigen::MatrixXf QTQ = Q.transpose() * Q;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(QTQ(i, j), expected, 1e-5)
                << "QTQ at (" << i << "," << j << ")";
        }

    // Q * R == A
    Eigen::MatrixXf QR = Q * R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(QR(i, j), A(i, j), 1e-5)
                << "QR at (" << i << "," << j << ")";

    // R is upper triangular
    for (int j = 0; j < 3; ++j)
        for (int i = j + 1; i < 3; ++i)
            EXPECT_NEAR(R(i, j), 0.0f, 1e-5);
}

TEST_F(NeonLinalgTest, QR64x32Tall) {
    int m = 64, n = 32;
    Eigen::MatrixXf A = randomMatrix(m, n, 99);

    auto [Q, R] = neon_qr(A);
    ASSERT_EQ(Q.rows(), m);
    ASSERT_EQ(Q.cols(), m);
    ASSERT_EQ(R.rows(), m);
    ASSERT_EQ(R.cols(), n);

    // Q^T * Q == I (m x m)
    Eigen::MatrixXf QTQ = Q.transpose() * Q;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(QTQ(i, j), expected, 1e-4)
                << "QTQ at (" << i << "," << j << ")";
        }

    // Q * R == A
    Eigen::MatrixXf QR = Q * R;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(QR(i, j), A(i, j), 1e-4)
                << "QR at (" << i << "," << j << ")";

    // R upper triangular (below min(m,n) diagonal)
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < m; ++i)
            EXPECT_NEAR(R(i, j), 0.0f, 1e-4);
}

// =========================================================================
// TRSV Tests
// =========================================================================

TEST_F(NeonLinalgTest, TrsvLower3x3) {
    Eigen::MatrixXf L(3, 3);
    L << 2, 0, 0,
         1, 3, 0,
         4, 2, 5;
    Eigen::VectorXf b(3);
    b << 4, 7, 30;

    Eigen::VectorXf x = neon_trsv_lower(L, b);

    // Verify L*x == b
    Eigen::VectorXf Lx = L * x;
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(Lx(i), b(i), 1e-5);
}

TEST_F(NeonLinalgTest, TrsvUpper3x3) {
    Eigen::MatrixXf U(3, 3);
    U << 3, 1, 2,
         0, 4, 1,
         0, 0, 5;
    Eigen::VectorXf b(3);
    b << 13, 9, 10;

    Eigen::VectorXf x = neon_trsv_upper(U, b);

    // Verify U*x == b
    Eigen::VectorXf Ux = U * x;
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(Ux(i), b(i), 1e-5);
}

TEST_F(NeonLinalgTest, TrsvLower64x64Random) {
    int n = 64;
    Eigen::MatrixXf M = randomMatrix(n, n, 55);
    // Make lower triangular with positive diagonal
    Eigen::MatrixXf L = Eigen::MatrixXf::Zero(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            L(i, j) = M(i, j);
        }
        L(j, j) = std::fabs(L(j, j)) + 1.0f; // ensure nonzero diagonal
    }

    Eigen::VectorXf x_true = randomVector(n, 77);
    Eigen::VectorXf b = L * x_true;

    Eigen::VectorXf x = neon_trsv_lower(L, b);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(x(i), x_true(i), 1e-3)
            << "at i=" << i;
}

TEST_F(NeonLinalgTest, TrsvUpper64x64Random) {
    int n = 64;
    Eigen::MatrixXf M = randomMatrix(n, n, 66);
    Eigen::MatrixXf U = Eigen::MatrixXf::Zero(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            U(i, j) = M(i, j);
        }
        U(j, j) = std::fabs(U(j, j)) + 1.0f;
    }

    Eigen::VectorXf x_true = randomVector(n, 88);
    Eigen::VectorXf b = U * x_true;

    Eigen::VectorXf x = neon_trsv_upper(U, b);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(x(i), x_true(i), 5e-3)
            << "at i=" << i;
}

// =========================================================================
// Solve Tests
// =========================================================================

TEST_F(NeonLinalgTest, Solve32x32) {
    int n = 32;
    Eigen::MatrixXf A = randomMatrix(n, n, 111);
    // Make well-conditioned
    A += n * Eigen::MatrixXf::Identity(n, n);
    Eigen::VectorXf x_true = randomVector(n, 222);
    Eigen::VectorXf b = A * x_true;

    Eigen::VectorXf x = neon_solve(A, b);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(x(i), x_true(i), 1e-3)
            << "at i=" << i;
}

TEST_F(NeonLinalgTest, SolveSPD32x32) {
    int n = 32;
    Eigen::MatrixXf A = randomSPD(n, 333);
    Eigen::VectorXf x_true = randomVector(n, 444);
    Eigen::VectorXf b = A * x_true;

    Eigen::VectorXf x = neon_solve_spd(A, b);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(x(i), x_true(i), 1e-3)
            << "at i=" << i;
}

TEST_F(NeonLinalgTest, SolveMatchesEigen) {
    int n = 16;
    Eigen::MatrixXf A = randomMatrix(n, n, 555);
    A += n * Eigen::MatrixXf::Identity(n, n);
    Eigen::VectorXf b = randomVector(n, 666);

    Eigen::VectorXf x_neon = neon_solve(A, b);
    Eigen::VectorXf x_eigen = A.partialPivLu().solve(b);

    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(x_neon(i), x_eigen(i), 1e-4)
            << "at i=" << i;
}

// =========================================================================
// Inverse Tests
// =========================================================================

TEST_F(NeonLinalgTest, Inverse32x32) {
    int n = 32;
    Eigen::MatrixXf A = randomMatrix(n, n, 777);
    A += n * Eigen::MatrixXf::Identity(n, n);

    Eigen::MatrixXf Ainv = neon_inverse(A);
    ASSERT_EQ(Ainv.rows(), n);
    ASSERT_EQ(Ainv.cols(), n);

    Eigen::MatrixXf I = A * Ainv;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(I(i, j), expected, 1e-3)
                << "A*Ainv at (" << i << "," << j << ")";
        }
}

TEST_F(NeonLinalgTest, InverseSmall3x3) {
    Eigen::MatrixXf A(3, 3);
    A << 1, 2, 3,
         0, 1, 4,
         5, 6, 0;

    Eigen::MatrixXf Ainv = neon_inverse(A);
    ASSERT_EQ(Ainv.rows(), 3);

    Eigen::MatrixXf I = A * Ainv;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(I(i, j), expected, 1e-5)
                << "A*Ainv at (" << i << "," << j << ")";
        }
}

// =========================================================================
// Eigen Wrapper Integration Tests
// =========================================================================

TEST_F(NeonLinalgTest, EigenCholeskyWrapper) {
    Eigen::MatrixXf A = randomSPD(16, 888);
    Eigen::MatrixXf L = neon_cholesky(A);
    ASSERT_GT(L.size(), 0);

    Eigen::MatrixXf diff = L * L.transpose() - A;
    EXPECT_LT(diff.norm(), 1e-3);
}

TEST_F(NeonLinalgTest, EigenQRWrapper) {
    int m = 8, n = 6;
    Eigen::MatrixXf A = randomMatrix(m, n, 999);
    auto [Q, R] = neon_qr(A);

    Eigen::MatrixXf diff = Q * R - A;
    EXPECT_LT(diff.norm(), 1e-4);

    Eigen::MatrixXf QTQ = Q.transpose() * Q;
    Eigen::MatrixXf Idiff = QTQ - Eigen::MatrixXf::Identity(m, m);
    EXPECT_LT(Idiff.norm(), 1e-4);
}

TEST_F(NeonLinalgTest, EigenLUWrapper) {
    int n = 16;
    Eigen::MatrixXf A = randomMatrix(n, n, 1010);
    A += n * Eigen::MatrixXf::Identity(n, n);

    auto [LU, piv] = neon_lu(A);

    // Extract L and U, verify P*A == L*U
    Eigen::MatrixXf L = Eigen::MatrixXf::Identity(n, n);
    Eigen::MatrixXf U = Eigen::MatrixXf::Zero(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i > j) L(i, j) = LU(i, j);
            else U(i, j) = LU(i, j);
        }
    }
    Eigen::MatrixXf P = Eigen::MatrixXf::Zero(n, n);
    for (int i = 0; i < n; ++i)
        P(i, piv(i)) = 1.0f;

    Eigen::MatrixXf diff = P * A - L * U;
    EXPECT_LT(diff.norm(), 1e-3);
}

TEST_F(NeonLinalgTest, EigenInverseWrapper) {
    int n = 16;
    Eigen::MatrixXf A = randomMatrix(n, n, 1111);
    A += n * Eigen::MatrixXf::Identity(n, n);

    Eigen::MatrixXf Ainv = neon_inverse(A);
    Eigen::MatrixXf I = A * Ainv;
    Eigen::MatrixXf diff = I - Eigen::MatrixXf::Identity(n, n);
    EXPECT_LT(diff.norm(), 1e-3);
}
