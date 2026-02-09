#include "optmath/neon_kernels.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <utility>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

// =========================================================================
// Internal NEON Helpers (static, file-local)
// =========================================================================

// y[i] += alpha * x[i] for i in [0, n)
static void neon_axpy_f32(float* y, const float* x, float alpha, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t valpha = vdupq_n_f32(alpha);
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t vx = vld1q_f32(x + i);
        vy = vmlaq_f32(vy, vx, valpha);
        vst1q_f32(y + i, vy);
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
#endif
}

// x[i] *= alpha for i in [0, n)
static void neon_scale_f32(float* x, float alpha, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t valpha = vdupq_n_f32(alpha);
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vx = vmulq_f32(vx, valpha);
        vst1q_f32(x + i, vx);
    }
    for (; i < n; ++i) {
        x[i] *= alpha;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
#endif
}

// Return index of element with maximum absolute value in x[0..n-1]
static std::size_t neon_iamax_f32(const float* x, std::size_t n) {
    if (n == 0) return 0;
#ifdef OPTMATH_USE_NEON
    std::size_t best_idx = 0;
    float best_val = std::fabs(x[0]);
    // Simple scalar scan — pivot search is not the bottleneck
    for (std::size_t i = 1; i < n; ++i) {
        float v = std::fabs(x[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return best_idx;
#else
    std::size_t best_idx = 0;
    float best_val = std::fabs(x[0]);
    for (std::size_t i = 1; i < n; ++i) {
        float v = std::fabs(x[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return best_idx;
#endif
}

// Swap rows r1 and r2 of an m-row, n-col column-major matrix with leading dim lda
static void neon_row_swap_f32(float* A, std::size_t lda, std::size_t n_cols,
                               std::size_t r1, std::size_t r2) {
    if (r1 == r2) return;
#ifdef OPTMATH_USE_NEON
    for (std::size_t j = 0; j < n_cols; ++j) {
        float tmp = A[r1 + j * lda];
        A[r1 + j * lda] = A[r2 + j * lda];
        A[r2 + j * lda] = tmp;
    }
#else
    for (std::size_t j = 0; j < n_cols; ++j) {
        float tmp = A[r1 + j * lda];
        A[r1 + j * lda] = A[r2 + j * lda];
        A[r2 + j * lda] = tmp;
    }
#endif
}

// =========================================================================
// Triangular Solve — TRSV (single RHS, column-major)
// =========================================================================

// Forward substitution: solve L*x = b (L lower triangular, non-unit diagonal)
// b is overwritten with x
void neon_trsv_lower_f32(float* b, const float* L, std::size_t n, std::size_t ldl) {
    for (std::size_t j = 0; j < n; ++j) {
        if (L[j + j * ldl] == 0.0f) return; // singular
        b[j] /= L[j + j * ldl];
        float scale = -b[j];
        std::size_t rem = n - j - 1;
        if (rem > 0) {
            neon_axpy_f32(b + j + 1, L + j + 1 + j * ldl, scale, rem);
        }
    }
}

// Backward substitution: solve U*x = b (U upper triangular)
// b is overwritten with x
void neon_trsv_upper_f32(float* b, const float* U, std::size_t n, std::size_t ldu) {
    for (std::size_t jj = n; jj > 0; --jj) {
        std::size_t j = jj - 1;
        if (U[j + j * ldu] == 0.0f) return; // singular
        b[j] /= U[j + j * ldu];
        float scale = -b[j];
        if (j > 0) {
            // Update b[0..j-1] -= b[j] * U[0..j-1, j]
            neon_axpy_f32(b, U + j * ldu, scale, j);
        }
    }
}

// Forward substitution with unit diagonal: solve L*x = b where diag(L) = 1
void neon_trsv_lower_unit_f32(float* b, const float* L, std::size_t n, std::size_t ldl) {
    for (std::size_t j = 0; j < n; ++j) {
        // diagonal is 1, no division needed
        float scale = -b[j];
        std::size_t rem = n - j - 1;
        if (rem > 0) {
            neon_axpy_f32(b + j + 1, L + j + 1 + j * ldl, scale, rem);
        }
    }
}

// Solve L^T * x = b using lower triangular L (backward sub on transpose)
void neon_trsv_lower_trans_f32(float* b, const float* L, std::size_t n, std::size_t ldl) {
    for (std::size_t jj = n; jj > 0; --jj) {
        std::size_t j = jj - 1;
        // b[j] -= dot(L[j+1:n, j], b[j+1:n])  -- but that's a column of L
        // In L^T, row j is column j of L, so:
        // b[j] -= sum_{i=j+1}^{n-1} L[i, j] * b[i]
        float sum = 0.0f;
        std::size_t rem = n - j - 1;
        if (rem > 0) {
            sum = neon_dot_f32(L + j + 1 + j * ldl, b + j + 1, rem);
        }
        b[j] = (b[j] - sum) / L[j + j * ldl];
    }
}

// =========================================================================
// Triangular Solve — TRSM (multiple RHS, column-major)
// =========================================================================

// Solve L * X = B where L is lower triangular, B has nrhs columns
// B is overwritten with X
void neon_trsm_lower_f32(float* B, const float* L, std::size_t n, std::size_t nrhs,
                          std::size_t ldl, std::size_t ldb) {
    for (std::size_t k = 0; k < nrhs; ++k) {
        neon_trsv_lower_f32(B + k * ldb, L, n, ldl);
    }
}

// Solve U * X = B where U is upper triangular, B has nrhs columns
void neon_trsm_upper_f32(float* B, const float* U, std::size_t n, std::size_t nrhs,
                          std::size_t ldu, std::size_t ldb) {
    for (std::size_t k = 0; k < nrhs; ++k) {
        neon_trsv_upper_f32(B + k * ldb, U, n, ldu);
    }
}

// =========================================================================
// Cholesky Decomposition (A = L * L^T, in-place, lower triangle)
// =========================================================================

int neon_cholesky_f32(float* A, std::size_t n, std::size_t lda) {
    for (std::size_t j = 0; j < n; ++j) {
        // Compute diagonal: A[j,j] -= dot(A[j, 0:j-1], A[j, 0:j-1])
        // In column-major lower triangle, row j of L is at A[j + k*lda] for k < j
        // But those are strided — gather the row into a temp buffer for dot
        float diag = A[j + j * lda];
        for (std::size_t k = 0; k < j; ++k) {
            diag -= A[j + k * lda] * A[j + k * lda];
        }
        if (diag <= 0.0f) {
            return static_cast<int>(j + 1); // not SPD, return 1-based index
        }
        A[j + j * lda] = std::sqrt(diag);
        float inv_diag = 1.0f / A[j + j * lda];

        // Update column j below diagonal
        for (std::size_t i = j + 1; i < n; ++i) {
            float sum = A[i + j * lda];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= A[i + k * lda] * A[j + k * lda];
            }
            A[i + j * lda] = sum * inv_diag;
        }
    }
    // Zero the strict upper triangle for clean output
    for (std::size_t j = 1; j < n; ++j) {
        for (std::size_t i = 0; i < j; ++i) {
            A[i + j * lda] = 0.0f;
        }
    }
    return 0;
}

// =========================================================================
// LU Decomposition with Partial Pivoting (in-place)
// =========================================================================

int neon_lu_f32(float* A, int* piv, std::size_t m, std::size_t n, std::size_t lda) {
    std::size_t mn = std::min(m, n);

    // Initialize pivot vector to identity permutation
    for (std::size_t i = 0; i < m; ++i) {
        piv[i] = static_cast<int>(i);
    }

    for (std::size_t j = 0; j < mn; ++j) {
        // Find pivot in column j, rows j..m-1
        std::size_t pivot_rel = neon_iamax_f32(A + j + j * lda, m - j);
        std::size_t pivot_row = j + pivot_rel;

        if (A[pivot_row + j * lda] == 0.0f) {
            return static_cast<int>(j + 1); // singular
        }

        // Swap rows
        if (pivot_row != j) {
            neon_row_swap_f32(A, lda, n, j, pivot_row);
            std::swap(piv[j], piv[pivot_row]);
        }

        // Scale column j below diagonal
        float inv_diag = 1.0f / A[j + j * lda];
        std::size_t rem = m - j - 1;
        if (rem > 0) {
            neon_scale_f32(A + j + 1 + j * lda, inv_diag, rem);
        }

        // Rank-1 update of trailing submatrix
        for (std::size_t k = j + 1; k < n; ++k) {
            if (rem > 0) {
                neon_axpy_f32(A + j + 1 + k * lda,
                              A + j + 1 + j * lda,
                              -A[j + k * lda],
                              rem);
            }
        }
    }
    return 0;
}

// =========================================================================
// QR Decomposition (Householder, in-place)
// =========================================================================

void neon_qr_f32(float* A, float* tau, std::size_t m, std::size_t n, std::size_t lda) {
    std::size_t mn = std::min(m, n);

    for (std::size_t j = 0; j < mn; ++j) {
        // Compute norm of A[j:m-1, j]
        std::size_t len = m - j;
        float* col = A + j + j * lda;

        float norm = neon_dot_f32(col, col, len);
        norm = std::sqrt(norm);

        if (norm == 0.0f) {
            tau[j] = 0.0f;
            continue;
        }

        // Choose sign to avoid cancellation
        float alpha = col[0];
        float sign = (alpha >= 0.0f) ? 1.0f : -1.0f;
        float beta = -sign * norm;

        // tau = (beta - alpha) / beta
        tau[j] = (beta - alpha) / beta;

        // v = col / (alpha - beta), with v[0] = 1
        float scale = 1.0f / (alpha - beta);
        for (std::size_t i = 1; i < len; ++i) {
            col[i] *= scale;
        }
        col[0] = beta; // Store R diagonal element

        // Apply reflector to trailing columns: A[j:m-1, j+1:n-1]
        // A := (I - tau * v * v^T) * A
        // For each column k: A[:,k] -= tau * (v^T * A[:,k]) * v
        for (std::size_t k = j + 1; k < n; ++k) {
            float* ak = A + j + k * lda;
            // dot = v^T * A[:,k], with v[0] = 1
            float dot = ak[0];
            for (std::size_t i = 1; i < len; ++i) {
                dot += col[i] * ak[i];
            }
            // ak -= tau * dot * v
            ak[0] -= tau[j] * dot;
            for (std::size_t i = 1; i < len; ++i) {
                ak[i] -= tau[j] * dot * col[i];
            }
        }
    }
}

// Extract explicit Q from stored Householder vectors
void neon_qr_extract_q_f32(float* Q, const float* A, const float* tau,
                             std::size_t m, std::size_t n, std::size_t lda, std::size_t ldq) {
    // Initialize Q to identity
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            Q[i + j * ldq] = (i == j) ? 1.0f : 0.0f;
        }
    }

    std::size_t mn = std::min(m, n);

    // Apply reflectors in reverse order
    for (std::size_t jj = mn; jj > 0; --jj) {
        std::size_t j = jj - 1;
        if (tau[j] == 0.0f) continue;

        std::size_t len = m - j;

        // Build v: v[0] = 1, v[1:] from A[j+1:m-1, j]
        // Apply to Q[j:m-1, j:m-1]: Q := Q * (I - tau * v * v^T)
        // For each column k of Q: Q[:,k] -= tau * (v^T * Q[:,k]) * v
        for (std::size_t k = j; k < m; ++k) {
            float* qk = Q + j + k * ldq;
            // dot = v^T * Q[:,k]
            float dot = qk[0]; // v[0] = 1
            for (std::size_t i = 1; i < len; ++i) {
                dot += A[j + i + j * lda] * qk[i];
            }
            // Q[:,k] -= tau * dot * v
            qk[0] -= tau[j] * dot;
            for (std::size_t i = 1; i < len; ++i) {
                qk[i] -= tau[j] * dot * A[j + i + j * lda];
            }
        }
    }
}

// =========================================================================
// General Solve via LU (A*x = b, A is n x n, b is n x 1)
// =========================================================================

int neon_solve_f32(float* A, float* b, std::size_t n, std::size_t lda) {
    std::vector<int> piv(n);

    int info = neon_lu_f32(A, piv.data(), n, n, lda);
    if (info != 0) return info;

    // Apply row permutation to b
    std::vector<float> tmp(n);
    for (std::size_t i = 0; i < n; ++i) {
        tmp[i] = b[piv[i]];
    }
    std::memcpy(b, tmp.data(), n * sizeof(float));

    // Forward substitution with unit-diagonal L
    neon_trsv_lower_unit_f32(b, A, n, lda);

    // Backward substitution with U
    neon_trsv_upper_f32(b, A, n, lda);

    return 0;
}

// =========================================================================
// SPD Solve via Cholesky (A*x = b, A is SPD n x n)
// =========================================================================

int neon_solve_spd_f32(float* A, float* b, std::size_t n, std::size_t lda) {
    int info = neon_cholesky_f32(A, n, lda);
    if (info != 0) return info;

    // Solve L * y = b
    neon_trsv_lower_f32(b, A, n, lda);

    // Solve L^T * x = y
    neon_trsv_lower_trans_f32(b, A, n, lda);

    return 0;
}

// =========================================================================
// Matrix Inverse via LU (Ainv = A^{-1})
// =========================================================================

int neon_inverse_f32(float* Ainv, const float* A, std::size_t n,
                      std::size_t lda, std::size_t ldinv) {
    // Copy A into workspace
    std::vector<float> LU(n * n);
    for (std::size_t j = 0; j < n; ++j) {
        std::memcpy(LU.data() + j * n, A + j * lda, n * sizeof(float));
    }

    std::vector<int> piv(n);
    int info = neon_lu_f32(LU.data(), piv.data(), n, n, n);
    if (info != 0) return info;

    // Set Ainv = permuted identity, then solve each column
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            Ainv[i + j * ldinv] = 0.0f;
        }
        Ainv[j + j * ldinv] = 1.0f;
    }

    // Apply permutation and solve
    for (std::size_t j = 0; j < n; ++j) {
        // Apply permutation to column j of identity
        std::vector<float> col(n, 0.0f);
        col[j] = 1.0f;
        std::vector<float> pcol(n);
        for (std::size_t i = 0; i < n; ++i) {
            pcol[i] = col[piv[i]];
        }
        std::memcpy(Ainv + j * ldinv, pcol.data(), n * sizeof(float));

        // Solve L * U * x = pcol
        neon_trsv_lower_unit_f32(Ainv + j * ldinv, LU.data(), n, n);
        neon_trsv_upper_f32(Ainv + j * ldinv, LU.data(), n, n);
    }

    return 0;
}

// =========================================================================
// Eigen Wrappers
// =========================================================================

Eigen::MatrixXf neon_cholesky(const Eigen::MatrixXf& A) {
    std::size_t n = static_cast<std::size_t>(A.rows());
    Eigen::MatrixXf L = A;
    // Eigen stores column-major by default
    int info = neon_cholesky_f32(L.data(), n, static_cast<std::size_t>(L.outerStride()));
    if (info != 0) {
        // Return empty matrix on failure
        return Eigen::MatrixXf();
    }
    return L;
}

std::pair<Eigen::MatrixXf, Eigen::VectorXi> neon_lu(const Eigen::MatrixXf& A) {
    std::size_t m = static_cast<std::size_t>(A.rows());
    std::size_t n = static_cast<std::size_t>(A.cols());
    Eigen::MatrixXf LU = A;
    Eigen::VectorXi piv(m);

    neon_lu_f32(LU.data(), piv.data(), m, n, static_cast<std::size_t>(LU.outerStride()));
    return {LU, piv};
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> neon_qr(const Eigen::MatrixXf& A) {
    std::size_t m = static_cast<std::size_t>(A.rows());
    std::size_t n = static_cast<std::size_t>(A.cols());
    std::size_t mn = std::min(m, n);

    Eigen::MatrixXf QR = A;
    std::vector<float> tau(mn);

    neon_qr_f32(QR.data(), tau.data(), m, n, static_cast<std::size_t>(QR.outerStride()));

    // Extract R (upper triangular part of QR)
    Eigen::MatrixXf R = Eigen::MatrixXf::Zero(m, n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i <= std::min(j, m - 1); ++i) {
            R(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                QR(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    // Extract Q
    Eigen::MatrixXf Q(m, m);
    neon_qr_extract_q_f32(Q.data(), QR.data(), tau.data(), m, n,
                            static_cast<std::size_t>(QR.outerStride()),
                            static_cast<std::size_t>(Q.outerStride()));

    return {Q, R};
}

Eigen::VectorXf neon_trsv_lower(const Eigen::MatrixXf& L, const Eigen::VectorXf& b) {
    Eigen::VectorXf x = b;
    neon_trsv_lower_f32(x.data(), L.data(),
                         static_cast<std::size_t>(L.rows()),
                         static_cast<std::size_t>(L.outerStride()));
    return x;
}

Eigen::VectorXf neon_trsv_upper(const Eigen::MatrixXf& U, const Eigen::VectorXf& b) {
    Eigen::VectorXf x = b;
    neon_trsv_upper_f32(x.data(), U.data(),
                         static_cast<std::size_t>(U.rows()),
                         static_cast<std::size_t>(U.outerStride()));
    return x;
}

Eigen::VectorXf neon_solve(const Eigen::MatrixXf& A, const Eigen::VectorXf& b) {
    std::size_t n = static_cast<std::size_t>(A.rows());
    Eigen::MatrixXf Acopy = A;
    Eigen::VectorXf x = b;
    neon_solve_f32(Acopy.data(), x.data(), n, static_cast<std::size_t>(Acopy.outerStride()));
    return x;
}

Eigen::VectorXf neon_solve_spd(const Eigen::MatrixXf& A, const Eigen::VectorXf& b) {
    std::size_t n = static_cast<std::size_t>(A.rows());
    Eigen::MatrixXf Acopy = A;
    Eigen::VectorXf x = b;
    neon_solve_spd_f32(Acopy.data(), x.data(), n, static_cast<std::size_t>(Acopy.outerStride()));
    return x;
}

Eigen::MatrixXf neon_inverse(const Eigen::MatrixXf& A) {
    std::size_t n = static_cast<std::size_t>(A.rows());
    Eigen::MatrixXf Ainv(n, n);
    int info = neon_inverse_f32(Ainv.data(), A.data(), n,
                                 static_cast<std::size_t>(A.outerStride()),
                                 static_cast<std::size_t>(Ainv.outerStride()));
    if (info != 0) {
        return Eigen::MatrixXf();
    }
    return Ainv;
}

} // namespace neon
} // namespace optmath
