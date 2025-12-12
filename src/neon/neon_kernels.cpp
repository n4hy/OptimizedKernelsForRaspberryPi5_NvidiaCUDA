#include "optmath/neon_kernels.hpp"
#include <cmath>
#include <algorithm>

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

// =========================================================================
// Core Intrinsics Implementations
// =========================================================================

float neon_dot_f32(const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Unrolled loop (4x4 = 16 elements per iter could be better, but we stick to 4 per iter for simplicity)
    // Actually, let's do 4x unroll (16 floats)
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, a0, b0);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        vsum = vmlaq_f32(vsum, a1, b1);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        vsum = vmlaq_f32(vsum, a2, b2);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        vsum = vmlaq_f32(vsum, a3, b3);
    }

    // Residual blocks of 4
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }

    float sum = vaddvq_f32(vsum);

    // Scalar tail
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

double neon_dot_f64(const double* a, const double* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float64x2_t vsum = vdupq_n_f64(0.0);
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t b0 = vld1q_f64(b + i);
        vsum = vmlaq_f64(vsum, a0, b0);

        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        vsum = vmlaq_f64(vsum, a1, b1);
    }

    // Residual block of 2
    if (i + 1 < n) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vsum = vmlaq_f64(vsum, va, vb);
        i += 2;
    }

    double sum = vaddvq_f64(vsum);

    // Scalar tail
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

void neon_add_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
#endif
}

void neon_sub_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
#endif
}

void neon_mul_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

void neon_div_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vdivq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
#endif
}

float neon_norm_f32(const float* a, std::size_t n) {
    // Norm = sqrt(dot(a, a))
    float dot = neon_dot_f32(a, a, n);
    return std::sqrt(dot);
}

float neon_reduce_sum_f32(const float* a, std::size_t n) {
    // Sum is dot with 1.0, but faster to just accumulate
#ifdef OPTMATH_USE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        vsum = vaddq_f32(vsum, vld1q_f32(a + i));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 4));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 8));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 12));
    }
    for (; i + 3 < n; i += 4) {
        vsum = vaddq_f32(vsum, vld1q_f32(a + i));
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; ++i) sum += a[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i=0; i<n; ++i) sum += a[i];
    return sum;
#endif
}

float neon_reduce_max_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_NEON
    float32x4_t vmax = vdupq_n_f32(-3.402823466e+38f); // Init with small num
    size_t i = 0;
    // Load first element to avoid dummy small num if preferred, but vdup is easier
    // Just handling remaining scalars carefully.

    for (; i + 3 < n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(a + i));
    }
    float max_val = vmaxvq_f32(vmax);
    for (; i < n; ++i) if(a[i] > max_val) max_val = a[i];
    return max_val;
#else
    float m = a[0];
    for(size_t i=1; i<n; ++i) if(a[i] > m) m = a[i];
    return m;
#endif
}

float neon_reduce_min_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_NEON
    float32x4_t vmin = vdupq_n_f32(3.402823466e+38f);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vmin = vminq_f32(vmin, vld1q_f32(a + i));
    }
    float min_val = vminvq_f32(vmin);
    for (; i < n; ++i) if(a[i] < min_val) min_val = a[i];
    return min_val;
#else
    float m = a[0];
    for(size_t i=1; i<n; ++i) if(a[i] < m) m = a[i];
    return m;
#endif
}

void neon_gemm_4x4_f32(float* C, const float* A, const float* B, std::size_t ldc) {
#ifdef OPTMATH_USE_NEON
    // A assumed column-major: A columns are contiguous
    // B assumed column-major: B columns are contiguous
    // C assumed column-major

    // Load C columns
    float32x4_t c0 = vld1q_f32(C);
    float32x4_t c1 = vld1q_f32(C + ldc);
    float32x4_t c2 = vld1q_f32(C + 2*ldc);
    float32x4_t c3 = vld1q_f32(C + 3*ldc);

    // Load A columns
    float32x4_t a0 = vld1q_f32(A);
    float32x4_t a1 = vld1q_f32(A + 4);
    float32x4_t a2 = vld1q_f32(A + 8);
    float32x4_t a3 = vld1q_f32(A + 12);

    // Load B elements. Since B is col-major, B[0], B[1], B[2], B[3] is the first column.
    // For C = A * B, the first column of C (c0) is A * col0_B.
    // c0 += a0*B[0,0] + a1*B[1,0] + a2*B[2,0] + a3*B[3,0]

    auto accumulate_col = [&](float32x4_t& c_col, const float* b_col_ptr) {
        c_col = vmlaq_n_f32(c_col, a0, b_col_ptr[0]);
        c_col = vmlaq_n_f32(c_col, a1, b_col_ptr[1]);
        c_col = vmlaq_n_f32(c_col, a2, b_col_ptr[2]);
        c_col = vmlaq_n_f32(c_col, a3, b_col_ptr[3]);
    };

    accumulate_col(c0, B);
    accumulate_col(c1, B + 4);
    accumulate_col(c2, B + 8);
    accumulate_col(c3, B + 12);

    vst1q_f32(C, c0);
    vst1q_f32(C + ldc, c1);
    vst1q_f32(C + 2*ldc, c2);
    vst1q_f32(C + 3*ldc, c3);
#else
    // Fallback scalar
    for(int j=0; j<4; ++j) {
        for(int i=0; i<4; ++i) {
            float sum = 0.0f;
            for(int k=0; k<4; ++k) {
                // Col major logic: A[i, k] -> A[i + k*4]
                sum += A[i + k*4] * B[k + j*4];
            }
            C[i + j*ldc] += sum;
        }
    }
#endif
}

void neon_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y) {
    // y[i] = sum(x[i+k] * h[k]) for k=0..n_h-1
    // We assume 'y' has size n_x - n_h + 1 (valid convolution) or similar.
    // The user must manage buffer sizes.
    // This simple kernel computes one output at a time, but vectorizes the dot product.

    // Output size
    size_t n_y = (n_x >= n_h) ? (n_x - n_h + 1) : 0;

    for (size_t i = 0; i < n_y; ++i) {
        y[i] = neon_dot_f32(x + i, h, n_h);
    }
}

void neon_relu_f32(float* data, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for(; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vmaxq_f32(v, vzero));
    }
    for(; i < n; ++i) {
        if(data[i] < 0.0f) data[i] = 0.0f;
    }
#else
    for(size_t i=0; i<n; ++i) if(data[i] < 0.0f) data[i] = 0.0f;
#endif
}

// Quick sigmoid approx: 1 / (1 + exp(-x))
// This is non-trivial in pure NEON without a math library or polynomial approx.
// We'll use std::exp for scalar tail and fallback, and maybe a polynomial for NEON if possible.
// For this task, we will just use scalar std::exp in loop to ensure correctness,
// unless we implement a vectorized exp.
// Given constraints, we will stick to a clean scalar loop implementation for complex math
// or use a very simple approximation if needed. The prompt asks for "approximations if necessary".
// Let's implement a scalar version for reliability, as writing a custom neon_exp is error-prone without testing.
// Wait, prompt says "use approximations if necessary but keep them numerically reasonable".
void neon_sigmoid_f32(float* data, std::size_t n) {
    for(size_t i=0; i<n; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

void neon_tanh_f32(float* data, std::size_t n) {
    for(size_t i=0; i<n; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

// =========================================================================
// Eigen Wrappers
// =========================================================================

float neon_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return 0.0f; // Minimal error handling
    return neon_dot_f32(a.data(), b.data(), a.size());
}

double neon_dot(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    if(a.size() != b.size()) return 0.0;
    return neon_dot_f64(a.data(), b.data(), a.size());
}

Eigen::VectorXf neon_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_add_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf neon_sub(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_sub_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf neon_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_mul_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf neon_div(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_div_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

float neon_norm(const Eigen::VectorXf& a) {
    return neon_norm_f32(a.data(), a.size());
}

float neon_reduce_sum(const Eigen::VectorXf& a) {
    return neon_reduce_sum_f32(a.data(), a.size());
}

float neon_reduce_max(const Eigen::VectorXf& a) {
    return neon_reduce_max_f32(a.data(), a.size());
}

float neon_reduce_min(const Eigen::VectorXf& a) {
    return neon_reduce_min_f32(a.data(), a.size());
}

Eigen::VectorXf neon_fir(const Eigen::VectorXf& x, const Eigen::VectorXf& h) {
    if (x.size() < h.size()) return Eigen::VectorXf();
    long out_size = x.size() - h.size() + 1;
    Eigen::VectorXf y(out_size);
    neon_fir_f32(x.data(), x.size(), h.data(), h.size(), y.data());
    return y;
}

void neon_relu(Eigen::VectorXf& x) {
    neon_relu_f32(x.data(), x.size());
}

void neon_sigmoid(Eigen::VectorXf& x) {
    neon_sigmoid_f32(x.data(), x.size());
}

void neon_tanh(Eigen::VectorXf& x) {
    neon_tanh_f32(x.data(), x.size());
}

Eigen::MatrixXf neon_gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    if (A.cols() != B.rows()) return Eigen::MatrixXf();

    // Result C
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(A.rows(), B.cols());

    // Simple tiled implementation calling 4x4 microkernel
    // We iterate over 4x4 blocks of C
    for (long j = 0; j < C.cols(); j += 4) {
        for (long i = 0; i < C.rows(); i += 4) {
            // For each block C[i:i+4, j:j+4]
            // Accumulate A[i:i+4, k:k+4] * B[k:k+4, j:j+4]
            for (long k = 0; k < A.cols(); k += 4) {
                // Check bounds
                if (i + 4 <= C.rows() && j + 4 <= C.cols() && k + 4 <= A.cols()) {
                    // Fast path: 4x4 aligned block
                     neon_gemm_4x4_f32(&C(i, j), &A(i, k), &B(k, j), C.outerStride());
                } else {
                    // Fallback for boundary blocks (naive multiply)
                    long i_lim = std::min(i + 4, (long)C.rows());
                    long j_lim = std::min(j + 4, (long)C.cols());
                    long k_lim = std::min(k + 4, (long)A.cols());

                    for (long jj = j; jj < j_lim; ++jj) {
                        for (long ii = i; ii < i_lim; ++ii) {
                            float sum = 0.0f;
                            for (long kk = k; kk < k_lim; ++kk) {
                                sum += A(ii, kk) * B(kk, jj);
                            }
                            C(ii, jj) += sum;
                        }
                    }
                }
            }
        }
    }
    return C;
}

Eigen::MatrixXf neon_mat_scale(const Eigen::MatrixXf& A, float s) {
    Eigen::MatrixXf C(A.rows(), A.cols());
#ifdef OPTMATH_USE_NEON
    float32x4_t vs = vdupq_n_f32(s);
    size_t n = A.size();
    const float* in = A.data();
    float* out = C.data();
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(in + i), vs));
    }
    for (; i < n; ++i) {
        out[i] = in[i] * s;
    }
#else
    C = A * s;
#endif
    return C;
}

Eigen::MatrixXf neon_mat_transpose(const Eigen::MatrixXf& A) {
    Eigen::MatrixXf C(A.cols(), A.rows());
#ifdef OPTMATH_USE_NEON
    // Blocked transpose 4x4 using NEON trn/zip/uzp intrinsics
    // vtrn1q_f32, vtrn2q_f32, etc.
    // For simplicity, we just use Eigen's implementation or a naive block transpose?
    // User requested "best build ... add every neon kernel you can find".
    // A specific 4x4 transpose kernel is good.

    // We assume C is col-major. C(i, j) = A(j, i).
    // If we iterate 4x4 blocks.

    for (int j = 0; j < A.cols(); j += 4) {
        for (int i = 0; i < A.rows(); i += 4) {
            if (i + 4 <= A.rows() && j + 4 <= A.cols()) {
                // Transpose 4x4 block from A(i,j) to C(j,i)
                // Load 4 columns of A
                float32x4_t c0 = vld1q_f32(&A(i, j));
                float32x4_t c1 = vld1q_f32(&A(i, j+1));
                float32x4_t c2 = vld1q_f32(&A(i, j+2));
                float32x4_t c3 = vld1q_f32(&A(i, j+3));

                // Transpose 4x4 in registers
                // 0: 00 10 20 30
                // 1: 01 11 21 31
                // ...

                // trn1/2 swaps elements
                float32x4_t t0 = vtrn1q_f32(c0, c1); // 00 01 20 21
                float32x4_t t1 = vtrn2q_f32(c0, c1); // 10 11 30 31
                float32x4_t t2 = vtrn1q_f32(c2, c3); // 02 03 22 23
                float32x4_t t3 = vtrn2q_f32(c2, c3); // 12 13 32 33

                // Swap 64-bit halves
                // C0' = 00 01 02 03 = Low(t0), Low(t2)
                // C1' = 10 11 12 13 = Low(t1), Low(t3)
                // C2' = 20 21 22 23 = High(t0), High(t2)
                // C3' = 30 31 32 33 = High(t1), High(t3)

                // vcombine? No, need vzip or similar.
                // Actually reinterpret as 64x2 and trn?
                // Let's use vcombine mechanism manually or just st4?
                // Actually, vtrn is efficient enough.
                // Construct from halves:
                // vextq?

                // Simpler: Use vst4q_f32 which interleaves!
                // Wait, vst4q takes {v0, v1, v2, v3} and stores them interleaved.
                // If we load 4 cols:
                // Col0: A00 A10 A20 A30
                // Col1: A01 A11 A21 A31
                // ...
                // If we interpret these as 4 vectors and do vst4, we get:
                // A00 A01 A02 A03, A10 A11 A12 A13...
                // Which is exactly the row-major order of the block,
                // i.e., the columns of the transposed block (since C is col-major).
                // So: Load 4 vectors (cols of A), Store using vst4 (interleave) into C (cols of C).

                // But vst4q stores linearly.
                // C is col-major. C's memory for the block is contiguous? No.
                // C(j, i) -> C's col j, row i.
                // The block in C starts at C(j, i).
                // Its columns are C(j, i)...C(j+3, i). These are stride separated by LDC.
                // So vst4 is not directly usable unless C is transposed in memory?
                // Wait, Transpose of A -> C.
                // C[j, i] = A[i, j].
                // We want to write into C.
                // C stores columns continuously.
                // Col j of C is C(0, j)...C(M, j).
                // We want to write A's row i into C's col i.
                // A's row i is strided.

                // So we have 4 vectors (Cols of A).
                // We transpose them in registers to get Rows of A.
                // Then store each Row of A as a Col of C.
                // Since C is col-major, a Col of C is contiguous.
                // So we just need to transpose in registers and store each vector to &C(j, i), &C(j, i+1)...

                // Re-doing transpose logic:
                // t0 = 00 01 20 21
                // t2 = 02 03 22 23
                // out0 = 00 01 02 03 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2))

                float32x4_t out0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
                float32x4_t out1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
                float32x4_t out2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
                float32x4_t out3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));

                vst1q_f32(&C(j, i), out0);
                vst1q_f32(&C(j, i+1), out1);
                vst1q_f32(&C(j, i+2), out2);
                vst1q_f32(&C(j, i+3), out3);

            } else {
                // Fallback
                for (int ii = i; ii < std::min(i+4, (int)A.rows()); ++ii) {
                    for (int jj = j; jj < std::min(j+4, (int)A.cols()); ++jj) {
                        C(jj, ii) = A(ii, jj);
                    }
                }
            }
        }
    }
#else
    C = A.transpose();
#endif
    return C;
}

Eigen::VectorXf neon_mat_vec_mul(const Eigen::MatrixXf& A, const Eigen::VectorXf& v) {
    if (A.cols() != v.size()) return Eigen::VectorXf();
    Eigen::VectorXf res = Eigen::VectorXf::Zero(A.rows());

#ifdef OPTMATH_USE_NEON
    // res = A * v
    // A is col-major. A = [col0 col1 ...]
    // res = sum(col_i * v_i)

    // We iterate over columns of A (and elements of v)
    // and accumulate into res.

    for (int j = 0; j < A.cols(); ++j) {
        float val = v[j];
        float32x4_t vval = vdupq_n_f32(val);

        int i = 0;
        float* r_ptr = res.data();
        const float* a_col = &A(0, j);

        for (; i + 3 < A.rows(); i += 4) {
            float32x4_t acc = vld1q_f32(r_ptr + i);
            float32x4_t col = vld1q_f32(a_col + i);
            acc = vmlaq_f32(acc, col, vval);
            vst1q_f32(r_ptr + i, acc);
        }
        for (; i < A.rows(); ++i) {
            res[i] += A(i, j) * val;
        }
    }
#else
    res = A * v;
#endif
    return res;
}

}
}
