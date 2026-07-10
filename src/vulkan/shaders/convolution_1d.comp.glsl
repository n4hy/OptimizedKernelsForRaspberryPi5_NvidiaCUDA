#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer BufX { float x[]; };
layout(std430, binding = 1) buffer BufH { float h[]; };
layout(std430, binding = 2) buffer BufY { float y[]; };

layout(push_constant) uniform PushConsts {
    uint n_x;
    uint n_h;
} p;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    // Output size assumed to be n_x - n_h + 1
    if (idx >= (p.n_x - p.n_h + 1)) return;

    // True valid-mode convolution: the kernel is flipped (h[n_h-1-k]), matching
    // the convention of convolution_2d. Use correlation_1d for the un-flipped
    // sliding dot product.
    float sum = 0.0;
    for (uint k = 0; k < p.n_h; ++k) {
        sum += x[idx + k] * h[p.n_h - 1u - k];
    }
    y[idx] = sum;
}