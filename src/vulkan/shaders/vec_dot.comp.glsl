#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer BufA { float a[]; };
layout(std430, binding = 1) buffer BufB { float b[]; };
layout(std430, binding = 2) buffer BufOut { float out_partials[]; };

layout(push_constant) uniform PushConsts {
    uint count;
} p;

shared float shared_sum[256];

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint lID = gl_LocalInvocationID.x;

    float val = 0.0;
    if (gID < p.count) {
        val = a[gID] * b[gID];
    }
    shared_sum[lID] = val;
    barrier();

    // Reduction in shared memory
    for (uint s = 128; s > 0; s >>= 1) {
        if (lID < s) {
            shared_sum[lID] += shared_sum[lID + s];
        }
        barrier();
    }

    // Write group result to output buffer
    if (lID == 0) {
        out_partials[gl_WorkGroupID.x] = shared_sum[0];
    }
}
