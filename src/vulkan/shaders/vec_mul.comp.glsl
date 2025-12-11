#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer BufA { float a[]; };
layout(std430, binding = 1) buffer BufB { float b[]; };
layout(std430, binding = 2) buffer BufOut { float out_data[]; };

layout(push_constant) uniform PushConsts {
    uint count;
} p;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= p.count) return;
    out_data[idx] = a[idx] * b[idx];
}
