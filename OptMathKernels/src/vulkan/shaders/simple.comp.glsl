#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputBuffer {
    float data[];
} inputBuffer;

layout(std430, binding = 1) buffer OutputBuffer {
    float data[];
} outputBuffer;

layout(push_constant) uniform PushConsts {
    float scalar;
    uint count;
} pushConsts;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pushConsts.count) return;

    outputBuffer.data[idx] = inputBuffer.data[idx] + pushConsts.scalar;
}
