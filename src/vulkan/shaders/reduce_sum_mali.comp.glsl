#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Reduction sum optimized for Mali-G720 Immortalis
// Uses 1024 threads per workgroup (Mali-G720 max)
// Two-phase reduction: subgroup-level first, then shared memory

layout(local_size_x = 1024) in;

layout(std430, binding = 0) readonly buffer Input {
    float dataIn[];
};
layout(std430, binding = 1) writeonly buffer Output {
    float dataOut[]; // Partials
};

layout(push_constant) uniform PushConsts {
    uint count;
} pc;

// Shared memory for inter-subgroup reduction
// Mali-G720 subgroup size is 16, so 1024/16 = 64 subgroups max
shared float subgroupSums[64];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint idx = gl_GlobalInvocationID.x;

    // Load value (0 if out of bounds)
    float val = 0.0;
    if (idx < pc.count) {
        val = dataIn[idx];
    }

    // Phase 1: Subgroup-level reduction using hardware subgroup ops
    float subgroupSum = subgroupAdd(val);

    // First thread in each subgroup writes to shared memory
    uint subgroupIdx = gl_SubgroupID;
    if (gl_SubgroupInvocationID == 0) {
        subgroupSums[subgroupIdx] = subgroupSum;
    }

    barrier();

    // Phase 2: First subgroup reduces all subgroup sums
    uint numSubgroups = gl_NumSubgroups;
    if (subgroupIdx == 0) {
        float partialSum = 0.0;
        if (gl_SubgroupInvocationID < numSubgroups) {
            partialSum = subgroupSums[gl_SubgroupInvocationID];
        }
        float totalSum = subgroupAdd(partialSum);

        // Write final result for this workgroup
        if (tid == 0) {
            dataOut[gl_WorkGroupID.x] = totalSum;
        }
    }
}
