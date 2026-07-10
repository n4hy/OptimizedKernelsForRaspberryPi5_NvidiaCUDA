#version 450
// Subgroup-shuffle reduction. The Broadcom V3D (Pi 5) does NOT support subgroup
// ARITHMETIC ops (subgroupAdd), but it does support SHUFFLE, so we build the
// intra-subgroup sum from a shuffle-xor butterfly instead of a shared-memory
// barrier tree. Each subgroup reduces its lanes with no barriers; the per-
// subgroup partials are then reduced by the first subgroup.
//
// Requires: subgroup BASIC + SHUFFLE, subgroupSize a power of two, and
// numSubgroups <= subgroupSize (true on V3D: 256/16 = 16 subgroups of 16). The
// host selects this shader only when those hold.
#extension GL_KHR_shader_subgroup_basic   : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input {
    float dataIn[];
};
layout(std430, binding = 1) writeonly buffer Output {
    float dataOut[]; // Partials, one per workgroup
};

layout(push_constant) uniform PushConsts {
    uint count;
} pc;

// One slot per subgroup; 256 is a safe upper bound (minSubgroupSize >= 1).
shared float partials[256];

float subgroupReduceAdd(float v) {
    for (uint offset = gl_SubgroupSize / 2u; offset > 0u; offset >>= 1u) {
        v += subgroupShuffleXor(v, offset);
    }
    return v; // every lane holds the subgroup sum
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float val = (idx < pc.count) ? dataIn[idx] : 0.0;

    val = subgroupReduceAdd(val);
    if (subgroupElect()) {
        partials[gl_SubgroupID] = val;
    }
    barrier();

    // First subgroup reduces the per-subgroup partials.
    if (gl_SubgroupID == 0u) {
        float p = (gl_SubgroupInvocationID < gl_NumSubgroups)
                  ? partials[gl_SubgroupInvocationID] : 0.0;
        p = subgroupReduceAdd(p);
        if (subgroupElect()) {
            dataOut[gl_WorkGroupID.x] = p;
        }
    }
}
