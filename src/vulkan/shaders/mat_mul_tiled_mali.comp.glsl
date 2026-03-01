#version 450
// Tiled Matrix Multiplication optimized for Mali-G720 Immortalis
// Uses 32x32 tiles (1024 threads per workgroup, Mali-G720 max)
// Shared memory: 2 * 32*32*4 = 8KB (fits in Mali-G720's 32KB shared mem)

layout(local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) readonly buffer InputA {
    float dataA[];
};
layout(std430, binding = 1) readonly buffer InputB {
    float dataB[];
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataOut[];
};

layout(push_constant) uniform PushConsts {
    uint M; // Rows of A / Rows of C
    uint K; // Cols of A = Rows of B
    uint N; // Cols of B / Cols of C
} pc;

#define TILE_SIZE 32

// Shared memory for tiles (8KB total)
shared float tileA[TILE_SIZE][TILE_SIZE];
shared float tileB[TILE_SIZE][TILE_SIZE];

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;

    uint localRow = gl_LocalInvocationID.x;
    uint localCol = gl_LocalInvocationID.y;

    float sum = 0.0;

    // Number of tiles along K dimension
    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory
        // A is M x K (col-major): A[row, k] -> index = k * M + row
        uint aCol = t * TILE_SIZE + localCol;
        if (row < pc.M && aCol < pc.K) {
            tileA[localRow][localCol] = dataA[aCol * pc.M + row];
        } else {
            tileA[localRow][localCol] = 0.0;
        }

        // Load tile of B into shared memory
        // B is K x N (col-major): B[k, col] -> index = col * K + k
        uint bRow = t * TILE_SIZE + localRow;
        if (bRow < pc.K && col < pc.N) {
            tileB[localRow][localCol] = dataB[col * pc.K + bRow];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        barrier();

        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier();
    }

    // Write result to global memory
    // C is M x N (col-major): C[row, col] -> index = col * M + row
    if (row < pc.M && col < pc.N) {
        dataOut[col * pc.M + row] = sum;
    }
}
