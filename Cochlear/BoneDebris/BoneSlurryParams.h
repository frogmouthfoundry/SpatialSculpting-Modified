/*
Abstract:
Shared C/Metal header for BoneSlurry metaball grid types.
*/
#pragma once

#include <simd/simd.h>

/// Per-particle data uploaded from CPU each frame.
struct BoneSlurryParticle {
    simd_float3 position;   // root-local space
    simd_float4 rotation;   // quaternion (x, y, z, w)
    simd_float3 scale;      // entity scale
};

/// Lightweight vertex for the bone slurry mesh (position + normal only).
struct BoneSlurryVertex {
    simd_float3 position;
    simd_float3 normal;
};

/// Grid parameters passed to all bone slurry compute kernels.
struct BoneSlurryGridParams {
    simd_float3 gridOrigin;      // world-space origin of the grid (min corner)
    simd_float3 voxelSize;       // per-axis voxel size (uniform cubic)
    simd_uint3  dimensions;      // grid dimensions (e.g. 64, 64, 64)
    unsigned int particleCount;  // number of active particles this frame
    unsigned int maxVertexCount; // vertex buffer capacity
    float isoValue;              // density threshold for surface extraction
};
