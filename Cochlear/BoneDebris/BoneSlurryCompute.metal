/*
Abstract:
Metal compute kernels for the BoneSlurry metaball grid system.
Four kernels:
  1. boneSlurryClearMesh  — zero vertex/index buffers
  2. boneSlurryClear      — zero the density buffer
  3. boneSlurrySplat      — accumulate ellipsoidal density from debris particles
  4. boneSlurryMarch      — extract isosurface mesh via marching cubes on the 64^3 density grid
*/

#include <metal_stdlib>
using namespace metal;

#include "BoneSlurryParams.h"
#include "../Mesh/MarchingCubesParams.h"

// ─────────────────────────────────────────────────────────────────────
// MARK: - Helpers
// ─────────────────────────────────────────────────────────────────────

static uint boneSlurryEdgeIndex(uint2 data, uint index) {
    return 0xfu & (index < 8 ? data.x >> ((index + 0) * 4) :
                               data.y >> ((index - 8) * 4));
}

static uint2 boneSlurryEdgeVertexPair(uint index) {
    uint v1 = index & 7;
    uint v2 = index < 8 ? ((index + 1) & 3) | (index & 4) : v1 + 4;
    return uint2(v1, v2);
}

static uint3 boneSlurryCubeVertex(uint index) {
    bool x = index & 1;
    bool y = index & 2;
    bool z = index & 4;
    return uint3(x ^ y, y, z);
}

/// Build a 3×3 rotation matrix from a quaternion (x, y, z, w).
static float3x3 boneSlurryQuatToMatrix(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;
    return float3x3(
        float3(1.0 - (yy + zz), xy + wz, xz - wy),
        float3(xy - wz, 1.0 - (xx + zz), yz + wx),
        float3(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}

/// Flat 3D index: z * dimX * dimY + y * dimX + x
static uint boneSlurryFlatIndex(uint3 coord, uint3 dims) {
    return coord.z * dims.x * dims.y + coord.y * dims.x + coord.x;
}

// Fixed-point scale factor for atomic density accumulation.
constant uint BONE_SLURRY_FIXED_POINT_SCALE = 1024u;

// ─────────────────────────────────────────────────────────────────────
// MARK: - Clear Kernels
// ─────────────────────────────────────────────────────────────────────

/// Zero the density buffer before each frame's splat pass.
[[kernel]]
void boneSlurryClear(device uint *density [[buffer(0)]],
                     constant uint &totalCount [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= totalCount) return;
    density[tid] = 0;
}

/// Zero the vertex and index buffers so unused entries produce degenerate triangles.
[[kernel]]
void boneSlurryClearMesh(device BoneSlurryVertex *vertices [[buffer(0)]],
                         device uint *indices [[buffer(1)]],
                         constant uint &maxVertexCount [[buffer(2)]],
                         uint vertexIndex [[thread_position_in_grid]]) {
    if (vertexIndex >= maxVertexCount) return;
    vertices[vertexIndex].position = float3(0);
    vertices[vertexIndex].normal = float3(0);
    indices[vertexIndex] = 0;
}

// ─────────────────────────────────────────────────────────────────────
// MARK: - Splat Kernel (Anisotropic Ellipsoidal)
// ─────────────────────────────────────────────────────────────────────

// Minimum influence radius in voxels — floor for spawn-size particles.
// 2.5 voxels with isoValue 0.75 gives ~0.9 voxel visual radius (small
// and round). Neighbouring particles overlap to merge smoothly.
constant float MIN_SPLAT_RADIUS_VOXELS = 2.5;

// Visual scale multiplier. At max growth (scale 3×), particles cover
// ~13.5 voxels → visual radius ~5 voxels. Growth from 0.9 → 5 = 5.5×.
constant float VISUAL_SCALE_MULTIPLIER = 30.0;

[[kernel]]
void boneSlurrySplat(device atomic_uint *density [[buffer(0)]],
                     constant BoneSlurryParticle *particles [[buffer(1)]],
                     constant BoneSlurryGridParams &params [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= params.particleCount) return;

    BoneSlurryParticle p = particles[tid];

    // Convert particle position to grid coordinates (floating point).
    float3 localPos = (p.position - params.gridOrigin) / params.voxelSize;

    // Skip particles far outside the grid.
    if (any(localPos < float3(-5.0)) || any(localPos > float3(params.dimensions) + 5.0)) return;

    // Compute half-extents in voxel space.
    // Base debris radius is 0.00015m (0.15mm) — matches BoneDebrisManager.debrisRadius.
    // Scale includes stretch + growth. VISUAL_SCALE_MULTIPLIER inflates the
    // metaball well beyond the physics body.
    float baseRadius = 0.00015;  // matches BoneDebrisManager.debrisRadius
    float3 worldHalfExtent = p.scale * baseRadius * VISUAL_SCALE_MULTIPLIER;
    float3 halfExtent = worldHalfExtent / params.voxelSize;

    // Enforce minimum influence radius so particles always register on the grid.
    halfExtent = max(halfExtent, float3(MIN_SPLAT_RADIUS_VOXELS));

    // Build rotation matrix from quaternion for anisotropic splat.
    float3x3 rotMat = boneSlurryQuatToMatrix(p.rotation);
    float3x3 invRot = transpose(rotMat);

    // AABB of influence in grid space (use per-axis half extents for tighter bound).
    // For rotated ellipsoid, use the max extent for the AABB.
    float maxHalfExtent = max(halfExtent.x, max(halfExtent.y, halfExtent.z));
    int3 minVoxel = int3(floor(localPos - maxHalfExtent));
    int3 maxVoxel = int3(ceil(localPos + maxHalfExtent));
    int3 dims = int3(params.dimensions);
    minVoxel = clamp(minVoxel, int3(0), dims - 1);
    maxVoxel = clamp(maxVoxel, int3(0), dims - 1);

    for (int z = minVoxel.z; z <= maxVoxel.z; z++) {
        for (int y = minVoxel.y; y <= maxVoxel.y; y++) {
            for (int x = minVoxel.x; x <= maxVoxel.x; x++) {
                float3 voxelCenter = float3(x, y, z) + 0.5;
                float3 delta = voxelCenter - localPos;

                // Transform delta into particle-local space for ellipsoidal test.
                float3 localDelta = invRot * delta;

                // Normalize by half-extents.
                float3 normalized = localDelta / halfExtent;
                float distSq = dot(normalized, normalized);

                if (distSq < 1.0) {
                    // Smooth cubic falloff: (1 - r²)² gives blobby metaball blending.
                    float t = 1.0 - distSq;
                    float contribution = t * t;
                    uint fixedPoint = uint(contribution * float(BONE_SLURRY_FIXED_POINT_SCALE));
                    uint idx = boneSlurryFlatIndex(uint3(x, y, z), params.dimensions);
                    atomic_fetch_add_explicit(&density[idx], fixedPoint, memory_order_relaxed);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MARK: - Marching Cubes Kernel (on density buffer)
// ─────────────────────────────────────────────────────────────────────

/// Read density from the flat buffer and convert from fixed-point to float.
static float boneSlurryReadDensity(device uint *density, uint3 coord, uint3 dims) {
    uint idx = boneSlurryFlatIndex(coord, dims);
    return float(density[idx]) / float(BONE_SLURRY_FIXED_POINT_SCALE);
}

/// Compute gradient-based normal via central differences on the density grid.
static float3 boneSlurryDensityNormal(device uint *density, uint3 coord, uint3 dims) {
    uint3 lo = max(coord, uint3(1)) - 1;
    uint3 hi = min(coord + 1, dims - 1);

    float dx = boneSlurryReadDensity(density, uint3(hi.x, coord.y, coord.z), dims)
             - boneSlurryReadDensity(density, uint3(lo.x, coord.y, coord.z), dims);
    float dy = boneSlurryReadDensity(density, uint3(coord.x, hi.y, coord.z), dims)
             - boneSlurryReadDensity(density, uint3(coord.x, lo.y, coord.z), dims);
    float dz = boneSlurryReadDensity(density, uint3(coord.x, coord.y, hi.z), dims)
             - boneSlurryReadDensity(density, uint3(coord.x, coord.y, lo.z), dims);

    // Gradient points toward increasing density; negate for outward-facing normal.
    return -float3(dx, dy, dz);
}

[[kernel]]
void boneSlurryMarch(device BoneSlurryVertex *vertices [[buffer(0)]],
                     device uint *indices [[buffer(1)]],
                     device uint *density [[buffer(2)]],
                     constant BoneSlurryGridParams &params [[buffer(3)]],
                     device uint2 *triangleTable [[buffer(4)]],
                     device atomic_uint &counter [[buffer(5)]],
                     uint3 voxelCoords [[thread_position_in_grid]]) {

    // The marching cubes grid is (dims-1)^3 cubes.
    if (any(voxelCoords >= params.dimensions - 1)) return;

    // Sample the 8 corners of the current cube.
    float samples[8];
    float3 sampleNormals[8];
    for (uint i = 0; i < 8; i++) {
        uint3 corner = voxelCoords + boneSlurryCubeVertex(i);
        samples[i] = boneSlurryReadDensity(density, corner, params.dimensions);
        sampleNormals[i] = boneSlurryDensityNormal(density, corner, params.dimensions);
    }

    // Build selector bitfield: bit i set if corner i is above isoValue.
    uint selector = 0;
    for (uint i = 0; i < 8; i++) {
        selector |= (samples[i] > params.isoValue) << i;
    }

    // All inside or all outside → no surface.
    if (selector == 0 || selector >= 0xff) return;

    // Interpolate edge positions and normals.
    float3 positions[12];
    float3 normals[12];

    for (uint i = 0; i < 12; i++) {
        uint2 pair = boneSlurryEdgeVertexPair(i);
        float s1 = samples[pair.x];
        float s2 = samples[pair.y];
        float3 v1 = float3(voxelCoords + boneSlurryCubeVertex(pair.x));
        float3 v2 = float3(voxelCoords + boneSlurryCubeVertex(pair.y));
        float denom = s2 - s1;
        float t = abs(denom) < 1e-6 ? 0.5 : (params.isoValue - s1) / denom;
        t = clamp(t, 0.0, 1.0);

        // Position in voxel coords → root-local space.
        float3 voxelPos = mix(v1, v2, t);
        positions[i] = voxelPos * params.voxelSize + params.gridOrigin;

        // Interpolated normal with safe normalization.
        float3 n = mix(sampleNormals[pair.x], sampleNormals[pair.y], t);
        float nLen = length(n);
        normals[i] = nLen > 1e-6 ? n / nLen : float3(0, 1, 0);
    }

    // Emit triangles from the lookup table.
    uint2 triangleData = triangleTable[selector];
    for (uint i = 0; i < 15; i += 3) {
        uint e0 = boneSlurryEdgeIndex(triangleData, i + 0);
        uint e1 = boneSlurryEdgeIndex(triangleData, i + 1);
        uint e2 = boneSlurryEdgeIndex(triangleData, i + 2);

        if (e0 == 15) return;

        uint triangleIndex = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
        uint vi0 = triangleIndex * 3;
        uint vi1 = vi0 + 1;
        uint vi2 = vi0 + 2;

        if (vi2 >= params.maxVertexCount) return;

        vertices[vi0].position = positions[e0];
        vertices[vi0].normal   = normals[e0];
        vertices[vi1].position = positions[e1];
        vertices[vi1].normal   = normals[e1];
        vertices[vi2].position = positions[e2];
        vertices[vi2].normal   = normals[e2];

        // Winding order matches existing convention.
        indices[vi0] = vi0;
        indices[vi1] = vi2;
        indices[vi2] = vi1;
    }
}
