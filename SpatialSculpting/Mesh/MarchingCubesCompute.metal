/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
An implementation of the Marching Cubes algorithm for meshing an isosurface.
*/

#include <metal_stdlib>

using namespace metal;

#include "MeshVertex.h"
#include "MarchingCubesParams.h"

uint edgeIndexFromTriangleTable(uint2 data, uint index) {
    return 0xfu & (index < 8 ? data.x >> ((index + 0) * 4) :
                               data.y >> ((index - 8) * 4));
}

uint2 edgeVertexPair(uint index) {
    // (0, 1) (1, 2) (2, 3) (3, 0)
    // (4, 5) (5, 6) (6, 7) (7, 4)
    // (0, 4) (1, 5) (2, 6) (3, 7)
    uint v1 = index & 7;
    uint v2 = index < 8 ? ((index + 1) & 3) | (index & 4) : v1 + 4;
    return uint2(v1, v2);
}

uint3 cubeVertex(uint index) {
    bool x = index & 1;
    bool y = index & 2;
    bool z = index & 4;
    return uint3(x ^ y, y, z);
}

float4 voxelValueWithNormal(uint3 voxelCoords, uint3 dimensions, texture3d<float, access::read> voxels) {
    // The current voxel coordinate minus one in both dimensions, guaranteed to be in bounds.
    uint3 voxelCoordsMinusOne = max(voxelCoords, 1) - 1;
    // The current voxel coordinate plus one in both dimensions, guaranteed to be in bounds.
    uint3 voxelCoordsPlusOne = min(voxelCoords + 1, dimensions - 1);

    // Sample the current pixel along with its six neighbors.
    float voxel = voxels.read(voxelCoords).r;
    float leftVoxel = voxels.read(uint3(voxelCoordsMinusOne.x, voxelCoords.yz)).r;
    float rightVoxel = voxels.read(uint3(voxelCoordsPlusOne.x, voxelCoords.yz)).r;
    float bottomVoxel = voxels.read(uint3(voxelCoords.x, voxelCoordsMinusOne.y, voxelCoords.z)).r;
    float topVoxel = voxels.read(uint3(voxelCoords.x, voxelCoordsPlusOne.y, voxelCoords.z)).r;
    float backVoxel = voxels.read(uint3(voxelCoords.xy, voxelCoordsMinusOne.z)).r;
    float frontVoxel = voxels.read(uint3(voxelCoords.xy, voxelCoordsPlusOne.z)).r;

    // Compute the normal direction.
    float3 normal = -float3(leftVoxel - rightVoxel, bottomVoxel - topVoxel, backVoxel - frontVoxel);

    // Pack the normal direction with the voxel value.
    return float4(normal, voxel);
}

[[kernel]]
void march(device MeshVertex *vertices [[buffer(0)]],
           device uint *indices [[buffer(1)]],
           texture3d<float, access::read> voxels [[texture(2)]],
           constant MarchingCubesParams &params [[buffer(3)]],
           device uint2 *triangleTable [[buffer(4)]],
           device atomic_uint &counter [[buffer(5)]],
           constant float &isoValue [[buffer(6)]],
           texture3d<half, access::sample> albedoTexture [[texture(7)]],
           texture3d<half, access::sample> normalTexture [[texture(8)]],
           texture3d<half, access::sample> roughnessTexture [[texture(9)]],
           texture3d<half, access::sample> confidenceTexture [[texture(10)]],
           uint3 chunkVoxelCoords [[thread_position_in_grid]]) {

    // Skip out of bounds threads.
    if (any(chunkVoxelCoords >= params.chunkDimensions - 1)) { return; }

    // Convert the chunk voxel coordinates to the actual voxel coordinates.
    uint3 voxelCoords = chunkVoxelCoords + uint3(0, 0, params.chunkStartZ);

    // Sample the voxels at each corner of the current cube.
    float4 samples[8];
    for (uint i = 0; i < 8; i++) {
        samples[i] = voxelValueWithNormal(voxelCoords + cubeVertex(i), params.dimensions, voxels);
    }

    // Encode which voxels of the cube are inside or outside of the isovalue with a bit field.
    uint selector = 0;
    for (uint i = 0; i < 8; i++) {
        selector |= (samples[i].w < isoValue) << i;
    }

    // Exit early if nothing can be constructed for this voxel
    // (all corners are inside or all corners are outside).
    if (selector == 0 || selector >= 0xff) { return; }

    // Get the position and normal of the isovalue along each edge of the cube.
    float3 positions[12];
    float3 normals[12];
    float4 colors[12];
    float roughnessValues[12];

    // Set up samplers and coordinate mapping for 3D material texture lookups.
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    constexpr sampler nearestSampler(coord::normalized, filter::nearest, address::clamp_to_edge);
    const float3 appBoundsMin = params.voxelStartPosition - params.voxelSize * 0.5;
    const float3 appBoundsMax = params.voxelStartPosition + params.voxelSize * (float3(params.dimensions) - 0.5);
    const float3 appExtent = max(appBoundsMax - appBoundsMin, float3(1e-6));
    const float3 sourceExtent = max(params.textureBoundsMax - params.textureBoundsMin, float3(1e-6));

    for (uint i = 0; i < 12; i++) {
        uint2 pair = edgeVertexPair(i);
        float4 sample1 = samples[pair.x];
        float4 sample2 = samples[pair.y];
        float3 vertex1 = float3(voxelCoords + cubeVertex(pair.x));
        float3 vertex2 = float3(voxelCoords + cubeVertex(pair.y));
        float denom = sample2.w - sample1.w;
        float t = abs(denom) < 1e-6 ? 0.5 : (isoValue - sample1.w) / denom;
        t = clamp(t, 0.0, 1.0);
        positions[i] = mix(vertex1, vertex2, t) * params.voxelSize + params.voxelStartPosition;

        // Compute gradient-based normal with safe normalization.
        float3 gradientNormal = mix(sample1.xyz, sample2.xyz, t);
        float gradientLength = length(gradientNormal);
        gradientNormal = gradientLength > 1e-6 ? gradientNormal / gradientLength : float3(0, 1, 0);

        // Map world position to normalized texture coordinates using manifest bounds.
        float3 worldUV = clamp((positions[i] - appBoundsMin) / appExtent, 0.0, 1.0);
        float3 sourcePosition = params.textureBoundsMin + worldUV * sourceExtent;
        float3 normalizedCoords = clamp((sourcePosition - params.textureBoundsMin) / sourceExtent, 0.0, 1.0);

        // Sample baked material volumes with linear interpolation.
        const half4 sampledAlbedo = albedoTexture.sample(linearSampler, normalizedCoords);
        const half4 sampledNormal = normalTexture.sample(linearSampler, normalizedCoords);
        const half4 sampledRoughness = roughnessTexture.sample(nearestSampler, normalizedCoords);
        const half sampledConfidence = confidenceTexture.sample(linearSampler, normalizedCoords).x;

        // Blend geometric normal with baked object-space normal using confidence.
        float confidence = clamp(float(sampledConfidence), 0.0, 1.0);
        float3 blended = gradientNormal + float3(sampledNormal.xyz) * (0.5 * confidence);
        float blendedLength = length(blended);
        normals[i] = blendedLength > 1e-6 ? blended / blendedLength : gradientNormal;

        // Confidence modulates how strongly baked albedo influences the final vertex color.
        float3 fallbackAlbedo = float3(0.8, 0.8, 0.8);
        colors[i] = float4(mix(fallbackAlbedo, float3(sampledAlbedo.xyz), confidence), 1.0);
        roughnessValues[i] = float(sampledRoughness.x);
    }

    // Get the triangle data for the current configuration of the inside and outside voxels.
    uint2 triangleData = triangleTable[selector];
    // Write the triangles to the vertex and index buffers.
    for (uint i = 0; i < 15; i += 3) {
        uint e0 = edgeIndexFromTriangleTable(triangleData, i + 0);
        uint e1 = edgeIndexFromTriangleTable(triangleData, i + 1);
        uint e2 = edgeIndexFromTriangleTable(triangleData, i + 2);

        if (e0 == 15) return;

        // Get the current triangle index from the atomic counter.
        uint triangleIndex = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);

        // Get the vertex indices of the triangle's vertices.
        uint vertexIndex0 = triangleIndex * 3;
        uint vertexIndex1 = vertexIndex0 + 1;
        uint vertexIndex2 = vertexIndex0 + 2;

        // Return early if there's no room for any more vertices.
        if (vertexIndex2 >= params.maxVertexCount) return;

        // Set the position, normal, color, and roughness of each vertex.
        vertices[vertexIndex0].position = positions[e0];
        vertices[vertexIndex0].normal = normals[e0];
        vertices[vertexIndex0].color = colors[e0];
        vertices[vertexIndex0].roughness = roughnessValues[e0];
        vertices[vertexIndex1].position = positions[e1];
        vertices[vertexIndex1].normal = normals[e1];
        vertices[vertexIndex1].color = colors[e1];
        vertices[vertexIndex1].roughness = roughnessValues[e1];
        vertices[vertexIndex2].position = positions[e2];
        vertices[vertexIndex2].normal = normals[e2];
        vertices[vertexIndex2].color = colors[e2];
        vertices[vertexIndex2].roughness = roughnessValues[e2];

        // Set the indices that describe the triangle.
        indices[vertexIndex0] = vertexIndex0;
        indices[vertexIndex1] = vertexIndex2;
        indices[vertexIndex2] = vertexIndex1;
    }
}

[[kernel]]
void clear(device MeshVertex *vertices [[buffer(0)]],
           device uint *indices [[buffer(1)]],
           constant MarchingCubesParams &params [[buffer(2)]],
           uint vertexIndex [[thread_position_in_grid]]) {
    // Skip out of bounds threads.
    if (vertexIndex >= params.maxVertexCount) { return; }

    // Reset vertices and indices.
    vertices[vertexIndex].position = 0;
    vertices[vertexIndex].normal = 0;
    vertices[vertexIndex].color = 0;
    vertices[vertexIndex].roughness = 0;
    indices[vertexIndex] = 0;
}
