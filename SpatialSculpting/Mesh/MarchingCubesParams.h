/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Parameters for marching cubes.
*/
#pragma once

#include <simd/simd.h>

struct MarchingCubesParams {
    simd_uint3 dimensions;
    simd_float3 voxelSize;
    simd_float3 voxelStartPosition;
    simd_float3 textureBoundsMin;
    simd_float3 textureBoundsMax;
    simd_uint3 chunkDimensions;
    unsigned int chunkStartZ;
    unsigned int maxVertexCount;
};
