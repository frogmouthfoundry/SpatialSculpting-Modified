/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Parameters for the sculpture volume.
*/

#pragma once

#include <simd/simd.h>

struct VolumeParams {
    simd_uint3 dimensions;
    simd_float3 voxelSize;
    simd_float3 voxelStartPosition;
    simd_float3 textureBoundsMin;
    simd_float3 textureBoundsMax;
};
