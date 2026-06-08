/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Parameters per mesh vertex.
*/
#pragma once

#include <simd/simd.h>

struct MeshVertex {
    simd_float3 position;
    simd_float3 normal;
    // Baked material channels sampled from 3D textures.
    simd_float4 color;
    float roughness;
};
