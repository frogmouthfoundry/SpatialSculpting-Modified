/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Parameters for sculpting to pass to compute shader.
*/

#pragma once

#include <simd/simd.h>

typedef enum {
  add, remove
} sculpt_mode;

struct SculptParams {
    sculpt_mode mode;
    simd_float4 toolPositionAndRadius;
    simd_float4 previousPositionAndHasPosition;
    /// Voxel-space offset for localized dispatch (thread coords are relative to this).
    simd_uint3 regionOffset;
};
