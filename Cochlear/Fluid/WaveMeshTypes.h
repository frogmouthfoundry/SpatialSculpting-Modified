/*
Shared C/Metal types for dynamic wave mesh updates.
*/

#pragma once

#include <simd/simd.h>

#ifndef __METAL__
typedef struct {
    float x;
    float y;
    float z;
} packed_float3;
#endif

struct WaveMeshVertex {
    packed_float3 position;
    packed_float3 normal;
    simd_packed_float2 uv;
};

struct WaveDescriptor {
    unsigned int segmentCount;
    float time;
    float waveDensity;
    float amplitude;
    float rippleX;
    float rippleZ;
    float rippleStrength;
    float rippleTime;
    float touchX0;
    float touchZ0;
    float touchX1;
    float touchZ1;
};
