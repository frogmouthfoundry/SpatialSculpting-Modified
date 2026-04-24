/*
Compute kernels for animated wave LowLevelMesh updates.
*/

#include <metal_stdlib>
using namespace metal;

#include "WaveMeshTypes.h"

float distance_to_segment(float2 p, float2 a, float2 b) {
    float2 ab = b - a;
    float2 ap = p - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0f, 1.0f);
    float2 closest = a + t * ab;
    return length(p - closest);
}

float organic_noise(float x, float z, float t) {
    return 0.0025f * sin(x * 12.0f + t * 0.3f)
         + 0.0020f * sin(z * 15.0f - t * 0.2f)
         + 0.0015f * sin((x + z) * 10.0f + t * 0.4f);
}

float wave_height(constant WaveDescriptor &wave, float x, float z) {
    float y = 0.0f;

    if (wave.rippleStrength > 0.0f) {
        float2 p = float2(x, z);
        float2 a = float2(wave.touchX0, wave.touchZ0);
        float2 b = float2(wave.touchX1, wave.touchZ1);
        float dist = distance_to_segment(p, a, b);

        float pressRadius = 0.08f;
        float pressDepth = 0.014f * wave.rippleStrength;
        float depression = -pressDepth * exp(-(dist * dist) / (2.0f * pressRadius * pressRadius));
        y += depression * exp(-wave.rippleTime * 5.0f);

        float waveSpeed = 0.18f;
        float waveFront = waveSpeed * wave.rippleTime;
        float frontWidth = 0.03f;
        float frontEnvelope = exp(-pow((dist - waveFront) / frontWidth, 2.0f));

        float ripple = wave.rippleStrength
                     * 0.02f
                     * frontEnvelope
                     * exp(-wave.rippleTime * 1.8f)
                     * sin((dist - waveFront) * 80.0f);
        y += ripple;
    }

    return y;
}

[[kernel]]
void update_wave_indices(device uint *indices [[buffer(0)]],
                         constant WaveDescriptor &wave [[buffer(1)]],
                         uint2 gridCoords [[thread_position_in_grid]]) {
    if ((gridCoords.x >= wave.segmentCount) || (gridCoords.y >= wave.segmentCount)) {
        return;
    }

    uint widthSegments = wave.segmentCount;
    uint widthVertexCount = widthSegments + 1;
    uint baseIndex = (gridCoords.y * widthSegments + gridCoords.x) * 6;
    uint baseVertex = gridCoords.y * widthVertexCount + gridCoords.x;

    indices[baseIndex + 0] = baseVertex;
    indices[baseIndex + 1] = baseVertex + widthVertexCount;
    indices[baseIndex + 2] = baseVertex + 1;
    indices[baseIndex + 3] = baseVertex + 1;
    indices[baseIndex + 4] = baseVertex + widthVertexCount;
    indices[baseIndex + 5] = baseVertex + widthVertexCount + 1;
}

[[kernel]]
void update_wave_vertices(device WaveMeshVertex *vertices [[buffer(0)]],
                          constant WaveDescriptor &wave [[buffer(1)]],
                          uint2 gridCoords [[thread_position_in_grid]]) {
    if ((gridCoords.x > wave.segmentCount) || (gridCoords.y > wave.segmentCount)) {
        return;
    }

    const float width = 1.0f;
    const float depth = 1.0f;
    const float widthSegments = wave.segmentCount;
    const float depthSegments = wave.segmentCount;
    const float segmentWidth = width / widthSegments;
    const float segmentDepth = depth / depthSegments;

    const uint widthVertexCount = widthSegments + 1;
    const uint vertexIndex = gridCoords.y * widthVertexCount + gridCoords.x;

    device WaveMeshVertex &vert = vertices[vertexIndex];

    const float x = gridCoords.x * segmentWidth - (width * 0.5f);
    const float z = gridCoords.y * segmentDepth - (depth * 0.5f);

    const float y = wave_height(wave, x, z) + organic_noise(x, z, wave.time);

    const float eps = 0.01f;
    const float y1 = wave_height(wave, x + eps, z) + organic_noise(x + eps, z, wave.time);
    const float y2 = wave_height(wave, x - eps, z) + organic_noise(x - eps, z, wave.time);
    const float dydx = y1 - y2;
    const float y3 = wave_height(wave, x, z + eps) + organic_noise(x, z + eps, wave.time);
    const float y4 = wave_height(wave, x, z - eps) + organic_noise(x, z - eps, wave.time);
    const float dydz = y3 - y4;
    const float dydy = eps * 2.0f;
    const float3 normal = normalize(float3(dydx, dydy, dydz));

    const float u = gridCoords.x / widthSegments;
    const float v = 1.0f - (gridCoords.y / depthSegments);

    vert.position = float3(x, y, z);
    vert.normal = normal;
    vert.uv = float2(u, v);
}
