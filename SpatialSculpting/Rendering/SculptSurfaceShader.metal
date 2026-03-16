/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Lit surface shader that displays compute-baked vertex albedo.
*/

#include <metal_stdlib>
#if __has_include(<RealityKit/RealityKit.h>)
#include <RealityKit/RealityKit.h>
#endif

using namespace metal;

#if __has_include(<RealityKit/RealityKit.h>)
[[visible]]
void sculptSurfaceShader(realitykit::surface_parameters params) {
    const half4 vertexColor = params.geometry().color();
    const float roughness = clamp(params.uniforms().custom_parameter().x, 0.04f, 1.0f);

    params.surface().set_base_color(vertexColor.rgb);
    params.surface().set_opacity(vertexColor.a);
    params.surface().set_metallic(0.0);
    params.surface().set_roughness(roughness);
}
#endif
