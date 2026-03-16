/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Extensions on `MeshVertex` for use with `LowLevelMesh`.
*/
import RealityKit

extension MeshVertex {
    static var vertexAttributes: [LowLevelMesh.Attribute] {
        let positionAttributeOffset = MemoryLayout.offset(of: \Self.position) ?? 0
        let normalAttributeOffset = MemoryLayout.offset(of: \Self.normal) ?? 16
        let colorAttributeOffset = MemoryLayout.offset(of: \Self.color) ?? 32

        let positionAttribute = LowLevelMesh.Attribute(semantic: .position, format: .float3, offset: positionAttributeOffset)
        let normalAttribute = LowLevelMesh.Attribute(semantic: .normal, format: .float3, offset: normalAttributeOffset)
        // Per-vertex albedo from baked 3D texture, passed to material.
        let colorAttribute = LowLevelMesh.Attribute(semantic: .color, format: .float4, offset: colorAttributeOffset)

        let vertexAttributes = [positionAttribute, normalAttribute, colorAttribute]

        return vertexAttributes
    }

    static let vertexLayouts: [LowLevelMesh.Layout] = [LowLevelMesh.Layout(bufferIndex: 0, bufferStride: MemoryLayout<Self>.stride)]
}
