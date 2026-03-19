/*
Abstract:
Extensions on BoneSlurryVertex for use with LowLevelMesh.
*/
import RealityKit

extension BoneSlurryVertex {
    static var vertexAttributes: [LowLevelMesh.Attribute] {
        let positionOffset = MemoryLayout.offset(of: \Self.position) ?? 0
        let normalOffset = MemoryLayout.offset(of: \Self.normal) ?? 12

        return [
            LowLevelMesh.Attribute(semantic: .position, format: .float3, offset: positionOffset),
            LowLevelMesh.Attribute(semantic: .normal, format: .float3, offset: normalOffset)
        ]
    }

    static let vertexLayouts: [LowLevelMesh.Layout] = [
        LowLevelMesh.Layout(bufferIndex: 0, bufferStride: MemoryLayout<Self>.stride)
    ]
}
