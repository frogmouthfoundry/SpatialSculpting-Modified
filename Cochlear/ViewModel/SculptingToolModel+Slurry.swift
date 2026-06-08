/*
Abstract:
Helpers for rebuilding the bone slurry visualization when stale GPU mesh state
needs to be discarded entirely.
*/

import RealityKit

extension SculptingToolModel {

    func replaceBoneSlurryGrid(volumeBoundsMin: SIMD3<Float>? = nil,
                               volumeBoundsMax: SIMD3<Float>? = nil) {
        boneSlurryGrid?.entity.removeFromParent()

        guard let newGrid = BoneSlurryGrid() else {
            boneSlurryGrid = nil
            sculptingTool.components[SculptingToolComponent.self]?.boneSlurryGrid = nil
            return
        }

        if let volumeBoundsMin, let volumeBoundsMax {
            newGrid.configure(volumeBoundsMin: volumeBoundsMin, volumeBoundsMax: volumeBoundsMax)
        }

        boneSlurryGrid = newGrid
        if let rootEntity {
            rootEntity.addChild(newGrid.entity)
        }
        sculptingTool.components[SculptingToolComponent.self]?.boneSlurryGrid = newGrid
    }
}
