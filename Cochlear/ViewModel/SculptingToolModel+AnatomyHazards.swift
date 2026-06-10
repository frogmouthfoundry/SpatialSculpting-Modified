/*
Abstract:
Procedural anatomy safezones and hazard reactions for the facial nerve and sigmoid sinus.
*/

import Foundation
import RealityKit

enum AnatomyHazardKind: Hashable {
    case facialNerve
    case sigmoid
}

struct AnatomySafezoneSphere {
    let kind: AnatomyHazardKind
    let entity: Entity
    let radius: Float
}

extension SculptingToolModel {

    func registerAnatomyEffectsRoot(_ root: Entity) {
        anatomyEffectsRoot = root
    }

    func preloadBloodSpurt() {
        Task { @MainActor in
            if let template = try? await Entity(named: "BloodSpurt") {
                self.bloodSpurtTemplate = template
            } else {
                print("[AnatomyHazard] Failed to pre-load BloodSpurt.usdz")
            }
        }
    }

    func registerSafezones(for kind: AnatomyHazardKind, entity: Entity) {
        anatomyHazardEntities[kind] = entity
        let bounds = entity.visualBounds(relativeTo: entity)
        let extents = bounds.extents
        let center = bounds.center

        let axisIndex: Int
        if extents.x >= extents.y && extents.x >= extents.z {
            axisIndex = 0
        } else if extents.y >= extents.x && extents.y >= extents.z {
            axisIndex = 1
        } else {
            axisIndex = 2
        }

        let sphereCount: Int
        let radius: Float
        switch kind {
        case .facialNerve:
            sphereCount = 4
            radius = max(minorExtent(for: extents, excluding: axisIndex) * 0.75, 0.008)
        case .sigmoid:
            sphereCount = 5
            radius = max(minorExtent(for: extents, excluding: axisIndex) * 0.9, 0.012)
        }

        let axisExtent = component(of: extents, at: axisIndex)
        let start = component(of: center, at: axisIndex) - axisExtent * 0.5
        let end = component(of: center, at: axisIndex) + axisExtent * 0.5

        for index in 0..<sphereCount {
            let t = sphereCount == 1 ? 0.5 : Float(index) / Float(sphereCount - 1)
            var localCenter = center
            setComponent(of: &localCenter, at: axisIndex, value: start + (end - start) * t)

            let safezone = Entity()
            safezone.name = kind == .facialNerve ? "FacialNerveSafezone_\(index)" : "SigmoidSafezone_\(index)"
            safezone.position = localCenter
            entity.addChild(safezone)
            anatomySafezones.append(AnatomySafezoneSphere(kind: kind, entity: safezone, radius: radius))
        }
    }

    func updateAnatomyHazards(toolPosition: SIMD3<Float>,
                              toolRadius: Float,
                              drillBitDirection: SIMD3<Float>) {
        guard let rootEntity else { return }

        var detectedKinds: Set<AnatomyHazardKind> = []
        var triggerPoints: [AnatomyHazardKind: SIMD3<Float>] = [:]

        for safezone in anatomySafezones {
            let safezonePosition = safezone.entity.position(relativeTo: rootEntity)
            let distance = simd_length(toolPosition - safezonePosition)
            guard distance <= (toolRadius + safezone.radius) else { continue }

            detectedKinds.insert(safezone.kind)
            if triggerPoints[safezone.kind] == nil {
                triggerPoints[safezone.kind] = toolPosition
            }
        }

        let newHazards = detectedKinds.subtracting(activeAnatomyHazards)
        activeAnatomyHazards = detectedKinds

        for hazard in newHazards {
            triggerScreenFlash(for: Set([hazard]))
            switch hazard {
            case .facialNerve:
                drillAudioModel?.triggerAlarm(duration: 1.0)
            case .sigmoid:
                drillAudioModel?.triggerBlood(duration: 1.0)
                if let position = triggerPoints[hazard] {
                    triggerBloodSpurt(at: position,
                                      direction: drillBitDirection)
                }
            }
        }
    }

    private func triggerScreenFlash(for kinds: Set<AnatomyHazardKind>) {
        hazardFlashTask?.cancel()
        hazardFlashTask = Task { @MainActor in
            defer {
                self.flashingAnatomyHazards = []
                self.setAnatomyHazardFlashActive(false)
                self.hazardFlashTask = nil
            }
            self.flashingAnatomyHazards = kinds
            self.setAnatomyHazardFlashActive(true)
            try? await Task.sleep(for: .milliseconds(500))
            self.setAnatomyHazardFlashActive(false)
            try? await Task.sleep(for: .milliseconds(500))
            self.setAnatomyHazardFlashActive(true)
            try? await Task.sleep(for: .milliseconds(500))
            self.setAnatomyHazardFlashActive(false)
        }
    }

    func applyAnatomyHazardWarningAppearance() {
        let warningMaterial = SimpleMaterial(color: .red, roughness: 0.3, isMetallic: false)

        for (kind, entity) in anatomyHazardEntities {
            let shouldFlash = isAnatomyHazardFlashActive && flashingAnatomyHazards.contains(kind)
            if shouldFlash {
                if cachedAnatomyHazardMaterials[kind] == nil {
                    cachedAnatomyHazardMaterials[kind] = cachedMaterials(for: entity)
                }
                applyMaterial(warningMaterial, toHierarchyOf: entity)
            } else if let cachedMaterials = cachedAnatomyHazardMaterials[kind] {
                restoreMaterials(cachedMaterials)
                cachedAnatomyHazardMaterials[kind] = nil
            }
        }
    }

    private func triggerBloodSpurt(at rootPosition: SIMD3<Float>,
                                   direction: SIMD3<Float>) {
        guard let effectsRoot = anatomyEffectsRoot,
              let bloodSpurtTemplate,
              let rootEntity else {
            return
        }

        let bloodSpurt = bloodSpurtTemplate.clone(recursive: true)
        bloodSpurt.name = "BloodSpurtEffect"
        bloodSpurt.position = effectsRoot.convert(position: rootPosition, from: rootEntity)
        bloodSpurt.orientation = safeBloodSpurtOrientation(for: direction)
        effectsRoot.addChild(bloodSpurt)

        Task { @MainActor in
            try? await Task.sleep(for: .seconds(5))
            bloodSpurt.removeFromParent()
        }
    }

    private func component(of vector: SIMD3<Float>, at index: Int) -> Float {
        switch index {
        case 0: return vector.x
        case 1: return vector.y
        default: return vector.z
        }
    }

    private func setComponent(of vector: inout SIMD3<Float>, at index: Int, value: Float) {
        switch index {
        case 0: vector.x = value
        case 1: vector.y = value
        default: vector.z = value
        }
    }

    private func minorExtent(for extents: SIMD3<Float>, excluding axisIndex: Int) -> Float {
        switch axisIndex {
        case 0: return max(extents.y, extents.z)
        case 1: return max(extents.x, extents.z)
        default: return max(extents.x, extents.y)
        }
    }

    private func cachedMaterials(for root: Entity) -> [(Entity, [any RealityKit.Material])] {
        var cache: [(Entity, [any RealityKit.Material])] = []
        func walk(_ entity: Entity) {
            if let model = entity.components[ModelComponent.self] {
                cache.append((entity, model.materials))
            }
            for child in entity.children {
                walk(child)
            }
        }
        walk(root)
        return cache
    }

    private func applyMaterial(_ material: any RealityKit.Material, toHierarchyOf root: Entity) {
        func walk(_ entity: Entity) {
            if var model = entity.components[ModelComponent.self] {
                model.materials = Array(repeating: material, count: max(model.materials.count, 1))
                entity.components.set(model)
            }
            for child in entity.children {
                walk(child)
            }
        }
        walk(root)
    }

    private func restoreMaterials(_ cachedMaterials: [(Entity, [any RealityKit.Material])]) {
        for (entity, materials) in cachedMaterials {
            if var model = entity.components[ModelComponent.self] {
                model.materials = materials
                entity.components.set(model)
            }
        }
    }

    private func safeBloodSpurtOrientation(for targetDirection: SIMD3<Float>) -> simd_quatf {
        let from = SIMD3<Float>(0, 1, 0)
        let viewerTilt = simd_quatf(angle: .pi / 6, axis: SIMD3<Float>(1, 0, 0))
        let yawAdjustment = simd_quatf(angle: -.pi / 12, axis: SIMD3<Float>(0, 1, 0))
        guard targetDirection.x.isFinite,
              targetDirection.y.isFinite,
              targetDirection.z.isFinite,
              simd_length_squared(targetDirection) > 1e-8 else {
            let fallback = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
            return simd_normalize(fallback * viewerTilt * yawAdjustment)
        }

        let to = simd_normalize(targetDirection)
        let dot = simd_dot(from, to)
        if dot > 0.9999 {
            let base = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1))
            return simd_normalize(base * viewerTilt * yawAdjustment)
        }
        if dot < -0.9999 {
            let base = simd_quatf(angle: .pi, axis: SIMD3<Float>(1, 0, 0))
            return simd_normalize(base * viewerTilt * yawAdjustment)
        }
        let rawAxis = simd_cross(from, to)
        let axisLengthSquared = simd_length_squared(rawAxis)
        guard axisLengthSquared > 1e-8,
              rawAxis.x.isFinite,
              rawAxis.y.isFinite,
              rawAxis.z.isFinite else {
            let fallback = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
            return simd_normalize(fallback * viewerTilt * yawAdjustment)
        }

        let axis = rawAxis / sqrt(axisLengthSquared)
        let angle = acos(simd_clamp(dot, -1.0, 1.0))
        guard angle.isFinite else {
            let fallback = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
            return simd_normalize(fallback * viewerTilt * yawAdjustment)
        }

        let baseOrientation = simd_normalize(simd_quatf(angle: angle, axis: axis))
        return simd_normalize(baseOrientation * viewerTilt * yawAdjustment)
    }
}
