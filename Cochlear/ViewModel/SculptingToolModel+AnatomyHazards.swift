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

    func updateAnatomyHazards(toolPosition: SIMD3<Float>, toolRadius: Float) {
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
            triggerScreenFlash()
            switch hazard {
            case .facialNerve:
                drillAudioModel?.triggerAlarm(duration: 1.0)
            case .sigmoid:
                if let position = triggerPoints[hazard] {
                    triggerBloodSpurt(at: position)
                }
            }
        }
    }

    private func triggerScreenFlash() {
        hazardFlashTask?.cancel()
        hazardFlashTask = Task { @MainActor in
            let flashOpacity: Double = 0.52
            for _ in 0..<3 {
                self.hazardFlashOpacity = flashOpacity
                try? await Task.sleep(for: .milliseconds(90))
                self.hazardFlashOpacity = 0
                try? await Task.sleep(for: .milliseconds(90))
            }
            self.hazardFlashOpacity = 0
            self.hazardFlashTask = nil
        }
    }

    private func triggerBloodSpurt(at rootPosition: SIMD3<Float>) {
        guard let effectsRoot = anatomyEffectsRoot,
              let bloodSpurtTemplate,
              let rootEntity else {
            return
        }

        let bloodSpurt = bloodSpurtTemplate.clone(recursive: true)
        bloodSpurt.name = "BloodSpurtEffect"
        bloodSpurt.position = effectsRoot.convert(position: rootPosition, from: rootEntity)
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
}
