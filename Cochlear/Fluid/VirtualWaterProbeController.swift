import RealityKit
import UIKit

@MainActor
final class VirtualWaterProbeController {

    struct SupportSampleResult {
        let isBlocked: Bool
        let resolutionVector: SIMD3<Float>
        let contactNormal: SIMD3<Float>
    }

    let radius: Float
    var horizontalReanchorDistance: Float = 0.10
    var initialAnchorOffsetDistance: Float = 0.15
    var settleSpeed: Float = 0.02
    var downhillAcceleration: Float = 0.03
    var contactDamping: Float = 8.0
    var freeDamping: Float = 2.5
    var maxLateralSpeed: Float = 0.03
    var pushoutBias: Float = 0.0015
    var debugOpacity: CGFloat = 0.85
    var isDebugVisible: Bool = false {
        didSet {
            if !isDebugVisible {
                debugEntity?.isEnabled = false
            } else if let debugPositionInRoot = waterPlacementPointInRoot ?? probePositionInRoot {
                updateDebugEntity(positionInRoot: debugPositionInRoot)
            }
        }
    }

    private let desiredRootFillDirection = simd_normalize(SIMD3<Float>(0, 0, -1))
    private let contactSampleCountThreshold: Int = 2
    private let deepPenetrationThreshold: Float = 0.001
    private let supportOffsets: [SIMD3<Float>] = [
        SIMD3<Float>(1, 0, 0),
        SIMD3<Float>(-1, 0, 0),
        SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(0, -1, 0),
        SIMD3<Float>(0, 0, 1),
        SIMD3<Float>(0, 0, -1),
        simd_normalize(SIMD3<Float>(1, 1, 0)),
        simd_normalize(SIMD3<Float>(1, -1, 0)),
        simd_normalize(SIMD3<Float>(-1, 1, 0)),
        simd_normalize(SIMD3<Float>(-1, -1, 0)),
        simd_normalize(SIMD3<Float>(1, 0, 1)),
        simd_normalize(SIMD3<Float>(-1, 0, 1)),
        simd_normalize(SIMD3<Float>(0, 1, 1)),
        simd_normalize(SIMD3<Float>(0, -1, 1)),
        simd_normalize(SIMD3<Float>(1, 0, -1)),
        simd_normalize(SIMD3<Float>(-1, 0, -1)),
        simd_normalize(SIMD3<Float>(0, 1, -1)),
        simd_normalize(SIMD3<Float>(0, -1, -1))
    ]

    private var lateralVelocity: SIMD3<Float> = .zero
    private(set) var probePositionInRoot: SIMD3<Float>? = nil
    private var anchorCarvePositionInRoot: SIMD3<Float>? = nil
    private var debugEntity: ModelEntity? = nil
    private var currentFillDirectionInRoot = simd_normalize(SIMD3<Float>(0, 0, -1))

    init(radius: Float = 0.008) {
        self.radius = radius
    }

    var waterPlacementPointInRoot: SIMD3<Float>? {
        guard let probePositionInRoot else { return nil }
        return probePositionInRoot - currentFillDirectionInRoot * radius
    }

    func reset() {
        probePositionInRoot = nil
        anchorCarvePositionInRoot = nil
        lateralVelocity = .zero
        debugEntity?.isEnabled = false
    }

    func update(rootEntity: Entity,
                collisionManager: CollisionManager,
                carvePositionInRoot: SIMD3<Float>?,
                isCarving: Bool,
                timestep: TimeInterval) {
        ensureDebugEntity(in: rootEntity)
        currentFillDirectionInRoot = desiredRootFillDirection

        if isCarving, let carvePositionInRoot {
            if probePositionInRoot == nil || shouldReanchor(to: carvePositionInRoot) {
                probePositionInRoot = carvePositionInRoot + currentFillDirectionInRoot * initialAnchorOffsetDistance
                anchorCarvePositionInRoot = carvePositionInRoot
                lateralVelocity = .zero
            }
        }

        guard var currentPosition = probePositionInRoot else {
            debugEntity?.isEnabled = false
            return
        }

        let dt = Float(max(0, timestep))
        let previousPosition = currentPosition
        if dt > 0 {
            currentPosition += currentFillDirectionInRoot * settleSpeed * dt
            if collisionManager.hasSDFCache,
               collisionManager.sampleSDF(at: currentPosition) == nil {
                currentPosition = previousPosition
            }
        }

        guard collisionManager.hasSDFCache else {
            probePositionInRoot = currentPosition
            updateDebugEntity(positionInRoot: waterPlacementPointInRoot ?? currentPosition)
            return
        }

        let contact = sampleSupport(at: currentPosition, collisionManager: collisionManager)
        if contact.isBlocked {
            currentPosition += contact.resolutionVector

            let downhill = currentFillDirectionInRoot - simd_dot(currentFillDirectionInRoot, contact.contactNormal) * contact.contactNormal
            if simd_length_squared(downhill) > 1e-8, dt > 0 {
                lateralVelocity += simd_normalize(downhill) * downhillAcceleration * dt
            }

            let damping = max(0, 1 - contactDamping * dt)
            lateralVelocity *= damping
        } else {
            let damping = max(0, 1 - freeDamping * dt)
            lateralVelocity *= damping
        }

        let lateralSpeed = simd_length(lateralVelocity)
        if lateralSpeed > maxLateralSpeed, lateralSpeed > 1e-8 {
            lateralVelocity = lateralVelocity / lateralSpeed * maxLateralSpeed
        }

        if dt > 0, simd_length_squared(lateralVelocity) > 1e-9 {
            let candidate = currentPosition + lateralVelocity * dt
            if collisionManager.sampleSDF(at: candidate) != nil {
                let lateralContact = sampleSupport(at: candidate, collisionManager: collisionManager)
                currentPosition = candidate
                if lateralContact.isBlocked {
                    currentPosition += lateralContact.resolutionVector
                }
            } else {
                lateralVelocity = .zero
            }
        }

        currentPosition = constrainCenterToSurfaceIfNeeded(currentPosition,
                                                           collisionManager: collisionManager)
        if let anchorCarvePositionInRoot {
            currentPosition.x = anchorCarvePositionInRoot.x
            currentPosition.y = anchorCarvePositionInRoot.y
        }
        probePositionInRoot = currentPosition
        updateDebugEntity(positionInRoot: waterPlacementPointInRoot ?? currentPosition)
    }

    private func shouldReanchor(to carvePositionInRoot: SIMD3<Float>) -> Bool {
        guard let anchorCarvePositionInRoot else { return true }
        let xyDistance = simd_length(SIMD2<Float>(carvePositionInRoot.x - anchorCarvePositionInRoot.x,
                                                  carvePositionInRoot.y - anchorCarvePositionInRoot.y))
        return xyDistance >= horizontalReanchorDistance
    }

    private func sampleSupport(at center: SIMD3<Float>,
                               collisionManager: CollisionManager) -> SupportSampleResult {
        var deepestPenetration: Float = 0
        var penetratingSampleCount = 0
        var weightedNormal: SIMD3<Float> = .zero
        var fallbackNormal: SIMD3<Float> = -currentFillDirectionInRoot

        for offset in supportOffsets {
            let samplePosition = center + offset * radius
            guard let sdfValue = collisionManager.sampleSDF(at: samplePosition) else { continue }
            guard sdfValue < 0 else { continue }

            let penetration = -sdfValue
            penetratingSampleCount += 1
            deepestPenetration = max(deepestPenetration, penetration)

            let gradient = collisionManager.sampleSDFGradient(at: samplePosition)
            if simd_length_squared(gradient) > 1e-10 {
                let normal = simd_normalize(gradient)
                weightedNormal += normal * penetration
                fallbackNormal = normal
            }
        }

        let isBlocked = penetratingSampleCount >= contactSampleCountThreshold ||
            deepestPenetration >= deepPenetrationThreshold

        guard isBlocked else {
            return SupportSampleResult(isBlocked: false,
                                       resolutionVector: .zero,
                                       contactNormal: fallbackNormal)
        }

        let contactNormal: SIMD3<Float>
        if simd_length_squared(weightedNormal) > 1e-10 {
            contactNormal = simd_normalize(weightedNormal)
        } else {
            contactNormal = fallbackNormal
        }

        let resolutionVector = contactNormal * (deepestPenetration + pushoutBias)
        return SupportSampleResult(isBlocked: true,
                                   resolutionVector: resolutionVector,
                                   contactNormal: contactNormal)
    }

    private func constrainCenterToSurfaceIfNeeded(_ center: SIMD3<Float>,
                                                  collisionManager: CollisionManager) -> SIMD3<Float> {
        guard let centerSDF = collisionManager.sampleSDF(at: center) else {
            return center
        }

        // For a sphere of radius r, the center should remain at least r away from
        // the surface in SDF space. If it gets closer than that, project it back out.
        let requiredCenterSDF = radius + pushoutBias
        guard centerSDF < requiredCenterSDF else {
            return center
        }

        let gradient = collisionManager.sampleSDFGradient(at: center)
        let outwardNormal: SIMD3<Float>
        if simd_length_squared(gradient) > 1e-10 {
            outwardNormal = simd_normalize(gradient)
        } else {
            outwardNormal = -currentFillDirectionInRoot
        }
        return center + outwardNormal * (requiredCenterSDF - centerSDF)
    }

    private func ensureDebugEntity(in rootEntity: Entity) {
        guard debugEntity == nil else { return }
        let material = SimpleMaterial(color: .red.withAlphaComponent(debugOpacity),
                                      roughness: 0.2,
                                      isMetallic: false)
        let entity = ModelEntity(mesh: .generateSphere(radius: radius),
                                 materials: [material])
        entity.name = "VirtualWaterProbeDebugSphere"
        entity.isEnabled = false
        rootEntity.addChild(entity)
        debugEntity = entity
    }

    private func updateDebugEntity(positionInRoot: SIMD3<Float>) {
        debugEntity?.position = positionInRoot
        debugEntity?.isEnabled = isDebugVisible
    }
}
