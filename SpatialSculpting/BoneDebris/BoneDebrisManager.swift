/*
Abstract:
Manages bone debris box volumes spawned when the sculpting tool contacts
the sculpted volume. Reads position and state directly from
SculptingToolComponent — no redundant position tracking.
*/

import RealityKit
import UIKit
import Combine
import QuartzCore

@MainActor @Observable
final class BoneDebrisManager {

    // MARK: - Constants

    static let debrisRadius: Float = 0.001  // 1 mm
    static let debrisStretch: Float = 2.5   // elongation along depth axis

    private let stepDistance: Float = 0.002
    private let minimumMovement: Float = 0.0005

    static let debrisGravity: SIMD3<Float> = SIMD3<Float>(-0.5, 0, -4.5)

    /// Maximum number of debris entities allowed at once.
    /// Oldest debris are removed when this limit is exceeded.
    private let maxDebrisCount: Int = 150

    // MARK: - Ejection Force (tunable per-axis)

    /// Base ejection force magnitude applied opposite to sculpting direction.
    var ejectionForceBase: Float = 0.0005
    /// Per-axis multipliers for ejection force. Z is 3x to push debris outward.
    var ejectionForceMultiplier: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 6.0)
    /// Frames to wait before applying the ejection impulse.
    private let ejectionDelayFrames: Int = 2

    /// Debris waiting for ejection force: (entity, opposite sculpt direction, frames remaining).
    private var pendingEjections: [(entity: ModelEntity, direction: SIMD3<Float>, framesLeft: Int)] = []

    // MARK: - Growth Animation (tunable)

    /// Per-axis scale increment per growth tick. X,Y grow at 2× the Z rate.
    var growthPerTick: SIMD3<Float> = SIMD3<Float>(0.10, 0.10, 0.05)
    /// Seconds between each growth tick.
    var growthInterval: TimeInterval = 0.2
    /// Per-axis maximum scale multiplier relative to spawn size.
    var growthMaxMultiplier: SIMD3<Float> = SIMD3<Float>(4.0, 4.0, 2.0)

    /// Debris currently growing: (entity, base scale at spawn, current per-axis multiplier, last tick time).
    private var growingDebris: [(entity: ModelEntity, baseScale: SIMD3<Float>, multiplier: SIMD3<Float>, lastTick: TimeInterval)] = []

    // MARK: - State

    var isEnabled: Bool = true

    var isGravityEnabled: Bool = false {
        didSet {
            guard oldValue != isGravityEnabled else { return }
            applyGravityStateToAllDebris()
        }
    }

    private var debrisMesh: MeshResource?
    private var debrisMaterial: (any Material)?
    private var physicsMaterial: PhysicsMaterialResource?

    /// Cached capsule shape — created once and reused for all debris entities.
    private var cachedDebrisShape: ShapeResource?

    weak var rootEntity: Entity?
    private var debrisContainer: Entity?

    private(set) var drawnEntities: [Entity] = []

    // MARK: - Init

    init() {}

    // MARK: - Setup

    func setup(rootEntity: Entity) {
        self.rootEntity = rootEntity

        var physicsSimulation = PhysicsSimulationComponent()
        physicsSimulation.gravity = Self.debrisGravity
        rootEntity.components.set(physicsSimulation)

        let container = Entity()
        container.name = "BoneDebrisContainer"
        rootEntity.addChild(container)
        self.debrisContainer = container

        debrisMesh = .generateSphere(radius: Self.debrisRadius)

        debrisMaterial = BoneDebrisMaterialLoader.loadDebrisMaterial()
        physicsMaterial = BoneDebrisMaterialLoader.loadDebrisPhysicsMaterial()

        // Create the capsule shape once and cache it.
        let capsuleHeight = Self.debrisRadius * 2.0 * (Self.debrisStretch - 1.0)
        cachedDebrisShape = ShapeResource.generateCapsule(
            height: capsuleHeight + Self.debrisRadius * 2.0,
            radius: Self.debrisRadius
        )
    }

    // MARK: - Per-Frame Update

    /// Called every frame. Reads position and state from SculptingToolComponent.
    func update(sculptingTool: Entity) {
        guard isEnabled else { return }
        guard let component = sculptingTool.components[SculptingToolComponent.self] else { return }
        guard component.isActive else { return }

        let currentPosition = sculptingTool.position

        guard let previousPosition = component.previousPosition else {
            // First frame of sculpt — place a single box.
            placeBox(at: currentPosition, direction: SIMD3<Float>(0, 0, 1))
            return
        }

        let delta = currentPosition - previousPosition
        let distance = simd_length(delta)
        guard distance >= minimumMovement else { return }

        let direction = simd_normalize(delta)
        let steps = max(1, Int(ceil(distance / stepDistance)))

        for i in 1...steps {
            let t = Float(i) / Float(steps)
            let pos = simd_mix(previousPosition, currentPosition, SIMD3<Float>(repeating: t))
            placeBox(at: pos, direction: direction)
        }
    }

    // MARK: - Box Placement

    /// Offset applied to push debris spawn point outward from the volume
    /// surface. Prevents debris from spawning inside the collision mesh
    /// (which would cause PhysX ejection). Direction: from volume center
    /// toward spawn point (volume is centered at origin).
    private let surfaceOffset: Float = 0.006  // 6 mm outward

    private func placeBox(at position: SIMD3<Float>, direction: SIMD3<Float>) {
        guard let mesh = debrisMesh,
              let material = debrisMaterial,
              let container = debrisContainer,
              let debrisShape = cachedDebrisShape else { return }

        // Enforce debris count limit — remove oldest first.
        while drawnEntities.count >= maxDebrisCount {
            let oldest = drawnEntities.removeFirst()
            oldest.removeFromParent()
        }

        // Offset spawn position outward from the volume center (origin).
        let outward = simd_length(position) > 0.001 ? simd_normalize(position) : SIMD3<Float>(0, 0, 1)
        let offsetPosition = position + outward * surfaceOffset

        let boxEntity = ModelEntity(mesh: mesh, materials: [material])
        boxEntity.name = "BoneDebris"

        let orientation = orientationAlongDirection(direction)
        // Stretch sphere into ellipsoid along local Z (depth) axis.
        let stretchScale = SIMD3<Float>(0.5, 1.0, 6.0)
        boxEntity.transform = Transform(
            scale: stretchScale,
            rotation: orientation,
            translation: offsetPosition
        )

        // Reuse cached shape for collision and physics.
        boxEntity.components.set(CollisionComponent(shapes: [debrisShape]))

        // Always spawn dynamic so ejection impulse works.
        // Gravity toggle can switch to static afterward.
        boxEntity.components.set(PhysicsBodyComponent(
            shapes: [debrisShape],
            mass: 0.001,
            material: physicsMaterial ?? .default,
            mode: .dynamic
        ))

        container.addChild(boxEntity)
        drawnEntities.append(boxEntity)

        // Queue ejection force: will fire after ejectionDelayFrames.
        // Direction is opposite to sculpting direction.
        let ejectionDir = -direction
        pendingEjections.append((entity: boxEntity, direction: ejectionDir, framesLeft: ejectionDelayFrames))

        // Queue for gradual growth from spawn scale to 200%.
        growingDebris.append((entity: boxEntity, baseScale: stretchScale, multiplier: SIMD3<Float>(1, 1, 1), lastTick: CACurrentMediaTime()))
    }

    // MARK: - Ejection Force Processing

    /// Call every frame to count down and apply ejection impulses to recently spawned debris.
    func processPendingEjections() {
        var remaining: [(entity: ModelEntity, direction: SIMD3<Float>, framesLeft: Int)] = []

        for var entry in pendingEjections {
            entry.framesLeft -= 1
            if entry.framesLeft <= 0 {
                // Apply per-axis scaled impulse.
                let scaledForce = entry.direction * ejectionForceMultiplier * ejectionForceBase
                entry.entity.addForce(scaledForce, relativeTo: nil)
            } else {
                remaining.append(entry)
            }
        }

        pendingEjections = remaining
    }

    // MARK: - Growth Processing

    /// Call every frame to gradually scale up debris over time.
    func processGrowth() {
        let now = CACurrentMediaTime()
        var stillGrowing: [(entity: ModelEntity, baseScale: SIMD3<Float>, multiplier: SIMD3<Float>, lastTick: TimeInterval)] = []

        for var entry in growingDebris {
            guard entry.entity.parent != nil else { continue } // removed debris

            let elapsed = now - entry.lastTick
            if elapsed >= growthInterval {
                // Grow each axis independently, clamped to its max.
                entry.multiplier = simd_min(entry.multiplier + growthPerTick, growthMaxMultiplier)
                entry.lastTick = now
                entry.entity.scale = entry.baseScale * entry.multiplier
            }

            // Keep tracking until all axes have reached their max.
            let reachedMax = entry.multiplier.x >= growthMaxMultiplier.x
                          && entry.multiplier.y >= growthMaxMultiplier.y
                          && entry.multiplier.z >= growthMaxMultiplier.z
            if !reachedMax {
                stillGrowing.append(entry)
            }
        }

        growingDebris = stillGrowing
    }

    private func orientationAlongDirection(_ direction: SIMD3<Float>) -> simd_quatf {
        let dir = simd_length(direction) > 0.0001 ? simd_normalize(direction) : SIMD3<Float>(0, 0, 1)
        let forward = SIMD3<Float>(0, 0, -1)
        let dot = simd_dot(forward, dir)

        if dot > 0.9999 {
            return simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        } else if dot < -0.9999 {
            return simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 1, 0))
        }

        let axis = simd_normalize(simd_cross(forward, dir))
        let angle = acos(simd_clamp(dot, -1.0, 1.0))
        return simd_quatf(angle: angle, axis: axis)
    }

    // MARK: - Cleanup

    func clearAllDebris() {
        for entity in drawnEntities { entity.removeFromParent() }
        drawnEntities.removeAll()
        pendingEjections.removeAll()
        growingDebris.removeAll()
    }

    func undoLastStroke(count: Int = 50) {
        let removeCount = min(count, drawnEntities.count)
        let entitiesToRemove = drawnEntities.suffix(removeCount)
        for entity in entitiesToRemove { entity.removeFromParent() }
        drawnEntities.removeLast(removeCount)
    }

    // MARK: - Gravity Toggle
    private func applyGravityStateToAllDebris() {
        guard let debrisShape = cachedDebrisShape else { return }
        let mode: PhysicsBodyMode = isGravityEnabled ? .dynamic : .static

        for entity in drawnEntities {
            guard let modelEntity = entity as? ModelEntity else { continue }
            modelEntity.components.set(PhysicsBodyComponent(
                shapes: [debrisShape],
                mass: 0.001,
                material: physicsMaterial ?? .default,
                mode: mode
            ))
        }
    }
}
