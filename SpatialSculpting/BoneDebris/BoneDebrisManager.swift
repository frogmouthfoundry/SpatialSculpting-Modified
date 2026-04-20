/*
Abstract:
Manages bone debris box volumes spawned when the sculpting tool contacts
the sculpted volume. Reads position and state directly from
SculptingToolComponent — no redundant position tracking.

SDF-based adhesion: debris "sticks" to the bone surface using forces
derived from the signed distance field gradient. Debris size is
modulated by SDF distance at spawn position.
*/

import RealityKit
import UIKit
import Metal
import Combine
import QuartzCore

@MainActor @Observable
final class BoneDebrisManager {

    // MARK: - Constants

    static let debrisRadius: Float = 0.00015  // 0.15 mm
    static let debrisStretch: Float = 1.5   // elongation along depth axis

    // Spawn rate ×2.5: smaller stepDistance + minimumMovement means more
    // debris particles are created per unit of drill travel.
    private let stepDistance: Float = 0.0016   // 50% lower spawn rate
    private let minimumMovement: Float = 0.0004

    static let debrisGravity: SIMD3<Float> = SIMD3<Float>(-0.5, 0, -4.5)

    /// Maximum number of debris entities allowed at once.
    /// Oldest debris are removed when this limit is exceeded.
    private let maxDebrisCount: Int = 400

    // MARK: - Ejection Force (tunable per-axis)

    /// Base ejection force magnitude applied opposite to sculpting direction.
    var ejectionForceBase: Float = 0.005  // outward ejection away from bone
    /// Per-axis multipliers for ejection force. Z is 3x to push debris outward.
    var ejectionForceMultiplier: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 6.0)
    /// Frames to wait before applying the ejection impulse.
    private let ejectionDelayFrames: Int = 2

    /// Debris waiting for ejection force: (entity, opposite sculpt direction, frames remaining).
    private var pendingEjections: [(entity: ModelEntity, direction: SIMD3<Float>, framesLeft: Int)] = []

    // MARK: - Growth Animation (tunable)

    /// Per-axis scale increment per growth tick. Slower growth for paste-like buildup.
    var growthPerTick: SIMD3<Float> = SIMD3<Float>(0.025, 0.025, 0.0125)  // reaches max in ~4s
    /// Seconds between each growth tick.
    var growthInterval: TimeInterval = 0.05
    /// Per-axis maximum scale multiplier relative to spawn size. Capped lower for realistic scale.
    var growthMaxMultiplier: SIMD3<Float> = SIMD3<Float>(3.0, 3.0, 2.0)

    /// Debris currently growing: (entity, base scale at spawn, current per-axis multiplier, last tick time).
    private var growingDebris: [(entity: ModelEntity, baseScale: SIMD3<Float>, multiplier: SIMD3<Float>, lastTick: TimeInterval)] = []

    // MARK: - SDF Adhesion Constants

    /// Stickiness factor: strength of the adhesion force pulling debris toward the surface.
    var adhesionStickiness: Float = 0.01

    /// Distance threshold (in model space) within which friction/stiction is applied.
    /// Debris closer than this to the surface will be damped aggressively.
    var adhesionFrictionThreshold: Float = 0.01

    /// Velocity damping factor applied when debris is within friction threshold.
    /// 0 = no damping, 1 = full stop. Applied per-frame as (1 - factor).
    var adhesionFrictionDamping: Float = 0.65

    /// Maximum SDF distance at which adhesion force is applied.
    /// Beyond this, debris is in free-fall (gravity only).
    var adhesionMaxDistance: Float = 0.04

    /// Size modulation: debris spawned at SDF=0 gets this scale multiplier,
    /// scaling down linearly as |SDF| increases toward adhesionMaxDistance.
    var sdfSizeScaleMin: Float = 0.5
    var sdfSizeScaleMax: Float = 1.5

    /// Central difference step size for SDF gradient computation (in voxel coords).
    private let gradientStep: Int = 1

    // MARK: - Volume Bounds (for debris culling)

    private var volumeBoundsMin: SIMD3<Float> = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
    private var volumeBoundsMax: SIMD3<Float> = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)

    // MARK: - SDF Data

    /// Volume spatial parameters (set during setup if voxelVolume is provided).
    private var hasSDF: Bool = false
    private var sdfDimensions: SIMD3<UInt32> = .zero
    private var sdfVoxelSize: SIMD3<Float> = .zero
    private var sdfVoxelStartPosition: SIMD3<Float> = .zero

    /// CPU-readable SDF data, refreshed periodically via blit from GPU.
    private var sdfData: [Float] = []
    /// Shared staging texture for CPU readback of SDF data.
    private var sdfStagingTexture: MTLTexture?

    /// Whether we're currently waiting for a blit to complete.
    private var isBlitting: Bool = false

    /// Set to true to request a blit from the compute system.
    var sdfBlitRequested: Bool = false

    /// Minimum interval between SDF blit requests.
    private let sdfBlitInterval: TimeInterval = 0.3
    private var lastSdfBlitTime: TimeInterval = 0

    // MARK: - Settling State Machine (Option C)

    /// Debris lifecycle: ejecting → settling → settled.
    /// Settled debris has its physics body frozen (kinematic) and is only
    /// moved by tool displacement (Option B).
    enum DebrisState {
        case ejecting    // just spawned, physics active, waiting for ejection impulse
        case settling    // velocity below threshold, counting frames to confirm
        case settled     // frozen in place, only displaced by tool
    }

    /// Per-entity state tracking for the settling system.
    private var debrisStates: [ObjectIdentifier: DebrisState] = [:]

    // MARK: - Growth / Visibility Data (readable by BoneSlurryGrid)

    /// Current visual growth multiplier per entity. Applied in uploadParticles
    /// instead of on the entity transform (which physics overwrites each frame).
    private(set) var growthMultipliers: [ObjectIdentifier: SIMD3<Float>] = [:]

    /// Spawn time per entity. Debris younger than `spawnVisibilityDelay` is
    /// excluded from the metaball grid so it appears invisible initially.
    private(set) var spawnTimes: [ObjectIdentifier: TimeInterval] = [:]

    /// Seconds after spawn before debris becomes visible in the metaball mesh.
    let spawnVisibilityDelay: TimeInterval = 0.15

    /// Velocity magnitude below which debris begins the settling countdown.
    private let settlingVelocityThreshold: Float = 0.005
    /// Number of consecutive frames below threshold to confirm settled.
    private let settlingFrameCount: Int = 10
    /// Per-entity frame counter for settling confirmation.
    private var settlingCounters: [ObjectIdentifier: Int] = [:]

    // MARK: - Tool Displacement (Option B)

    /// Radius around the tool tip within which settled debris is pushed outward.
    var toolDisplacementRadius: Float = 0.008  // 8 mm
    /// Force magnitude for displacing settled slurry.
    var toolDisplacementForce: Float = 0.0003

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

    /// Setup without SDF access (fallback — no adhesion).
    func setup(rootEntity: Entity) {
        setupCommon(rootEntity: rootEntity)
    }

    /// Setup with SDF access from the voxel volume. Enables adhesion forces.
    func setup(rootEntity: Entity, voxelVolume: VoxelVolume) {
        setupCommon(rootEntity: rootEntity)

        // Store volume bounds for debris culling.
        let dims = SIMD3<Float>(voxelVolume.dimensions)
        self.volumeBoundsMin = voxelVolume.voxelStartPosition - voxelVolume.voxelSize * 0.5
        self.volumeBoundsMax = voxelVolume.voxelStartPosition + voxelVolume.voxelSize * (dims - 0.5)

        // Store volume spatial parameters for CPU SDF sampling.
        self.hasSDF = true
        self.sdfDimensions = voxelVolume.dimensions
        self.sdfVoxelSize = voxelVolume.voxelSize
        self.sdfVoxelStartPosition = voxelVolume.voxelStartPosition

        // Create shared staging texture for CPU readback (same pattern as CollisionManager).
        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .r32Float
        desc.width = Int(sdfDimensions.x)
        desc.height = Int(sdfDimensions.y)
        desc.depth = Int(sdfDimensions.z)
        desc.usage = []
        desc.storageMode = .shared
        self.sdfStagingTexture = metalDevice?.makeTexture(descriptor: desc)

        // Pre-allocate SDF data array.
        let totalVoxels = Int(sdfDimensions.x) * Int(sdfDimensions.y) * Int(sdfDimensions.z)
        self.sdfData = [Float](repeating: Float.greatestFiniteMagnitude, count: totalVoxels)

        print("[BoneDebrisManager] SDF adhesion enabled: \(sdfDimensions.x)×\(sdfDimensions.y)×\(sdfDimensions.z)")
    }

    private func setupCommon(rootEntity: Entity) {
        self.rootEntity = rootEntity

        var physicsSimulation = PhysicsSimulationComponent()
        physicsSimulation.gravity = Self.debrisGravity
        rootEntity.components.set(physicsSimulation)

        let container = Entity()
        container.name = "BoneDebrisContainer"
        rootEntity.addChild(container)
        self.debrisContainer = container

        debrisMesh = .generateSphere(radius: Self.debrisRadius)

        var pbrMaterial = PhysicallyBasedMaterial()
        pbrMaterial.baseColor = .init(tint: UIColor(red: 0xE3/255.0, green: 0xDA/255.0, blue: 0xC9/255.0, alpha: 1.0))
        pbrMaterial.roughness = .init(floatLiteral: 0.7)
        pbrMaterial.metallic = .init(floatLiteral: 0.0)
        debrisMaterial = pbrMaterial
        physicsMaterial = try? PhysicsMaterialResource.generate(
            staticFriction: 0.5, dynamicFriction: 0.4, restitution: 0.1
        )

        // Create the capsule shape once and cache it.
        let capsuleHeight = Self.debrisRadius * 2.0 * (Self.debrisStretch - 1.0)
        cachedDebrisShape = ShapeResource.generateCapsule(
            height: capsuleHeight + Self.debrisRadius * 2.0,
            radius: Self.debrisRadius
        )
    }

    // MARK: - SDF Blit Management

    /// Returns the staging texture for the compute system to blit SDF data into.
    var debrisStagingTexture: MTLTexture? { sdfStagingTexture }

    /// Request an SDF blit if enough time has passed since the last one.
    func requestSdfBlitIfNeeded() {
        guard hasSDF, !isBlitting else { return }
        let now = CACurrentMediaTime()
        guard now - lastSdfBlitTime >= sdfBlitInterval else { return }
        lastSdfBlitTime = now
        isBlitting = true
        sdfBlitRequested = true
    }

    /// Called by the compute system's completion handler after blit finishes.
    nonisolated func onSdfBlitComplete() {
        Task { @MainActor in
            self.readSdfFromStagingTexture()
            self.isBlitting = false
        }
    }

    /// Read SDF float data from the shared staging texture into the CPU array.
    private func readSdfFromStagingTexture() {
        guard let texture = sdfStagingTexture else { return }
        let w = Int(sdfDimensions.x)
        let h = Int(sdfDimensions.y)
        let d = Int(sdfDimensions.z)

        sdfData.withUnsafeMutableBufferPointer { buf in
            texture.getBytes(
                buf.baseAddress!,
                bytesPerRow: w * MemoryLayout<Float>.size,
                bytesPerImage: w * h * MemoryLayout<Float>.size,
                from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                size: MTLSize(width: w, height: h, depth: d)),
                mipmapLevel: 0,
                slice: 0
            )
        }
    }

    // MARK: - CPU SDF Sampling

    /// Sample the SDF value at a world-space position. Returns a large positive value if out of bounds.
    private func sampleSDF(at worldPosition: SIMD3<Float>) -> Float {
        guard hasSDF, !sdfData.isEmpty else { return Float.greatestFiniteMagnitude }

        let w = Int(sdfDimensions.x)
        let h = Int(sdfDimensions.y)
        let d = Int(sdfDimensions.z)

        // Convert world position to voxel coordinates.
        let voxelCoordF = (worldPosition - sdfVoxelStartPosition) / sdfVoxelSize
        let x = Int(simd_clamp(voxelCoordF.x, 0, Float(w - 1)))
        let y = Int(simd_clamp(voxelCoordF.y, 0, Float(h - 1)))
        let z = Int(simd_clamp(voxelCoordF.z, 0, Float(d - 1)))

        return sdfData[z * w * h + y * w + x]
    }

    /// Compute the SDF gradient (surface normal direction) at a world-space position
    /// using central differences. Returns normalized gradient vector.
    private func sdfGradient(at worldPosition: SIMD3<Float>) -> SIMD3<Float> {
        guard hasSDF, !sdfData.isEmpty else { return SIMD3<Float>(0, 0, 1) }

        let w = Int(sdfDimensions.x)
        let h = Int(sdfDimensions.y)
        let d = Int(sdfDimensions.z)
        let step = gradientStep

        // Convert world position to voxel coordinates.
        let voxelCoordF = (worldPosition - sdfVoxelStartPosition) / sdfVoxelSize
        let cx = Int(simd_clamp(voxelCoordF.x, 0, Float(w - 1)))
        let cy = Int(simd_clamp(voxelCoordF.y, 0, Float(h - 1)))
        let cz = Int(simd_clamp(voxelCoordF.z, 0, Float(d - 1)))

        // Helper to read SDF with bounds clamping.
        func readSDF(_ x: Int, _ y: Int, _ z: Int) -> Float {
            let sx = max(0, min(w - 1, x))
            let sy = max(0, min(h - 1, y))
            let sz = max(0, min(d - 1, z))
            return sdfData[sz * w * h + sy * w + sx]
        }

        // Central differences along each axis.
        let dx = readSDF(cx + step, cy, cz) - readSDF(cx - step, cy, cz)
        let dy = readSDF(cx, cy + step, cz) - readSDF(cx, cy - step, cz)
        let dz = readSDF(cx, cy, cz + step) - readSDF(cx, cy, cz - step)

        let gradient = SIMD3<Float>(dx, dy, dz)
        let len = simd_length(gradient)
        return len > 1e-6 ? gradient / len : SIMD3<Float>(0, 0, 1)
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
            let oldId = ObjectIdentifier(oldest)
            debrisStates.removeValue(forKey: oldId)
            settlingCounters.removeValue(forKey: oldId)
            growthMultipliers.removeValue(forKey: oldId)
            spawnTimes.removeValue(forKey: oldId)
            oldest.removeFromParent()
        }

        // Offset spawn position outward from the volume center (origin).
        let outward = simd_length(position) > 0.001 ? simd_normalize(position) : SIMD3<Float>(0, 0, 1)
        let offsetPosition = position + outward * surfaceOffset

        let boxEntity = ModelEntity(mesh: mesh, materials: [material])
        boxEntity.name = "BoneDebris"

        let orientation = orientationAlongDirection(direction)

        // Initial scale = unit (mesh is already at debrisRadius).
        // debrisStretch elongates along Z. SDF modulates overall size.
        let sdfSizeScale = computeSDFSizeScale(at: position)
        let stretchScale = SIMD3<Float>(1.0, 1.0, Self.debrisStretch) * sdfSizeScale

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

        // Hide physics entities — the metaball mesh is the visual representation.
        boxEntity.components.set(OpacityComponent(opacity: 0.0))

        container.addChild(boxEntity)
        drawnEntities.append(boxEntity)

        // Initialize settling state, growth multiplier, and spawn time.
        let entityId = ObjectIdentifier(boxEntity)
        debrisStates[entityId] = .ejecting
        growthMultipliers[entityId] = stretchScale  // initial visual scale
        spawnTimes[entityId] = CACurrentMediaTime()

        // Queue ejection force: combine outward-from-surface (radial from volume
        // center) with opposite-drill-direction so debris flies toward the user.
        // SDF gradient is unreliable at freshly carved surfaces due to blit lag.
        let ejectionDir = simd_normalize(outward - direction)
        pendingEjections.append((entity: boxEntity, direction: ejectionDir, framesLeft: ejectionDelayFrames))

        // Queue for gradual growth from spawn scale to max.
        growingDebris.append((entity: boxEntity, baseScale: stretchScale, multiplier: SIMD3<Float>(1, 1, 1), lastTick: CACurrentMediaTime()))
    }

    /// Compute a size scale factor based on SDF distance at the spawn position.
    /// Debris closer to the surface (SDF ≈ 0) gets a larger scale.
    private func computeSDFSizeScale(at position: SIMD3<Float>) -> Float {
        guard hasSDF, !sdfData.isEmpty else { return 1.0 }

        let d = abs(sampleSDF(at: position))
        // Normalize distance: 0 at surface, 1 at adhesionMaxDistance.
        let t = simd_clamp(d / adhesionMaxDistance, 0, 1)
        // Interpolate: closer to surface = larger size.
        return simd_mix(sdfSizeScaleMax, sdfSizeScaleMin, t)
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
    /// Growth multipliers are stored in `growthMultipliers` dictionary and
    /// applied in uploadParticles — NOT on the entity transform, because
    /// RealityKit physics overwrites dynamic entity transforms each frame.
    private var growthLogCounter: Int = 0

    func processGrowth() {
        let now = CACurrentMediaTime()
        var stillGrowing: [(entity: ModelEntity, baseScale: SIMD3<Float>, multiplier: SIMD3<Float>, lastTick: TimeInterval)] = []

        for var entry in growingDebris {
            guard entry.entity.parent != nil else { continue }
            let id = ObjectIdentifier(entry.entity)

            let elapsed = now - entry.lastTick
            if elapsed >= growthInterval {
                entry.multiplier = simd_min(entry.multiplier + growthPerTick, growthMaxMultiplier)
                entry.lastTick = now
            }

            growthMultipliers[id] = entry.baseScale * entry.multiplier

            let reachedMax = entry.multiplier.x >= growthMaxMultiplier.x
                          && entry.multiplier.y >= growthMaxMultiplier.y
                          && entry.multiplier.z >= growthMaxMultiplier.z
            if !reachedMax {
                stillGrowing.append(entry)
            }
        }

        growthLogCounter += 1
        if growthLogCounter % 60 == 0 && !growingDebris.isEmpty {
            let sample = growingDebris[0]
            let id = ObjectIdentifier(sample.entity)
            let stored = growthMultipliers[id] ?? .zero
            print("[Growth] growing=\(growingDebris.count) mult=\(sample.multiplier) stored=\(stored) base=\(sample.baseScale)")
        }

        growingDebris = stillGrowing
    }

    // MARK: - SDF Adhesion Processing

    /// Call every frame to apply SDF-based adhesion forces and friction to all debris.
    /// Debris near the bone surface is pulled toward it and damped; debris far away
    /// falls freely under gravity.
    func processAdhesion() {
        guard hasSDF, !sdfData.isEmpty else { return }
        guard isEnabled else { return }

        for entity in drawnEntities {
            guard let modelEntity = entity as? ModelEntity,
                  modelEntity.parent != nil else { continue }

            let position = modelEntity.position(relativeTo: debrisContainer)

            // Sample SDF and compute gradient at debris position.
            let sdfValue = sampleSDF(at: position)
            let absDist = abs(sdfValue)

            // Only apply adhesion within range.
            guard absDist < adhesionMaxDistance else { continue }

            let surfaceNormal = sdfGradient(at: position)

            // Adhesion force: pull debris toward the surface.
            // Force direction: opposite to gradient (toward decreasing SDF = toward surface).
            // Strength scales with proximity: strongest at surface, fading at adhesionMaxDistance.
            let proximityFactor = 1.0 - (absDist / adhesionMaxDistance)
            let adhesionForce = -surfaceNormal * adhesionStickiness * proximityFactor
            modelEntity.addForce(adhesionForce, relativeTo: nil)

            // Friction/stiction: aggressively damp velocity when very close to surface.
            if absDist < adhesionFrictionThreshold {
                // Read current linear velocity and damp it.
                if var physics = modelEntity.components[PhysicsMotionComponent.self] {
                    physics.linearVelocity *= (1.0 - adhesionFrictionDamping)
                    // Also damp angular velocity to prevent spinning on surface.
                    physics.angularVelocity *= (1.0 - adhesionFrictionDamping * 0.5)
                    modelEntity.components.set(physics)
                }
            }
        }
    }

    // MARK: - Settling State Processing (Option C)

    /// Transitions debris through ejecting → settling → settled states.
    /// Settled debris has its physics body switched to kinematic, preventing drift.
    func processSettling() {
        guard isEnabled else { return }

        for entity in drawnEntities {
            guard let modelEntity = entity as? ModelEntity,
                  modelEntity.parent != nil else { continue }

            let id = ObjectIdentifier(modelEntity)
            let currentState = debrisStates[id] ?? .ejecting

            switch currentState {
            case .ejecting:
                // Check if ejection impulse has been applied (no longer in pendingEjections).
                let stillPending = pendingEjections.contains { $0.entity === modelEntity }
                if !stillPending {
                    debrisStates[id] = .settling
                    settlingCounters[id] = 0
                }

            case .settling:
                // Only freeze debris that is near the bone surface (SDF-based).
                // Prevents debris from freezing mid-air when adhesion friction
                // damps velocity before it reaches a resting surface.
                let position = modelEntity.position(relativeTo: debrisContainer)
                let sdfDist = abs(sampleSDF(at: position))
                let nearSurface = sdfDist < adhesionFrictionThreshold

                let velocity = modelEntity.components[PhysicsMotionComponent.self]?.linearVelocity ?? .zero
                let speed = simd_length(velocity)

                if speed < settlingVelocityThreshold && nearSurface {
                    let count = (settlingCounters[id] ?? 0) + 1
                    settlingCounters[id] = count
                    if count >= settlingFrameCount {
                        if var physics = modelEntity.components[PhysicsBodyComponent.self] {
                            physics.mode = .kinematic
                            modelEntity.components.set(physics)
                        }
                        if var motion = modelEntity.components[PhysicsMotionComponent.self] {
                            motion.linearVelocity = .zero
                            motion.angularVelocity = .zero
                            modelEntity.components.set(motion)
                        }
                        debrisStates[id] = .settled
                        settlingCounters.removeValue(forKey: id)
                    }
                } else {
                    settlingCounters[id] = 0
                }

            case .settled:
                // Already frozen — nothing to do here.
                // Tool displacement is handled in processToolDisplacement().
                break
            }
        }
    }

    // MARK: - Tool Displacement Processing (Option B)

    /// Pushes settled debris outward when the tool passes near them.
    /// Creates the "smearing" effect seen in real surgery where the burr
    /// drags and pushes accumulated slurry.
    func processToolDisplacement(toolPosition: SIMD3<Float>) {
        guard isEnabled else { return }

        let radiusSq = toolDisplacementRadius * toolDisplacementRadius

        for entity in drawnEntities {
            guard let modelEntity = entity as? ModelEntity,
                  modelEntity.parent != nil else { continue }

            let id = ObjectIdentifier(modelEntity)
            let state = debrisStates[id] ?? .ejecting

            // Only displace settled or settling debris.
            guard state == .settled || state == .settling else { continue }

            let debrisPos = modelEntity.position(relativeTo: debrisContainer)
            let delta = debrisPos - toolPosition
            let distSq = simd_length_squared(delta)

            guard distSq < radiusSq, distSq > 1e-8 else { continue }

            let dist = sqrt(distSq)
            let pushDir = delta / dist  // normalized direction away from tool
            let falloff = 1.0 - (dist / toolDisplacementRadius)  // stronger when closer

            if state == .settled {
                // Unfreeze temporarily: switch back to dynamic for the push.
                if var physics = modelEntity.components[PhysicsBodyComponent.self] {
                    physics.mode = .dynamic
                    modelEntity.components.set(physics)
                }
                // Reset to settling so it can re-settle after displacement.
                debrisStates[id] = .settling
                settlingCounters[id] = 0
            }

            let force = pushDir * toolDisplacementForce * falloff
            modelEntity.addForce(force, relativeTo: nil)
        }
    }

    // MARK: - Out-of-Bounds Culling

    /// Remove debris that has left the sculpting volume bounds.
    func cullOutOfBoundsDebris() {
        let bMin = volumeBoundsMin
        let bMax = volumeBoundsMax
        var i = 0
        while i < drawnEntities.count {
            let entity = drawnEntities[i]
            let pos = entity.position(relativeTo: debrisContainer)
            if pos.x < bMin.x || pos.y < bMin.y || pos.z < bMin.z ||
               pos.x > bMax.x || pos.y > bMax.y || pos.z > bMax.z {
                let id = ObjectIdentifier(entity)
                debrisStates.removeValue(forKey: id)
                settlingCounters.removeValue(forKey: id)
                growthMultipliers.removeValue(forKey: id)
                spawnTimes.removeValue(forKey: id)
                entity.removeFromParent()
                drawnEntities.remove(at: i)
            } else {
                i += 1
            }
        }
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
        debrisStates.removeAll()
        settlingCounters.removeAll()
        growthMultipliers.removeAll()
        spawnTimes.removeAll()
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
