/*
Abstract:
Manages the continuous drawing of thin bone debris box volumes when the
sculpting tool tip contacts the sculpted volume. Each box is 5 mm × 5 mm × 2 mm
(width × height × depth). Boxes are placed at intervals equal to or smaller
than the box depth so the debris trail appears seamless.

Only bone debris entities receive custom gravity — the main sculpting volume
is never affected.
*/

import RealityKit
import UIKit
import Combine

/// Manages the lifecycle of drawn bone debris box entities in the scene.
@MainActor @Observable
final class BoneDebrisManager {
    
    // MARK: - Constants
    
    /// Box dimensions in meters (width, height, depth).
    static let boxWidth:  Float = 0.005  // 5 mm
    static let boxHeight: Float = 0.005  // 5 mm
    static let boxDepth:  Float = 0.020 // 10 mm
    
    /// Maximum spacing between consecutive boxes to guarantee a seamless trail.
    private let stepDistance: Float = 0.002
    
    /// Minimum distance the tool must move before placing a new box.
    private let minimumMovement: Float = 0.0005 // 0.5 mm
    
    /// The custom gravity vector for bone debris (negative-Z at 9.81 m/s²).
    static let debrisGravity: SIMD3<Float> = SIMD3<Float>(0, 0, -9.81)
    
    // MARK: - State
    
    /// Whether bone debris drawing is currently enabled by the user.
    var isEnabled: Bool = true
    
    /// Whether custom negative-Z gravity is applied to the bone debris boxes.
    /// When toggled, all existing and future debris boxes are updated.
    /// Only debris entities are affected — the main sculpting volume is never changed.
    var isGravityEnabled: Bool = false {
        didSet {
            guard oldValue != isGravityEnabled else { return }
            applyGravityStateToAllDebris()
        }
    }
    
    /// Whether the tool is actively drawing (button held + inside volume).
    private(set) var isDrawing: Bool = false
    
    /// The previous world-space position where a box was placed.
    private var previousPosition: SIMD3<Float>?
    
    /// The previous drawing direction (used to orient boxes along the trail).
    private var previousDirection: SIMD3<Float>?
    
    /// Shared mesh resource for all debris boxes (created once, reused).
    private var boxMesh: MeshResource?
    
    /// Cached material loaded from the bundle (or the fallback color).
    private var debrisMaterial: (any Material)?
    
    /// Optional physics material.
    private var physicsMaterial: PhysicsMaterialResource?
    
    /// The parent entity to which all drawn debris boxes are added.
    weak var rootEntity: Entity?
    
    /// A dedicated parent entity for all debris children (makes cleanup easy).
    private var debrisContainer: Entity?
    
    /// The bounding box min of the sculpting volume in local (root-entity) space.
    private let volumeMin: SIMD3<Float>
    
    /// The bounding box max of the sculpting volume in local (root-entity) space.
    private let volumeMax: SIMD3<Float>
    
    /// All entities that have been drawn in the current session (for undo/clear).
    private(set) var drawnEntities: [Entity] = []
    
    // MARK: - Initialiser
    
    /// - Parameters:
    ///   - volumeMin: Lower corner of the sculpted volume bounding box (local space).
    ///   - volumeMax: Upper corner of the sculpted volume bounding box (local space).
    init(volumeMin: SIMD3<Float> = SIMD3<Float>(repeating: -0.5),
         volumeMax: SIMD3<Float> = SIMD3<Float>(repeating:  0.5)) {
        self.volumeMin = volumeMin
        self.volumeMax = volumeMax
    }
    
    // MARK: - Setup
    
    /// Call once after the root entity is available (e.g. inside `RealityView`).
    func setup(rootEntity: Entity) {
        self.rootEntity = rootEntity
        
        // Apply a PhysicsSimulationComponent with custom -Z gravity to the ROOT
        // entity. This ensures all physics bodies (mesh chunk statics + debris
        // dynamics) share the SAME simulation and can collide with each other.
        // Static bodies are unaffected by gravity, so only the dynamic debris
        // boxes will actually fall in -Z.
        var physicsSimulation = PhysicsSimulationComponent()
        physicsSimulation.gravity = Self.debrisGravity
        rootEntity.components.set(physicsSimulation)
        
        // Create a plain container entity for organizational cleanup
        // (no PhysicsSimulationComponent — everything uses root's simulation).
        let container = Entity()
        container.name = "BoneDebrisContainer"
        
        rootEntity.addChild(container)
        self.debrisContainer = container
        
        // Pre-create the shared box mesh.
        boxMesh = .generateBox(
            width:  Self.boxWidth,
            height: Self.boxHeight,
            depth:  Self.boxDepth,
            cornerRadius: 0.0002  // Very slight rounding for a polished look.
        )
        
        // Load material and physics material.
        debrisMaterial = BoneDebrisMaterialLoader.loadDebrisMaterial()
        physicsMaterial = BoneDebrisMaterialLoader.loadDebrisPhysicsMaterial()
    }
    
    // MARK: - Per-Frame Update
    
    /// Called every frame from the scene update loop.
    /// - Parameters:
    ///   - toolPosition: The sculpting tool tip position in the root entity's local space.
    ///   - isSculptingActive: Whether the user is currently pressing the sculpt button.
    func update(toolPosition: SIMD3<Float>, isSculptingActive: Bool) {
        guard isEnabled else {
            endTrail()
            return
        }
        
        let insideVolume = isInsideVolume(position: toolPosition)
        
        if isSculptingActive && insideVolume {
            if !isDrawing {
                beginTrail(at: toolPosition)
            } else {
                continueTrail(to: toolPosition)
            }
        } else {
            if isDrawing {
                endTrail()
            }
        }
    }
    
    // MARK: - Trail Lifecycle
    
    /// Starts a new debris trail at the given position.
    private func beginTrail(at position: SIMD3<Float>) {
        isDrawing = true
        previousPosition = position
        previousDirection = nil
        
        // Place the first box.
        placeBox(at: position, direction: SIMD3<Float>(0, 0, 1))
    }
    
    /// Continues the trail by interpolating boxes from the previous position to the current one.
    private func continueTrail(to position: SIMD3<Float>) {
        guard let prev = previousPosition else {
            previousPosition = position
            return
        }
        
        let delta = position - prev
        let distance = simd_length(delta)
        
        // Don't place boxes if the tool hasn't moved enough.
        guard distance >= minimumMovement else { return }
        
        let direction = simd_normalize(delta)
        
        // Calculate how many boxes we need to fill the gap.
        let steps = max(1, Int(ceil(distance / stepDistance)))
        
        for i in 1...steps {
            let t = Float(i) / Float(steps)
            let interpolatedPosition = simd_mix(prev, position, SIMD3<Float>(repeating: t))
            
            // Only place if the interpolated position is still inside the volume.
            if isInsideVolume(position: interpolatedPosition) {
                placeBox(at: interpolatedPosition, direction: direction)
            }
        }
        
        previousPosition = position
        previousDirection = direction
    }
    
    /// Ends the current debris trail.
    private func endTrail() {
        isDrawing = false
        previousPosition = nil
        previousDirection = nil
    }
    
    // MARK: - Box Placement
    
    /// Places a single bone debris box entity at the given position, oriented along the given direction.
    private func placeBox(at position: SIMD3<Float>, direction: SIMD3<Float>) {
        guard let mesh = boxMesh,
              let material = debrisMaterial,
              let container = debrisContainer else { return }
        
        let boxEntity = ModelEntity(mesh: mesh, materials: [material])
        boxEntity.name = "BoneDebrisBox"
        
        // Orient the box so that its thin (depth) axis aligns with the drawing direction.
        let orientation = orientationAlongDirection(direction)
        boxEntity.transform = Transform(
            scale: SIMD3<Float>(repeating: 1.0),
            rotation: orientation,
            translation: position
        )
        
        // Set up collision and physics.
        let boxShape = ShapeResource.generateBox(
            width:  Self.boxWidth,
            height: Self.boxHeight,
            depth:  Self.boxDepth
        )
        boxEntity.components.set(CollisionComponent(shapes: [boxShape]))
        
        if isGravityEnabled {
            // Dynamic mode: the root's PhysicsSimulationComponent provides
            // custom -Z gravity to all dynamic children automatically.
            boxEntity.components.set(PhysicsBodyComponent(
                shapes: [boxShape],
                mass: 0.001,
                material: physicsMaterial ?? .default,
                mode: .dynamic
            ))
        } else {
            // Static mode: boxes stay in place, no gravity.
            boxEntity.components.set(PhysicsBodyComponent(
                shapes: [boxShape],
                mass: 0.001,
                material: physicsMaterial ?? .default,
                mode: .static
            ))
        }
        
        container.addChild(boxEntity)
        drawnEntities.append(boxEntity)
    }
    
    /// Computes a quaternion that rotates the default forward direction (-Z)
    /// to align with the given `direction` vector.
    private func orientationAlongDirection(_ direction: SIMD3<Float>) -> simd_quatf {
        let dir = simd_length(direction) > 0.0001 ? simd_normalize(direction) : SIMD3<Float>(0, 0, 1)
        
        // Default forward in RealityKit is -Z.
        let forward = SIMD3<Float>(0, 0, -1)
        
        let dot = simd_dot(forward, dir)
        
        // Handle near-parallel and near-anti-parallel cases.
        if dot > 0.9999 {
            return simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        } else if dot < -0.9999 {
            return simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 1, 0))
        }
        
        let axis = simd_normalize(simd_cross(forward, dir))
        let angle = acos(simd_clamp(dot, -1.0, 1.0))
        return simd_quatf(angle: angle, axis: axis)
    }
    
    // MARK: - Volume Hit Test
    
    /// Returns `true` if the position lies within the sculpting volume bounding box.
    func isInsideVolume(position: SIMD3<Float>) -> Bool {
        return position.x >= volumeMin.x && position.x <= volumeMax.x &&
               position.y >= volumeMin.y && position.y <= volumeMax.y &&
               position.z >= volumeMin.z && position.z <= volumeMax.z
    }
    
    // MARK: - Cleanup
    
    /// Removes all drawn bone debris boxes from the scene.
    func clearAllDebris() {
        for entity in drawnEntities {
            entity.removeFromParent()
        }
        drawnEntities.removeAll()
        endTrail()
    }
    
    /// Removes the most recent continuous trail (undo last stroke).
    func undoLastStroke(count: Int = 50) {
        let removeCount = min(count, drawnEntities.count)
        let entitiesToRemove = drawnEntities.suffix(removeCount)
        for entity in entitiesToRemove {
            entity.removeFromParent()
        }
        drawnEntities.removeLast(removeCount)
    }
    
    // MARK: - Gravity Toggle
    
    /// Switches all existing bone debris boxes between static (no gravity) and
    /// dynamic (affected by the root's -Z PhysicsSimulationComponent gravity).
    /// Only affects bone debris entities — the main sculpting volume is never touched.
    private func applyGravityStateToAllDebris() {
        for entity in drawnEntities {
            guard let modelEntity = entity as? ModelEntity else { continue }
            
            let boxShape = ShapeResource.generateBox(
                width:  Self.boxWidth,
                height: Self.boxHeight,
                depth:  Self.boxDepth
            )
            
            if isGravityEnabled {
                // Switch to dynamic mode — root's PhysicsSimulationComponent
                // provides custom -Z gravity automatically.
                modelEntity.components.set(PhysicsBodyComponent(
                    shapes: [boxShape],
                    mass: 0.001,
                    material: physicsMaterial ?? .default,
                    mode: .dynamic
                ))
            } else {
                // Switch back to static mode.
                modelEntity.components.set(PhysicsBodyComponent(
                    shapes: [boxShape],
                    mass: 0.001,
                    material: physicsMaterial ?? .default,
                    mode: .static
                ))
            }
        }
    }
}
