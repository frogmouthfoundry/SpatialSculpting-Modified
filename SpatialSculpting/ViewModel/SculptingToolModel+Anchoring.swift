/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Functions related to anchoring with RealityKit and ARKit.
*/

import ARKit
import RealityKit
@preconcurrency import GameController

@MainActor let trackingStateColor: [AccessoryAnchor.TrackingState: UIColor] = [
    .positionOrientationTracked: .green,
    .orientationTracked: .yellow,
    .positionOrientationTrackedLowAccuracy: .orange,
    .untracked: .red
]

// Get the ARKit accessory anchor from a RealityKit AnchorEntity.
@MainActor func getAccessoryAnchor(entity: AnchorEntity) -> AccessoryAnchor? {
    if let accessoryAnchor = entity.components[ARKitAnchorComponent.self]?.anchor as? AccessoryAnchor {
        return accessoryAnchor
    }
    return nil
}

extension SculptingToolModel {

    // MARK: - Drill Shaft Collision Visual Feedback

    /// Walk the drill model entity tree and cache every ModelComponent's materials.
    func cacheDrillMaterials() {
        guard let drillModel = drillModelEntity else { return }
        var cache: [(Entity, [any RealityKit.Material])] = []
        func walk(_ entity: Entity) {
            if let model = entity.components[ModelComponent.self] {
                cache.append((entity, model.materials))
            }
            for child in entity.children {
                walk(child)
            }
        }
        walk(drillModel)
        _cachedDrillMaterials = cache
    }

    /// Tint the entire drill model red to indicate shaft collision.
    func tintDrillRed() {
        guard !_isDrillTintedRed else { return }
        _isDrillTintedRed = true
        guard let drillModel = drillModelEntity else { return }
        let redMaterial = SimpleMaterial(color: .red, roughness: 0.3, isMetallic: false)
        func walk(_ entity: Entity) {
            if var model = entity.components[ModelComponent.self] {
                model.materials = Array(repeating: redMaterial, count: model.materials.count)
                entity.components.set(model)
            }
            for child in entity.children {
                walk(child)
            }
        }
        walk(drillModel)
    }

    /// Restore original drill materials after shaft collision clears.
    func restoreDrillMaterials() {
        guard _isDrillTintedRed else { return }
        _isDrillTintedRed = false
        for (entity, materials) in _cachedDrillMaterials {
            if var model = entity.components[ModelComponent.self] {
                model.materials = materials
                entity.components.set(model)
            }
        }
    }

    // Add a visual tooltip to indicate where sculpting occurs.
    // Also add a tracking state indicator to indicate when tracking may be
    // failing due to reduced sensor coverage.
    @MainActor
    func addSculptingTooltip(to entity: AnchorEntity) {
        // Tooltip sphere removed — the drill ball now serves as the visual indicator.
        sculptingEntity = entity
    }
    
    /// Load the Drill.usdz model from the bundle and attach it to the anchor,
    /// together with a spinning drill ball at the tip.
    @MainActor
    func attachDrillModel(to anchor: AnchorEntity) async {
        // Load drill model from bundle
        guard let drillModel = try? await ModelEntity(named: "Drill") else {
            print("Failed to load Drill.usdz")
            return
        }

        // Scale so the drill is roughly 17.5 cm long (matching ToolChange_Drill)
        let targetLength: Float = 0.175
        let bounds = drillModel.visualBounds(relativeTo: nil)
        let modelLength = max(bounds.extents.x, max(bounds.extents.y, bounds.extents.z))
        if modelLength > 0 {
            let scaleFactor = targetLength / modelLength
            drillModel.scale = SIMD3<Float>(repeating: scaleFactor)
        }

        // Position so the tip of the model sits near the anchor origin.
        // The drill model extends along +Z; shift it so the front (min Z)
        // is 5 cm in front of the anchor.
        let scaledBounds = drillModel.visualBounds(relativeTo: nil)
        let center = scaledBounds.center
        let extents = scaledBounds.extents
        drillModel.position = SIMD3<Float>(
            -center.x,
            -center.y,
            -center.z + extents.z / 2 - 0.05
        )

        anchor.addChild(drillModel)
        drillModelEntity = drillModel

        // Cache original materials for shaft collision tint/restore.
        cacheDrillMaterials()

        // Compute the drill tip position in anchor-local space.
        // The positioning math above places the model's min-Z face at Z = -0.05.
        // In general: tipZ = drillModel.position.z + scaledBounds.min.z
        let tipZ = drillModel.position.z + scaledBounds.min.z
        // X/Y offset compensates for the drill overlay being slightly off the stylus axis.
        let tipPosition = SIMD3<Float>(-0.003, 0.001, tipZ) //orig(-0.005,0.001, tipZ)

        // Create spinning drill ball at the tip, centered on the shaft axis.
        let drillBall = DrillRotationComponent.createDrillBall(rpm: 400)
        drillBall.position = tipPosition
        anchor.addChild(drillBall)
        drillBallEntity = drillBall

        // Store the tip offset so updateSculptingTool() can use it for sculpting position.
        drillBallLocalOffset = tipPosition

        print("Drill model and rotating ball attached at tip \(tipPosition)")
    }
    
    // Anchor via AnchorEntity to a GCDevice.
    // Set up stylus or controller inputs.
    @MainActor
    func setupSpatialAccessory(device: GCDevice, hapticsModel: HapticsModel) async throws {
        let source = try await AnchoringComponent.AccessoryAnchoringSource(device: device)
        
        guard let location = source.locationName(named: "aim") ?? source.locationName(named: "tip") else {
            return
        }
        
        let sculptingEntity = AnchorEntity(.accessory(from: source, location: location),
                                           trackingMode: .predicted,
                                           physicsSimulation: .none)
        
        sculptingEntity.name = "SculptingEntity"
        
        rootEntity?.addChild(sculptingEntity)
        
        addSculptingTooltip(to: sculptingEntity)
        
        // Attach the drill overlay model and spinning ball
        await attachDrillModel(to: sculptingEntity)
        
        // Set up inputs to take in controller or stylus style inputs.
        if let stylus = device as? GCStylus {
            setupStylusInputs(stylus: stylus, hapticsModel: hapticsModel)
        } else if let controller = device as? GCController {
            setupControllerInputs(controller: controller, hapticsModel: hapticsModel)
        }

        // Haptics disabled for now
        // hapticsModel.startIdleVibration()
    }
    
}
