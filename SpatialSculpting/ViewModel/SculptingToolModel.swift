/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
A model of the sculpting.
*/

import SwiftUI
import RealityKit
import ARKit

@MainActor @Observable
final class SculptingToolModel {
    // Min and max radii of the sculpting tool.
    let minRadius: Float = 0.002
    let maxRadius: Float = 0.5

    var rootEntity: Entity? = nil // The root entity in the RealityView.

    let sculptingTool = Entity(components: [ModelComponent(mesh: .generateSphere(radius: 0.001), materials: [SimpleMaterial()]),
                                            OpacityComponent(opacity: 0.01)])
    
    var sculptingEntity: AnchorEntity? = nil
    var trackingStateIndicator: ModelEntity? = nil
    
    var additiveIcon: Entity? = nil
    var subtractiveIcon: Entity? = nil
    var enlargeIcon: Entity? = nil
    var reduceIcon: Entity? = nil

    var hapticsModel: HapticsModel? = nil

    // Sculpt material management
    var loadedRoughnessValue: Float = 0.5
    var loadedAlbedoTint: SIMD3<Float> = SIMD3<Float>(repeating: 1.0)
    private var bundledSculptMaterial: (any RealityKit.Material)? = nil
    private var bundledMaterialLoadAttempted: Bool = false
    private let sculptRealityAssetName = "SculptGraph"
    private let sculptProxyEntityName = "SculptGraphProxy"

    // Drill overlay entities
    var drillModelEntity: Entity? = nil
    var drillBallEntity: ModelEntity? = nil

    /// Anchor-local offset from the accessory origin to the drill ball center.
    /// Computed dynamically from the USDZ model bounds in `attachDrillModel`.
    var drillBallLocalOffset = SIMD3<Float>(0, 0, -0.04)

    // Shaft collision detection
    let shaftCollisionDetector = ShaftCollisionDetector()
    /// Cached original materials for the drill model (populated after attach).
    var _cachedDrillMaterials: [(Entity, [any RealityKit.Material])] = []
    /// Whether the drill is currently tinted red due to shaft collision.
    var _isDrillTintedRed: Bool = false

    // Tracks carving state for logging and particle bursts
    private var wasCarving: Bool = false

    /// Pre-loaded BoneDust template entity — cloned on each burst to avoid disk I/O.
    private var boneDustTemplate: Entity? = nil
    
    /// Manages the drawing of bone debris box volumes when the tool contacts the sculpted volume.
    let boneDebrisManager = BoneDebrisManager()
    
    /// Manages collision generation for the sculpting volume via CPU marching cubes.
    let collisionManager = CollisionManager()

    /// 64^3 metaball density grid for unified debris visualization.
    let boneSlurryGrid: BoneSlurryGrid? = BoneSlurryGrid()

    // MARK: - Sculpt Material

    /// Attempt to load the SculptGraphMaterial from Reality Kit assets once.
    func prepareBundledSculptMaterialIfNeeded() {
#if os(visionOS)
        guard !bundledMaterialLoadAttempted else { return }
        bundledMaterialLoadAttempted = true
        Task { @MainActor in
            do {
                let root = try await Entity(named: sculptRealityAssetName, in: .main)
                guard let proxy = root.findEntity(named: sculptProxyEntityName),
                      let model = proxy.components[ModelComponent.self],
                      let material = model.materials.first else {
                    print("Loaded \(sculptRealityAssetName).reality, but \(sculptProxyEntityName) material was not found.")
                    updateSculptMeshMaterials()
                    return
                }
                bundledSculptMaterial = material
                print("Loaded sculpt material from \(sculptRealityAssetName).reality entity \(sculptProxyEntityName).")
            } catch {
                print("Bundled sculpt material not loaded; using lit PBR fallback. Error: \(error)")
            }
            updateSculptMeshMaterials()
        }
#endif
    }

    /// Build the best available material for mesh chunks.
    func makeSculptMaterial() -> any RealityKit.Material {
#if os(visionOS)
        if let loadedMaterial = bundledSculptMaterial {
            return loadedMaterial
        }
        var material = PhysicallyBasedMaterial()
        let tint = UIColor(red: CGFloat(max(0, min(1, loadedAlbedoTint.x))),
                           green: CGFloat(max(0, min(1, loadedAlbedoTint.y))),
                           blue: CGFloat(max(0, min(1, loadedAlbedoTint.z))),
                           alpha: 1.0)
        material.baseColor = .init(tint: tint)
        material.roughness = .init(scale: max(0.05, min(1.0, loadedRoughnessValue)))
        material.metallic = .init(scale: 0.0)
        return material
#else
        do {
            var material = try CustomMaterial(surfaceShader: .named("sculptSurfaceShader"),
                                              geometryModifier: nil,
                                              lightingModel: .lit)
            material.custom.value = SIMD4<Float>(max(0.04, min(1.0, loadedRoughnessValue)), 0, 0, 0)
            return material
        } catch {
            print("Failed to create sculpt custom material (falling back to SimpleMaterial): \(error)")
            return SimpleMaterial(color: .white,
                                  roughness: max(0.05, min(1.0, loadedRoughnessValue)),
                                  isMetallic: false)
        }
#endif
    }

    /// Refresh mesh chunk materials after importing package data.
    func updateSculptMeshMaterials() {
        guard let rootEntity = rootEntity else { return }
        let material = makeSculptMaterial()
        for child in rootEntity.children where child.name == "SculptMeshChunk" {
            if var model = child.components[ModelComponent.self] {
                model.materials = [material]
                child.components.set(model)
            }
        }
    }

    // Raycast from the accessory to the add/subtract elements of the toolbar to swap to
    // the different controls.
    func selectToolbarElement(sculptingEntity: AnchorEntity) {
        guard let scene = sculptingEntity.scene else {
            return
        }
        let raycastOrigin = sculptingEntity.position(relativeTo: nil)
        let raycastForward = -simd_make_float3(sculptingEntity.transformMatrix(relativeTo: nil).columns.2)
        if let raycastHit = scene.raycast(origin: raycastOrigin, direction: raycastForward, length: 10, relativeTo: nil).first {
            if raycastHit.entity.name == "AdditiveIcon" {
                sculptingTool.components[SculptingToolComponent.self]?.mode = .add
            } else if raycastHit.entity.name == "SubtractiveIcon" {
                sculptingTool.components[SculptingToolComponent.self]?.mode = .remove
            } else if raycastHit.entity.name == "EnlargeIcon" {
                if let radius = sculptingTool.components[SculptingToolComponent.self]?.radius {
                    sculptingTool.components[SculptingToolComponent.self]?.radius = simd_clamp(radius + 0.01, minRadius, maxRadius)
                }
            } else if raycastHit.entity.name == "ReduceIcon" {
                if let radius = sculptingTool.components[SculptingToolComponent.self]?.radius {
                    sculptingTool.components[SculptingToolComponent.self]?.radius = simd_clamp(radius - 0.01, minRadius, maxRadius)
                }
            }
        }
    }
    
    // Display the toolbar, offsetting based on the hand holding the accessory.
    @MainActor
    func displayToolbar(transform: Transform, accessoryAnchor: AccessoryAnchor) {
        guard let chirality = accessoryAnchor.heldChirality else {
            return
        }
        
        guard let additiveIcon = additiveIcon,
              let subtractiveIcon = subtractiveIcon,
              let reduceIcon = reduceIcon,
              let enlargeIcon = enlargeIcon else {
            return
        }
                
        let zTranslation: Float = -0.1
        let xTranslation: Float = {
            switch chirality {
            case .left:
                return 0.05 // Display a toolbar 5 cm to the right.
            case .right:
                return -0.05 // Display a toolbar 5 cm to the left.
            default:
                return 0.0 // Do not offset the toolbar.
            }
        }()
        
        subtractiveIcon.isEnabled = true
        additiveIcon.isEnabled = true
        reduceIcon.isEnabled = true
        enlargeIcon.isEnabled = true
        
        additiveIcon.position = transform.translation + SIMD3<Float>(xTranslation, 0.05, zTranslation)
        subtractiveIcon.position = transform.translation + SIMD3<Float>(xTranslation, 0, zTranslation)
        reduceIcon.position = transform.translation + SIMD3<Float>(xTranslation, -0.05, zTranslation)
        enlargeIcon.position = transform.translation + SIMD3<Float>(xTranslation, -0.10, zTranslation)
    }
    
    func handlePalettePress(pressed: Bool) {
        guard let sculptingEntity = sculptingEntity else {
            return
        }
        
        guard let accessoryAnchor = getAccessoryAnchor(entity: sculptingEntity) else {
            return
        }
        
        if pressed {
            Task { @MainActor in
                displayToolbar(transform: sculptingTool.transform, accessoryAnchor: accessoryAnchor)
            }
        } else {
            selectToolbarElement(sculptingEntity: sculptingEntity)
            additiveIcon?.isEnabled = false
            subtractiveIcon?.isEnabled = false
            reduceIcon?.isEnabled = false
            enlargeIcon?.isEnabled = false
        }
    }
    
    // Show an indicator when tracking is non 6 DOF.
    // Gets an ARKit AccessoryAnchor from a RealityKit AnchorEntity
    // and extracts relevant interaction data for warning of possible bad data.
    func updateTrackingStateIndicatorIfDirty(sculptingEntity: AnchorEntity) {
        guard let accessoryAnchor = getAccessoryAnchor(entity: sculptingEntity) else {
            return
        }
        
        if var sculptingToolComponent = sculptingTool.components[SculptingToolComponent.self] {
            if sculptingToolComponent.trackingState != accessoryAnchor.trackingState {
                sculptingToolComponent.trackingState = accessoryAnchor.trackingState
                // If we have lost 6 DOF tracking, display the tracking warning
                if accessoryAnchor.trackingState != .positionOrientationTracked {
                    let trackingColor = [SimpleMaterial(color: trackingStateColor[sculptingToolComponent.trackingState]!, isMetallic: false)]
                    trackingStateIndicator?.isEnabled = true
                    trackingStateIndicator?.components[ModelComponent.self]?.materials = trackingColor
                } else {
                    trackingStateIndicator?.isEnabled = false
                }
            }
        }
    }
    
    // Every frame, update the interactions with in-app content
    // from the sculpting tool to the virtual clay.
    func updateSculptingTool() {
        // Process collision updates before the sculptingEntity guard
        // so initial/scheduled regeneration works even without a connected stylus.
        collisionManager.processUpdatesIfNeeded()

        // Process pending debris ejection forces (frame countdown).
        boneDebrisManager.processPendingEjections()

        // Process debris growth animation.
        boneDebrisManager.processGrowth()

        // Process SDF adhesion forces on all debris.
        boneDebrisManager.processAdhesion()

        // Request periodic SDF blit for debris adhesion.
        boneDebrisManager.requestSdfBlitIfNeeded()

        // If collision manager needs a blit, set the request on the component.
        if collisionManager.blitRequested,
           let stagingTexture = collisionManager.collisionStagingTexture {
            collisionManager.blitRequested = false
            let manager = collisionManager
            sculptingTool.components[SculptingToolComponent.self]?.collisionBlitRequest = (
                stagingTexture,
                { @Sendable in manager.onBlitComplete() }
            )
        }

        // If debris manager needs a blit, set the request on the component.
        if boneDebrisManager.sdfBlitRequested,
           let stagingTexture = boneDebrisManager.debrisStagingTexture {
            boneDebrisManager.sdfBlitRequested = false
            let manager = boneDebrisManager
            sculptingTool.components[SculptingToolComponent.self]?.debrisBlitRequest = (
                stagingTexture,
                { @Sendable in manager.onSdfBlitComplete() }
            )
        }

        // Upload debris particle data to bone slurry grid for metaball visualization.
        if let root = rootEntity {
            boneSlurryGrid?.uploadParticles(from: boneDebrisManager,
                                            drillPosition: sculptingTool.position,
                                            rootEntity: root)
        }

        guard let sculptingEntity = sculptingEntity else {
            return
        }
        guard let rootEntity = rootEntity, let matrix = try? sculptingEntity.transform(from: rootEntity) else {
            return
        }
        // Update the sculpting tool with the sculpting tool's transform.
        // This ensures it can carve into the correct part of virtual clay.
        sculptingTool.transform = Transform(matrix: simd_float4x4(matrix))
        
        // Offset sculpting position to match the drill ball tip (centered on shaft axis).
        // drillBallLocalOffset is computed from the USDZ bounds in attachDrillModel().
        let rotatedOffset = sculptingTool.transform.rotation.act(drillBallLocalOffset)
        sculptingTool.position += rotatedOffset

        // --- Shaft collision detection ---
        // The shaft extends from the tip backward along the drill's local +Z axis
        // (the USDZ model body goes in +Z from the tip).
        let shaftDirection = sculptingTool.transform.rotation.act(SIMD3<Float>(0, 0, 1))
        let shaftResult = shaftCollisionDetector.test(
            tipPosition: sculptingTool.position,
            shaftDirection: simd_normalize(shaftDirection),
            collisionManager: collisionManager
        )

        if shaftResult.isColliding {
            // Physical blocking: push the tool out of the volume.
            sculptingTool.position += shaftResult.correctionVector
            // Visual feedback: tint drill red.
            tintDrillRed()
            // Haptic feedback: sharp warning pulse.
            hapticsModel?.startShaftCollisionFeedback()
        } else {
            // Clear visual and haptic warnings.
            restoreDrillMaterials()
            hapticsModel?.stopShaftCollisionFeedback()
        }

        // Always sculpt when the device is present
        sculptingTool.components[SculptingToolComponent.self]?.isActive = true
        
        // Log carving state using the SDF value sampled on the GPU.
        // SDF <= 0 means the tool is at or inside the mesh surface.
        if let sculptor = sculptingTool.components[SculptingToolComponent.self]?.sculptor {
            let sdf = sculptor.lastSampledSDF
            let isCarving = sdf <= 0
            if isCarving != wasCarving {
                print(isCarving ? "[Drill] Carving" : "[Drill] Idle")
                if isCarving {
                    triggerBoneDustBurst()
                    boneDebrisManager.update(sculptingTool: sculptingTool)
                }
                wasCarving = isCarving
            }
            // Throttled collision update while carving.
            if isCarving {
                collisionManager.markDirty()
            }
        }
        
        /*
        // --- Bone debris update ---
        // Read the current active state from the sculpting tool component.
        let isActive = sculptingTool.components[SculptingToolComponent.self]?.isActive ?? false
        boneDebrisManager.update(sculptingTool: sculptingTool)
         */
    }
    
    // Add the attachment to the root entity.
    func addEntityAttachmentToRoot(entity: Entity, name: String) {
        entity.position = .zero
        entity.isEnabled = false
        entity.components.set(CollisionComponent(shapes: [.generateBox(size: .init(repeating: 0.05))]))
        entity.name = name
        rootEntity?.addChild(entity)
    }

    // MARK: - Bone Dust Particles

    /// Pre-load the BoneDust template once. Call during setup.
    func preloadBoneDust() {
        Task { @MainActor in
            if let template = try? await Entity(named: "BoneDust") {
                self.boneDustTemplate = template
            } else {
                print("[BoneDust] Failed to pre-load BoneDust.usdz")
            }
        }
    }

    /// Spawn a clone of the pre-loaded BoneDust entity at the drill tip, let it play, then destroy it.
    private func triggerBoneDustBurst() {
        guard let root = rootEntity,
              let template = boneDustTemplate else { return }
        let dustEntity = template.clone(recursive: true)
        dustEntity.position = sculptingTool.position
        root.addChild(dustEntity)

        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(1500))
            dustEntity.removeFromParent()
        }
    }

}
