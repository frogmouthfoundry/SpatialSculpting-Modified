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

    nonisolated func spatialAccessoryDebugDescription(for device: GCDevice) -> String {
        let id = ObjectIdentifier(device)
        let kind = device is GCStylus ? "stylus" : "controller"
        let category = device.productCategory
        let vendor = device.vendorName ?? "unknown-vendor"
        return "\(kind) vendor=\(vendor) category=\(category) objectID=\(id)"
    }

    func removeSpatialAccessoryAnchor(for accessoryID: ObjectIdentifier) {
        print("[Accessory] Removing anchor objectID=\(accessoryID) activeID=\(String(describing: activeSpatialAccessoryID)) activeKind=\(String(describing: activeSpatialAccessoryKind))")
        if let anchor = spatialAccessoryAnchors[accessoryID] {
            anchor.removeFromParent()
            if sculptingEntity === anchor {
                sculptingEntity = nil
                drillModelEntity = nil
                drillBallEntity = nil
                removeShaftCollisionSnapshotDrill()
                drillModelBaseLocalTransform = nil
                drillModelDefaultLocalTransform = nil
                drillBallDefaultLocalTransform = nil
            }
        }
        spatialAccessoryAnchors.removeValue(forKey: accessoryID)
        configuredSpatialAccessoryIDs.remove(accessoryID)
        if activeSpatialAccessoryID == accessoryID {
            activeSpatialAccessoryID = nil
            activeSpatialAccessoryKind = nil
        }
    }

    func invalidateAccessoryTrackingState() {
        print("[Accessory] Invalidating accessory tracking state")
        let accessoryIDs = Array(spatialAccessoryAnchors.keys)
        for accessoryID in accessoryIDs {
            removeSpatialAccessoryAnchor(for: accessoryID)
        }
        removeShaftCollisionSnapshotDrill()
        configuredSpatialAccessoryIDs.removeAll()
        spatialAccessoryAnchors.removeAll()
        activeSpatialAccessoryID = nil
        activeSpatialAccessoryKind = nil
        trackingStateIndicator?.isEnabled = false
        clearCarvingAndTrackingRuntimeState()
        hapticsModel?.stopSculptVibration()
        hapticsModel?.stopShaftCollisionFeedback()
        updateDrillCarvingAudio(isCarving: false)
        updateDrillTrackingAudio(for: .untracked)
        sculptingTool.components[SculptingToolComponent.self]?.isActive = false
        resetShaftCollisionStateIfNeeded()
    }

    private var shaftCollisionWarningUIColor: UIColor {
        UIColor(red: 1.0, green: 0.76, blue: 0.18, alpha: 1.0)
    }

    private var anatomyHazardFlashUIColor: UIColor {
        .red
    }

    private var activeWarningUIColor: UIColor? {
        if isAnatomyHazardFlashActive {
            return anatomyHazardFlashUIColor
        }
        if isShaftCollisionWarningActive {
            return shaftCollisionWarningUIColor
        }
        return nil
    }

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
        if let drillBall = drillBallEntity,
           let model = drillBall.components[ModelComponent.self] {
            cache.append((drillBall, model.materials))
        }
        _cachedDrillMaterials = cache
    }

    /// Tint the drill model to the active warning state.
    func tintDrillRed() {
        isShaftCollisionWarningActive = true
        applyDrillWarningAppearance()
    }

    /// Restore original drill materials after shaft collision clears.
    func restoreDrillMaterials() {
        isShaftCollisionWarningActive = false
        applyDrillWarningAppearance()
    }

    private func setEntityTreeOpacity(_ entity: Entity, opacity: Float) {
        entity.components.set(OpacityComponent(opacity: opacity))
        for child in entity.children {
            setEntityTreeOpacity(child, opacity: opacity)
        }
    }

    func setLiveDrillOverlayVisible(_ isVisible: Bool) {
        let opacity: Float = isVisible ? 1.0 : 0.0
        if let drillModelEntity {
            setEntityTreeOpacity(drillModelEntity, opacity: opacity)
        }
        if let drillBallEntity {
            setEntityTreeOpacity(drillBallEntity, opacity: opacity)
        }
    }

    private func applyWarningTint(to entity: Entity, color: UIColor) {
        let warningMaterial = SimpleMaterial(color: color, roughness: 0.3, isMetallic: false)
        if var model = entity.components[ModelComponent.self] {
            model.materials = Array(repeating: warningMaterial, count: max(model.materials.count, 1))
            entity.components.set(model)
        }
        for child in entity.children {
            applyWarningTint(to: child, color: color)
        }
    }

    private func removeNamedDescendants(from entity: Entity, named targetName: String) {
        for child in Array(entity.children) {
            if child.name == targetName {
                child.removeFromParent()
            } else {
                removeNamedDescendants(from: child, named: targetName)
            }
        }
    }

    func removeShaftCollisionSnapshotDrill() {
        shaftCollisionSnapshotRoot?.removeFromParent()
        shaftCollisionSnapshotRoot = nil
        shaftCollisionSnapshotModel = nil
        shaftCollisionSnapshotBall = nil
        isShaftCollisionSnapshotVisible = false
        setLiveDrillOverlayVisible(true)
    }

    func rebuildShaftCollisionSnapshotDrillIfNeeded() {
        guard let root = rootEntity,
              let drillModelEntity,
              let drillBallEntity else { return }

        removeShaftCollisionSnapshotDrill()

        let snapshotRoot = Entity()
        snapshotRoot.name = "ShaftCollisionSnapshotRoot"
        snapshotRoot.isEnabled = false

        let snapshotModel = drillModelEntity.clone(recursive: true)
        let snapshotBall = drillBallEntity.clone(recursive: true)

        removeNamedDescendants(from: snapshotModel, named: "DrillTrackingAudioEmitter")
        removeNamedDescendants(from: snapshotModel, named: "DrillContactAudioEmitter")
        let initialColor = activeWarningUIColor ?? shaftCollisionWarningUIColor
        applyWarningTint(to: snapshotModel, color: initialColor)
        applyWarningTint(to: snapshotBall, color: initialColor)
        if var rotation = snapshotBall.components[DrillRotationComponent.self] {
            rotation.rpm = 0
            snapshotBall.components.set(rotation)
        }

        snapshotRoot.addChild(snapshotModel)
        snapshotRoot.addChild(snapshotBall)
        root.addChild(snapshotRoot)

        shaftCollisionSnapshotRoot = snapshotRoot
        shaftCollisionSnapshotModel = snapshotModel
        shaftCollisionSnapshotBall = snapshotBall
    }

    private func applyShaftCollisionSnapshotWarningAppearance() {
        guard isShaftCollisionSnapshotVisible else { return }
        let desiredColor = activeWarningUIColor ?? shaftCollisionWarningUIColor
        if let snapshotModel = shaftCollisionSnapshotModel {
            applyWarningTint(to: snapshotModel, color: desiredColor)
        }
        if let snapshotBall = shaftCollisionSnapshotBall {
            applyWarningTint(to: snapshotBall, color: desiredColor)
        }
    }

    func showShaftCollisionSnapshot(modelTransform: Transform?, ballTransform: Transform?) {
        if shaftCollisionSnapshotRoot == nil {
            rebuildShaftCollisionSnapshotDrillIfNeeded()
        }

        if let modelTransform {
            shaftCollisionSnapshotModel?.transform = modelTransform
        }
        if let ballTransform {
            shaftCollisionSnapshotBall?.transform = ballTransform
        }
        shaftCollisionSnapshotRoot?.isEnabled = true
        isShaftCollisionSnapshotVisible = true
        applyShaftCollisionSnapshotWarningAppearance()
        setLiveDrillOverlayVisible(false)
    }

    func hideShaftCollisionSnapshot() {
        shaftCollisionSnapshotRoot?.isEnabled = false
        isShaftCollisionSnapshotVisible = false
        setLiveDrillOverlayVisible(true)
    }

    /// Tint all sculpt mesh chunks to the current warning state.
    func tintSculptMeshRed() {
        isShaftCollisionWarningActive = true
        applySculptMeshWarningAppearance()
    }

    /// Restore original sculpt mesh chunk materials after collision clears.
    func restoreSculptMeshMaterials() {
        isShaftCollisionWarningActive = false
        applySculptMeshWarningAppearance()
    }

    func setAnatomyHazardFlashActive(_ isActive: Bool) {
        isAnatomyHazardFlashActive = isActive
        applyDrillWarningAppearance()
        applyShaftCollisionSnapshotWarningAppearance()
        applySculptMeshWarningAppearance()
        applyAnatomyHazardWarningAppearance()
    }

    private func applyDrillWarningAppearance() {
        let desiredColor = activeWarningUIColor

        guard let desiredColor else {
            guard _isDrillTintedRed else { return }
            _isDrillTintedRed = false
            for (entity, materials) in _cachedDrillMaterials {
                if var model = entity.components[ModelComponent.self] {
                    model.materials = materials
                    entity.components.set(model)
                }
            }
            return
        }

        if _cachedDrillMaterials.isEmpty {
            cacheDrillMaterials()
        }

        let warningMaterial = SimpleMaterial(color: desiredColor, roughness: 0.3, isMetallic: false)
        if let drillModel = drillModelEntity {
            func walk(_ entity: Entity) {
                if var model = entity.components[ModelComponent.self] {
                    model.materials = Array(repeating: warningMaterial, count: max(model.materials.count, 1))
                    entity.components.set(model)
                }
                for child in entity.children {
                    walk(child)
                }
            }
            walk(drillModel)
        }
        if var ballModel = drillBallEntity?.components[ModelComponent.self] {
            ballModel.materials = Array(repeating: warningMaterial, count: max(ballModel.materials.count, 1))
            if let drillBall = drillBallEntity {
                drillBall.components.set(ballModel)
            }
        }
        _isDrillTintedRed = true
    }

    private func applySculptMeshWarningAppearance() {
        guard let root = rootEntity else { return }
        let sculptMeshChunks = root.children.filter { $0.name == "SculptMeshChunk" }

        if _cachedSculptMeshMaterials.isEmpty {
            for child in sculptMeshChunks {
                if let model = child.components[ModelComponent.self] {
                    _cachedSculptMeshMaterials.append((child, model.materials))
                }
            }
        }

        let desiredColor = activeWarningUIColor

        guard let desiredColor else {
            guard _isSculptMeshTintedRed else { return }
            _isSculptMeshTintedRed = false
            let sculptMaterial = makeSculptMaterial()
            for entity in sculptMeshChunks {
                if var model = entity.components[ModelComponent.self] {
                    model.materials = Array(repeating: sculptMaterial, count: max(model.materials.count, 1))
                    entity.components.set(model)
                }
            }
            _cachedSculptMeshMaterials.removeAll()
            return
        }

        let warningMaterial = SimpleMaterial(color: desiredColor, roughness: 0.3, isMetallic: false)
        for (entity, materials) in _cachedSculptMeshMaterials {
            if var model = entity.components[ModelComponent.self] {
                model.materials = Array(repeating: warningMaterial, count: max(materials.count, 1))
                entity.components.set(model)
            }
        }
        _isSculptMeshTintedRed = true
    }

    /// Show the current cylindrical shaft detector zone in root space.
    func updateShaftCollisionDebugMarker(center: SIMD3<Float>,
                                         axis: SIMD3<Float>,
                                         length: Float,
                                         radius: Float,
                                         collisionPoint: SIMD3<Float>) {
        guard let root = rootEntity else { return }
        if shaftCollisionDebugMarker == nil {
            let marker = ModelEntity(
                mesh: .generateBox(size: SIMD3<Float>(radius * 2, radius * 2, length)),
                materials: [SimpleMaterial(color: .yellow.withAlphaComponent(0.55), roughness: 0.1, isMetallic: false)]
            )
            marker.name = "ShaftCollisionDebugMarker"
            root.addChild(marker)
            shaftCollisionDebugMarker = marker
        }
        let normalizedAxis = simd_length_squared(axis) > 1e-8 ? simd_normalize(axis) : SIMD3<Float>(0, 0, 1)
        let from = SIMD3<Float>(0, 0, 1)
        let dot = simd_dot(from, normalizedAxis)
        let rotation: simd_quatf
        if dot > 0.9999 {
            rotation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        } else if dot < -0.9999 {
            rotation = simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 1, 0))
        } else {
            let rotationAxis = simd_normalize(simd_cross(from, normalizedAxis))
            rotation = simd_quatf(angle: acos(simd_clamp(dot, -1.0, 1.0)), axis: rotationAxis)
        }
        shaftCollisionDebugMarker?.transform = Transform(
            scale: SIMD3<Float>(repeating: 1),
            rotation: rotation,
            translation: center
        )
        shaftCollisionDebugMarker?.isEnabled = true
        _ = collisionPoint
    }

    func hideShaftCollisionDebugMarker() {
        shaftCollisionDebugMarker?.isEnabled = false
    }

    func updateDrillTipDebugMarker(position: SIMD3<Float>) {
        guard isSlurryDebugEnabled else {
            drillTipDebugMarker?.isEnabled = false
            return
        }
        guard let root = rootEntity else { return }
        if drillTipDebugMarker == nil {
            let marker = ModelEntity(
                mesh: .generateSphere(radius: 0.0025),
                materials: [SimpleMaterial(color: .green.withAlphaComponent(0.9), roughness: 0.1, isMetallic: false)]
            )
            marker.name = "DrillTipDebugMarker"
            root.addChild(marker)
            drillTipDebugMarker = marker
        }
        drillTipDebugMarker?.position = position
        drillTipDebugMarker?.isEnabled = true
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
        drillModelBaseLocalTransform = drillModel.transform
        drillModelDefaultLocalTransform = drillModel.transform

        // Compute the drill tip position in anchor-local space.
        // The positioning math above places the model's min-Z face at Z = -0.05.
        // In general: tipZ = drillModel.position.z + scaledBounds.min.z
        let tipZ = drillModel.position.z + scaledBounds.min.z
        // X/Y offset compensates for the drill overlay being slightly off the stylus axis.
        let tipPosition = SIMD3<Float>(-0.003, 0.001, tipZ) //orig(-0.005,0.001, tipZ)

        // Create spinning drill ball at the tip, centered on the shaft axis.
        let drillBall = DrillRotationComponent.createDrillBall(rpm: 400)
        drillBall.position = tipPosition
        let selectedScale = SIMD3<Float>(repeating: selectedDrillBitScale)
        drillBall.scale = selectedScale
        anchor.addChild(drillBall)
        drillBallEntity = drillBall
        drillBallDefaultLocalTransform = drillBall.transform
        drillBallDefaultLocalTransform?.scale = selectedScale

        // Cache original materials for shaft collision tint/restore (model + bit).
        cacheDrillMaterials()

        // Build the pooled warning snapshot once so shaft collision can freeze
        // a visual copy without touching the live tracked drill hierarchy.
        rebuildShaftCollisionSnapshotDrillIfNeeded()

        // Store the tip offset so updateSculptingTool() can use it for sculpting position.
        drillBallLocalOffset = tipPosition

        // Attach spatial audio emitters to the drill overlay so playback follows
        // the tool reliably in visionOS.
        drillAudioModel?.attachSpatialAudio(to: drillModel, tipLocalPosition: tipPosition)

        print("Drill model and rotating ball attached at tip \(tipPosition)")
    }
    
    // Anchor via AnchorEntity to a GCDevice.
    // Set up stylus or controller inputs.
    @MainActor
    func setupSpatialAccessory(device: GCDevice, hapticsModel: HapticsModel) async throws {
        spatialAccessorySetupAttemptCount += 1
        let accessoryID = ObjectIdentifier(device)
        let accessoryKind: SpatialAccessoryKind = (device is GCStylus) ? .stylus : .controller
        print("[Accessory] setupSpatialAccessory attempt=\(spatialAccessorySetupAttemptCount) \(spatialAccessoryDebugDescription(for: device)) activeKind=\(String(describing: activeSpatialAccessoryKind)) activeID=\(String(describing: activeSpatialAccessoryID)) configuredCount=\(configuredSpatialAccessoryIDs.count)")

        if configuredSpatialAccessoryIDs.contains(accessoryID) {
            print("[Accessory] Skipping setup: exact object already configured \(spatialAccessoryDebugDescription(for: device))")
            return
        }

        if activeSpatialAccessoryKind == accessoryKind, sculptingEntity != nil {
            print("[Accessory] Skipping setup: active accessory of same kind already installed \(String(describing: activeSpatialAccessoryKind))")
            return
        }

        if accessoryKind == .controller, activeSpatialAccessoryKind == .stylus {
            print("[Accessory] Skipping controller setup because stylus is already active")
            return
        }

        if accessoryKind == .stylus,
           let activeID = activeSpatialAccessoryID,
           activeSpatialAccessoryKind == .controller {
            print("[Accessory] Replacing active controller with stylus")
            removeSpatialAccessoryAnchor(for: activeID)
        }

        let source = try await AnchoringComponent.AccessoryAnchoringSource(device: device)
        
        guard let location = source.locationName(named: "aim") ?? source.locationName(named: "tip") else {
            return
        }
        
        let sculptingEntity = AnchorEntity(.accessory(from: source, location: location),
                                           trackingMode: .predicted,
                                           physicsSimulation: .none)
        
        sculptingEntity.name = "SculptingEntity"
        
        (accessoryRootEntity ?? rootEntity)?.addChild(sculptingEntity)
        configuredSpatialAccessoryIDs.insert(accessoryID)
        spatialAccessoryAnchors[accessoryID] = sculptingEntity
        activeSpatialAccessoryID = accessoryID
        activeSpatialAccessoryKind = accessoryKind
        print("[Accessory] Installed active accessory kind=\(accessoryKind) objectID=\(accessoryID) totalAnchors=\(spatialAccessoryAnchors.count)")
        
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
