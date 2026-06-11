/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
A model of the sculpting.
*/

import SwiftUI
import RealityKit
import ARKit
import QuartzCore

@MainActor @Observable
final class SculptingToolModel {
    enum SpatialAccessoryKind {
        case stylus
        case controller
    }

    // Min and max radii of the sculpting tool.
    let minRadius: Float = 0.002
    let maxRadius: Float = 0.5

    @ObservationIgnored var rootEntity: Entity? = nil // The root entity in the RealityView.
    @ObservationIgnored var accessoryRootEntity: Entity? = nil // Untransformed scene root for accessory anchors.

    @ObservationIgnored let sculptingTool = Entity(components: [ModelComponent(mesh: .generateSphere(radius: 0.001), materials: [SimpleMaterial()]),
                                                                OpacityComponent(opacity: 0.01)])
    
    @ObservationIgnored var sculptingEntity: AnchorEntity? = nil
    @ObservationIgnored var trackingStateIndicator: ModelEntity? = nil
    
    @ObservationIgnored var additiveIcon: Entity? = nil
    @ObservationIgnored var subtractiveIcon: Entity? = nil
    @ObservationIgnored var enlargeIcon: Entity? = nil
    @ObservationIgnored var reduceIcon: Entity? = nil

    @ObservationIgnored var hapticsModel: HapticsModel? = nil
    @ObservationIgnored var drillAudioModel: DrillAudioModel? = nil
    @ObservationIgnored var shouldClearDebris: Bool = false
    @ObservationIgnored var isSpatialToolbarPanelVisible: Bool = false

    // Sculpt material management
    var loadedRoughnessValue: Float = 0.5
    var loadedAlbedoTint: SIMD3<Float> = SIMD3<Float>(repeating: 1.0)
    private var bundledSculptMaterial: (any RealityKit.Material)? = nil
    private var bundledMaterialLoadAttempted: Bool = false
    private let sculptRealityAssetName = "SculptGraph"
    private let sculptProxyEntityName = "SculptGraphProxy"

    // Drill overlay entities
    @ObservationIgnored var drillModelEntity: Entity? = nil
    @ObservationIgnored var drillBallEntity: ModelEntity? = nil
    @ObservationIgnored var shaftCollisionSnapshotRoot: Entity? = nil
    @ObservationIgnored var shaftCollisionSnapshotModel: Entity? = nil
    @ObservationIgnored var shaftCollisionSnapshotBall: ModelEntity? = nil
    @ObservationIgnored var isShaftCollisionSnapshotVisible: Bool = false
    var selectedDrillBitScale: Float = 1.0
    var drillModelBaseLocalTransform: Transform? = nil

    /// Anchor-local offset from the accessory origin to the drill ball center.
    /// Computed dynamically from the USDZ model bounds in `attachDrillModel`.
    var drillBallLocalOffset = SIMD3<Float>(0, 0, -0.04)

    // Shaft collision detection
    @ObservationIgnored let shaftCollisionDetector = ShaftCollisionDetector()
    /// Cached original materials for the drill model (populated after attach).
    var _cachedDrillMaterials: [(Entity, [any RealityKit.Material])] = []
    /// Whether the drill is currently tinted red due to shaft collision.
    var _isDrillTintedRed: Bool = false
    /// Cached original materials for sculpt mesh chunks while collision tinting is active.
    var _cachedSculptMeshMaterials: [(Entity, [any RealityKit.Material])] = []
    /// Whether sculpt mesh chunks are currently tinted red due to shaft collision.
    var _isSculptMeshTintedRed: Bool = false
    /// Whether shaft collision warning visuals are currently active.
    @ObservationIgnored var isShaftCollisionWarningActive: Bool = false
    /// Whether anatomy-hazard sculpt flash visuals are currently active.
    @ObservationIgnored var isAnatomyHazardFlashActive: Bool = false
    /// Temporary debug visual for the cylindrical shaft collision detector.
    @ObservationIgnored var shaftCollisionDebugMarker: ModelEntity? = nil
    /// Temporary debug visual for the live drill tip position.
    @ObservationIgnored var drillTipDebugMarker: ModelEntity? = nil
    @ObservationIgnored var isSlurryDebugEnabled: Bool = false
    var hazardFlashOpacity: Double = 0
    @ObservationIgnored var anatomyEffectsRoot: Entity? = nil
    @ObservationIgnored var anatomySafezones: [AnatomySafezoneSphere] = []
    @ObservationIgnored var activeAnatomyHazards: Set<AnatomyHazardKind> = []
    @ObservationIgnored var flashingAnatomyHazards: Set<AnatomyHazardKind> = []
    @ObservationIgnored var anatomyHazardEntities: [AnatomyHazardKind: Entity] = [:]
    @ObservationIgnored var cachedAnatomyHazardMaterials: [AnatomyHazardKind: [(Entity, [any RealityKit.Material])]] = [:]
    @ObservationIgnored var hazardFlashTask: Task<Void, Never>? = nil
    @ObservationIgnored var bloodSpurtTemplate: Entity? = nil
    @ObservationIgnored var configuredSpatialAccessoryIDs: Set<ObjectIdentifier> = []
    @ObservationIgnored var spatialAccessoryAnchors: [ObjectIdentifier: AnchorEntity] = [:]
    @ObservationIgnored var activeSpatialAccessoryID: ObjectIdentifier? = nil
    @ObservationIgnored var activeSpatialAccessoryKind: SpatialAccessoryKind? = nil
    @ObservationIgnored var didRegisterAccessoryNotifications: Bool = false
    @ObservationIgnored var spatialAccessorySetupAttemptCount: Int = 0
    @ObservationIgnored var accessoryTrackingSession: SpatialTrackingSession? = nil
    @ObservationIgnored var accessoryTrackingSessionTask: Task<Void, Never>? = nil
    @ObservationIgnored var accessoryTrackingRestartTask: Task<Void, Never>? = nil
    @ObservationIgnored var accessoryDiscoveryRescanTask: Task<Void, Never>? = nil
    @ObservationIgnored var accessoryTrackingShouldBeRunning: Bool = false

    // Shaft collision freeze/recovery state.
    private let shaftFreezeDuration: TimeInterval = 0.3
    private let shaftRecoveryDuration: TimeInterval = 0.2
    private var isShaftFrozen: Bool = false
    private var isShaftRecovering: Bool = false
    private var freezeEndTime: TimeInterval = 0
    private var recoveryStartTime: TimeInterval = 0
    private var frozenToolTransform: Transform = Transform()
    private var frozenDrillModelTransform: Transform? = nil
    private var frozenDrillBallTransform: Transform? = nil
    private var recoveryStartToolTransform: Transform = Transform()
    private var recoveryStartDrillModelTransform: Transform? = nil
    private var recoveryStartDrillBallTransform: Transform? = nil
    private var isDrillOverlayDetachedForFreeze: Bool = false
    private var cachedDrillBallRPMBeforeFreeze: Int? = nil
    @ObservationIgnored var isShaftCollisionDetectionSuspended: Bool = true
    private var shaftCollisionDetectionResumeTime: TimeInterval = .greatestFiniteMagnitude
    var drillModelDefaultLocalTransform: Transform? = nil
    var drillBallDefaultLocalTransform: Transform? = nil
    private var lastTrackingAudioState: AccessoryAnchor.TrackingState? = nil
    private(set) var isCurrentlyCarving: Bool = false

    // Tracks carving state for logging and particle bursts
    private var wasCarving: Bool = false
    private let carvingRadiusMargin: Float = 0.0005
    private let carvingFramesRequired: Int = 2
    private let idleFramesRequired: Int = 2
    private var carvingFrameStreak: Int = 0
    private var idleFrameStreak: Int = 0
    private let debrisSimulationInterval: TimeInterval = 1.0 / 30.0
    private var lastDebrisSimulationTime: TimeInterval = 0
    private let boneDustBurstInterval: TimeInterval = 0.25
    private var lastBoneDustBurstTime: TimeInterval = -.greatestFiniteMagnitude
    private let boneDustNormalResampleBurstInterval: Int = 5
    private var cachedBoneDustSurfaceNormal: SIMD3<Float>? = nil
    private var boneDustBurstsSinceLastNormalSample: Int = .max

    /// Pre-loaded BoneDust template entity used to populate the reusable burst pool.
    private var boneDustTemplate: Entity? = nil
    private let boneDustPoolSize: Int = 12
    private var availableBoneDustEntities: [Entity] = []
    private var activeBoneDustActivationIDs: [ObjectIdentifier: UInt64] = [:]
    private var activeBoneDustEntities: [ObjectIdentifier: Entity] = [:]
    private var nextBoneDustActivationID: UInt64 = 0
    
    /// Manages the drawing of bone debris box volumes when the tool contacts the sculpted volume.
    @ObservationIgnored let boneDebrisManager = BoneDebrisManager()
    
    /// Manages collision generation for the sculpting volume via CPU marching cubes.
    @ObservationIgnored let collisionManager = CollisionManager()

    /// 64^3 metaball density grid for unified debris visualization.
    /// Created lazily after the RealityView scene is live to avoid launch-time
    /// LowLevelMesh/Metal initialization during Observation setup.
    @ObservationIgnored var boneSlurryGrid: BoneSlurryGrid? = nil

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
        _cachedSculptMeshMaterials.removeAll()
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
        guard pressed else { return }

        additiveIcon?.isEnabled = false
        subtractiveIcon?.isEnabled = false
        reduceIcon?.isEnabled = false
        enlargeIcon?.isEnabled = false
        isSpatialToolbarPanelVisible.toggle()
    }
    
    // Show an indicator when tracking is non 6 DOF.
    // Gets an ARKit AccessoryAnchor from a RealityKit AnchorEntity
    // and extracts relevant interaction data for warning of possible bad data.
    func updateTrackingStateIndicatorIfDirty(sculptingEntity: AnchorEntity) {
        guard let accessoryAnchor = getAccessoryAnchor(entity: sculptingEntity) else {
            return
        }

        if lastTrackingAudioState != accessoryAnchor.trackingState {
            lastTrackingAudioState = accessoryAnchor.trackingState
            updateDrillTrackingAudio(for: accessoryAnchor.trackingState)
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
                sculptingTool.components.set(sculptingToolComponent)
            }
        }
    }
    
    // Every frame, update the interactions with in-app content
    // from the sculpting tool to the virtual clay.
    func updateSculptingTool() {
        // Process pending debris ejection forces (frame countdown).
        let now = CACurrentMediaTime()
        let hasDebrisWork = boneDebrisManager.hasActiveDebrisWork
        if hasDebrisWork, now - lastDebrisSimulationTime >= debrisSimulationInterval {
            lastDebrisSimulationTime = now
            boneDebrisManager.processPendingEjections()
            boneDebrisManager.processGrowth()
            boneDebrisManager.processAdhesion()
            boneDebrisManager.processSettling()
            boneDebrisManager.processToolDisplacement(toolPosition: sculptingTool.position)
            boneDebrisManager.requestSdfBlitIfNeeded()
        }

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
        // Also cull debris that has left the volume bounds.
        if hasDebrisWork {
            boneDebrisManager.cullOutOfBoundsDebris()
        }
        if hasDebrisWork, let root = rootEntity {
            boneSlurryGrid?.uploadParticles(from: boneDebrisManager,
                                            rootEntity: root)
        }

        guard let sculptingEntity = sculptingEntity else {
            return
        }

        updateTrackingStateIndicatorIfDirty(sculptingEntity: sculptingEntity)
        guard let rootEntity = rootEntity,
              let liveAnchorMatrix = try? sculptingEntity.transform(from: rootEntity) else {
            return
        }
        // Live anchor transform from accessory in root-local space.
        let liveAnchorTransform = Transform(matrix: simd_float4x4(liveAnchorMatrix))

        // Keep two transform sources:
        // 1) the tracked anchor-derived pose, which is the true live stylus target
        // 2) the currently rendered drill pose, used only in normal anchored mode
        let trackedModelTransform = currentLiveDrillModelTransform(anchorTransform: liveAnchorTransform)
        let trackedBallTransform = currentLiveDrillBallTransform(anchorTransform: liveAnchorTransform)
        let renderedModelTransform = currentRenderedDrillModelTransform(relativeTo: rootEntity)
        let renderedBallTransform = currentRenderedDrillBallTransform(relativeTo: rootEntity)

        let liveModelTransform = (isShaftFrozen || isShaftRecovering || isDrillOverlayDetachedForFreeze)
            ? trackedModelTransform
            : (renderedModelTransform ?? trackedModelTransform)
        let liveBallTransform = (isShaftFrozen || isShaftRecovering || isDrillOverlayDetachedForFreeze)
            ? trackedBallTransform
            : (renderedBallTransform ?? trackedBallTransform)

        // Compute live sculpting tool pose (tip-centered) from anchor.
        var liveToolTransform = liveAnchorTransform
        if let ballTransform = liveBallTransform {
            // Use drill-bit world position/orientation as the exact carving tip pose.
            liveToolTransform.translation = ballTransform.translation
            liveToolTransform.rotation = ballTransform.rotation
        } else {
            // Fallback to anchor-local offset math while detached/recovering.
            let rotatedOffset = liveToolTransform.rotation.act(drillBallLocalOffset)
            let rootScale = rootEntity.transform.scale
            let safeScale = SIMD3<Float>(
                abs(rootScale.x) > 1e-6 ? rootScale.x : 1.0,
                abs(rootScale.y) > 1e-6 ? rootScale.y : 1.0,
                abs(rootScale.z) > 1e-6 ? rootScale.z : 1.0
            )
            liveToolTransform.translation += rotatedOffset / safeScale
        }

        // --- Shaft collision detection ---
        if isShaftCollisionDetectionSuspended, now >= shaftCollisionDetectionResumeTime {
            isShaftCollisionDetectionSuspended = false
            shaftCollisionDetectionResumeTime = .greatestFiniteMagnitude
        }

        let accessoryTrackingState = getAccessoryAnchor(entity: sculptingEntity)?.trackingState
        let canRunShaftCollisionDetection =
            !isShaftCollisionDetectionSuspended &&
            collisionManager.hasSDFCache &&
            accessoryTrackingState == .positionOrientationTracked

        // The shaft extends from the tip backward along the drill's local +Z axis
        // (the USDZ model body goes in +Z from the tip).
        let shaftAxis = (liveModelTransform?.rotation ?? liveToolTransform.rotation).act(SIMD3<Float>(0, 0, 1))
        let shaftDirection = simd_length_squared(shaftAxis) > 1e-8 ? simd_normalize(shaftAxis) : SIMD3<Float>(0, 0, 1)
        let shaftDetectorCenter = liveToolTransform.translation + shaftDirection * shaftCollisionDetector.detectorCenterOffset
        collisionManager.updateLocalCollisionFocus(centerInRoot: shaftDetectorCenter)
        // Collision/SDF refresh is only needed once a tracked tool exists.
        collisionManager.processUpdatesIfNeeded()
        if !collisionManager.hasSDFCache {
            collisionManager.scheduleRegeneration()
        }

        let shaftResult: ShaftCollisionResult
        if canRunShaftCollisionDetection {
            shaftResult = shaftCollisionDetector.test(
                tipPosition: liveToolTransform.translation,
                shaftDirection: shaftDirection,
                collisionManager: collisionManager
            )
        } else {
            resetShaftCollisionStateIfNeeded()
            shaftResult = ShaftCollisionResult(isColliding: false,
                                               correctionVector: .zero,
                                               collisionPoint: .zero,
                                               penetrationDepth: 0)
        }

        if shaftResult.isColliding {
            // DEBUG ONLY: Uncomment to visualize the temporary cylindrical shaft
            // collision detector while tuning alignment.
            // let detectorCenter = liveToolTransform.translation + shaftDirection * shaftCollisionDetector.detectorCenterOffset
            // updateShaftCollisionDebugMarker(center: detectorCenter,
            //                                 axis: shaftDirection,
            //                                 length: shaftCollisionDetector.detectorLength,
            //                                 radius: shaftCollisionDetector.detectorRadius,
            //                                 collisionPoint: shaftResult.collisionPoint)
            // Visual feedback: keep the tracked drill hidden and show a frozen
            // warning snapshot at the collision pose while tinting the sculpt volume.
            if !isShaftCollisionSnapshotVisible {
                showShaftCollisionSnapshot(modelTransform: liveModelTransform,
                                           ballTransform: liveBallTransform)
            }
            tintSculptMeshRed()
            // Haptic feedback: sharp warning pulse.
            hapticsModel?.startShaftCollisionFeedback()
        } else {
            // DEBUG ONLY: Paired with the commented visualization above.
            // hideShaftCollisionDebugMarker()
            // Clear visual and haptic warnings.
            hideShaftCollisionSnapshot()
            restoreSculptMeshMaterials()
            hapticsModel?.stopShaftCollisionFeedback()
        }

        // Always keep the underlying tool following live tracking. Shaft collision
        // blocks carving, but no longer detaches or freezes the tracked hierarchy.
        sculptingTool.transform = liveToolTransform

        updateDrillTipDebugMarker(position: sculptingTool.position)
        boneDebrisManager.updateDebrisSpawnPreview(sculptingTool: sculptingTool)

        let drillBitDirection = -shaftDirection
        if let toolComponent = sculptingTool.components[SculptingToolComponent.self] {
            updateAnatomyHazards(toolPosition: sculptingTool.position,
                                 toolRadius: toolComponent.radius,
                                 drillBitDirection: drillBitDirection)
        }

        // Pause carving/tool logic while frozen or recovering.
        let carvingBlocked = shaftResult.isColliding
        sculptingTool.components[SculptingToolComponent.self]?.isActive = !carvingBlocked

        if carvingBlocked {
            isCurrentlyCarving = false
            hapticsModel?.stopSculptVibration()
            updateDrillCarvingAudio(isCarving: false)
            carvingFrameStreak = 0
            idleFrameStreak = 0
            if wasCarving {
                print("[Drill] Idle")
                wasCarving = false
            }
            return
        }
        
        // Log carving state using the SDF value sampled on the GPU.
        // SDF <= 0 means the tool is at or inside the mesh surface.
        if let sculptingToolComponent = sculptingTool.components[SculptingToolComponent.self] {
            let sculptor = sculptingToolComponent.sculptor
            let sdf = sculptor.lastSampledSDF
            let isCarving = updateCarvingState(sampledSDF: sdf,
                                               toolRadius: sculptingToolComponent.radius,
                                               isToolActive: sculptingToolComponent.isActive)
            isCurrentlyCarving = isCarving
            if isCarving != wasCarving {
                print(isCarving ? "[Drill] Carving" : "[Drill] Idle")
                if isCarving {
                    boneDebrisManager.update(sculptingTool: sculptingTool)
                    hapticsModel?.startSculptVibration()
                    updateDrillCarvingAudio(isCarving: true)
                    triggerBoneDustBurstIfNeeded(force: true,
                                                 shaftDirection: shaftDirection)
                } else {
                    stopAllBoneDustBursts()
                    hapticsModel?.stopSculptVibration()
                    updateDrillCarvingAudio(isCarving: false)
                }
                wasCarving = isCarving
            }
            if isCarving {
                hapticsModel?.startSculptVibration()
                updateDrillCarvingAudio(isCarving: true)
                triggerBoneDustBurstIfNeeded(force: false,
                                             shaftDirection: shaftDirection)
            } else {
                stopAllBoneDustBursts()
                hapticsModel?.stopSculptVibration()
                updateDrillCarvingAudio(isCarving: false)
            }
            // Throttled collision update while carving.
            if isCarving {
                collisionManager.markDirty()
            }
        } else {
            isCurrentlyCarving = false
        }
        
        /*
        // --- Bone debris update ---
        // Read the current active state from the sculpting tool component.
        let isActive = sculptingTool.components[SculptingToolComponent.self]?.isActive ?? false
        boneDebrisManager.update(sculptingTool: sculptingTool)
         */
    }

    // MARK: - Shaft Collision Freeze / Recovery
    // Deprecated path: the drill hierarchy is no longer detached or interpolated
    // for shaft collision. The app now keeps tracking live and shows a pooled
    // warning snapshot instead. These helpers remain only as fallback/reference.

    private func startShaftFreeze(now: TimeInterval,
                                  liveToolTransform: Transform,
                                  liveModelTransform: Transform?,
                                  liveBallTransform: Transform?) {
        guard let root = rootEntity else { return }

        // Capture the exact aligned freeze pose before detaching.
        frozenToolTransform = liveToolTransform
        frozenDrillModelTransform = liveModelTransform ?? drillModelEntity?.transform
        frozenDrillBallTransform = liveBallTransform ?? drillBallEntity?.transform
        if let ballTransform = frozenDrillBallTransform {
            frozenToolTransform.translation = ballTransform.translation
            frozenToolTransform.rotation = ballTransform.rotation
        }

        if !isDrillOverlayDetachedForFreeze {
            detachDrillOverlayFromAnchor(to: root)
        }

        isShaftRecovering = false
        isShaftFrozen = true
        freezeEndTime = now + shaftFreezeDuration
    }

    private func startShaftRecovery(now: TimeInterval) {
        isShaftFrozen = false
        isShaftRecovering = true
        recoveryStartTime = now

        recoveryStartToolTransform = sculptingTool.transform
        recoveryStartDrillModelTransform = drillModelEntity?.transform
        recoveryStartDrillBallTransform = drillBallEntity?.transform
    }

    private func finishShaftRecovery() {
        isShaftRecovering = false
        reattachDrillOverlayToAnchor()
    }

    func suspendShaftCollisionDetection() {
        isShaftCollisionDetectionSuspended = true
        shaftCollisionDetectionResumeTime = .greatestFiniteMagnitude
        resetShaftCollisionStateIfNeeded()
    }

    func resumeShaftCollisionDetection(after delay: TimeInterval) {
        isShaftCollisionDetectionSuspended = true
        shaftCollisionDetectionResumeTime = CACurrentMediaTime() + max(0, delay)
        resetShaftCollisionStateIfNeeded()
    }

    func resetShaftCollisionStateIfNeeded() {
        if isShaftFrozen || isShaftRecovering {
            isShaftFrozen = false
            isShaftRecovering = false
        }
        freezeEndTime = 0
        recoveryStartTime = 0
        hideShaftCollisionSnapshot()
        restoreDrillMaterials()
        restoreSculptMeshMaterials()
        hapticsModel?.stopShaftCollisionFeedback()
        if isDrillOverlayDetachedForFreeze {
            reattachDrillOverlayToAnchor()
        }
    }

    private func detachDrillOverlayFromAnchor(to root: Entity) {
        drillModelEntity?.setParent(root, preservingWorldTransform: true)
        drillBallEntity?.setParent(root, preservingWorldTransform: true)
        if var rotation = drillBallEntity?.components[DrillRotationComponent.self] {
            cachedDrillBallRPMBeforeFreeze = rotation.rpm
            rotation.rpm = 0
            drillBallEntity?.components.set(rotation)
        }
        isDrillOverlayDetachedForFreeze = true
    }

    private func reattachDrillOverlayToAnchor() {
        guard isDrillOverlayDetachedForFreeze, let anchor = sculptingEntity else { return }
        drillModelEntity?.setParent(anchor, preservingWorldTransform: false)
        drillBallEntity?.setParent(anchor, preservingWorldTransform: false)
        if let defaultModelTransform = drillModelDefaultLocalTransform {
            drillModelEntity?.transform = defaultModelTransform
        }
        if let defaultBallTransform = drillBallDefaultLocalTransform {
            drillBallEntity?.transform = defaultBallTransform
        }
        if var rotation = drillBallEntity?.components[DrillRotationComponent.self] {
            rotation.rpm = cachedDrillBallRPMBeforeFreeze ?? rotation.rpm
            drillBallEntity?.components.set(rotation)
        }
        cachedDrillBallRPMBeforeFreeze = nil
        isDrillOverlayDetachedForFreeze = false
    }

    private func liveDrillModelWorldTransform(anchorTransform: Transform) -> Transform? {
        guard let local = drillModelDefaultLocalTransform else { return nil }
        let worldMatrix = anchorTransform.matrix * local.matrix
        return Transform(matrix: worldMatrix)
    }

    private func liveDrillBallWorldTransform(anchorTransform: Transform) -> Transform? {
        guard let local = drillBallDefaultLocalTransform else { return nil }
        let worldMatrix = anchorTransform.matrix * local.matrix
        return Transform(matrix: worldMatrix)
    }

    private func currentLiveDrillModelTransform(anchorTransform: Transform) -> Transform? {
        if !isDrillOverlayDetachedForFreeze {
            return liveDrillModelWorldTransform(anchorTransform: anchorTransform)
        }
        return liveDrillModelWorldTransform(anchorTransform: anchorTransform)
    }

    private func currentLiveDrillBallTransform(anchorTransform: Transform) -> Transform? {
        if !isDrillOverlayDetachedForFreeze {
            return liveDrillBallWorldTransform(anchorTransform: anchorTransform)
        }
        return liveDrillBallWorldTransform(anchorTransform: anchorTransform)
    }

    private func currentRenderedDrillModelTransform(relativeTo rootEntity: Entity) -> Transform? {
        guard let drillModelEntity,
              let matrix = try? drillModelEntity.transform(from: rootEntity) else { return nil }
        return Transform(matrix: simd_float4x4(matrix))
    }

    private func currentRenderedDrillBallTransform(relativeTo rootEntity: Entity) -> Transform? {
        guard let drillBallEntity,
              let matrix = try? drillBallEntity.transform(from: rootEntity) else { return nil }
        return Transform(matrix: simd_float4x4(matrix))
    }

    private func interpolateTransform(_ from: Transform, _ to: Transform, _ t: Float) -> Transform {
        let tt = max(0, min(1, t))
        return Transform(
            scale: simd_mix(from.scale, to.scale, SIMD3<Float>(repeating: tt)),
            rotation: simd_slerp(from.rotation, to.rotation, tt),
            translation: simd_mix(from.translation, to.translation, SIMD3<Float>(repeating: tt))
        )
    }

    func clearCarvingAndTrackingRuntimeState() {
        lastTrackingAudioState = nil
        isCurrentlyCarving = false
        wasCarving = false
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
                self.buildBoneDustPoolIfNeeded()
            } else {
                print("[BoneDust] Failed to pre-load BoneDust.usdz")
            }
        }
    }

    private func buildBoneDustPoolIfNeeded() {
        guard availableBoneDustEntities.isEmpty,
              activeBoneDustActivationIDs.isEmpty,
              let template = boneDustTemplate else { return }
        for index in 0..<boneDustPoolSize {
            let dustEntity = template.clone(recursive: true)
            dustEntity.name = "BoneDustPooled_\(index)"
            dustEntity.isEnabled = false
            availableBoneDustEntities.append(dustEntity)
        }
    }

    private func triggerBoneDustBurstIfNeeded(force: Bool, shaftDirection: SIMD3<Float>) {
        let now = CACurrentMediaTime()
        guard force || now - lastBoneDustBurstTime >= boneDustBurstInterval else { return }
        lastBoneDustBurstTime = now
        triggerBoneDustBurst(shaftDirection: shaftDirection)
    }

    /// Borrow a pooled BoneDust entity, orient it to the carved surface normal,
    /// then return it to the pool after its burst lifetime elapses.
    private func triggerBoneDustBurst(shaftDirection: SIMD3<Float>) {
        guard let root = rootEntity,
              boneDustTemplate != nil else { return }
        buildBoneDustPoolIfNeeded()
        guard let dustEntity = availableBoneDustEntities.popLast() else { return }
        let surfaceNormal = preferredBoneDustSurfaceNormal(fallbackShaftDirection: shaftDirection)
        nextBoneDustActivationID &+= 1
        let activationID = nextBoneDustActivationID
        let entityID = ObjectIdentifier(dustEntity)
        activeBoneDustActivationIDs[entityID] = activationID
        activeBoneDustEntities[entityID] = dustEntity
        dustEntity.transform = Transform(
            scale: dustEntity.scale,
            rotation: rotationAligningUpVector(to: surfaceNormal),
            translation: sculptingTool.position
        )
        dustEntity.isEnabled = true
        root.addChild(dustEntity)

        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(1500))
            guard self.activeBoneDustActivationIDs[entityID] == activationID else { return }
            self.activeBoneDustActivationIDs.removeValue(forKey: entityID)
            self.activeBoneDustEntities.removeValue(forKey: entityID)
            dustEntity.isEnabled = false
            dustEntity.removeFromParent()
            self.availableBoneDustEntities.append(dustEntity)
        }
    }

    private func stopAllBoneDustBursts() {
        guard !activeBoneDustEntities.isEmpty else { return }
        let activeEntities = activeBoneDustEntities
        activeBoneDustActivationIDs.removeAll()
        activeBoneDustEntities.removeAll()
        for (entityID, entity) in activeEntities {
            entity.isEnabled = false
            entity.removeFromParent()
            if !availableBoneDustEntities.contains(where: { ObjectIdentifier($0) == entityID }) {
                availableBoneDustEntities.append(entity)
            }
        }
    }

    private func preferredBoneDustSurfaceNormal(fallbackShaftDirection: SIMD3<Float>) -> SIMD3<Float> {
        let shouldResample = cachedBoneDustSurfaceNormal == nil ||
            boneDustBurstsSinceLastNormalSample >= boneDustNormalResampleBurstInterval
        if shouldResample {
            let gradient = collisionManager.sampleSDFGradient(at: sculptingTool.position)
            if simd_length_squared(gradient) > 1e-8 {
                let normal = simd_normalize(gradient)
                cachedBoneDustSurfaceNormal = normal
                boneDustBurstsSinceLastNormalSample = 0
                return normal
            }
        }
        boneDustBurstsSinceLastNormalSample &+= 1
        if let cachedBoneDustSurfaceNormal {
            return cachedBoneDustSurfaceNormal
        }
        return simd_normalize(-fallbackShaftDirection)
    }

    private func updateCarvingState(sampledSDF: Float,
                                    toolRadius: Float,
                                    isToolActive: Bool) -> Bool {
        guard isToolActive, sampledSDF.isFinite else {
            carvingFrameStreak = 0
            idleFrameStreak = 0
            return false
        }

        // The sculpting sphere influences the volume when its radius overlaps the
        // surface. Since `sampleSDF` measures distance from the tool center, actual
        // carve/contact starts when `sampledSDF <= toolRadius`.
        let enterThreshold = toolRadius - carvingRadiusMargin
        let exitThreshold = toolRadius + carvingRadiusMargin
        let wantsCarving = sampledSDF <= enterThreshold
        let wantsIdle = sampledSDF >= exitThreshold

        if wantsCarving {
            carvingFrameStreak += 1
            idleFrameStreak = 0
        } else if wantsIdle {
            idleFrameStreak += 1
            carvingFrameStreak = 0
        } else {
            carvingFrameStreak = 0
            idleFrameStreak = 0
        }

        if wasCarving {
            return idleFrameStreak < idleFramesRequired
        }
        return carvingFrameStreak >= carvingFramesRequired
    }

    private func rotationAligningUpVector(to targetUp: SIMD3<Float>) -> simd_quatf {
        let from = SIMD3<Float>(0, 1, 0)
        let to = simd_length_squared(targetUp) > 1e-8 ? simd_normalize(targetUp) : from
        let dot = simd_dot(from, to)
        if dot > 0.9999 {
            return simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1))
        }
        if dot < -0.9999 {
            return simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 0, 1))
        }
        let axis = simd_normalize(simd_cross(from, to))
        return simd_quatf(angle: acos(simd_clamp(dot, -1.0, 1.0)), axis: axis)
    }

}
