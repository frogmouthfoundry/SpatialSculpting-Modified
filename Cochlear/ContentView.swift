/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
The volume for sculpting and UI for controls.
*/

import SwiftUI
import ARKit
import RealityKit
import GameController
import CoreHaptics
import QuartzCore

private enum ScenePerformanceTier: Equatable {
    case normal
    case zoomed
}

struct ContentView: View {
    var root: Entity = Entity(components: [ComputeSystemComponent(computeSystem: SculptingToolSystem())])
    @State private var fluidSceneRoot: Entity = {
        let entity = Entity()
        entity.name = "FluidSceneRoot"
        return entity
    }()
    @State private var staticSceneRoot: Entity = {
        let entity = Entity()
        entity.name = "StaticSceneRoot"
        return entity
    }()
    @State private var interactiveAnatomyRoot: Entity = {
        let entity = Entity()
        entity.name = "InteractiveAnatomyRoot"
        return entity
    }()

    @State var sculpting: SculptingToolModel = SculptingToolModel()
    @State var haptics: HapticsModel = HapticsModel()
    @State var drillAudio: DrillAudioModel = DrillAudioModel()

    let marchingCubesMesh: MarchingCubesMesh!
    let sculptor: MarchingCubesMeshSculptor!

    @State var saveDocument: VolumeDocument? = nil
    @State var isSaving = false
    @State private var fluidWaveMesh: AnimatedWaveMesh? = nil
    @State private var fluidWaveEntity: ModelEntity? = nil
    @State private var defaultFluidLayerPosition: SIMD3<Float>? = nil
    @State private var waterProbeController: VirtualWaterProbeController = VirtualWaterProbeController(radius: 0.008)
    @State private var fluidRevealDebrisThreshold: Int = 6
    @State private var firstWaterCarveZInRoot: Float? = nil
    @State private var lastFluidRippleTimestamp: TimeInterval = 0
    @State private var isFluidLayerHiddenForDebrisReset: Bool = true
    @State private var selectedBitSizeMM: Int = 6
    @State private var didApplyBitSizeToDrillBall: Bool = false
    @State private var didLoadStaticSceneContent: Bool = false
    @State private var didLoadInteractiveAnatomyContent: Bool = false
    @State private var didQueueDeferredStartupWork: Bool = false
    @State private var didScheduleStartupReload: Bool = false
    @State private var isSlurryDebugUIEnabled: Bool = false
    @State private var isWaterDebugUIEnabled: Bool = false
    @State private var sceneZoomFactor: Float = 1.0
    @State private var scenePerformanceTier: ScenePerformanceTier = .normal
    @State private var sceneRollAngleRadians: Float = 0

    // Volume transparency toggle (50% transparent when on).
    @State private var isVolumeTransparent: Bool = false

    // Volume scale tracking: 1.0 = original size; each press changes by 0.1.
    // Minimum allowed = 0.4 (40% of original). Maximum = 1.0 (original size).
    @State private var volumeScaleFactor: Float = 1.0

    // Build initial voxel volume from bundled package manifest when available.
    private static func initialVolumeConfig() -> (dimensions: SIMD3<UInt32>,
                                                  voxelSize: SIMD3<Float>,
                                                  voxelStart: SIMD3<Float>,
                                                  textureBoundsMin: SIMD3<Float>,
                                                  textureBoundsMax: SIMD3<Float>) {
        let fallbackDimensions = SIMD3<UInt32>(128, 128, 128)
        let fallbackExtent = SIMD3<Float>(0.8, 0.8, 0.8)
        let fallbackVoxelSize = fallbackExtent / SIMD3<Float>(fallbackDimensions)
        let fallbackStart = -SIMD3<Float>(fallbackDimensions) * fallbackVoxelSize / 2
        let fallbackBoundsMin = fallbackStart - fallbackVoxelSize * 0.5
        let fallbackBoundsMax = fallbackStart + fallbackVoxelSize * (SIMD3<Float>(fallbackDimensions) - 0.5)
        let targetVolumeSize: Float = 0.8

        guard let url = Bundle.main.url(forResource: "MyVolume", withExtension: "sculptpkg") else {
            print("Initial volume config: using fallback (missing MyVolume.sculptpkg).")
            return (fallbackDimensions, fallbackVoxelSize, fallbackStart, fallbackBoundsMin, fallbackBoundsMax)
        }
        do {
            let manifest = try VolumeDocument.parsePackageManifest(at: url)
            let dims = SIMD3<UInt32>(UInt32(manifest.grid.width),
                                     UInt32(manifest.grid.height),
                                     UInt32(manifest.grid.depth))
            guard manifest.bounds.min.count == 3, manifest.bounds.max.count == 3 else {
                print("Initial volume config: manifest bounds invalid count; using fallback bounds.")
                let voxelSize = fallbackExtent / SIMD3<Float>(dims)
                let voxelStart = -SIMD3<Float>(dims) * voxelSize / 2
                let boundsMin = voxelStart - voxelSize * 0.5
                let boundsMax = voxelStart + voxelSize * (SIMD3<Float>(dims) - 0.5)
                return (dims, voxelSize, voxelStart, boundsMin, boundsMax)
            }

            let bmin = SIMD3<Float>(manifest.bounds.min[0], manifest.bounds.min[1], manifest.bounds.min[2])
            let bmax = SIMD3<Float>(manifest.bounds.max[0], manifest.bounds.max[1], manifest.bounds.max[2])
            let sourceExtent = simd_max(bmax - bmin, SIMD3<Float>(repeating: 1e-5))
            let maxExtent = max(sourceExtent.x, max(sourceExtent.y, sourceExtent.z))
            let uniformScale = targetVolumeSize / max(maxExtent, 1e-5)
            let fittedExtent = sourceExtent * uniformScale
            let fittedMin = -fittedExtent * 0.5
            let voxelSize = fittedExtent / SIMD3<Float>(dims)
            let voxelStart = fittedMin + voxelSize * 0.5
            print("Initial volume config from manifest: grid=\(dims), voxelSize=\(voxelSize)")
            return (dims, voxelSize, voxelStart, bmin, bmax)
        } catch {
            print("Initial volume config: manifest parse failed, using fallback. Error: \(error)")
            return (fallbackDimensions, fallbackVoxelSize, fallbackStart, fallbackBoundsMin, fallbackBoundsMax)
        }
    }

    init() {
        let config = Self.initialVolumeConfig()

        guard let voxelVolume = try? VoxelVolume(dimensions: config.dimensions,
                                                 voxelSize: config.voxelSize,
                                                 voxelStartPosition: config.voxelStart,
                                                 textureBoundsMin: config.textureBoundsMin,
                                                 textureBoundsMax: config.textureBoundsMax) else {
            self.marchingCubesMesh = nil
            self.sculptor = nil
            print("Failed to create volume.")
            return
        }

        self.marchingCubesMesh = try? MarchingCubesMesh(voxelVolume: voxelVolume)
        self.sculptor = MarchingCubesMeshSculptor(marchingCubesMesh: marchingCubesMesh)
    }

    func createMeshChunkEntity(meshChunk: MarchingCubesMeshChunk) throws -> Entity {
        let mesh = try MeshResource(from: meshChunk.mesh)
        let meshChunkEntity = Entity()
        // Use the lit sculpt material so compute-baked vertex color appears on first render.
        meshChunkEntity.components.set(ModelComponent(mesh: mesh, materials: [sculpting.makeSculptMaterial()]))
        meshChunkEntity.name = "SculptMeshChunk"
        return meshChunkEntity
    }

    private func createFluidLayerIfNeeded() {
        guard fluidWaveEntity == nil else { return }
        do {
            let waveMesh = try AnimatedWaveMesh()
            waveMesh.segmentCount = sceneZoomFactor >= 2.0 ? 64 : 128
            waveMesh.waveDensity = 3.0
            waveMesh.amplitude = 0.0
            waveMesh.speed = 1.0

            let meshResource = try MeshResource(from: waveMesh.lowLevelMesh)
            var material = PhysicallyBasedMaterial()
            material.baseColor.tint = .init(red: 0.78, green: 0.76, blue: 0.78, alpha: 1.0)
            if let normalTexture = try? TextureResource.load(named: "water 0397cbormal") {
                material.normal.texture = .init(normalTexture)
            } else {
                print("Fluid normal texture not found: water 0397cbormal")
            }
            material.roughness.scale = 0.20
            material.metallic.scale = 0.0
            material.blending = .transparent(opacity: 0.82)
            material.faceCulling = .back

            let entity = ModelEntity(mesh: meshResource, materials: [material])
            entity.name = "FluidLayer"
            if let voxelVolume = marchingCubesMesh?.voxelVolume {
                let dims = SIMD3<Float>(voxelVolume.dimensions)
                let sculptMin = voxelVolume.voxelStartPosition - voxelVolume.voxelSize * 0.5
                let sculptMax = voxelVolume.voxelStartPosition + voxelVolume.voxelSize * (dims - 0.5)
                let sculptExtent = sculptMax - sculptMin
                entity.position = SIMD3<Float>(
                    (sculptMin.x + sculptMax.x) * 0.5,
                    (sculptMin.y + sculptMax.y) * 0.5,
                    sculptMax.z - 0.15
                )
                entity.scale = SIMD3<Float>(
                    sculptExtent.x * 0.4032,
                    1.0,
                    sculptExtent.y * 0.4032
                )
            } else {
                entity.position = SIMD3<Float>(0, 0, 0.23)
                entity.scale = SIMD3<Float>(0.32256, 1.0, 0.32256)
            }
            entity.orientation = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(1, 0, 0))
            entity.isEnabled = !isFluidLayerHiddenForDebrisReset
            fluidSceneRoot.addChild(entity)

            fluidWaveMesh = waveMesh
            fluidWaveEntity = entity
            defaultFluidLayerPosition = entity.position
            applyPerformanceGovernor()
        } catch {
            print("Failed to create fluid layer: \(error)")
        }
    }

    private func updateFluidLayer(timestep: TimeInterval) {
        if fluidWaveMesh?.needsFrameUpdate == true {
            fluidWaveMesh?.update(timestep)
        }
        updateVirtualWaterProbe(timestep: timestep)
        if isFluidLayerHiddenForDebrisReset,
           sculpting.boneDebrisManager.spawnedDebrisSinceLastClear >= fluidRevealDebrisThreshold {
            setFluidLayerVisible(true)
            isFluidLayerHiddenForDebrisReset = false
        }
        triggerFluidRippleIfNeeded()
    }

    private func setFluidLayerVisible(_ isVisible: Bool) {
        fluidWaveEntity?.isEnabled = isVisible
    }

    private func resetFluidAccumulation() {
        waterProbeController.reset()
        firstWaterCarveZInRoot = nil
        if let defaultFluidLayerPosition {
            fluidWaveEntity?.position = defaultFluidLayerPosition
        }
        setFluidLayerVisible(false)
        isFluidLayerHiddenForDebrisReset = true
    }

    private func updateVirtualWaterProbe(timestep: TimeInterval) {
        guard let rootEntity = sculpting.rootEntity else { return }

        let carvePositionInRoot: SIMD3<Float>? = sculpting.isCurrentlyCarving ? sculpting.sculptingTool.position : nil
        if let carvePositionInRoot, firstWaterCarveZInRoot == nil {
            firstWaterCarveZInRoot = carvePositionInRoot.z
        }
        waterProbeController.update(rootEntity: rootEntity,
                                    collisionManager: sculpting.collisionManager,
                                    carvePositionInRoot: carvePositionInRoot,
                                    isCarving: sculpting.isCurrentlyCarving,
                                    timestep: timestep)

        guard let fluidWaveEntity,
              let targetPointInRoot = waterProbeController.waterPlacementPointInRoot else { return }

        var clampedPointInRoot = targetPointInRoot
        if let firstWaterCarveZInRoot {
            clampedPointInRoot.z = min(firstWaterCarveZInRoot - 0.15, targetPointInRoot.z)
        }

        let targetPointInFluid = fluidSceneRoot.convert(position: clampedPointInRoot, from: rootEntity)
        fluidWaveEntity.position.z = targetPointInFluid.z
    }

    private func sculptVolumeLocalBounds() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        guard let voxelVolume = marchingCubesMesh?.voxelVolume else { return nil }
        let dims = SIMD3<Float>(voxelVolume.dimensions)
        let sculptMin = voxelVolume.voxelStartPosition - voxelVolume.voxelSize * 0.5
        let sculptMax = voxelVolume.voxelStartPosition + voxelVolume.voxelSize * (dims - 0.5)
        return (sculptMin, sculptMax)
    }

    private func sculptVolumePositiveZFacePivotWorld() -> SIMD3<Float> {
        guard let bounds = sculptVolumeLocalBounds() else { return .zero }
        let worldBounds = transformedBounds(corners: boundsCorners(min: bounds.min, max: bounds.max),
                                            matrix: root.transform.matrix)
        return SIMD3<Float>(
            (worldBounds.min.x + worldBounds.max.x) * 0.5,
            (worldBounds.min.y + worldBounds.max.y) * 0.5,
            worldBounds.max.z
        )
    }

    private func setUniformScale(_ entity: Entity, scale: Float) {
        entity.transform.scale = SIMD3<Float>(repeating: scale)
    }

    private func translateScalableSceneRoots(z deltaZ: Float) {
        guard abs(deltaZ) > 0.0001 else { return }

        root.position.z += deltaZ
        staticSceneRoot.position.z += deltaZ
        interactiveAnatomyRoot.position.z += deltaZ
        fluidSceneRoot.position.z += deltaZ
    }

    private func syncDrillVisualZoom() {
        let zoomFactor = sceneZoomFactor
        let bitScale = sculpting.selectedDrillBitScale * zoomFactor

        if let baseModelTransform = sculpting.drillModelBaseLocalTransform {
            let tipPivot = sculpting.drillBallLocalOffset
            var zoomedModelTransform = baseModelTransform
            zoomedModelTransform.scale = baseModelTransform.scale * zoomFactor
            zoomedModelTransform.translation = tipPivot + (baseModelTransform.translation - tipPivot) * zoomFactor
            sculpting.drillModelDefaultLocalTransform = zoomedModelTransform

            if let drillModel = sculpting.drillModelEntity,
               drillModel.parent === sculpting.sculptingEntity {
                drillModel.transform = zoomedModelTransform
            }
        }

        if var defaultBallTransform = sculpting.drillBallDefaultLocalTransform {
            defaultBallTransform.translation = sculpting.drillBallLocalOffset
            defaultBallTransform.scale = SIMD3<Float>(repeating: bitScale)
            sculpting.drillBallDefaultLocalTransform = defaultBallTransform
        }

        if let drillBall = sculpting.drillBallEntity,
           drillBall.parent === sculpting.sculptingEntity {
            drillBall.position = sculpting.drillBallLocalOffset
            drillBall.scale = SIMD3<Float>(repeating: bitScale)
        }
    }

    private func applySceneZoomScale() {
        setUniformScale(root, scale: volumeScaleFactor * sceneZoomFactor)
        setUniformScale(staticSceneRoot, scale: sceneZoomFactor)
        setUniformScale(interactiveAnatomyRoot, scale: sceneZoomFactor)
        setUniformScale(fluidSceneRoot, scale: sceneZoomFactor)
    }

    private func applySceneZoom(_ newZoomFactor: Float) {
        guard abs(newZoomFactor - sceneZoomFactor) > 0.0001 else { return }

        let anchoredFrontZ = sculptVolumePositiveZFacePivotWorld().z
        sceneZoomFactor = newZoomFactor
        applySceneZoomScale()
        syncDrillVisualZoom()
        applyPerformanceGovernor()
        let shiftedFrontZ = sculptVolumePositiveZFacePivotWorld().z
        translateScalableSceneRoots(z: anchoredFrontZ - shiftedFrontZ)
        if sculpting.sculptingEntity != nil {
            sculpting.collisionManager.scheduleRegeneration()
        }
    }

    private func resetZoomedSceneTransformsToBaseline() {
        root.transform.translation = .zero
        staticSceneRoot.transform = Transform()
        interactiveAnatomyRoot.transform = Transform()
        fluidSceneRoot.transform = Transform()
    }

    private func updateRootGravityFromCurrentRotation() {
        let worldGravity = BoneDebrisManager.debrisGravity
        let inverseRotation = root.transform.rotation.inverse
        let localGravity = inverseRotation.act(worldGravity)
        var sim = root.components[PhysicsSimulationComponent.self] ?? PhysicsSimulationComponent()
        sim.gravity = localGravity
        root.components.set(sim)
    }

    private func applySceneWideRollIncrement(radians: Float, updateState: Bool = true) {
        guard abs(radians) > 0.0001 else { return }

        let increment = simd_quatf(angle: radians, axis: SIMD3<Float>(0, 0, 1))
        root.transform.rotation = increment * root.transform.rotation
        staticSceneRoot.transform.rotation = increment * staticSceneRoot.transform.rotation
        interactiveAnatomyRoot.transform.rotation = increment * interactiveAnatomyRoot.transform.rotation
        fluidSceneRoot.transform.rotation = increment * fluidSceneRoot.transform.rotation

        if updateState {
            sceneRollAngleRadians += radians
        }

        updateRootGravityFromCurrentRotation()
    }

    private var selectedBitRadiusMeters: Float {
        Float(selectedBitSizeMM) / 1000.0
    }

    private func applySelectedBitSize() {
        let radius = selectedBitRadiusMeters
        let scale = radius / DrillRotationComponent.defaultBallRadius
        sculpting.sculptingTool.components[SculptingToolComponent.self]?.radius = radius
        sculpting.boneDebrisManager.newDebrisSizeScale = scale
        sculpting.selectedDrillBitScale = scale

        if sculpting.drillBallEntity != nil {
            syncDrillVisualZoom()
            didApplyBitSizeToDrillBall = true
        } else {
            didApplyBitSizeToDrillBall = false
        }
    }

    private func setBitSize(mm: Int) {
        selectedBitSizeMM = mm
        applySelectedBitSize()
    }

    @ViewBuilder
    private func bitSizeButton(_ mm: Int) -> some View {
        Button("\(mm)mm") {
            setBitSize(mm: mm)
        }
        .buttonStyle(.borderedProminent)
        .tint(selectedBitSizeMM == mm ? .green : .gray)
    }

    private func zoomButton(_ factor: Float) -> some View {
        let isActive = abs(sceneZoomFactor - factor) < 0.001
        let label = abs(factor - 1.0) < 0.001
            ? "1.0x"
            : (abs(factor - 1.5) < 0.001 ? "1.5x" : "2.0x")

        return Button(label) {
            applySceneZoom(factor)
        }
        .buttonStyle(.borderedProminent)
        .tint(isActive ? .green : .gray)
    }

    private func triggerFluidRippleIfNeeded() {
        guard let waveMesh = fluidWaveMesh, let fluidEntity = fluidWaveEntity else { return }
        guard fluidEntity.isEnabled else { return }
        let toolPositionInFluidSpace = fluidEntity.convert(position: sculpting.sculptingTool.position, from: root)
        guard abs(toolPositionInFluidSpace.x) < 0.5,
              abs(toolPositionInFluidSpace.y) < 0.5,
              abs(toolPositionInFluidSpace.z) < 0.03 else {
            return
        }

        let now = CACurrentMediaTime()
        guard now - lastFluidRippleTimestamp > 0.07 else { return }
        lastFluidRippleTimestamp = now
        waveMesh.triggerRipple(at: SIMD2<Float>(toolPositionInFluidSpace.x, toolPositionInFluidSpace.y))
    }

    private func ensureBoneSlurryGridIfNeeded() {
        guard sculpting.boneSlurryGrid == nil,
              !sculpting.boneDebrisManager.drawnEntities.isEmpty else { return }

        if let voxelVolume = marchingCubesMesh?.voxelVolume {
            let slurryBounds = slurryBounds(for: voxelVolume)
            sculpting.replaceBoneSlurryGrid(volumeBoundsMin: slurryBounds.min,
                                            volumeBoundsMax: slurryBounds.max)
        } else {
            sculpting.replaceBoneSlurryGrid()
        }
        applyPerformanceGovernor()
    }

    private func performanceTier(for zoomFactor: Float) -> ScenePerformanceTier {
        zoomFactor >= 2.0 ? .zoomed : .normal
    }

    private func applyPerformanceGovernor() {
        let tier = performanceTier(for: sceneZoomFactor)
        let tierChanged = tier != scenePerformanceTier
        scenePerformanceTier = tier

        let desiredFluidSegments = tier == .zoomed ? 64 : 128
        if let waveMesh = fluidWaveMesh, waveMesh.segmentCount != desiredFluidSegments {
            waveMesh.segmentCount = desiredFluidSegments
            waveMesh.update(0)
        }

        sculpting.boneSlurryGrid?.setPerformanceTier(tier == .zoomed ? .zoomed : .normal)
        sculpting.collisionManager.setPerformanceTier(tier == .zoomed ? .zoomed : .normal)
        sculpting.boneDebrisManager.setPerformanceTier(tier == .zoomed ? .zoomed : .normal)
        if tierChanged, sculpting.sculptingEntity != nil {
            sculpting.collisionManager.scheduleRegeneration()
        }
    }

    private func boundsCorners(min: SIMD3<Float>, max: SIMD3<Float>) -> [SIMD3<Float>] {
        [
            SIMD3<Float>(min.x, min.y, min.z),
            SIMD3<Float>(min.x, min.y, max.z),
            SIMD3<Float>(min.x, max.y, min.z),
            SIMD3<Float>(min.x, max.y, max.z),
            SIMD3<Float>(max.x, min.y, min.z),
            SIMD3<Float>(max.x, min.y, max.z),
            SIMD3<Float>(max.x, max.y, min.z),
            SIMD3<Float>(max.x, max.y, max.z)
        ]
    }

    private func transformedBounds(corners: [SIMD3<Float>], matrix: simd_float4x4) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        var minCorner = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxCorner = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for corner in corners {
            let transformed = matrix * SIMD4<Float>(corner, 1)
            let point = SIMD3<Float>(transformed.x, transformed.y, transformed.z)
            minCorner = simd_min(minCorner, point)
            maxCorner = simd_max(maxCorner, point)
        }

        return (minCorner, maxCorner)
    }

    @MainActor
    private func queueDeferredStartupWorkIfNeeded() {
        guard !didQueueDeferredStartupWork else { return }
        didQueueDeferredStartupWork = true

        Task { @MainActor in
            await Task.yield()
            await populateStaticSceneIfNeeded()
            try? await Task.sleep(for: .milliseconds(100))
            await populateInteractiveAnatomyIfNeeded()
            try? await Task.sleep(for: .milliseconds(100))
            sculpting.preloadBoneDust()
            sculpting.preloadBloodSpurt()
        }
    }

    @MainActor
    private func scheduleStartupReloadIfNeeded() {
        guard !didScheduleStartupReload else { return }
        didScheduleStartupReload = true

        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(250))
            performReload()
        }
    }

    private func slurryBounds(for voxelVolume: VoxelVolume) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        let dims = SIMD3<Float>(voxelVolume.dimensions)
        let sculptLocalMin = voxelVolume.voxelStartPosition - voxelVolume.voxelSize * 0.5
        let sculptLocalMax = voxelVolume.voxelStartPosition + voxelVolume.voxelSize * (dims - 0.5)
        let sculptLocalExtent = sculptLocalMax - sculptLocalMin
        let sculptLocalCenterYZ = SIMD2<Float>(
            (sculptLocalMin.y + sculptLocalMax.y) * 0.5,
            (sculptLocalMin.z + sculptLocalMax.z) * 0.5
        )

        // Start from the currently reduced slurry size, then enlarge it by 25%.
        let baseAxisScale = Float(pow(0.25, 1.0 / 3.0)) * 1.25 * 0.5
        let axisScale = baseAxisScale * 1.25
        let slurryLocalExtent = sculptLocalExtent * axisScale

        // Anchor against the sculpt volume's pre-load +X face so the existing
        // root rotation carries the slurry onto the displayed +Z side.
        let slurryLocalMaxX = sculptLocalMax.x + 0.05
        let slurryLocalMinX = slurryLocalMaxX - slurryLocalExtent.x
        let slurryLocalMinY = sculptLocalCenterYZ.x - slurryLocalExtent.y * 0.5
        let slurryLocalMaxY = sculptLocalCenterYZ.x + slurryLocalExtent.y * 0.5
        let slurryLocalMinZ = sculptLocalCenterYZ.y - slurryLocalExtent.z * 0.5
        let slurryLocalMaxZ = sculptLocalCenterYZ.y + slurryLocalExtent.z * 0.5

        return (
            SIMD3<Float>(slurryLocalMinX, slurryLocalMinY, slurryLocalMinZ),
            SIMD3<Float>(slurryLocalMaxX, slurryLocalMaxY, slurryLocalMaxZ)
        )
    }

    @MainActor
    private func populateStaticSceneIfNeeded() async {
        guard !didLoadStaticSceneContent else { return }

        if let earStructure = try? await Entity(named: "EarStructure") {
            earStructure.scale *= 2.25
            earStructure.transform.translation += SIMD3(0.09, 0.02, 0.15)
            staticSceneRoot.addChild(earStructure)
        } else {
            print("Failed to find Ear Structure")
        }

        if let stagingProps = try? await Entity(named: "Staging") {
            stagingProps.scale *= 1.8
            stagingProps.transform.translation += SIMD3(0.14, 0, 0.16)
            let rotationQuaternion = simd_quatf(angle: -3.36, axis: SIMD3<Float>(0, 0, 1))
            stagingProps.transform.rotation += rotationQuaternion
            staticSceneRoot.addChild(stagingProps)
        } else {
            print("Failed to find staging props")
        }

        didLoadStaticSceneContent = true
    }

    @MainActor
    private func populateInteractiveAnatomyIfNeeded() async {
        guard !didLoadInteractiveAnatomyContent else { return }

        if let facialNerve = try? await Entity(named: "FacialNerve") {
            facialNerve.scale *= 2.25
            facialNerve.transform.translation += SIMD3(0.09, 0.02, 0.15)
            interactiveAnatomyRoot.addChild(facialNerve)
            sculpting.registerSafezones(for: .facialNerve, entity: facialNerve)
        } else {
            print("Failed to find FacialNerve")
        }

        if let sigmoidSinus = try? await Entity(named: "Sigmoid") {
            sigmoidSinus.scale *= 2.25
            sigmoidSinus.transform.translation += SIMD3(0.09, 0.02, 0.14)
            interactiveAnatomyRoot.addChild(sigmoidSinus)
            sculpting.registerSafezones(for: .sigmoid, entity: sigmoidSinus)
        } else {
            print("Failed to find Simoid")
        }

        didLoadInteractiveAnatomyContent = true
    }

    func sculptingVolume() -> some View {
        RealityView { content, attachments in
            // Initialize visionOS bundled reality-material path (if bundled).
            sculpting.prepareBundledSculptMaterialIfNeeded()

            if staticSceneRoot.parent == nil {
                content.add(staticSceneRoot)
            }
            if interactiveAnatomyRoot.parent == nil {
                content.add(interactiveAnatomyRoot)
            }
            if fluidSceneRoot.parent == nil {
                content.add(fluidSceneRoot)
            }
            sculpting.registerAnatomyEffectsRoot(interactiveAnatomyRoot)

            // Create an entity to render each mesh chunk.
            if let meshChunks = marchingCubesMesh?.meshChunks {
                for meshChunk in meshChunks {
                    if let meshChunkEntity = try? createMeshChunkEntity(meshChunk: meshChunk) {
                        root.addChild(meshChunkEntity)
                    }
                }
            }

            sculpting.drillAudioModel = drillAudio
            drillAudio.prepareAudioIfNeeded()
            drillAudio.attachHazardAudio(to: interactiveAnatomyRoot)

            var toolComponent = SculptingToolComponent(sculptor: sculptor)
            toolComponent.boneSlurryGrid = sculpting.boneSlurryGrid
            sculpting.sculptingTool.components.set(toolComponent)

            root.addChild(sculpting.sculptingTool)

            // Add bone slurry mesh entity to root for rendering.
            if let slurryEntity = sculpting.boneSlurryGrid?.entity {
                root.addChild(slurryEntity)
            }
            sculpting.boneSlurryGrid?.setDebugRootEntity(root)

            content.add(root)
            sculpting.rootEntity = root

            // Set up the bone debris manager with the root entity and SDF access.
            if let voxelVolume = marchingCubesMesh?.voxelVolume {
                sculpting.boneDebrisManager.setup(rootEntity: root, voxelVolume: voxelVolume)
            } else {
                sculpting.boneDebrisManager.setup(rootEntity: root)
            }

            // Set up the collision manager with direct SDF access.
            if let voxelVolume = marchingCubesMesh?.voxelVolume {
                sculpting.collisionManager.setup(rootEntity: root, voxelVolume: voxelVolume)
            }
            applyPerformanceGovernor()

            // Update sculpting tool and check for tracking quality each frame.
            _ = content.subscribe(to: SceneEvents.Update.self) {
                event in
                queueDeferredStartupWorkIfNeeded()
                scheduleStartupReloadIfNeeded()
                if sculpting.shouldClearDebris {
                    sculpting.shouldClearDebris = false
                    handleClearDebris()
                }
                if sculpting.drillBallEntity != nil, !didApplyBitSizeToDrillBall {
                    applySelectedBitSize()
                }
                ensureBoneSlurryGridIfNeeded()
                sculpting.updateSculptingTool()
                createFluidLayerIfNeeded()
                updateFluidLayer(timestep: event.deltaTime)
            }

            if let additiveAttachment = attachments.entity(for: "Additive") {
                sculpting.addEntityAttachmentToRoot(entity: additiveAttachment, name: "AdditiveIcon")
                sculpting.additiveIcon = additiveAttachment
            }

            if let subtractiveAttachment = attachments.entity(for: "Subtractive") {
                sculpting.addEntityAttachmentToRoot(entity: subtractiveAttachment, name: "SubtractiveIcon")
                sculpting.subtractiveIcon = subtractiveAttachment
            }

            if let enlargeAttachment = attachments.entity(for: "Enlarge") {
                sculpting.addEntityAttachmentToRoot(entity: enlargeAttachment, name: "EnlargeIcon")
                sculpting.enlargeIcon = enlargeAttachment
            }

            if let reduceAttachment = attachments.entity(for: "Reduce") {
                sculpting.addEntityAttachmentToRoot(entity: reduceAttachment, name: "ReduceIcon")
                sculpting.reduceIcon = reduceAttachment
            }

            // Iterate over all the currently connected supported spatial accessories.
            // Also, handle notifications of incoming connections.
            await sculpting.handleGameControllerSetup(hapticsModel: haptics)
        } attachments: {
            Attachment(id: "Additive") {
                ToolbarElement(name: "Add")
            }

            Attachment(id: "Subtractive") {
                ToolbarElement(name: "Subtract")
            }

            Attachment(id: "Enlarge") {
                ToolbarElement(name: "Enlarge")
            }

            Attachment(id: "Reduce") {
                ToolbarElement(name: "Reduce")
            }
        }.task {
            // Get transforms of accessories in the app process.
            let configuration = SpatialTrackingSession.Configuration(tracking: [.accessory])
            let session = SpatialTrackingSession()
            await session.run(configuration)
        }
    }

    func saveButton() -> some View {
        Button {
            sculpting.save { document in
                Task { @MainActor in
                    self.saveDocument = document
                    self.isSaving = true
                }
            }
        } label: {
            Text("Save")
        }
        .fileExporter(isPresented: $isSaving,
                      document: saveDocument) { result in
            switch result {
            case .success(let url):
                print("Saved to \(url)")
            case .failure(let error):
                print("Error saving: \(error)")
            }
            isSaving = false
            saveDocument = nil
        } onCancellation: {
            isSaving = false
            saveDocument = nil
        }
    }

    func openButton() -> some View {
        Button {
            performReload()
        } label: {
            Text("Reload")
        }
    }

    @MainActor
    private func performReload() {
        do {
            // Load a packaged sculpt volume from app bundle.
            try sculpting.loadBundledPackage(named: "MyVolume")
            applyLoadedVolumePresentationTransform()
            resetFluidAccumulation()
            if sculpting.sculptingEntity != nil {
                sculpting.collisionManager.scheduleRegeneration()
            }
        } catch {
            // Fallback to legacy .volume file.
            if let url = Bundle.main.url(forResource: "MyModel", withExtension: "volume") {
                do {
                    try sculpting.loadFromURL(url)
                    applyLoadedVolumePresentationTransform()
                    resetFluidAccumulation()
                    if sculpting.sculptingEntity != nil {
                        sculpting.collisionManager.scheduleRegeneration()
                    }
                } catch {
                    print("Failed to open document: \(error)")
                }
            } else {
                print("Failed to open bundled package: \(error)")
            }
        }
    }

    private func applyLoadedVolumePresentationTransform() {
        let preservedZoomFactor = sceneZoomFactor
        let preservedRollAngle = sceneRollAngleRadians
        sceneZoomFactor = 1.0
        sceneRollAngleRadians = 0
        resetZoomedSceneTransformsToBaseline()
        root.transform.rotation = simd_quatf(angle: (.pi * 3.0) / 2.0, axis: SIMD3<Float>(0, 1, 0))
        volumeScaleFactor = 0.75
        applyVolumeScale(volumeScaleFactor)
        if abs(preservedZoomFactor - 1.0) > 0.0001 {
            applySceneZoom(preservedZoomFactor)
        }
        if abs(preservedRollAngle) > 0.0001 {
            applySceneWideRollIncrement(radians: preservedRollAngle)
        } else {
            updateRootGravityFromCurrentRotation()
        }
    }

    func clearButton() -> some View {
        Button {
            sculpting.sculptingTool.components[SculptingToolComponent.self]?.clear = true
            if sculpting.sculptingEntity != nil {
                sculpting.collisionManager.scheduleRegeneration()
            }
        } label: {
            Text("Clear")
        }
    }

    func resetButton() -> some View {
        Button {
            sculpting.sculptingTool.components[SculptingToolComponent.self]?.reset = true
            if sculpting.sculptingEntity != nil {
                sculpting.collisionManager.scheduleRegeneration()
            }
        } label: {
            Text("Reset")
        }
    }

    func toggleDebrisDrawingButton() -> some View {
        Button {
            sculpting.boneDebrisManager.isEnabled.toggle()
        } label: {
            Text(sculpting.boneDebrisManager.isEnabled ? "Debris: ON" : "Debris: OFF")
        }
    }

    func clearDebrisButton() -> some View {
        Button {
            handleClearDebris()
        } label: {
            Text("Clear Debris")
        }
    }

    private func setSlurryDebugEnabled(_ isEnabled: Bool) {
        isSlurryDebugUIEnabled = isEnabled
        sculpting.isSlurryDebugEnabled = isEnabled
        sculpting.boneDebrisManager.setSlurryDebugEnabled(isEnabled)
        sculpting.boneSlurryGrid?.setSlurryDebugEnabled(isEnabled)
        if isEnabled {
            sculpting.updateDrillTipDebugMarker(position: sculpting.sculptingTool.position)
            sculpting.boneDebrisManager.updateDebrisSpawnPreview(sculptingTool: sculpting.sculptingTool)
        } else {
            sculpting.drillTipDebugMarker?.isEnabled = false
        }
    }

    private func setWaterDebugEnabled(_ isEnabled: Bool) {
        isWaterDebugUIEnabled = isEnabled
        waterProbeController.isDebugVisible = isEnabled
    }

    func slurryDebugButton() -> some View {
        Button {
            setSlurryDebugEnabled(!isSlurryDebugUIEnabled)
        } label: {
            Text(isSlurryDebugUIEnabled ? "Slurry Grid ON" : "Slurry Grid Off")
        }
        .buttonStyle(.borderedProminent)
        .tint(isSlurryDebugUIEnabled ? .green : .gray)
    }

    func waterDebugButton() -> some View {
        Button {
            setWaterDebugEnabled(!isWaterDebugUIEnabled)
        } label: {
            Text(isWaterDebugUIEnabled ? "Water Pos ON" : "Water Pos Off")
        }
        .buttonStyle(.borderedProminent)
        .tint(isWaterDebugUIEnabled ? .green : .gray)
    }

    private func handleClearDebris() {
        sculpting.boneDebrisManager.clearAllDebris()
        sculpting.sculptingTool.components[SculptingToolComponent.self]?.previousPosition = nil
        resetFluidAccumulation()

        if let voxelVolume = marchingCubesMesh?.voxelVolume {
            let slurryBounds = slurryBounds(for: voxelVolume)
            sculpting.replaceBoneSlurryGrid(volumeBoundsMin: slurryBounds.min,
                                            volumeBoundsMax: slurryBounds.max)
        } else {
            sculpting.replaceBoneSlurryGrid()
        }
    }

    func toggleGravityButton() -> some View {
        Button {
            sculpting.boneDebrisManager.isGravityEnabled.toggle()
        } label: {
            Text(sculpting.boneDebrisManager.isGravityEnabled ? "Gravity: ON" : "Gravity: OFF")
        }
    }


    func dropSphereButton() -> some View {
        Button {
            sculpting.collisionManager.dropTestSphere()
        } label: {
            Text("Drop Sphere")
        }
    }

    // Rotate the sculpting volume 90° around the given world-space axis.
    // After rotating, counter-rotate the gravity vector so it stays fixed
    // in world space (PhysicsSimulationComponent.gravity is in local space).
    private func rotateVolume(axis: SIMD3<Float>) {
        let increment = simd_quatf(angle: .pi / 2, axis: normalize(axis))
        root.transform.rotation = increment * root.transform.rotation

        updateRootGravityFromCurrentRotation()
    }

    func rotateXButton() -> some View {
        Button { rotateVolume(axis: SIMD3<Float>(1, 0, 0)) } label: { Text("Rot X") }
    }

    func rotateYButton() -> some View {
        Button { rotateVolume(axis: SIMD3<Float>(0, 1, 0)) } label: { Text("Rot Y") }
    }

    func rotateZButton() -> some View {
        Button { rotateVolume(axis: SIMD3<Float>(0, 0, 1)) } label: { Text("Rot Z") }
    }

    func cameraRollButton(_ degrees: Float) -> some View {
        let label = degrees > 0 ? "+45 deg" : "-45 deg"
        return Button {
            applySceneWideRollIncrement(radians: degrees * (.pi / 180))
        } label: {
            Text(label)
        }
    }

    // MARK: - Volume Transparency

    /// Apply or remove 50% transparency on all sculpt mesh chunks.
    private func applyVolumeTransparency(_ transparent: Bool) {
        let opacity: Float = transparent ? 0.5 : 1.0
        for child in root.children where child.name == "SculptMeshChunk" {
            if transparent {
                child.components.set(OpacityComponent(opacity: opacity))
            } else {
                child.components.remove(OpacityComponent.self)
            }
        }
    }

    func toggleTransparencyButton() -> some View {
        Button {
            isVolumeTransparent.toggle()
            applyVolumeTransparency(isVolumeTransparent)
        } label: {
            Text(isVolumeTransparent ? "Xray ON" : "Xray Off")
        }
        .buttonStyle(.borderedProminent)
        .tint(isVolumeTransparent ? .green : .gray)
    }

    // MARK: - Volume Scale

    /// Scale the sculpting volume (mesh chunks + bone slurry + collision) uniformly.
    /// The base scale combines with the scene zoom factor.
    private func applyVolumeScale(_ scale: Float) {
        root.transform.scale = SIMD3<Float>(repeating: scale * sceneZoomFactor)
    }

    func shrinkVolumeButton() -> some View {
        Button {
            let newScale = max(0.4, volumeScaleFactor - 0.1)
            volumeScaleFactor = newScale
            applyVolumeScale(newScale)
        } label: {
            Text("Shrink")
        }
        .disabled(volumeScaleFactor <= 0.41)
    }

    func growVolumeButton() -> some View {
        Button {
            let newScale = min(1.0, volumeScaleFactor + 0.1)
            volumeScaleFactor = newScale
            applyVolumeScale(newScale)
        } label: {
            Text("Grow")
        }
        .disabled(volumeScaleFactor >= 0.99)
    }

    //Final Consolidated View
    var body: some View {
        ZStack {

            //Sculpt Volume
            sculptingVolume()
                .ornament(attachmentAnchor: .scene(.bottomFront)) {
                    VStack {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Bit Size")
                                .font(.caption)
                            HStack {
                                bitSizeButton(6)
                                bitSizeButton(4)
                                bitSizeButton(2)
                            }
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Zoom")
                                .font(.caption)
                            HStack {
                                zoomButton(1.0)
                                zoomButton(1.5)
                                zoomButton(2.0)
                            }
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Camera Rotation")
                                .font(.caption)
                            HStack {
                                cameraRollButton(45)
                                cameraRollButton(-45)
                            }
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("General")
                                .font(.caption)
                            HStack {
                                openButton()
                                clearDebrisButton()
                            }
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Debug")
                                .font(.caption2)
                            HStack {
                                toggleTransparencyButton()
                                slurryDebugButton()
                                waterDebugButton()
                            }
                            HStack {
                                rotateXButton()
                                rotateYButton()
                                rotateZButton()
                            }
                            HStack {
                                shrinkVolumeButton()
                                growVolumeButton()
                            }
                        }
                    }.padding().glassBackgroundEffect()
                }
            //end Sculpting Volume
        }
    }
}
