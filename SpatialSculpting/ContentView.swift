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

struct ContentView: View {
    var root: Entity = Entity(components: [ComputeSystemComponent(computeSystem: SculptingToolSystem())])

    @State var sculpting: SculptingToolModel = SculptingToolModel()
    @State var haptics: HapticsModel = HapticsModel()

    let marchingCubesMesh: MarchingCubesMesh!
    let sculptor: MarchingCubesMeshSculptor!

    @State var saveDocument: VolumeDocument? = nil
    @State var isSaving = false
    @State private var fluidWaveMesh: AnimatedWaveMesh? = nil
    @State private var fluidWaveEntity: ModelEntity? = nil
    @State private var lastFluidRippleTimestamp: TimeInterval = 0

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
            waveMesh.segmentCount = 128
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
            material.faceCulling = .none

            let entity = ModelEntity(mesh: meshResource, materials: [material])
            entity.name = "FluidLayer"
            entity.position = SIMD3<Float>(0, -0.22, 0)
            entity.scale = SIMD3<Float>(0.64, 0.8, 0.64)
            root.addChild(entity)

            fluidWaveMesh = waveMesh
            fluidWaveEntity = entity
        } catch {
            print("Failed to create fluid layer: \(error)")
        }
    }

    private func updateFluidLayer(timestep: TimeInterval) {
        fluidWaveMesh?.update(timestep)
        triggerFluidRippleIfNeeded()
    }

    private func triggerFluidRippleIfNeeded() {
        guard let waveMesh = fluidWaveMesh, let fluidEntity = fluidWaveEntity else { return }
        let toolPositionInFluidSpace = fluidEntity.convert(position: sculpting.sculptingTool.position, from: root)
        guard abs(toolPositionInFluidSpace.x) < 0.5,
              abs(toolPositionInFluidSpace.z) < 0.5,
              abs(toolPositionInFluidSpace.y) < 0.03 else {
            return
        }

        let now = CACurrentMediaTime()
        guard now - lastFluidRippleTimestamp > 0.07 else { return }
        lastFluidRippleTimestamp = now
        waveMesh.triggerRipple(at: SIMD2<Float>(toolPositionInFluidSpace.x, toolPositionInFluidSpace.z))
    }

    func sculptingVolume() -> some View {
        RealityView { content, attachments in
            // Initialize visionOS bundled reality-material path (if bundled).
            sculpting.prepareBundledSculptMaterialIfNeeded()

            // Create an entity to render each mesh chunk.
            if let meshChunks = marchingCubesMesh?.meshChunks {
                for meshChunk in meshChunks {
                    if let meshChunkEntity = try? createMeshChunkEntity(meshChunk: meshChunk) {
                        root.addChild(meshChunkEntity)
                    }
                }
            }

            var toolComponent = SculptingToolComponent(sculptor: sculptor)
            toolComponent.boneSlurryGrid = sculpting.boneSlurryGrid
            sculpting.sculptingTool.components.set(toolComponent)

            root.addChild(sculpting.sculptingTool)

            // Add bone slurry mesh entity to root for rendering.
            if let slurryEntity = sculpting.boneSlurryGrid?.entity {
                root.addChild(slurryEntity)
            }

            // Add animated fluid layer mesh.
            createFluidLayerIfNeeded()

            content.add(root)
            sculpting.rootEntity = root

            // Set up the bone debris manager with the root entity and SDF access.
            if let voxelVolume = marchingCubesMesh?.voxelVolume {
                sculpting.boneDebrisManager.setup(rootEntity: root, voxelVolume: voxelVolume)
                // Lock slurry grid to the same bounds as the sculpting volume.
                let dims = SIMD3<Float>(voxelVolume.dimensions)
                let bMin = voxelVolume.voxelStartPosition - voxelVolume.voxelSize * 0.5
                let bMax = voxelVolume.voxelStartPosition + voxelVolume.voxelSize * (dims - 0.5)
                sculpting.boneSlurryGrid?.configure(volumeBoundsMin: bMin, volumeBoundsMax: bMax)
            } else {
                sculpting.boneDebrisManager.setup(rootEntity: root)
            }

            // Pre-load bone dust particle template to avoid disk I/O during sculpting.
            sculpting.preloadBoneDust()

            // Set up the collision manager with direct SDF access.
            if let voxelVolume = marchingCubesMesh?.voxelVolume {
                sculpting.collisionManager.setup(rootEntity: root, voxelVolume: voxelVolume)
            }
            // Schedule initial collision generation (delay for GPU to render first mesh).
            sculpting.collisionManager.scheduleRegeneration()

            // Update sculpting tool and check for tracking quality each frame.
            _ = content.subscribe(to: SceneEvents.Update.self) {
                event in
                sculpting.updateSculptingTool()
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
            do {
                // Load a packaged sculpt volume from app bundle.
                try sculpting.loadBundledPackage(named: "MyVolume")
                sculpting.collisionManager.scheduleRegeneration()
            } catch {
                // Fallback to legacy .volume file.
                if let url = Bundle.main.url(forResource: "MyModel", withExtension: "volume") {
                    do {
                        try sculpting.loadFromURL(url)
                        sculpting.collisionManager.scheduleRegeneration()
                    } catch {
                        print("Failed to open document: \(error)")
                    }
                } else {
                    print("Failed to open bundled package: \(error)")
                }
            }
        } label: {
            Text("Load")
        }
    }

    func clearButton() -> some View {
        Button {
            sculpting.sculptingTool.components[SculptingToolComponent.self]?.clear = true
            sculpting.collisionManager.scheduleRegeneration()
        } label: {
            Text("Clear")
        }
    }

    func resetButton() -> some View {
        Button {
            sculpting.sculptingTool.components[SculptingToolComponent.self]?.reset = true
            sculpting.collisionManager.scheduleRegeneration()
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
            sculpting.boneDebrisManager.clearAllDebris()
        } label: {
            Text("Clear Debris")
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

        // Express the desired world-space gravity in the root's new local space.
        let worldGravity = BoneDebrisManager.debrisGravity
        let inverseRotation = root.transform.rotation.inverse
        let localGravity = inverseRotation.act(worldGravity)
        var sim = root.components[PhysicsSimulationComponent.self] ?? PhysicsSimulationComponent()
        sim.gravity = localGravity
        root.components.set(sim)
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
            Text(isVolumeTransparent ? "Opaque" : "Transparent")
        }
    }

    // MARK: - Volume Scale

    /// Scale the sculpting volume (mesh chunks + bone slurry + collision) uniformly.
    /// Drill overlay, drill ball, and sculpting tool head are NOT affected since
    /// they are positioned relative to the accessory anchor, not the root entity.
    private func applyVolumeScale(_ scale: Float) {
        root.transform.scale = SIMD3<Float>(repeating: scale)
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
                        HStack {
                            saveButton()
                            openButton()
                            clearDebrisButton()
                        }
                        HStack {
                            rotateXButton()
                            rotateYButton()
                            rotateZButton()
                        }
                        HStack {
                            toggleTransparencyButton()
                            shrinkVolumeButton()
                            growVolumeButton()
                        }
                    }.padding().glassBackgroundEffect()
                }
            //end Sculpting Volume

            /*
            //Additional 3D Content
            RealityView { content in

                guard let stageEntity = try? Entity.load(named: "EarStructure") else {
                    print("Failed to find Ear Structure")
                    return }

                stageEntity.scale *= 0.3
                //stageEntity.transform.translation += SIMD3(0,0,0.1)
                content.add(stageEntity)
            }
            //end Additional 3D Content
             */
        }
    }
}
