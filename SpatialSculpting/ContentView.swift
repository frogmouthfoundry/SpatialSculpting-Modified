/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The volume for sculpting and UI for controls.
*/

import SwiftUI
import ARKit
import RealityKit
import GameController
import CoreHaptics

struct ContentView: View {
    var root: Entity = Entity(components: [ComputeSystemComponent(computeSystem: SculptingToolSystem())])
    
    @State var sculpting: SculptingToolModel = SculptingToolModel()
    @State var haptics: HapticsModel = HapticsModel()
    
    let marchingCubesMesh: MarchingCubesMesh!
    let sculptor: MarchingCubesMeshSculptor!

    @State var saveDocument: VolumeDocument? = nil
    @State var isSaving = false

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
            
            sculpting.sculptingTool.components.set(SculptingToolComponent(sculptor: sculptor))
            
            root.addChild(sculpting.sculptingTool)
            
            content.add(root)
            sculpting.rootEntity = root
            
            // Set up the bone debris manager with the root entity.
            sculpting.boneDebrisManager.setup(rootEntity: root)

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
                _ in
                sculpting.updateSculptingTool()
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
            Text("Open")
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
    
    func diagnoseButton() -> some View {
        Button {
            if let root = sculpting.rootEntity {
                PhysicsDiagnostics.runAll(rootEntity: root)
            }
        } label: {
            Text("Diagnose")
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
                            clearButton()
                            resetButton()
                            toggleDebrisDrawingButton()
                            clearDebrisButton()
                            toggleGravityButton()
                            diagnoseButton()
                            dropSphereButton()
                        }
                        HStack {
                            rotateXButton()
                            rotateYButton()
                            rotateZButton()
                        }
                    }.padding().glassBackgroundEffect()
                }
            //end Sculpting Volume
            
            /*
            //Additional 3D Content
            RealityView { content in
                
                guard let stageEntity = try? Entity.load(named: "Staging") else {
                    print("❌ Failed to find Staging Entity")
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
