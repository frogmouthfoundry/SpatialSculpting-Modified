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

    init() {
        let dimensions = SIMD3<UInt32>(128, 128, 128)
        let voxelSize = SIMD3<Float>(1.0, 1.0, 1.0) / SIMD3<Float>(dimensions)
        let voxelStartPosition = -SIMD3<Float>(dimensions) * voxelSize / 2
        
        guard let voxelVolume = try? VoxelVolume(dimensions: dimensions, voxelSize: voxelSize, voxelStartPosition: voxelStartPosition) else {
            self.marchingCubesMesh = nil
            self.sculptor = nil
            print("Failed to create volume.")
            return
        }
        
        self.marchingCubesMesh = try? MarchingCubesMesh(voxelVolume: voxelVolume)
        self.sculptor = MarchingCubesMeshSculptor(marchingCubesMesh: marchingCubesMesh)
        
    }
    
    /*
    //ORIGINAL createMeshChunkEntity
    func createMeshChunkEntity(meshChunk: MarchingCubesMeshChunk) throws -> Entity {
        let mesh = try MeshResource(from: meshChunk.mesh)
        let meshChunkEntity = Entity()
        meshChunkEntity.components.set(ModelComponent(mesh: mesh, materials: [SimpleMaterial(color: #colorLiteral(red: 0.6000000238, green: 0.6000000238, blue: 0.6000000238, alpha: 1), roughness: 0.8, isMetallic: false)]))
        return meshChunkEntity
    }
     */
    
    
    //NEW MESH CHUNK ENTITY FUNCTION
    func createMeshChunkEntity(meshChunk: MarchingCubesMeshChunk) async throws -> Entity {
        let mesh = try await MeshResource(from: meshChunk.mesh)
        let meshChunkEntity = Entity()

        guard let textureResource = try? TextureResource.load(named: "MyAlbedo") else {
            meshChunkEntity.components.set(
                ModelComponent(mesh: mesh, materials: [SimpleMaterial(color: .brown, roughness: 0.95, isMetallic: false)])
            )
            return meshChunkEntity
        }

        var material = PhysicallyBasedMaterial()
        material.baseColor = PhysicallyBasedMaterial.BaseColor(texture: .init(textureResource))

        if let normalMap = try? TextureResource.load(named: "MyNormal") {
            material.normal = .init(texture: .init(normalMap))
        }

        material.metallic = .init(floatLiteral: 0.0)
        material.roughness = .init(floatLiteral: 0.95)

        meshChunkEntity.components.set(ModelComponent(mesh: mesh, materials: [material]))
        
        return meshChunkEntity
    }//end New createMeshChunk function
     
    func sculptingVolume() -> some View {
        RealityView { content, attachments in
            
            
            // Create an entity to render each mesh chunk.
            if let meshChunks = marchingCubesMesh?.meshChunks {
                for meshChunk in meshChunks {
                    if let meshChunkEntity = try? await createMeshChunkEntity(meshChunk: meshChunk) {
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
            guard let url = Bundle.main.url(forResource: "MyModel", withExtension: "volume") else {
                print("Failed to find MyModel.")
                return
            }
            
            do {
                try sculpting.loadFromURL(url)
                sculpting.collisionManager.scheduleRegeneration()
            } catch {
                print("Failed to open document: \(error)")
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
    
    //Final Consolidated View
    var body: some View {
        ZStack {
            
            //Sculpt Volume
            sculptingVolume()
                .ornament(attachmentAnchor: .scene(.bottomFront)) {
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
