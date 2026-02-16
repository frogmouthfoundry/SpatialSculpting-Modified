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
    @State var isOpening = false
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
        
        //initialize to loaded model
        guard let url = Bundle.main.url(forResource: "MyModel", withExtension: "volume") else {
            print("Failed to find MyModel.")
            return
        }
        
        do {
            try sculpting.loadFromURL(url)
        } catch {
            print("Failed to load model document: \(error)")
        }
        //
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
    func createMeshChunkEntity(meshChunk: MarchingCubesMeshChunk) throws -> Entity {
        let mesh = try MeshResource(from: meshChunk.mesh)
        let meshChunkEntity = Entity()

        // Load a texture from your asset catalog or bundle
        guard let textureResource = try? TextureResource.load(named: "MyAlbedo") else {
            // Fallback to a flat brown color if the texture is missing
            meshChunkEntity.components.set(
                ModelComponent(mesh: mesh, materials: [SimpleMaterial(color: .brown, roughness: 0.95, isMetallic: false)])
            )
            // Still add collision even on fallback path.
            addChunkPhysics(to: meshChunkEntity, meshChunk: meshChunk)
            return meshChunkEntity
        }

        // Create a physically based material and assign the texture as the base color using the proper API
        var material = PhysicallyBasedMaterial()
        // Base color takes a BaseColor value. Provide a texture by wrapping the TextureResource in a MaterialParameters.Texture
        material.baseColor = PhysicallyBasedMaterial.BaseColor(texture: .init(textureResource))

        // Try to load and apply a normal map
        if let normalMap = try? TextureResource.load(named: "MyNormal") {
            material.normal = .init(texture: .init(normalMap))
        }

        // Optional: add normal/metallic/roughness if you have them
        // if let normalMap = try? TextureResource.load(named: "StoneNormal") {
        //     material.normal = .init(texture: .init(normalMap))
        // }
        material.metallic = .init(floatLiteral: 0.0)
        material.roughness = .init(floatLiteral: 0.95)

        meshChunkEntity.components.set(ModelComponent(mesh: mesh, materials: [material]))
        
        // Add collision + static physics body so bone debris can collide with the volume.
        addChunkPhysics(to: meshChunkEntity, meshChunk: meshChunk)
        
        return meshChunkEntity
    }//end New createMeshChunk function
    
    /// Adds a static collision box and physics body to a mesh chunk entity.
    /// The box covers the chunk's bounding region so bone debris bounces off the sculpted volume.
    private func addChunkPhysics(to entity: Entity, meshChunk: MarchingCubesMeshChunk) {
        let params = meshChunk.params
        
        // Reconstruct the chunk's bounding box from its params
        // (mirrors the bounds calculation in MarchingCubesMesh.createMeshChunk).
        let voxelSize = params.voxelSize
        let chunkDims = SIMD3<Float>(params.chunkDimensions)
        let chunkVoxelStart = params.voxelStartPosition
                              + Float(params.chunkStartZ) * SIMD3<Float>(0, 0, voxelSize.z)
        
        let boundsMin = chunkVoxelStart - voxelSize / 2
        let boundsMax = chunkVoxelStart + voxelSize * chunkDims + voxelSize / 2
        let boundsSize = boundsMax - boundsMin
        let boundsCenter = (boundsMin + boundsMax) / 2
        
        // Create a box shape covering this chunk's region, offset to the correct center.
        let shape = ShapeResource.generateBox(
            width:  boundsSize.x,
            height: boundsSize.y,
            depth:  boundsSize.z
        ).offsetBy(rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                   translation: boundsCenter)
        
        entity.components.set(CollisionComponent(shapes: [shape]))
        entity.components.set(PhysicsBodyComponent(
            shapes: [shape],
            mass: 0,
            material: .default,
            mode: .static
        ))
    }
     

    func sculptingVolume() -> some View {
        RealityView { content, attachments in
            
            
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
            isOpening = true
        } label: {
            Text("Open")
        }
        .fileImporter(isPresented: $isOpening, allowedContentTypes: [VolumeDocument.utType], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let success):
                do {
                    //let url = success[0].absoluteURL
                    
                    //Hijack the load URL for own load model
                    guard let url = Bundle.main.url(forResource: "MyModel", withExtension: "volume") else {
                        print("Failed to find MyModel.")
                        return
                    }
                    
                    try sculpting.loadFromURL(url)
                } catch {
                    print("Failed to open document: \(error)")
                }
            case .failure(let error):
                print("Error opening: \(error)")
            }
            isOpening = false
        } onCancellation: {
            isOpening = false
        }
    }

    func clearButton() -> some View {
        Button {
            sculpting.sculptingTool.components[SculptingToolComponent.self]?.clear = true
        } label: {
            Text("Clear")
        }
    }
    
    func resetButton() -> some View {
        Button {
            sculpting.sculptingTool.components[SculptingToolComponent.self]?.reset = true
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
    
    //Final View
    var body: some View {
        ZStack {
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
                    }.padding().glassBackgroundEffect()
                }
        }
    }
}
