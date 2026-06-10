/*
Abstract:
Manages a slurry density grid locked to a reduced region inside the sculpting volume bounds.
Bone debris particles are splatted into the grid as anisotropic
metaballs, then a secondary marching cubes pass extracts a unified
mesh with a wet PBR material.

GPU dispatch order each frame:
  1. boneSlurryClearMesh — zero vertex/index buffers (degenerate unused triangles)
  2. boneSlurryClear     — zero the density buffer
  3. boneSlurrySplat     — accumulate ellipsoidal density from debris
  4. boneSlurryMarch     — extract isosurface via marching cubes
*/

import RealityKit
import Metal
import UIKit
import QuartzCore

enum BoneSlurryPerformanceTier: Equatable {
    case normal
    case zoomed
}

@MainActor
final class BoneSlurryGrid {

    // MARK: - Grid Constants

    static let gridDimension: UInt32 = 64

    /// Maximum number of vertices the bone slurry mesh can hold.
    private let maxVertexCapacity: Int = 300_000

    /// Maximum number of debris particles to upload.
    private let maxParticleCount: Int = 100

    /// Iso-value threshold for surface extraction.
    /// Lower value = surfaces merge at greater distance → more paste-like.
    /// High isoValue extracts surface near particle centers — small visual size
    /// despite broad density coverage. Growth expands the region where density
    /// exceeds this threshold → visible surface grows. 0.75 with MIN_SPLAT 2.5
    /// gives ~0.9 voxel visual radius at spawn, ~5 voxels at max growth.
    var isoValue: Float = 0.75

    // MARK: - Volume Bounds (set once via configure)

    /// Fixed grid origin and voxel size, locked to the sculpting volume.
    /// Set by configure(volumeBoundsMin:volumeBoundsMax:) at startup.
    private var volumeConfigured: Bool = false
    private var volumeBoundsMin: SIMD3<Float> = .zero
    private var volumeBoundsMax: SIMD3<Float> = .zero
    private var gridOriginInRoot: SIMD3<Float> = .zero
    private let particleBoundsMargin: Float = 0.02

    // MARK: - Metal Resources

    private let densityBuffer: MTLBuffer
    private let particleBuffer: MTLBuffer
    private let counterBuffer: MTLBuffer
    private let triangleTableBuffer: MTLBuffer

    // MARK: - Compute Pipelines

    private let clearMeshPipeline: MTLComputePipelineState
    private let clearDensityPipeline: MTLComputePipelineState
    private let splatPipeline: MTLComputePipelineState
    private let marchPipeline: MTLComputePipelineState

    // MARK: - Dispatch Sizes

    private let totalVoxels: UInt32

    private let clearMeshThreadsPerThreadgroup: MTLSize
    private let clearDensityThreadgroups: MTLSize
    private let clearDensityThreadsPerThreadgroup: MTLSize
    private let marchThreadsPerThreadgroup: MTLSize

    // MARK: - Output Mesh

    var mesh: LowLevelMesh
    let entity: ModelEntity
    private var meshBounds: BoundingBox

    // MARK: - State

    private var params: BoneSlurryGridParams
    private let uploadInterval: TimeInterval = 1.0 / 30.0
    private var lastUploadTime: TimeInterval = 0
    private var meshExtractionInterval: TimeInterval = 1.0 / 12.0
    private var lastMeshExtractionTime: TimeInterval = 0
    private var meshNeedsExtraction: Bool = false
    private var lastParticleSignature: UInt64 = 0
    private var lastUploadedParticleCount: UInt32 = 0
    private var lastExtractedIsoValue: Float = .greatestFiniteMagnitude
    private var isMeshHiddenForEmptyState: Bool = true
    private var activeMaxVertexCount: Int = 300_000
    private var performanceTier: BoneSlurryPerformanceTier = .normal

    // MARK: - Debug

    private var debugFrameCounter: Int = 0
    /// Temporary debug visual for the slurry-grid bounds.
    private var slurryBoundsDebugMarker: ModelEntity?
    private weak var debugRootEntity: Entity?
    private var isSlurryDebugEnabled: Bool = false

    // MARK: - Init

    init?() {
        guard let device = metalDevice else {
            print("[BoneSlurryGrid] No Metal device")
            return nil
        }

        let dim = Self.gridDimension
        let totalVoxelCount = dim * dim * dim  // 64^3 = 262,144
        self.totalVoxels = totalVoxelCount

        // --- Metal Buffers ---

        guard let densityBuf = device.makeBuffer(
            length: Int(totalVoxelCount) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            print("[BoneSlurryGrid] Failed to create density buffer")
            return nil
        }
        self.densityBuffer = densityBuf
        densityBuf.label = "BoneSlurryDensity"

        guard let particleBuf = device.makeBuffer(
            length: maxParticleCount * MemoryLayout<BoneSlurryParticle>.stride,
            options: .storageModeShared
        ) else {
            print("[BoneSlurryGrid] Failed to create particle buffer")
            return nil
        }
        self.particleBuffer = particleBuf
        particleBuf.label = "BoneSlurryParticles"

        guard let counterBuf = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            print("[BoneSlurryGrid] Failed to create counter buffer")
            return nil
        }
        self.counterBuffer = counterBuf
        counterBuf.label = "BoneSlurryCounter"

        let tableLength = MarchingCubesData.triangleTable.count * MemoryLayout<UInt64>.stride
        guard let tableBuf = device.makeBuffer(
            bytes: MarchingCubesData.triangleTable,
            length: tableLength
        ) else {
            print("[BoneSlurryGrid] Failed to create triangle table buffer")
            return nil
        }
        self.triangleTableBuffer = tableBuf
        tableBuf.label = "BoneSlurryTriangleTable"

        // --- Compute Pipelines ---

        guard let clearMeshPipe = makeComputePipeline(named: "boneSlurryClearMesh"),
              let clearDensityPipe = makeComputePipeline(named: "boneSlurryClear"),
              let splatPipe = makeComputePipeline(named: "boneSlurrySplat"),
              let marchPipe = makeComputePipeline(named: "boneSlurryMarch") else {
            print("[BoneSlurryGrid] Failed to create compute pipelines")
            return nil
        }
        self.clearMeshPipeline = clearMeshPipe
        self.clearDensityPipeline = clearDensityPipe
        self.splatPipeline = splatPipe
        self.marchPipeline = marchPipe

        // --- Dispatch Sizes ---

        let maxVerts = maxVertexCapacity

        let clearMeshTpTg = MTLSize(width: 512, height: 1, depth: 1)
        self.clearMeshThreadsPerThreadgroup = clearMeshTpTg

        let clearDensTpTg = MTLSize(width: 512, height: 1, depth: 1)
        let clearDensTg = MTLSize(
            width: (Int(totalVoxelCount) + clearDensTpTg.width - 1) / clearDensTpTg.width,
            height: 1, depth: 1
        )
        self.clearDensityThreadgroups = clearDensTg
        self.clearDensityThreadsPerThreadgroup = clearDensTpTg

        let marchTpTg = MTLSize(width: 4, height: 4, depth: 4)
        self.marchThreadsPerThreadgroup = marchTpTg

        // --- LowLevelMesh ---

        let largeBounds = BoundingBox(
            min: SIMD3<Float>(repeating: -5.0),
            max: SIMD3<Float>(repeating: 5.0)
        )
        self.meshBounds = largeBounds
        guard let llMesh = Self.makeLowLevelMesh(vertexCapacity: maxVerts,
                                                 indexCount: maxVerts,
                                                 bounds: largeBounds) else {
            print("[BoneSlurryGrid] Failed to create LowLevelMesh")
            return nil
        }
        self.mesh = llMesh

        // --- Entity with PBR Material (W1 + W2) ---
        var material = PhysicallyBasedMaterial()
        material.baseColor = .init(tint: UIColor(red: 0.95, green: 0.92, blue: 0.87, alpha: 1.0))
        material.roughness = .init(scale: 0.4)
        material.metallic = .init(scale: 0.02)

        let meshResource = try? MeshResource(from: mesh)
        if let meshRes = meshResource {
            self.entity = ModelEntity(mesh: meshRes, materials: [material])
        } else {
            self.entity = ModelEntity()
            print("[BoneSlurryGrid] Failed to create MeshResource from LowLevelMesh")
        }
        entity.name = "BoneSlurryMesh"

        // --- Default Params (overwritten by configure) ---
        self.params = BoneSlurryGridParams(
            gridOrigin: .zero,
            voxelSize: SIMD3<Float>(repeating: 0.001),
            dimensions: SIMD3<UInt32>(repeating: dim),
            particleCount: 0,
            maxVertexCount: UInt32(maxVerts),
            isoValue: isoValue
        )

        self.activeMaxVertexCount = maxVerts

        print("[BoneSlurryGrid] Initialized: \(dim)^3 grid, maxVerts=\(maxVerts)")
    }

    // MARK: - Configuration

    /// Lock the slurry grid to the sculpting volume bounds. Call once at setup.
    func configure(volumeBoundsMin: SIMD3<Float>, volumeBoundsMax: SIMD3<Float>) {
        self.volumeBoundsMin = volumeBoundsMin
        self.volumeBoundsMax = volumeBoundsMax
        self.gridOriginInRoot = volumeBoundsMin
        self.volumeConfigured = true

        let extent = volumeBoundsMax - volumeBoundsMin
        let dim = Float(Self.gridDimension)
        params.gridOrigin = volumeBoundsMin
        params.voxelSize = extent / dim
        params.maxVertexCount = UInt32(activeMaxVertexCount)
        entity.position = .zero
        updateMeshBounds(min: volumeBoundsMin, max: volumeBoundsMax)
        updateSlurryBoundsDebugMarker(volumeBoundsMin: volumeBoundsMin,
                                      volumeBoundsMax: volumeBoundsMax)

        print("[BoneSlurryGrid] Configured: bounds=\(volumeBoundsMin)...\(volumeBoundsMax), voxelSize=\(params.voxelSize)")
    }

    func setPerformanceTier(_ tier: BoneSlurryPerformanceTier) {
        guard performanceTier != tier else { return }
        performanceTier = tier
        switch tier {
        case .normal:
            meshExtractionInterval = 1.0 / 12.0
            activeMaxVertexCount = maxVertexCapacity
        case .zoomed:
            meshExtractionInterval = 1.0 / 8.0
            activeMaxVertexCount = 180_000
        }
        params.maxVertexCount = UInt32(activeMaxVertexCount)
        meshNeedsExtraction = true
        lastMeshExtractionTime = 0
    }

    /// Debug-only root used so the bounds marker stays visible even when the
    /// slurry mesh entity itself is hidden.
    func setDebugRootEntity(_ rootEntity: Entity?) {
        debugRootEntity = rootEntity
        if let marker = slurryBoundsDebugMarker,
           let rootEntity,
           marker.parent !== rootEntity {
            marker.removeFromParent()
            rootEntity.addChild(marker)
        }
        slurryBoundsDebugMarker?.isEnabled = isSlurryDebugEnabled
    }

    func setSlurryDebugEnabled(_ isEnabled: Bool) {
        isSlurryDebugEnabled = isEnabled
        if isEnabled, volumeConfigured {
            updateSlurryBoundsDebugMarker(volumeBoundsMin: volumeBoundsMin,
                                          volumeBoundsMax: volumeBoundsMax)
        } else {
            slurryBoundsDebugMarker?.isEnabled = false
        }
    }

    private func updateSlurryBoundsDebugMarker(volumeBoundsMin: SIMD3<Float>,
                                               volumeBoundsMax: SIMD3<Float>) {
        guard isSlurryDebugEnabled else {
            slurryBoundsDebugMarker?.isEnabled = false
            return
        }
        let extent = max(volumeBoundsMax - volumeBoundsMin, SIMD3<Float>(repeating: 0.0005))
        let center = (volumeBoundsMin + volumeBoundsMax) * 0.5
        if slurryBoundsDebugMarker == nil {
            let marker = ModelEntity(
                mesh: .generateBox(size: extent),
                materials: [SimpleMaterial(color: .cyan.withAlphaComponent(0.12), roughness: 0.1, isMetallic: false)]
            )
            marker.name = "SlurryBoundsDebugMarker"
            if let debugRootEntity {
                debugRootEntity.addChild(marker)
            } else {
                entity.addChild(marker)
            }
            slurryBoundsDebugMarker = marker
        } else if var model = slurryBoundsDebugMarker?.components[ModelComponent.self] {
            model.mesh = .generateBox(size: extent)
            slurryBoundsDebugMarker?.components.set(model)
        }
        slurryBoundsDebugMarker?.position = center
        slurryBoundsDebugMarker?.isEnabled = true
    }

    // MARK: - Per-Frame Particle Upload (CPU → GPU)

    /// Reads debris entity transforms and uploads them to the particle buffer.
    /// Particle positions stay in root-local space so the GPU emits slurry
    /// vertices in the same coordinate space as the sculpt volume and drill.
    /// Scale comes from the growth multiplier dictionary (not the entity
    /// transform, which physics overwrites). Debris younger than
    /// spawnVisibilityDelay is excluded.
    func uploadParticles(from debrisManager: BoneDebrisManager,
                         rootEntity: Entity) {
        let now = CACurrentMediaTime()
        guard now - lastUploadTime >= uploadInterval else { return }
        lastUploadTime = now

        params.isoValue = isoValue
        guard volumeConfigured else { return }

        let entities = debrisManager.drawnEntities
        let limit = min(entities.count, maxParticleCount)
        let delay = debrisManager.spawnVisibilityDelay

        let ptr = particleBuffer.contents().bindMemory(
            to: BoneSlurryParticle.self,
            capacity: maxParticleCount
        )

        var validCount: Int = 0
        var signature: UInt64 = 0xcbf29ce484222325
        for i in 0..<limit {
            guard let modelEntity = entities[i] as? ModelEntity,
                  modelEntity.parent != nil else { continue }

            let id = ObjectIdentifier(modelEntity)

            // Skip debris that hasn't reached visibility age.
            if let spawnTime = debrisManager.spawnTimes[id],
               now - spawnTime < delay { continue }

            let pos = modelEntity.position(relativeTo: rootEntity)

            // Skip particles outside the volume bounds.
            if pos.x < (volumeBoundsMin.x - particleBoundsMargin) ||
               pos.y < (volumeBoundsMin.y - particleBoundsMargin) ||
               pos.z < (volumeBoundsMin.z - particleBoundsMargin) ||
               pos.x > (volumeBoundsMax.x + particleBoundsMargin) ||
               pos.y > (volumeBoundsMax.y + particleBoundsMargin) ||
               pos.z > (volumeBoundsMax.z + particleBoundsMargin) {
                continue
            }

            let rot = modelEntity.orientation(relativeTo: rootEntity)
            // Use growth multiplier (decoupled from physics transform).
            let scl = debrisManager.growthMultipliers[id] ?? SIMD3<Float>(repeating: 1.0)
            ptr[validCount] = BoneSlurryParticle(
                position: pos,
                rotation: SIMD4<Float>(rot.imag.x, rot.imag.y, rot.imag.z, rot.real),
                scale: scl
            )
            signature = mixedParticleSignature(signature: signature,
                                               position: pos,
                                               rotation: rot,
                                               scale: scl)
            validCount += 1
        }

        params.particleCount = UInt32(validCount)
        let particleCount = params.particleCount
        let isoChanged = abs(lastExtractedIsoValue - params.isoValue) > 0.0001
        if particleCount != lastUploadedParticleCount || signature != lastParticleSignature || isoChanged {
            meshNeedsExtraction = true
            lastUploadedParticleCount = particleCount
            lastParticleSignature = signature
        }

        debugFrameCounter += 1
        if debugFrameCounter <= 0 {
            // Log scale range to verify growth reaches GPU
            var minScl: Float = Float.greatestFiniteMagnitude
            var maxScl: Float = 0
            for j in 0..<validCount {
                let s = ptr[j].scale
                let mag = s.x * s.y * s.z
                minScl = min(minScl, mag)
                maxScl = max(maxScl, mag)
            }
            let minStr = String(format: "%.4f", minScl)
            let maxStr = String(format: "%.4f", maxScl)
            print("[BoneSlurryGrid] frame=\(debugFrameCounter) particles=\(validCount)/\(entities.count) scaleVol=\(minStr)...\(maxStr)")
        }
    }

    func clearAllParticlesAndMesh() {
        params.particleCount = 0
        params.maxVertexCount = UInt32(activeMaxVertexCount)
        lastUploadedParticleCount = 0
        lastParticleSignature = 0
        lastExtractedIsoValue = .greatestFiniteMagnitude
        lastMeshExtractionTime = 0
        meshNeedsExtraction = false
        rebuildMesh(indexCount: 0)
        entity.isEnabled = false
        isMeshHiddenForEmptyState = true
    }

    // MARK: - GPU Dispatch (called from ComputeSystem)

    func update(computeContext: inout ComputeUpdateContext) {
        let now = CACurrentMediaTime()
        guard params.particleCount > 0 else {
            // Do not rebuild the LowLevelMesh every empty frame. That causes
            // heavy RealityKit/Metal churn during launch and after load, when
            // the slurry often has zero particles for extended periods.
            if !isMeshHiddenForEmptyState {
                entity.isEnabled = false
                isMeshHiddenForEmptyState = true
            }
            return
        }
        guard meshNeedsExtraction,
              now - lastMeshExtractionTime >= meshExtractionInterval else { return }

        guard let computeEncoder = computeContext.computeEncoder() else { return }
        params.maxVertexCount = UInt32(activeMaxVertexCount)
        mesh.parts.replaceAll([LowLevelMesh.Part(indexCount: activeMaxVertexCount, bounds: meshBounds)])
        entity.isEnabled = true
        isMeshHiddenForEmptyState = false
        lastMeshExtractionTime = now
        meshNeedsExtraction = false
        lastExtractedIsoValue = params.isoValue

        let vertexBuffer = mesh.replace(bufferIndex: 0, using: computeContext.commandBuffer)
        let indexBuffer = mesh.replaceIndices(using: computeContext.commandBuffer)

        // 1. Clear mesh
        var maxVerts = UInt32(activeMaxVertexCount)
        computeEncoder.setComputePipelineState(clearMeshPipeline)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&maxVerts, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.dispatchThreadgroups(clearMeshThreadgroups(for: activeMaxVertexCount),
                                            threadsPerThreadgroup: clearMeshThreadsPerThreadgroup)

        // 2. Clear density
        var totalCount = totalVoxels
        computeEncoder.setComputePipelineState(clearDensityPipeline)
        computeEncoder.setBuffer(densityBuffer, offset: 0, index: 0)
        computeEncoder.setBytes(&totalCount, length: MemoryLayout<UInt32>.stride, index: 1)
        computeEncoder.dispatchThreadgroups(clearDensityThreadgroups, threadsPerThreadgroup: clearDensityThreadsPerThreadgroup)

        computeEncoder.memoryBarrier(scope: .buffers)

        // 3. Splat particles
        computeEncoder.setComputePipelineState(splatPipeline)
        computeEncoder.setBuffer(densityBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(particleBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&params, length: MemoryLayout<BoneSlurryGridParams>.stride, index: 2)

        let splatThreads = Int(params.particleCount)
        let splatTpTg = MTLSize(width: min(splatThreads, 64), height: 1, depth: 1)
        let splatTg = MTLSize(
            width: (splatThreads + splatTpTg.width - 1) / splatTpTg.width,
            height: 1, depth: 1
        )
        computeEncoder.dispatchThreadgroups(splatTg, threadsPerThreadgroup: splatTpTg)

        computeEncoder.memoryBarrier(scope: .buffers)

        // 4. Marching cubes
        counterBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        computeEncoder.setComputePipelineState(marchPipeline)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(densityBuffer, offset: 0, index: 2)
        computeEncoder.setBytes(&params, length: MemoryLayout<BoneSlurryGridParams>.stride, index: 3)
        computeEncoder.setBuffer(triangleTableBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(counterBuffer, offset: 0, index: 5)
        computeEncoder.dispatchThreadgroups(marchThreadgroups(),
                                            threadsPerThreadgroup: marchThreadsPerThreadgroup)

        // Periodic telemetry
        let shouldLog = false
        if shouldLog {
            let counterBufLocal = counterBuffer
            let frameNum = debugFrameCounter
            let particleCountLocal = params.particleCount
            let maxVertsLocal = params.maxVertexCount
            computeContext.commandBuffer.addCompletedHandler { _ in
                let realTriCount = counterBufLocal.contents().load(as: UInt32.self)
                let status = realTriCount > 0 ? "OK" : "ZERO"
                print("[BoneSlurryGrid] \(status) frame=\(frameNum) tris=\(realTriCount)/\(maxVertsLocal/3) particles=\(particleCountLocal)")
            }
        }
    }

    private static func makeLowLevelMesh(vertexCapacity: Int,
                                         indexCount: Int,
                                         bounds: BoundingBox) -> LowLevelMesh? {
        let meshDesc = LowLevelMesh.Descriptor(
            vertexCapacity: vertexCapacity,
            vertexAttributes: BoneSlurryVertex.vertexAttributes,
            vertexLayouts: BoneSlurryVertex.vertexLayouts,
            indexCapacity: vertexCapacity
        )
        guard let llMesh = try? LowLevelMesh(descriptor: meshDesc) else { return nil }
        llMesh.parts.replaceAll([LowLevelMesh.Part(indexCount: indexCount, bounds: bounds)])
        return llMesh
    }

    private func rebuildMesh(indexCount: Int) {
        guard let replacementMesh = Self.makeLowLevelMesh(vertexCapacity: maxVertexCapacity,
                                                          indexCount: indexCount,
                                                          bounds: meshBounds) else {
            mesh.parts.replaceAll([LowLevelMesh.Part(indexCount: indexCount, bounds: meshBounds)])
            return
        }

        mesh = replacementMesh
        guard let meshResource = try? MeshResource(from: replacementMesh),
              var model = entity.components[ModelComponent.self] else {
            print("[BoneSlurryGrid] Failed to rebuild slurry mesh resource")
            return
        }

        model.mesh = meshResource
        entity.components.set(model)
    }

    private func mixedParticleSignature(signature: UInt64,
                                        position: SIMD3<Float>,
                                        rotation: simd_quatf,
                                        scale: SIMD3<Float>) -> UInt64 {
        var hash = signature
        hash = mixHash(hash, quantized(position.x, step: 0.002))
        hash = mixHash(hash, quantized(position.y, step: 0.002))
        hash = mixHash(hash, quantized(position.z, step: 0.002))
        hash = mixHash(hash, quantized(rotation.imag.x, step: 0.08))
        hash = mixHash(hash, quantized(rotation.imag.y, step: 0.08))
        hash = mixHash(hash, quantized(rotation.imag.z, step: 0.08))
        hash = mixHash(hash, quantized(rotation.real, step: 0.08))
        hash = mixHash(hash, quantized(scale.x, step: 0.05))
        hash = mixHash(hash, quantized(scale.y, step: 0.05))
        hash = mixHash(hash, quantized(scale.z, step: 0.05))
        return hash
    }

    private func quantized(_ value: Float, step: Float) -> UInt64 {
        UInt64(bitPattern: Int64((value / step).rounded(.toNearestOrEven)))
    }

    private func mixHash(_ hash: UInt64, _ value: UInt64) -> UInt64 {
        let prime: UInt64 = 1099511628211
        return (hash ^ value) &* prime
    }

    private func clearMeshThreadgroups(for maxVertices: Int) -> MTLSize {
        MTLSize(
            width: (maxVertices + clearMeshThreadsPerThreadgroup.width - 1) / clearMeshThreadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
    }

    private func marchThreadgroups() -> MTLSize {
        let cubes = Int(Self.gridDimension - 1)
        return MTLSize(
            width: (cubes + marchThreadsPerThreadgroup.width - 1) / marchThreadsPerThreadgroup.width,
            height: (cubes + marchThreadsPerThreadgroup.height - 1) / marchThreadsPerThreadgroup.height,
            depth: (cubes + marchThreadsPerThreadgroup.depth - 1) / marchThreadsPerThreadgroup.depth
        )
    }

    private func updateMeshBounds(min: SIMD3<Float>, max: SIMD3<Float>) {
        let padding = params.voxelSize + SIMD3<Float>(repeating: 0.01)
        meshBounds = BoundingBox(min: min - padding, max: max + padding)
        mesh.parts.replaceAll([LowLevelMesh.Part(indexCount: activeMaxVertexCount, bounds: meshBounds)])
    }
}
