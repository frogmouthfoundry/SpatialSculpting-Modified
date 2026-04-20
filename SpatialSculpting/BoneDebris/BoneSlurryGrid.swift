/*
Abstract:
Manages a 64^3 density grid locked to the sculpting volume bounds.
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

@MainActor
final class BoneSlurryGrid {

    // MARK: - Grid Constants

    static let gridDimension: UInt32 = 128

    /// Maximum number of vertices the bone slurry mesh can hold.
    private let maxVertexCapacity: Int = 300_000

    /// Maximum number of debris particles to upload.
    private let maxParticleCount: Int = 400

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

    private let clearMeshThreadgroups: MTLSize
    private let clearMeshThreadsPerThreadgroup: MTLSize
    private let clearDensityThreadgroups: MTLSize
    private let clearDensityThreadsPerThreadgroup: MTLSize
    private let marchThreadgroups: MTLSize
    private let marchThreadsPerThreadgroup: MTLSize

    // MARK: - Output Mesh

    let mesh: LowLevelMesh
    let entity: ModelEntity

    // MARK: - State

    private var params: BoneSlurryGridParams

    // MARK: - Debug

    private var debugFrameCounter: Int = 0

    // MARK: - Init

    init?() {
        guard let device = metalDevice else {
            print("[BoneSlurryGrid] No Metal device")
            return nil
        }

        let dim = Self.gridDimension
        let totalVoxelCount = dim * dim * dim  // 262,144
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
        let clearMeshTg = MTLSize(
            width: (maxVerts + clearMeshTpTg.width - 1) / clearMeshTpTg.width,
            height: 1, depth: 1
        )
        self.clearMeshThreadgroups = clearMeshTg
        self.clearMeshThreadsPerThreadgroup = clearMeshTpTg

        let clearDensTpTg = MTLSize(width: 512, height: 1, depth: 1)
        let clearDensTg = MTLSize(
            width: (Int(totalVoxelCount) + clearDensTpTg.width - 1) / clearDensTpTg.width,
            height: 1, depth: 1
        )
        self.clearDensityThreadgroups = clearDensTg
        self.clearDensityThreadsPerThreadgroup = clearDensTpTg

        let marchTpTg = MTLSize(width: 4, height: 4, depth: 4)
        let cubes = Int(dim - 1)
        let marchTg = MTLSize(
            width: (cubes + marchTpTg.width - 1) / marchTpTg.width,
            height: (cubes + marchTpTg.height - 1) / marchTpTg.height,
            depth: (cubes + marchTpTg.depth - 1) / marchTpTg.depth
        )
        self.marchThreadgroups = marchTg
        self.marchThreadsPerThreadgroup = marchTpTg

        // --- LowLevelMesh ---

        let meshDesc = LowLevelMesh.Descriptor(
            vertexCapacity: maxVerts,
            vertexAttributes: BoneSlurryVertex.vertexAttributes,
            vertexLayouts: BoneSlurryVertex.vertexLayouts,
            indexCapacity: maxVerts
        )
        guard let llMesh = try? LowLevelMesh(descriptor: meshDesc) else {
            print("[BoneSlurryGrid] Failed to create LowLevelMesh")
            return nil
        }
        self.mesh = llMesh

        let largeBounds = BoundingBox(
            min: SIMD3<Float>(repeating: -5.0),
            max: SIMD3<Float>(repeating: 5.0)
        )
        mesh.parts.replaceAll([LowLevelMesh.Part(indexCount: maxVerts, bounds: largeBounds)])

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

        print("[BoneSlurryGrid] Initialized: \(dim)^3 grid, maxVerts=\(maxVerts)")
    }

    // MARK: - Configuration

    /// Lock the slurry grid to the sculpting volume bounds. Call once at setup.
    func configure(volumeBoundsMin: SIMD3<Float>, volumeBoundsMax: SIMD3<Float>) {
        self.volumeBoundsMin = volumeBoundsMin
        self.volumeBoundsMax = volumeBoundsMax
        self.volumeConfigured = true

        let extent = volumeBoundsMax - volumeBoundsMin
        let dim = Float(Self.gridDimension)
        params.gridOrigin = volumeBoundsMin
        params.voxelSize = extent / dim

        print("[BoneSlurryGrid] Configured: bounds=\(volumeBoundsMin)...\(volumeBoundsMax), voxelSize=\(params.voxelSize)")
    }

    // MARK: - Per-Frame Particle Upload (CPU → GPU)

    /// Reads debris entity transforms and uploads them to the particle buffer.
    /// Grid origin/voxelSize are fixed to the sculpting volume — only particles
    /// within bounds are uploaded. Scale comes from the growth multiplier
    /// dictionary (not the entity transform, which physics overwrites).
    /// Debris younger than spawnVisibilityDelay is excluded.
    func uploadParticles(from debrisManager: BoneDebrisManager,
                         rootEntity: Entity) {
        params.isoValue = isoValue
        guard volumeConfigured else { return }

        let entities = debrisManager.drawnEntities
        let limit = min(entities.count, maxParticleCount)
        let now = CACurrentMediaTime()
        let delay = debrisManager.spawnVisibilityDelay

        let ptr = particleBuffer.contents().bindMemory(
            to: BoneSlurryParticle.self,
            capacity: maxParticleCount
        )

        var validCount: Int = 0
        for i in 0..<limit {
            guard let modelEntity = entities[i] as? ModelEntity,
                  modelEntity.parent != nil else { continue }

            let id = ObjectIdentifier(modelEntity)

            // Skip debris that hasn't reached visibility age.
            if let spawnTime = debrisManager.spawnTimes[id],
               now - spawnTime < delay { continue }

            let pos = modelEntity.position(relativeTo: rootEntity)

            // Skip particles outside the volume bounds.
            if pos.x < volumeBoundsMin.x || pos.y < volumeBoundsMin.y || pos.z < volumeBoundsMin.z ||
               pos.x > volumeBoundsMax.x || pos.y > volumeBoundsMax.y || pos.z > volumeBoundsMax.z {
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
            validCount += 1
        }

        params.particleCount = UInt32(validCount)

        debugFrameCounter += 1
        if debugFrameCounter <= 30 || debugFrameCounter % 60 == 0 {
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

    // MARK: - GPU Dispatch (called from ComputeSystem)

    func update(computeContext: inout ComputeUpdateContext) {
        guard let computeEncoder = computeContext.computeEncoder() else { return }

        let vertexBuffer = mesh.replace(bufferIndex: 0, using: computeContext.commandBuffer)
        let indexBuffer = mesh.replaceIndices(using: computeContext.commandBuffer)

        // 1. Clear mesh
        var maxVerts = UInt32(maxVertexCapacity)
        computeEncoder.setComputePipelineState(clearMeshPipeline)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&maxVerts, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.dispatchThreadgroups(clearMeshThreadgroups, threadsPerThreadgroup: clearMeshThreadsPerThreadgroup)

        guard params.particleCount > 0 else { return }

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
        computeEncoder.dispatchThreadgroups(marchThreadgroups, threadsPerThreadgroup: marchThreadsPerThreadgroup)

        // Periodic telemetry
        let shouldLog = debugFrameCounter <= 30 || debugFrameCounter % 60 == 0
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
}
