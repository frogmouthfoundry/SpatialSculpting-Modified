/*
Abstract:
Manages a 64^3 density grid that follows the debris centroid.
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

    static let gridDimension: UInt32 = 64

    /// Default grid extent (used for init only; actual extent is dynamic per-frame).
    private static let defaultExtent: Float = 0.05  // 5 cm cube
    private static let defaultVoxelSize: Float = defaultExtent / Float(gridDimension)

    /// Maximum number of vertices the bone slurry mesh can hold.
    private let maxVertexCapacity: Int = 300_000

    /// Maximum number of debris particles to upload.
    private let maxParticleCount: Int = 150

    /// Iso-value threshold for surface extraction.
    /// Density > isoValue → inside the metaball.
    var isoValue: Float = 0.25

    // MARK: - Metal Resources

    /// Density buffer: 64^3 atomic_uint values for splatting.
    private let densityBuffer: MTLBuffer

    /// Particle data uploaded from CPU each frame.
    private let particleBuffer: MTLBuffer

    /// Atomic counter for marching cubes vertex allocation.
    private let counterBuffer: MTLBuffer

    /// Reuses the existing triangle table from MarchingCubesData.
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

        // Triangle table from existing MarchingCubesData.
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

        // Clear mesh: one thread per vertex slot.
        let clearMeshTpTg = MTLSize(width: 512, height: 1, depth: 1)
        let clearMeshTg = MTLSize(
            width: (maxVerts + clearMeshTpTg.width - 1) / clearMeshTpTg.width,
            height: 1, depth: 1
        )
        self.clearMeshThreadgroups = clearMeshTg
        self.clearMeshThreadsPerThreadgroup = clearMeshTpTg

        // Clear density: one thread per voxel.
        let clearDensTpTg = MTLSize(width: 512, height: 1, depth: 1)
        let clearDensTg = MTLSize(
            width: (Int(totalVoxelCount) + clearDensTpTg.width - 1) / clearDensTpTg.width,
            height: 1, depth: 1
        )
        self.clearDensityThreadgroups = clearDensTg
        self.clearDensityThreadsPerThreadgroup = clearDensTpTg

        // March: one thread per cube = (dim-1)^3.
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

        // Set indexCount to full capacity — unused vertices are zeroed by clear pass
        // (degenerate triangles at origin, invisible). This matches the existing
        // MarchingCubesMesh pattern exactly and avoids completed-handler timing issues.
        // Use a large bounding box so RealityKit never frustum-culls the mesh.
        let largeBounds = BoundingBox(
            min: SIMD3<Float>(repeating: -2.0),
            max: SIMD3<Float>(repeating: 2.0)
        )
        mesh.parts.replaceAll([LowLevelMesh.Part(indexCount: maxVerts, bounds: largeBounds)])

        // --- Entity with PBR Material ---

        // Wet bone slurry: white/cream, medium roughness, low metallic.
        var material = PhysicallyBasedMaterial()
        material.baseColor = .init(tint: UIColor(red: 0.95, green: 0.92, blue: 0.85, alpha: 1.0))
        material.roughness = .init(scale: 0.7)
        material.metallic = .init(scale: 0.05)

        let meshResource = try? MeshResource(from: mesh)
        if let meshRes = meshResource {
            self.entity = ModelEntity(mesh: meshRes, materials: [material])
        } else {
            self.entity = ModelEntity()
            print("[BoneSlurryGrid] Failed to create MeshResource from LowLevelMesh")
        }
        entity.name = "BoneSlurryMesh"

        // --- Default Params ---

        let vs = SIMD3<Float>(repeating: Self.defaultVoxelSize)
        self.params = BoneSlurryGridParams(
            gridOrigin: .zero,
            voxelSize: vs,
            dimensions: SIMD3<UInt32>(repeating: dim),
            particleCount: 0,
            maxVertexCount: UInt32(maxVerts),
            isoValue: isoValue
        )

        print("[BoneSlurryGrid] Initialized: \(dim)^3 grid, defaultVoxel=\(Self.defaultVoxelSize)m, maxVerts=\(maxVerts)")
    }

    /// Padding added around the particle bounding box so edge particles
    /// still have room for their full ellipsoidal splat influence.
    private let boundsPadding: Float = 0.02  // 2 cm

    /// Minimum grid extent to prevent degenerate voxel sizes when few particles are clustered.
    private let minGridExtent: Float = 0.02  // 2 cm

    /// Maximum grid extent. Beyond this, voxels become too coarse for meaningful density.
    /// At 0.15m / 64 ≈ 2.3mm voxels — combined with MIN_SPLAT_RADIUS_VOXELS = 3.0
    /// and VISUAL_SCALE_MULTIPLIER = 10, each particle covers ≥6 voxels.
    private let maxGridExtent: Float = 0.15  // 15 cm

    /// Maximum distance from drill position to include a particle in the grid.
    /// Debris that have flown beyond this range under gravity are excluded.
    private let particleMaxRange: Float = 0.5  // 50 cm

    // MARK: - Per-Frame Particle Upload (CPU → GPU)

    /// Reads debris entity transforms and uploads them to the particle buffer.
    /// The grid dynamically fits the bounding box of ALL particles (capped at
    /// maxGridExtent to maintain voxel resolution), centered on the particle centroid.
    func uploadParticles(from debrisManager: BoneDebrisManager,
                         drillPosition: SIMD3<Float>,
                         rootEntity: Entity) {
        params.isoValue = isoValue

        // Upload particle positions/rotations/scales in root-local space.
        let entities = debrisManager.drawnEntities
        let count = min(entities.count, maxParticleCount)
        params.particleCount = UInt32(count)

        guard count > 0 else {
            debugFrameCounter += 1
            if debugFrameCounter <= 10 || debugFrameCounter % 60 == 0 {
                print("[BoneSlurryGrid] frame=\(debugFrameCounter) — no particles")
            }
            return
        }

        let ptr = particleBuffer.contents().bindMemory(to: BoneSlurryParticle.self, capacity: maxParticleCount)

        // First pass: collect positions and filter out runaway particles.
        // Debris under gravity can fly meters away — those are useless for the grid.
        // Use drillPosition as anchor: only include particles within particleMaxRange.
        var validCount: Int = 0
        var bbMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var bbMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        var centroid = SIMD3<Float>.zero

        for i in 0..<count {
            guard let modelEntity = entities[i] as? ModelEntity,
                  modelEntity.parent != nil else { continue }
            // All positions in root-local space.
            let pos = modelEntity.position(relativeTo: rootEntity)

            // Skip particles that have flown too far from the sculpting area.
            let distFromDrill = simd_length(pos - drillPosition)
            if distFromDrill > particleMaxRange { continue }

            let rot = modelEntity.orientation(relativeTo: rootEntity)
            let scl = modelEntity.scale(relativeTo: rootEntity)

            ptr[validCount] = BoneSlurryParticle(
                position: pos,
                rotation: SIMD4<Float>(rot.imag.x, rot.imag.y, rot.imag.z, rot.real),
                scale: scl
            )

            bbMin = simd_min(bbMin, pos)
            bbMax = simd_max(bbMax, pos)
            centroid += pos
            validCount += 1
        }

        // Update particle count to actual valid count (no gaps, no runaways).
        params.particleCount = UInt32(validCount)
        guard validCount > 0 else {
            debugFrameCounter += 1
            return
        }
        centroid /= Float(validCount)

        // Expand bounding box by padding to accommodate particle extents.
        bbMin -= SIMD3<Float>(repeating: boundsPadding)
        bbMax += SIMD3<Float>(repeating: boundsPadding)

        // Compute grid extent: use the largest axis to keep voxels cubic.
        let extent = bbMax - bbMin
        var gridSize = max(max(extent.x, extent.y), max(extent.z, minGridExtent))

        // Cap the grid extent to maintain voxel resolution.
        gridSize = min(gridSize, maxGridExtent)
        let center = centroid

        // Update grid origin and per-axis voxel size.
        params.gridOrigin = center - SIMD3<Float>(repeating: gridSize * 0.5)
        params.voxelSize = SIMD3<Float>(repeating: gridSize / Float(Self.gridDimension))

        // Debug logging — frequent at startup, then periodic.
        debugFrameCounter += 1
        if debugFrameCounter <= 30 || debugFrameCounter % 60 == 0 {
            print("[BoneSlurryGrid] frame=\(debugFrameCounter) particles=\(validCount)/\(count), gridSize=\(String(format: "%.4f", gridSize))m, voxelSize=\(String(format: "%.5f", params.voxelSize.x))m")
            print("[BoneSlurryGrid]   centroid=\(centroid), gridOrigin=\(params.gridOrigin)")
            if validCount > 0 {
                let p0 = ptr[0]
                print("[BoneSlurryGrid]   particle[0] pos=\(p0.position) scale=\(p0.scale)")
                let gridLocal = (p0.position - params.gridOrigin) / params.voxelSize
                print("[BoneSlurryGrid]   particle[0] gridLocal=\(gridLocal)")
            }
        }
    }

    // MARK: - GPU Dispatch (called from ComputeSystem)

    /// Dispatches four compute passes: clearMesh → clearDensity → splat → march.
    /// Must be called from within a ComputeSystem's update method.
    func update(computeContext: inout ComputeUpdateContext) {
        guard let computeEncoder = computeContext.computeEncoder() else { return }

        // Acquire vertex and index buffers for the LowLevelMesh.
        let vertexBuffer = mesh.replace(bufferIndex: 0, using: computeContext.commandBuffer)
        let indexBuffer = mesh.replaceIndices(using: computeContext.commandBuffer)

        // ── 1. Clear mesh vertex/index buffers ──
        // Zeroed vertices produce degenerate triangles at the origin (invisible).
        var maxVerts = UInt32(maxVertexCapacity)
        computeEncoder.setComputePipelineState(clearMeshPipeline)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&maxVerts, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.dispatchThreadgroups(clearMeshThreadgroups, threadsPerThreadgroup: clearMeshThreadsPerThreadgroup)

        // If no particles, stop after clearing (mesh will be empty degenerate triangles).
        guard params.particleCount > 0 else { return }

        // ── 2. Clear density buffer ──
        var totalCount = totalVoxels
        computeEncoder.setComputePipelineState(clearDensityPipeline)
        computeEncoder.setBuffer(densityBuffer, offset: 0, index: 0)
        computeEncoder.setBytes(&totalCount, length: MemoryLayout<UInt32>.stride, index: 1)
        computeEncoder.dispatchThreadgroups(clearDensityThreadgroups, threadsPerThreadgroup: clearDensityThreadsPerThreadgroup)

        // Memory barrier: ensure density clear is visible before splat writes.
        computeEncoder.memoryBarrier(scope: .buffers)

        // ── 3. Splat particles into density grid ──
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

        // Memory barrier: ensure splat density writes are visible before march reads.
        computeEncoder.memoryBarrier(scope: .buffers)

        // ── 4. Marching cubes on density grid ──
        // Read previous frame's triangle count BEFORE resetting counter.
        // (Previous frame's GPU has completed by now; current frame is still being encoded.)
        let prevFrameTriCount = counterBuffer.contents().load(as: UInt32.self)
        // Reset the atomic counter for this frame's march.
        counterBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        computeEncoder.setComputePipelineState(marchPipeline)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(densityBuffer, offset: 0, index: 2)
        computeEncoder.setBytes(&params, length: MemoryLayout<BoneSlurryGridParams>.stride, index: 3)
        computeEncoder.setBuffer(triangleTableBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(counterBuffer, offset: 0, index: 5)
        computeEncoder.dispatchThreadgroups(marchThreadgroups, threadsPerThreadgroup: marchThreadsPerThreadgroup)

        // Debug: log triangle count and density stats from PREVIOUS frame.
        // prevFrameTriCount was read BEFORE we reset the counter above.
        // Density buffer still contains previous frame's splat output
        // (current frame's clear hasn't executed on GPU yet).
        if debugFrameCounter <= 30 || debugFrameCounter % 60 == 0 {
            // Scan density buffer for previous frame's splat results.
            let densityPtr = densityBuffer.contents().bindMemory(to: UInt32.self, capacity: Int(totalVoxels))
            var maxDensityRaw: UInt32 = 0
            var nonZeroCount: Int = 0
            for i in 0..<Int(totalVoxels) {
                let d = densityPtr[i]
                if d > 0 { nonZeroCount += 1 }
                if d > maxDensityRaw { maxDensityRaw = d }
            }
            let maxDensityFloat = Float(maxDensityRaw) / 1024.0

            print("[BoneSlurryGrid] GPU dispatch: particles=\(params.particleCount), prevFrameTriangles=\(prevFrameTriCount)")
            print("[BoneSlurryGrid]   density: nonZero=\(nonZeroCount)/\(totalVoxels), maxRaw=\(maxDensityRaw), maxFloat=\(String(format: "%.3f", maxDensityFloat))")
            print("[BoneSlurryGrid]   params: gridOrigin=\(params.gridOrigin), voxelSize=\(params.voxelSize), dims=\(params.dimensions)")
            print("[BoneSlurryGrid]   params: isoValue=\(params.isoValue), maxVertexCount=\(params.maxVertexCount)")
        }
    }
}
