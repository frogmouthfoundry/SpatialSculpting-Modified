/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Generates compute commands to dispatch for marching cubes.
*/

import RealityKit
import Metal

@MainActor
struct MarchingCubesMeshSculptor {
    // Compute pipeline corresponding to the Metal compute shader function `reset`.
    //
    // See `SculptVoxelsComputeShader.metal`.
    private let resetPipeline: MTLComputePipelineState!
    // Compute pipeline corresponding to the Metal compute shader function `clearVolume`.
    //
    // See `SculptVoxelsComputeShader.metal`.
    private let clearPipeline: MTLComputePipelineState!
    // Compute pipeline corresponding to the Metal compute shader function `sculpt`.
    //
    // See `SculptVoxelsComputeShader.metal`.
    private let sculptPipeline: MTLComputePipelineState!
    // Compute pipeline for sampling the SDF at a single point.
    private let sampleSDFPipeline: MTLComputePipelineState!
    // CPU-readable buffer that receives the sampled SDF value (1 float).
    let sdfResultBuffer: MTLBuffer!
    
    let marchingCubesMesh: MarchingCubesMesh
    
    init(marchingCubesMesh: MarchingCubesMesh) {
        resetPipeline = makeComputePipeline(named: "reset")
        clearPipeline = makeComputePipeline(named: "clearVolume")
        sculptPipeline = makeComputePipeline(named: "sculpt")
        sampleSDFPipeline = makeComputePipeline(named: "sampleSDF")
        sdfResultBuffer = metalDevice?.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)
        self.marchingCubesMesh = marchingCubesMesh
    }
    
    func reset(computeContext: inout ComputeUpdateContext) {
        let volume = marchingCubesMesh.voxelVolume
        // Get the command buffer and compute encoder for dispatching commands to the GPU.
        guard let computeEncoder = computeContext.computeEncoder() else {
            return
        }

        // Set the compute shader pipeline to `reset`.
        computeEncoder.setComputePipelineState(resetPipeline)
        
        // Pass a writable version of the voxels texture to the compute shader.
        computeEncoder.setTexture(volume.voxelTexture, index: 0)
        
        // Pass the volume parameters to the compute shader.
        var volumeParams = volume.volumeParams
        computeEncoder.setBytes(&volumeParams, length: MemoryLayout<VolumeParams>.size, index: 1)
        
        // Dispatch the compute shader.
        computeEncoder.dispatchThreadgroups(volume.idealThreadgroupCount,
                                            threadsPerThreadgroup: volume.idealThreadsPerThreadgroup)

        // Mark mesh dirty and regenerate.
        marchingCubesMesh.isDirty = true
        marchingCubesMesh.update(computeContext: &computeContext)
    }

    func clear(computeContext: inout ComputeUpdateContext) {
        let volume = marchingCubesMesh.voxelVolume
        // Get the command buffer and compute encoder for dispatching commands to the GPU.
        guard let computeEncoder = computeContext.computeEncoder() else {
            return
        }

        // Set the compute shader pipeline to `clear`.
        computeEncoder.setComputePipelineState(clearPipeline)

        // Pass a writable version of the voxels texture to the compute shader.
        computeEncoder.setTexture(volume.voxelTexture, index: 0)

        // Pass the volume parameters to the compute shader.
        var volumeParams = volume.volumeParams
        computeEncoder.setBytes(&volumeParams, length: MemoryLayout<VolumeParams>.size, index: 1)

        // Dispatch the compute shader.
        computeEncoder.dispatchThreadgroups(volume.idealThreadgroupCount,
                                            threadsPerThreadgroup: volume.idealThreadsPerThreadgroup)

        // Mark mesh dirty and regenerate.
        marchingCubesMesh.isDirty = true
        marchingCubesMesh.update(computeContext: &computeContext)
    }

    func sculpt(sculptParams: SculptParams, computeContext: inout ComputeUpdateContext) {
        let volume = marchingCubesMesh.voxelVolume
        guard let computeEncoder = computeContext.computeEncoder() else { return }

        // Compute a tight AABB in voxel space around the tool capsule.
        // Only dispatch threads within this region instead of the full 128³ volume.
        let toolPos = SIMD3<Float>(sculptParams.toolPositionAndRadius.x,
                                   sculptParams.toolPositionAndRadius.y,
                                   sculptParams.toolPositionAndRadius.z)
        let toolRadius = sculptParams.toolPositionAndRadius.w
        let hasPrevious = sculptParams.previousPositionAndHasPosition.w != 0
        let prevPos = SIMD3<Float>(sculptParams.previousPositionAndHasPosition.x,
                                   sculptParams.previousPositionAndHasPosition.y,
                                   sculptParams.previousPositionAndHasPosition.z)

        // World-space AABB of the capsule + smooth blend margin.
        let smoothMargin: Float = 0.002
        let margin = toolRadius + smoothMargin
        let minWorld: SIMD3<Float>
        let maxWorld: SIMD3<Float>
        if hasPrevious {
            minWorld = simd_min(toolPos, prevPos) - SIMD3<Float>(repeating: margin)
            maxWorld = simd_max(toolPos, prevPos) + SIMD3<Float>(repeating: margin)
        } else {
            minWorld = toolPos - SIMD3<Float>(repeating: margin)
            maxWorld = toolPos + SIMD3<Float>(repeating: margin)
        }

        // Convert to voxel coordinates, clamped to volume bounds.
        let dims = SIMD3<Float>(volume.dimensions)
        let minVoxelF = (minWorld - volume.voxelStartPosition) / volume.voxelSize
        let maxVoxelF = (maxWorld - volume.voxelStartPosition) / volume.voxelSize
        let minVoxel = SIMD3<UInt32>(simd_clamp(floor(minVoxelF), SIMD3<Float>.zero, dims - 1))
        let maxVoxel = SIMD3<UInt32>(simd_clamp(ceil(maxVoxelF), SIMD3<Float>.zero, dims - 1))
        let regionSize = maxVoxel &- minVoxel &+ 1

        // Set up sculpt params with the region offset.
        var sculptParams = sculptParams
        sculptParams.regionOffset = minVoxel

        computeEncoder.setComputePipelineState(sculptPipeline)
        computeEncoder.setTexture(volume.voxelTexture, index: 0)
        computeEncoder.setTexture(volume.voxelTexture, index: 1)

        var volumeParams = volume.volumeParams
        computeEncoder.setBytes(&volumeParams, length: MemoryLayout<VolumeParams>.size, index: 2)
        computeEncoder.setBytes(&sculptParams, length: MemoryLayout<SculptParams>.size, index: 3)

        // Dispatch only over the localized region.
        let tpg = MTLSize(width: 8, height: 8, depth: 8)
        let threadgroups = MTLSize(
            width:  (Int(regionSize.x) + tpg.width  - 1) / tpg.width,
            height: (Int(regionSize.y) + tpg.height - 1) / tpg.height,
            depth:  (Int(regionSize.z) + tpg.depth  - 1) / tpg.depth
        )
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: tpg)

        // Mark mesh dirty and regenerate.
        marchingCubesMesh.isDirty = true
        marchingCubesMesh.update(computeContext: &computeContext)
    }

    /// Dispatch a single-thread compute kernel that samples the SDF at `worldPosition`
    /// and writes the result into `sdfResultBuffer`. The value is available after the
    /// command buffer commits (read it next frame via `lastSampledSDF`).
    func sampleSDF(at worldPosition: SIMD3<Float>, computeContext: inout ComputeUpdateContext) {
        let volume = marchingCubesMesh.voxelVolume
        guard let computeEncoder = computeContext.computeEncoder() else { return }

        computeEncoder.setComputePipelineState(sampleSDFPipeline)
        computeEncoder.setTexture(volume.voxelTexture, index: 0)

        var volumeParams = volume.volumeParams
        computeEncoder.setBytes(&volumeParams, length: MemoryLayout<VolumeParams>.size, index: 0)

        var pos = worldPosition
        computeEncoder.setBytes(&pos, length: MemoryLayout<SIMD3<Float>>.size, index: 1)

        computeEncoder.setBuffer(sdfResultBuffer, offset: 0, index: 2)

        computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }

    /// Read the SDF value written by the previous frame's `sampleSDF` dispatch.
    /// Negative or near-zero means the tool is at/inside the mesh surface.
    var lastSampledSDF: Float {
        guard let ptr = sdfResultBuffer?.contents().bindMemory(to: Float.self, capacity: 1) else {
            return Float.greatestFiniteMagnitude
        }
        return ptr.pointee
    }

    func save(destinationTexture: MTLTexture, computeContext: inout ComputeUpdateContext, onCompletion: @Sendable @escaping () throws -> Void) {
        guard let blitEncoder = computeContext.blitEncoder() else {
            return
        }
        
        blitEncoder.copy(from: marchingCubesMesh.voxelVolume.voxelTexture, to: destinationTexture)

        computeContext.commandBuffer.addCompletedHandler { _ in
            do {
                try onCompletion()
            } catch {
                print("Error saving texture: \(error)")
            }
        }
    }

    func load(sourceTexture: MTLTexture, computeContext: inout ComputeUpdateContext) {
        guard let blitEncoder = computeContext.blitEncoder() else {
            return
        }
        blitEncoder.copy(from: sourceTexture, to: marchingCubesMesh.voxelVolume.voxelTexture)

        // Mark mesh dirty and regenerate.
        marchingCubesMesh.isDirty = true
        marchingCubesMesh.update(computeContext: &computeContext)
    }

    // Load complete sculpt package texture payload (SDF + albedo + normal + confidence + roughness).
    func load(payload: SculptLoadPayload, computeContext: inout ComputeUpdateContext) {
        guard let blitEncoder = computeContext.blitEncoder() else {
            return
        }
        let volume = marchingCubesMesh.voxelVolume
        blitEncoder.copy(from: payload.sdfTexture, to: volume.voxelTexture)
        blitEncoder.copy(from: payload.albedoTexture, to: volume.albedoTexture)
        blitEncoder.copy(from: payload.normalTexture, to: volume.normalTexture)
        blitEncoder.copy(from: payload.confidenceTexture, to: volume.confidenceTexture)
        blitEncoder.copy(from: payload.roughnessTexture, to: volume.roughnessTexture)
        volume.updateTextureBounds(min: payload.textureBoundsMin, max: payload.textureBoundsMax)

        // Mark mesh dirty and regenerate.
        marchingCubesMesh.isDirty = true
        marchingCubesMesh.update(computeContext: &computeContext)
    }

    /// Blit the SDF texture to a shared staging texture for CPU collision readback.
    /// The completion handler fires after the command buffer commits.
    func blitForCollision(destinationTexture: MTLTexture,
                          computeContext: inout ComputeUpdateContext,
                          onCompletion: @Sendable @escaping () -> Void) {
        guard let blitEncoder = computeContext.blitEncoder() else { return }
        blitEncoder.copy(from: marchingCubesMesh.voxelVolume.voxelTexture, to: destinationTexture)
        computeContext.commandBuffer.addCompletedHandler { _ in
            onCompletion()
        }
    }
}
