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
        
        // Update the marching cubes mesh.
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

        // Update the marching cubes mesh.
        marchingCubesMesh.update(computeContext: &computeContext)
    }

    func sculpt(sculptParams: SculptParams, computeContext: inout ComputeUpdateContext) {
        let volume = marchingCubesMesh.voxelVolume
        // Get the command buffer and compute encoder for dispatching commands to the GPU.
        guard let computeEncoder = computeContext.computeEncoder() else {
            return
        }

        // Set the compute shader pipeline to `sculpt`.
        computeEncoder.setComputePipelineState(sculptPipeline)

        // Pass readable and writable versions of the voxels texture to the compute shader.
        computeEncoder.setTexture(volume.voxelTexture, index: 0)
        computeEncoder.setTexture(volume.voxelTexture, index: 1)
        
        // Pass the volume and sculpt parameters to the compute shader.
        var volumeParams = volume.volumeParams
        var sculptParams = sculptParams
        computeEncoder.setBytes(&volumeParams, length: MemoryLayout<VolumeParams>.size, index: 2)
        computeEncoder.setBytes(&sculptParams, length: MemoryLayout<SculptParams>.size, index: 3)
        
        // Dispatch the compute shader.
        computeEncoder.dispatchThreadgroups(volume.idealThreadgroupCount,
                                            threadsPerThreadgroup: volume.idealThreadsPerThreadgroup)
        
        // Update the marching cubes mesh.
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

        // Update the marching cubes mesh.
        marchingCubesMesh.update(computeContext: &computeContext)
    }
}
