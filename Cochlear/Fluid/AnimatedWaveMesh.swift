/*
Animated LowLevelMesh wave surface updated by Metal compute kernels.
*/

import Metal
import RealityKit

@MainActor
final class AnimatedWaveMesh: ComputeSystem {
    enum Error: Swift.Error {
        case missingMetalDevice
        case missingVertexPipeline
        case missingIndexPipeline
    }

    let maxSegmentCount: Int = 128
    let lowLevelMesh: LowLevelMesh

    var waveDensity: Float = 3.0
    var amplitude: Float = 0.0
    var speed: Float = 1.0

    var segmentCount: Int = 128 {
        didSet { needsTopologyUpdate = true }
    }

    private let updateVerticesPipeline: MTLComputePipelineState
    private let updateIndicesPipeline: MTLComputePipelineState
    private var needsTopologyUpdate: Bool = true
    private var time: TimeInterval = 0.0

    private(set) var ripplePoint: SIMD2<Float>? = nil
    private var touchStart: SIMD2<Float> = .zero
    private var touchEnd: SIMD2<Float> = .zero
    private var rippleTime: Float = 0.0

    var needsFrameUpdate: Bool {
        needsTopologyUpdate || amplitude != 0 || ripplePoint != nil
    }

    init() throws {
        guard metalDevice != nil else { throw Error.missingMetalDevice }
        guard let verticesPipeline = makeComputePipeline(named: "update_wave_vertices") else {
            throw Error.missingVertexPipeline
        }
        guard let indicesPipeline = makeComputePipeline(named: "update_wave_indices") else {
            throw Error.missingIndexPipeline
        }

        self.updateVerticesPipeline = verticesPipeline
        self.updateIndicesPipeline = indicesPipeline

        let vertex = MemoryLayout<WaveMeshVertex>.self
        let attributes: [LowLevelMesh.Attribute] = [
            .init(semantic: .position, format: .float3, offset: vertex.offset(of: \.position)!),
            .init(semantic: .normal, format: .float3, offset: vertex.offset(of: \.normal)!),
            .init(semantic: .uv0, format: .float2, offset: vertex.offset(of: \.uv)!)
        ]
        let layouts: [LowLevelMesh.Layout] = [
            .init(bufferIndex: 0, bufferOffset: 0, bufferStride: vertex.stride)
        ]

        let vertexCapacity = (maxSegmentCount + 1) * (maxSegmentCount + 1)
        let indexCapacity = maxSegmentCount * maxSegmentCount * 6
        let descriptor = LowLevelMesh.Descriptor(
            vertexCapacity: vertexCapacity,
            vertexAttributes: attributes,
            vertexLayouts: layouts,
            indexCapacity: indexCapacity,
            indexType: .uint32
        )
        self.lowLevelMesh = try LowLevelMesh(descriptor: descriptor)
    }

    func triggerRipple(at localPoint: SIMD2<Float>) {
        touchStart = touchEnd
        touchEnd = localPoint
        ripplePoint = localPoint
        rippleTime = 0
    }

    private func advanceSimulation(_ timestep: TimeInterval) {
        time += timestep

        if ripplePoint != nil {
            rippleTime += Float(timestep)
            if rippleTime > 2.0 {
                ripplePoint = nil
                rippleTime = 0
            }
        }
    }

    func update(computeContext: inout ComputeUpdateContext) {
        guard needsFrameUpdate else { return }

        advanceSimulation(computeContext.sceneUpdateContext.deltaTime)

        guard let commandEncoder = computeContext.computeEncoder() else {
            return
        }

        let activeSegmentCount = max(0, min(segmentCount, maxSegmentCount))
        let indexCount = activeSegmentCount * activeSegmentCount * 6

        var waveDescriptor = WaveDescriptor(
            segmentCount: UInt32(activeSegmentCount),
            time: Float(time) * speed,
            waveDensity: waveDensity,
            amplitude: amplitude,
            rippleX: ripplePoint?.x ?? 0,
            rippleZ: ripplePoint?.y ?? 0,
            rippleStrength: ripplePoint != nil ? 0.6 : 0.0,
            rippleTime: rippleTime,
            touchX0: touchStart.x,
            touchZ0: touchStart.y,
            touchX1: touchEnd.x,
            touchZ1: touchEnd.y
        )

        let requiresUniformDispatch = !(metalDevice?.supportsFamily(.apple4) ?? false)
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)

        commandEncoder.setBytes(&waveDescriptor, length: MemoryLayout<WaveDescriptor>.size, index: 1)

        let vertexBuffer = lowLevelMesh.replace(bufferIndex: 0, using: computeContext.commandBuffer)
        commandEncoder.setComputePipelineState(updateVerticesPipeline)
        commandEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        let vertexThreads = MTLSize(width: activeSegmentCount + 1, height: activeSegmentCount + 1, depth: 1)
        if requiresUniformDispatch {
            let groups = MTLSize(
                width: (vertexThreads.width + threadgroupSize.width - 1) / threadgroupSize.width,
                height: (vertexThreads.height + threadgroupSize.height - 1) / threadgroupSize.height,
                depth: 1
            )
            commandEncoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroupSize)
        } else {
            commandEncoder.dispatchThreads(vertexThreads, threadsPerThreadgroup: threadgroupSize)
        }

        if needsTopologyUpdate {
            let indexBuffer = lowLevelMesh.replaceIndices(using: computeContext.commandBuffer)
            commandEncoder.setComputePipelineState(updateIndicesPipeline)
            commandEncoder.setBuffer(indexBuffer, offset: 0, index: 0)
            let indexThreads = MTLSize(width: activeSegmentCount, height: activeSegmentCount, depth: 1)
            if requiresUniformDispatch {
                let groups = MTLSize(
                    width: (indexThreads.width + threadgroupSize.width - 1) / threadgroupSize.width,
                    height: (indexThreads.height + threadgroupSize.height - 1) / threadgroupSize.height,
                    depth: 1
                )
                commandEncoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroupSize)
            } else {
                commandEncoder.dispatchThreads(indexThreads, threadsPerThreadgroup: threadgroupSize)
            }

            let bounds = BoundingBox(min: SIMD3<Float>(-0.5, -1.0, -0.5),
                                     max: SIMD3<Float>(0.5, 1.0, 0.5))
            lowLevelMesh.parts.replaceAll([
                LowLevelMesh.Part(indexOffset: 0,
                                  indexCount: indexCount,
                                  topology: .triangle,
                                  materialIndex: 0,
                                  bounds: bounds)
            ])
            needsTopologyUpdate = false
        }
    }
}
