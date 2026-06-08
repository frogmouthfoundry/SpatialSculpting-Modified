/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
The volumetric representation of the sculpture.
*/

import RealityKit
import Metal

enum VoxelVolumeError: Error {
    case failedToCreateTexture
}

// A 3D texture that represents a volume.
// The value of each voxel represents the distance from an isosurface.
@MainActor
final class VoxelVolume {
    var voxelTexture: MTLTexture
    // 3D material volumes loaded from sculpt package.
    var albedoTexture: MTLTexture
    var normalTexture: MTLTexture
    var confidenceTexture: MTLTexture
    var roughnessTexture: MTLTexture

    let dimensions: SIMD3<UInt32>
    let voxelSize: SIMD3<Float>
    let voxelStartPosition: SIMD3<Float>
    var textureBoundsMin: SIMD3<Float>
    var textureBoundsMax: SIMD3<Float>

    var volumeParams: VolumeParams {
        VolumeParams(dimensions: dimensions,
                     voxelSize: voxelSize,
                     voxelStartPosition: voxelStartPosition,
                     textureBoundsMin: textureBoundsMin,
                     textureBoundsMax: textureBoundsMax)
    }

    let idealThreadgroupCount: MTLSize
    let idealThreadsPerThreadgroup: MTLSize

    init(dimensions: SIMD3<UInt32>,
         voxelSize: SIMD3<Float>,
         voxelStartPosition: SIMD3<Float>,
         textureBoundsMin: SIMD3<Float>? = nil,
         textureBoundsMax: SIMD3<Float>? = nil) throws {
        self.dimensions = dimensions
        self.voxelSize = voxelSize
        self.voxelStartPosition = voxelStartPosition
        let derivedBoundsMin = voxelStartPosition - voxelSize * 0.5
        let derivedBoundsMax = voxelStartPosition + voxelSize * (SIMD3<Float>(dimensions) - 0.5)
        self.textureBoundsMin = textureBoundsMin ?? derivedBoundsMin
        self.textureBoundsMax = textureBoundsMax ?? derivedBoundsMax

        self.voxelTexture = try Self.makeVolumeTexture(dimensions: dimensions,
                                                       pixelFormat: .r32Float,
                                                       usage: [.shaderRead, .shaderWrite])
        self.albedoTexture = try Self.makeVolumeTexture(dimensions: dimensions,
                                                        pixelFormat: .rgba8Unorm,
                                                        usage: [.shaderRead, .shaderWrite],
                                                        storageMode: .shared)
        self.normalTexture = try Self.makeVolumeTexture(dimensions: dimensions,
                                                        pixelFormat: .rgba8Snorm,
                                                        usage: [.shaderRead, .shaderWrite],
                                                        storageMode: .shared)
        self.confidenceTexture = try Self.makeVolumeTexture(dimensions: dimensions,
                                                            pixelFormat: .r8Unorm,
                                                            usage: [.shaderRead, .shaderWrite],
                                                            storageMode: .shared)
        self.roughnessTexture = try Self.makeVolumeTexture(dimensions: dimensions,
                                                           pixelFormat: .r8Unorm,
                                                           usage: [.shaderRead, .shaderWrite],
                                                           storageMode: .shared)

        // Fill material textures with sensible defaults.
        Self.fillTexture(texture: self.albedoTexture, value: [204, 204, 204, 255], bytesPerPixel: 4)
        Self.fillTexture(texture: self.normalTexture, value: [0, 0, 127, 127], bytesPerPixel: 4)
        Self.fillTexture(texture: self.confidenceTexture, value: [255], bytesPerPixel: 1)
        Self.fillTexture(texture: self.roughnessTexture, value: [128], bytesPerPixel: 1)

        self.idealThreadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 8)
        self.idealThreadgroupCount = MTLSize(width: (Int(dimensions.x) + idealThreadsPerThreadgroup.width - 1) / idealThreadsPerThreadgroup.width,
                                             height: (Int(dimensions.y) + idealThreadsPerThreadgroup.height - 1) / idealThreadsPerThreadgroup.height,
                                             depth: (Int(dimensions.z) + idealThreadsPerThreadgroup.depth - 1) / idealThreadsPerThreadgroup.depth)
    }

    func updateTextureBounds(min: SIMD3<Float>, max: SIMD3<Float>) {
        textureBoundsMin = min
        textureBoundsMax = max
    }

    private static func makeVolumeTexture(dimensions: SIMD3<UInt32>,
                                          pixelFormat: MTLPixelFormat,
                                          usage: MTLTextureUsage,
                                          storageMode: MTLStorageMode = .private) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = pixelFormat
        descriptor.width = Int(dimensions.x)
        descriptor.height = Int(dimensions.y)
        descriptor.depth = Int(dimensions.z)
        descriptor.usage = usage
        descriptor.storageMode = storageMode

        guard let texture = metalDevice?.makeTexture(descriptor: descriptor) else {
            throw VoxelVolumeError.failedToCreateTexture
        }
        return texture
    }

    private static func fillTexture(texture: MTLTexture, value: [UInt8], bytesPerPixel: Int) {
        let bytesPerRow = texture.width * bytesPerPixel
        let bytesPerImage = bytesPerRow * texture.height
        var image = [UInt8](repeating: 0, count: bytesPerImage * texture.depth)
        for i in stride(from: 0, to: image.count, by: bytesPerPixel) {
            for c in 0..<bytesPerPixel {
                image[i + c] = value[c]
            }
        }
        image.withUnsafeBytes { ptr in
            texture.replace(region: MTLRegionMake3D(0, 0, 0, texture.width, texture.height, texture.depth),
                            mipmapLevel: 0,
                            slice: 0,
                            withBytes: ptr.baseAddress!,
                            bytesPerRow: bytesPerRow,
                            bytesPerImage: bytesPerImage)
        }
    }
}
