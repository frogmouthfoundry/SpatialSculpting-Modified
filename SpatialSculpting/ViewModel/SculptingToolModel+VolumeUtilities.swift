/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Handle loading and saving out to a document.
*/

@preconcurrency import Metal
import MetalKit

extension SculptingToolModel {
    // MARK: - Staging Texture Helpers

    private func makeSharedTexture(from original: MTLTexture,
                                   pixelFormat: MTLPixelFormat? = nil) -> MTLTexture? {
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = original.textureType
        textureDescriptor.pixelFormat = pixelFormat ?? original.pixelFormat
        textureDescriptor.width = original.width
        textureDescriptor.height = original.height
        textureDescriptor.depth = original.depth
        textureDescriptor.usage = []
        textureDescriptor.storageMode = .shared
        return metalDevice?.makeTexture(descriptor: textureDescriptor)
    }

    private func mapDimensions(for map: SculptPackageManifest.VolumeMap,
                               fallback: SculptPackageManifest.Grid) -> SIMD3<Int> {
        SIMD3<Int>(map.width ?? fallback.width,
                   map.height ?? fallback.height,
                   map.depth ?? fallback.depth)
    }

    // MARK: - Resampling Utilities

    private func resampleR32Float(_ source: Data,
                                  sourceDims: SIMD3<Int>,
                                  targetDims: SIMD3<Int>) -> Data {
        let src = source.withUnsafeBytes { raw in
            Array(UnsafeBufferPointer<Float>(start: raw.bindMemory(to: Float.self).baseAddress,
                                             count: source.count / MemoryLayout<Float>.size))
        }
        var dst = [Float](repeating: 0, count: targetDims.x * targetDims.y * targetDims.z)
        let sx = Float(max(sourceDims.x - 1, 1))
        let sy = Float(max(sourceDims.y - 1, 1))
        let sz = Float(max(sourceDims.z - 1, 1))
        let tx = Float(max(targetDims.x - 1, 1))
        let ty = Float(max(targetDims.y - 1, 1))
        let tz = Float(max(targetDims.z - 1, 1))
        func srcIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
            ((z * sourceDims.y + y) * sourceDims.x + x)
        }
        func dstIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
            ((z * targetDims.y + y) * targetDims.x + x)
        }
        for z in 0..<targetDims.z {
            let pz = (Float(z) / tz) * sz
            let z0 = Int(floor(pz))
            let z1 = min(z0 + 1, sourceDims.z - 1)
            let fz = pz - Float(z0)
            for y in 0..<targetDims.y {
                let py = (Float(y) / ty) * sy
                let y0 = Int(floor(py))
                let y1 = min(y0 + 1, sourceDims.y - 1)
                let fy = py - Float(y0)
                for x in 0..<targetDims.x {
                    let px = (Float(x) / tx) * sx
                    let x0 = Int(floor(px))
                    let x1 = min(x0 + 1, sourceDims.x - 1)
                    let fx = px - Float(x0)
                    let c000 = src[srcIndex(x0, y0, z0)]
                    let c100 = src[srcIndex(x1, y0, z0)]
                    let c010 = src[srcIndex(x0, y1, z0)]
                    let c110 = src[srcIndex(x1, y1, z0)]
                    let c001 = src[srcIndex(x0, y0, z1)]
                    let c101 = src[srcIndex(x1, y0, z1)]
                    let c011 = src[srcIndex(x0, y1, z1)]
                    let c111 = src[srcIndex(x1, y1, z1)]
                    let c00 = c000 + (c100 - c000) * fx
                    let c10 = c010 + (c110 - c010) * fx
                    let c01 = c001 + (c101 - c001) * fx
                    let c11 = c011 + (c111 - c011) * fx
                    let c0 = c00 + (c10 - c00) * fy
                    let c1 = c01 + (c11 - c01) * fy
                    dst[dstIndex(x, y, z)] = c0 + (c1 - c0) * fz
                }
            }
        }
        return Data(bytes: dst, count: dst.count * MemoryLayout<Float>.size)
    }

    private func resampleRGBA8(_ source: Data,
                               sourceDims: SIMD3<Int>,
                               targetDims: SIMD3<Int>) -> Data {
        var dst = Data(count: targetDims.x * targetDims.y * targetDims.z * 4)
        let sx = Float(max(sourceDims.x - 1, 1))
        let sy = Float(max(sourceDims.y - 1, 1))
        let sz = Float(max(sourceDims.z - 1, 1))
        let tx = Float(max(targetDims.x - 1, 1))
        let ty = Float(max(targetDims.y - 1, 1))
        let tz = Float(max(targetDims.z - 1, 1))
        func srcIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
            ((z * sourceDims.y + y) * sourceDims.x + x) * 4
        }
        func dstIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
            ((z * targetDims.y + y) * targetDims.x + x) * 4
        }
        source.withUnsafeBytes { srcRaw in
            let src = srcRaw.bindMemory(to: UInt8.self)
            dst.withUnsafeMutableBytes { dstRaw in
                let out = dstRaw.bindMemory(to: UInt8.self)
                for z in 0..<targetDims.z {
                    let pz = (Float(z) / tz) * sz
                    let z0 = Int(floor(pz))
                    let z1 = min(z0 + 1, sourceDims.z - 1)
                    let fz = pz - Float(z0)
                    for y in 0..<targetDims.y {
                        let py = (Float(y) / ty) * sy
                        let y0 = Int(floor(py))
                        let y1 = min(y0 + 1, sourceDims.y - 1)
                        let fy = py - Float(y0)
                        for x in 0..<targetDims.x {
                            let px = (Float(x) / tx) * sx
                            let x0 = Int(floor(px))
                            let x1 = min(x0 + 1, sourceDims.x - 1)
                            let fx = px - Float(x0)
                            let i000 = srcIndex(x0, y0, z0)
                            let i100 = srcIndex(x1, y0, z0)
                            let i010 = srcIndex(x0, y1, z0)
                            let i110 = srcIndex(x1, y1, z0)
                            let i001 = srcIndex(x0, y0, z1)
                            let i101 = srcIndex(x1, y0, z1)
                            let i011 = srcIndex(x0, y1, z1)
                            let i111 = srcIndex(x1, y1, z1)
                            let o = dstIndex(x, y, z)
                            for c in 0..<4 {
                                let c000 = Float(src[i000 + c])
                                let c100 = Float(src[i100 + c])
                                let c010 = Float(src[i010 + c])
                                let c110 = Float(src[i110 + c])
                                let c001 = Float(src[i001 + c])
                                let c101 = Float(src[i101 + c])
                                let c011 = Float(src[i011 + c])
                                let c111 = Float(src[i111 + c])
                                let c00 = c000 + (c100 - c000) * fx
                                let c10 = c010 + (c110 - c010) * fx
                                let c01 = c001 + (c101 - c001) * fx
                                let c11 = c011 + (c111 - c011) * fx
                                let c0 = c00 + (c10 - c00) * fy
                                let c1 = c01 + (c11 - c01) * fy
                                out[o + c] = UInt8(clamping: Int((c0 + (c1 - c0) * fz).rounded()))
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    private func resampleNearest(_ source: Data,
                                 sourceDims: SIMD3<Int>,
                                 targetDims: SIMD3<Int>,
                                 bytesPerPixel: Int) -> Data {
        var dst = Data(count: targetDims.x * targetDims.y * targetDims.z * bytesPerPixel)
        let sx = Float(max(sourceDims.x - 1, 1))
        let sy = Float(max(sourceDims.y - 1, 1))
        let sz = Float(max(sourceDims.z - 1, 1))
        let tx = Float(max(targetDims.x - 1, 1))
        let ty = Float(max(targetDims.y - 1, 1))
        let tz = Float(max(targetDims.z - 1, 1))
        source.withUnsafeBytes { srcRaw in
            let src = srcRaw.bindMemory(to: UInt8.self)
            dst.withUnsafeMutableBytes { dstRaw in
                let out = dstRaw.bindMemory(to: UInt8.self)
                for z in 0..<targetDims.z {
                    let zz = Int(((Float(z) / tz) * sz).rounded())
                    for y in 0..<targetDims.y {
                        let yy = Int(((Float(y) / ty) * sy).rounded())
                        for x in 0..<targetDims.x {
                            let xx = Int(((Float(x) / tx) * sx).rounded())
                            let srcBase = ((zz * sourceDims.y + yy) * sourceDims.x + xx) * bytesPerPixel
                            let dstBase = ((z * targetDims.y + y) * targetDims.x + x) * bytesPerPixel
                            for c in 0..<bytesPerPixel {
                                out[dstBase + c] = src[srcBase + c]
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    private func reorderLegacyVolume(_ source: Data,
                                     dims: SIMD3<Int>,
                                     bytesPerPixel: Int) -> Data {
        var dst = Data(count: source.count)
        source.withUnsafeBytes { srcRaw in
            let src = srcRaw.bindMemory(to: UInt8.self)
            dst.withUnsafeMutableBytes { dstRaw in
                let out = dstRaw.bindMemory(to: UInt8.self)
                for z in 0..<dims.z {
                    for y in 0..<dims.y {
                        for x in 0..<dims.x {
                            let srcBase = ((x * dims.y + y) * dims.z + z) * bytesPerPixel
                            let dstBase = ((z * dims.y + y) * dims.x + x) * bytesPerPixel
                            for c in 0..<bytesPerPixel {
                                out[dstBase + c] = src[srcBase + c]
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    private func packageUsesNativeLayout(_ manifest: SculptPackageManifest) -> Bool {
        manifest.layout == "x_fastest"
    }

    private func packageUsesCenteredVoxelOrigin(_ manifest: SculptPackageManifest) -> Bool {
        manifest.voxelOrigin == "center"
    }

    private func forceOpaqueAlpha(_ data: Data) -> Data {
        var out = data
        out.withUnsafeMutableBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for i in stride(from: 0, to: bytes.count, by: 4) {
                bytes[i + 3] = 255
            }
        }
        return out
    }

    // MARK: - Save / Load

    // Save a finished sculpture out as a VolumeDocument.
    @MainActor
    func save(onCompleted: @Sendable @escaping (VolumeDocument) -> Void) {
        guard var sculptingToolComponent = sculptingTool.components[SculptingToolComponent.self] else {
            return
        }

        let originalTexture = sculptingToolComponent.sculptor.marchingCubesMesh.voxelVolume.voxelTexture
        guard let destinationTexture = makeSharedTexture(from: originalTexture) else {
            return
        }

        sculptingToolComponent.saveToTexture = (destinationTexture, {
            let document = try VolumeDocument(texture: destinationTexture)
            onCompleted(document)
        })
        sculptingTool.components.set(sculptingToolComponent)
    }

    // Load in a VolumeDocument or sculpt package.
    func loadFromURL(_ url: URL) throws {
        guard var sculptingToolComponent = sculptingTool.components[SculptingToolComponent.self] else {
            return
        }

        let originalTexture = sculptingToolComponent.sculptor.marchingCubesMesh.voxelVolume.voxelTexture

        // Support both legacy SDF documents and sculpt package folders.
        var isDirectory: ObjCBool = false
        let isFolder = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
        let hasManifest = FileManager.default.fileExists(atPath: url.appendingPathComponent("manifest.json").path)
        if url.pathExtension.lowercased() == "sculptpkg" || (isFolder && hasManifest) {
            let payload = try makeLoadPayload(fromPackageURL: url,
                                              volumeTexture: originalTexture)
            loadedRoughnessValue = payload.averageRoughness
            loadedAlbedoTint = payload.averageAlbedo
            updateSculptMeshMaterials()
            sculptingToolComponent.loadPayload = payload
        } else {
            guard let sourceTexture = makeSharedTexture(from: originalTexture) else {
                return
            }
            try VolumeDocument.loadFromURL(url, texture: sourceTexture)
            sculptingToolComponent.loadFromTexture = sourceTexture
        }

        sculptingTool.components.set(sculptingToolComponent)
    }

    // MARK: - Package Loading

    private func makeLoadPayload(fromPackageURL packageURL: URL,
                                 volumeTexture: MTLTexture) throws -> SculptLoadPayload {
        let manifest = try VolumeDocument.withAccessibleURL(packageURL) {
            try VolumeDocument.parsePackageManifest(at: packageURL)
        }
        let usesNativeLayout = packageUsesNativeLayout(manifest)
        let manifestBoundsMin: SIMD3<Float>?
        let manifestBoundsMax: SIMD3<Float>?
        if manifest.bounds.min.count == 3, manifest.bounds.max.count == 3 {
            let bmin = SIMD3<Float>(manifest.bounds.min[0], manifest.bounds.min[1], manifest.bounds.min[2])
            let bmax = SIMD3<Float>(manifest.bounds.max[0], manifest.bounds.max[1], manifest.bounds.max[2])
            manifestBoundsMin = bmin
            manifestBoundsMax = bmax
            print("Manifest bounds: \(bmin) / \(bmax)")
        } else {
            manifestBoundsMin = nil
            manifestBoundsMax = nil
            print("Manifest diagnostics: bounds missing or invalid count.")
        }
        let targetDims = SIMD3<Int>(volumeTexture.width, volumeTexture.height, volumeTexture.depth)
        let sdfSourceDims = mapDimensions(for: manifest.sdf, fallback: manifest.grid)
        let albedoSourceDims = mapDimensions(for: manifest.albedo, fallback: manifest.grid)

        guard let volume = sculptingTool.components[SculptingToolComponent.self]?.sculptor.marchingCubesMesh.voxelVolume else {
            throw VolumeDocumentError.invalidPackage("Missing voxel volume while preparing package load.")
        }
        let derivedMin = volume.voxelStartPosition - volume.voxelSize * 0.5
        let derivedMax = volume.voxelStartPosition + volume.voxelSize * (SIMD3<Float>(volume.dimensions) - 0.5)
        let textureBoundsMin = manifestBoundsMin ?? derivedMin
        let textureBoundsMax = manifestBoundsMax ?? derivedMax

        guard let sdfTexture = makeSharedTexture(from: volumeTexture, pixelFormat: .r32Float),
              let albedoTexture = makeSharedTexture(from: volumeTexture, pixelFormat: .rgba8Unorm),
              let normalTexture = makeSharedTexture(from: volumeTexture, pixelFormat: .rgba8Snorm),
              let confidenceTexture = makeSharedTexture(from: volumeTexture, pixelFormat: .r8Unorm),
              let roughnessTexture = makeSharedTexture(from: volumeTexture, pixelFormat: .r8Unorm) else {
            throw VolumeDocumentError.invalidPackage("Failed to allocate staging textures.")
        }

        // -- SDF --
        let sdfData = try VolumeDocument.withAccessibleURL(packageURL) {
            try Data(contentsOf: packageURL.appendingPathComponent(manifest.sdf.file, isDirectory: false), options: [.alwaysMapped])
        }
        let expectedSdfBytes = sdfSourceDims.x * sdfSourceDims.y * sdfSourceDims.z * 4
        guard sdfData.count == expectedSdfBytes else {
            throw VolumeDocumentError.unexpectedFileSize(sdfData.count, expectedSdfBytes)
        }
        let orientedSdfData = usesNativeLayout ? sdfData : reorderLegacyVolume(sdfData, dims: sdfSourceDims, bytesPerPixel: 4)
        let stagedSdfData: Data
        if sdfSourceDims == targetDims {
            stagedSdfData = orientedSdfData
        } else {
            print("Resampling SDF \(sdfSourceDims) -> \(targetDims)")
            stagedSdfData = resampleR32Float(orientedSdfData, sourceDims: sdfSourceDims, targetDims: targetDims)
        }
        try VolumeDocument.loadRawData(stagedSdfData, into: sdfTexture)

        // -- Albedo --
        let albedoData = try VolumeDocument.withAccessibleURL(packageURL) {
            try Data(contentsOf: packageURL.appendingPathComponent(manifest.albedo.file, isDirectory: false), options: [.alwaysMapped])
        }
        let expectedAlbedoBytes = albedoSourceDims.x * albedoSourceDims.y * albedoSourceDims.z * 4
        guard albedoData.count == expectedAlbedoBytes else {
            throw VolumeDocumentError.unexpectedFileSize(albedoData.count, expectedAlbedoBytes)
        }
        let orientedAlbedoData = usesNativeLayout ? albedoData : reorderLegacyVolume(albedoData, dims: albedoSourceDims, bytesPerPixel: 4)
        let stagedAlbedoData: Data
        if albedoSourceDims == targetDims {
            stagedAlbedoData = forceOpaqueAlpha(orientedAlbedoData)
        } else {
            print("Resampling albedo \(albedoSourceDims) -> \(targetDims)")
            stagedAlbedoData = forceOpaqueAlpha(resampleRGBA8(orientedAlbedoData, sourceDims: albedoSourceDims, targetDims: targetDims))
        }
        try VolumeDocument.loadRawData(stagedAlbedoData, into: albedoTexture)

        // -- Normal --
        let voxelCount = targetDims.x * targetDims.y * targetDims.z
        if let normalMap = manifest.normal {
            let normalSourceDims = mapDimensions(for: normalMap, fallback: manifest.grid)
            let normalData = try VolumeDocument.withAccessibleURL(packageURL) {
                try Data(contentsOf: packageURL.appendingPathComponent(normalMap.file, isDirectory: false), options: [.alwaysMapped])
            }
            let expectedNormalBytes = normalSourceDims.x * normalSourceDims.y * normalSourceDims.z * 4
            guard normalData.count == expectedNormalBytes else {
                throw VolumeDocumentError.unexpectedFileSize(normalData.count, expectedNormalBytes)
            }
            let orientedNormalData = usesNativeLayout ? normalData : reorderLegacyVolume(normalData, dims: normalSourceDims, bytesPerPixel: 4)
            let stagedNormalData = normalSourceDims == targetDims
                ? orientedNormalData
                : resampleNearest(orientedNormalData, sourceDims: normalSourceDims, targetDims: targetDims, bytesPerPixel: 4)
            try VolumeDocument.loadRawData(stagedNormalData, into: normalTexture)
        } else {
            var generated = Data(count: voxelCount * 4)
            generated.withUnsafeMutableBytes { raw in
                let bytes = raw.bindMemory(to: UInt8.self)
                for i in stride(from: 0, to: bytes.count, by: 4) {
                    bytes[i + 0] = 0
                    bytes[i + 1] = 0
                    bytes[i + 2] = 127
                    bytes[i + 3] = 127
                }
            }
            try VolumeDocument.loadRawData(generated, into: normalTexture)
            print("Normal map missing in package; using default object-space normal volume.")
        }

        // -- Confidence --
        let confidenceData: Data
        if let confidenceMap = manifest.confidence {
            let confidenceSourceDims = mapDimensions(for: confidenceMap, fallback: manifest.grid)
            confidenceData = try VolumeDocument.withAccessibleURL(packageURL) {
                try Data(contentsOf: packageURL.appendingPathComponent(confidenceMap.file, isDirectory: false), options: [.alwaysMapped])
            }
            let expectedConfidenceBytes = confidenceSourceDims.x * confidenceSourceDims.y * confidenceSourceDims.z
            guard confidenceData.count == expectedConfidenceBytes else {
                throw VolumeDocumentError.unexpectedFileSize(confidenceData.count, expectedConfidenceBytes)
            }
            let orientedConfidenceData = usesNativeLayout ? confidenceData : reorderLegacyVolume(confidenceData, dims: confidenceSourceDims, bytesPerPixel: 1)
            let stagedConfidenceData = confidenceSourceDims == targetDims
                ? orientedConfidenceData
                : resampleNearest(orientedConfidenceData, sourceDims: confidenceSourceDims, targetDims: targetDims, bytesPerPixel: 1)
            try VolumeDocument.loadRawData(stagedConfidenceData, into: confidenceTexture)
        } else {
            confidenceData = Data(repeating: 255, count: voxelCount)
            try VolumeDocument.loadRawData(confidenceData, into: confidenceTexture)
            print("Confidence map missing in package; assuming fully confident surface volume.")
        }

        // -- Roughness --
        let roughnessData: Data
        if let roughnessMap = manifest.roughness ?? manifest.specular {
            let roughnessSourceDims = mapDimensions(for: roughnessMap, fallback: manifest.grid)
            roughnessData = try VolumeDocument.withAccessibleURL(packageURL) {
                try Data(contentsOf: packageURL.appendingPathComponent(roughnessMap.file, isDirectory: false), options: [.alwaysMapped])
            }
            let expectedRoughnessBytes = roughnessSourceDims.x * roughnessSourceDims.y * roughnessSourceDims.z
            guard roughnessData.count == expectedRoughnessBytes else {
                throw VolumeDocumentError.unexpectedFileSize(roughnessData.count, expectedRoughnessBytes)
            }
            let orientedRoughnessData = usesNativeLayout ? roughnessData : reorderLegacyVolume(roughnessData, dims: roughnessSourceDims, bytesPerPixel: 1)
            let stagedRoughnessData = roughnessSourceDims == targetDims
                ? orientedRoughnessData
                : resampleNearest(orientedRoughnessData, sourceDims: roughnessSourceDims, targetDims: targetDims, bytesPerPixel: 1)
            try VolumeDocument.loadRawData(stagedRoughnessData, into: roughnessTexture)
        } else {
            roughnessData = Data(repeating: 128, count: voxelCount)
            try VolumeDocument.loadRawData(roughnessData, into: roughnessTexture)
            print("Roughness map missing in package; using default roughness volume.")
        }

        // Compute averages for fallback material tinting.
        let roughnessAverage = min(max(Float(roughnessData.reduce(UInt64(0)) { $0 + UInt64($1) }) / max(Float(roughnessData.count), 1.0) / 255.0, 0.0), 1.0)

        var rAcc: Float = 0
        var gAcc: Float = 0
        var bAcc: Float = 0
        let pixelCount = max(stagedAlbedoData.count / 4, 1)
        for i in stride(from: 0, to: stagedAlbedoData.count, by: 4) {
            rAcc += Float(stagedAlbedoData[i + 0])
            gAcc += Float(stagedAlbedoData[i + 1])
            bAcc += Float(stagedAlbedoData[i + 2])
        }
        let averageAlbedo = SIMD3<Float>(rAcc, gAcc, bAcc) / (Float(pixelCount) * 255.0)

        print("Loaded package: albedo avg rgb=\(averageAlbedo), roughness avg=\(roughnessAverage)")

        return SculptLoadPayload(sdfTexture: sdfTexture,
                                 albedoTexture: albedoTexture,
                                 normalTexture: normalTexture,
                                 confidenceTexture: confidenceTexture,
                                 roughnessTexture: roughnessTexture,
                                 textureBoundsMin: textureBoundsMin,
                                 textureBoundsMax: textureBoundsMax,
                                 averageRoughness: roughnessAverage,
                                 averageAlbedo: averageAlbedo)
    }

    // Load packaged volume directly from app bundle.
    func loadBundledPackage(named: String) throws {
        guard let url = Bundle.main.url(forResource: named, withExtension: "sculptpkg") else {
            throw VolumeDocumentError.invalidPackage("Missing bundled package: \(named).sculptpkg")
        }
        try loadFromURL(url)
    }
}
