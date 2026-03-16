/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
The file format for saving sculptures to.
*/

import SwiftUI
import UniformTypeIdentifiers
import Metal

enum VolumeDocumentError: Error {
    case unexpectedPixelFormat(MTLPixelFormat)
    case missingFileContents
    case unexpectedFileSize(Int, Int)
    case failedToAccessSecurityScopedResource
    case unsupportedPixelFormat(MTLPixelFormat)
    case invalidPackage(String)
    case missingPackageFile(String)
}

struct VolumeDocument: FileDocument {
    // Support selecting a package folder from Open.
    static let packageUTType = UTType(exportedAs: "com.spatial-sculpting.sculptpkg", conformingTo: .folder)
    static let utType = UTType(exportedAs: "com.spatial-sculpting.volume-document", conformingTo: nil)
    static var readableContentTypes: [UTType] { [utType] }
    var data: Data

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw VolumeDocumentError.missingFileContents
        }
        self.data = data
    }

    init(texture: MTLTexture) throws {
        guard texture.pixelFormat == .r32Float else {
            throw VolumeDocumentError.unexpectedPixelFormat(texture.pixelFormat)
        }
        let bytesPerRow = 4 * texture.width
        let bytesPerImage = bytesPerRow * texture.height
        let sizeBytes = bytesPerImage * texture.depth
        var data = Data(count: sizeBytes)
        data.withUnsafeMutableBytes { pointer in
            texture.getBytes(pointer.baseAddress!,
                             bytesPerRow: bytesPerRow,
                             bytesPerImage: bytesPerImage,
                             from: MTLRegionMake3D(0, 0, 0, texture.width, texture.height, texture.depth),
                             mipmapLevel: 0,
                             slice: 0)
        }

        self.data = data
    }

    @MainActor
    static func loadFromURL(_ url: URL, texture: MTLTexture) throws {
        guard texture.pixelFormat == .r32Float else {
            throw VolumeDocumentError.unexpectedPixelFormat(texture.pixelFormat)
        }
        let data = try withAccessibleURL(url) {
            try Data(contentsOf: url, options: [.alwaysMapped])
        }
        try loadRawData(data, into: texture)
    }

    // Generalized raw loader for additional 3D map textures.
    static func loadRawData(_ data: Data, into texture: MTLTexture) throws {
        let bpp = try Self.bytesPerPixel(texture.pixelFormat)
        let bytesPerRow = bpp * texture.width
        let bytesPerImage = bytesPerRow * texture.height
        let sizeBytes = bytesPerImage * texture.depth

        guard data.count == sizeBytes else {
            throw VolumeDocumentError.unexpectedFileSize(data.count, sizeBytes)
        }

        data.withUnsafeBytes { pointer in
            texture.replace(region: MTLRegionMake3D(0, 0, 0, texture.width, texture.height, texture.depth),
                            mipmapLevel: 0,
                            slice: 0,
                            withBytes: pointer.baseAddress!,
                            bytesPerRow: bytesPerRow,
                            bytesPerImage: bytesPerImage)
        }
    }

    static func bytesPerPixel(_ format: MTLPixelFormat) throws -> Int {
        switch format {
        case .r8Unorm:
            return 1
        case .rgba8Unorm, .rgba8Snorm:
            return 4
        case .r32Float:
            return 4
        default:
            throw VolumeDocumentError.unsupportedPixelFormat(format)
        }
    }

    static func parsePackageManifest(at url: URL) throws -> SculptPackageManifest {
        let manifestURL = url.appendingPathComponent("manifest.json", isDirectory: false)
        guard FileManager.default.fileExists(atPath: manifestURL.path()) else {
            throw VolumeDocumentError.missingPackageFile("manifest.json")
        }
        let data = try Data(contentsOf: manifestURL, options: [.alwaysMapped])
        do {
            return try JSONDecoder().decode(SculptPackageManifest.self, from: data)
        } catch {
            throw VolumeDocumentError.invalidPackage("Invalid manifest.json: \(error)")
        }
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: self.data)
    }

    // Allow reading bundle resources without security-scoped access.
    static func withAccessibleURL<T>(_ url: URL, _ body: () throws -> T) throws -> T {
        if url.path.hasPrefix(Bundle.main.bundleURL.path) {
            return try body()
        }
        guard url.startAccessingSecurityScopedResource() else {
            throw VolumeDocumentError.failedToAccessSecurityScopedResource
        }
        defer {
            url.stopAccessingSecurityScopedResource()
        }
        return try body()
    }
}

// Manifest model for multi-map sculpt package import.
struct SculptPackageManifest: Codable {
    struct Grid: Codable {
        let width: Int
        let height: Int
        let depth: Int
    }

    struct Bounds: Codable {
        let min: [Float]
        let max: [Float]
    }

    struct VolumeMap: Codable {
        let file: String
        let format: String
        let space: String?
        let width: Int?
        let height: Int?
        let depth: Int?
    }

    let version: Int
    let layout: String?
    let voxelOrigin: String?
    let grid: Grid
    let bounds: Bounds
    let sdf: VolumeMap
    let albedo: VolumeMap
    let normal: VolumeMap?
    let confidence: VolumeMap?
    let roughness: VolumeMap?
    // Keep optional legacy compatibility with earlier packages.
    let specular: VolumeMap?
}
