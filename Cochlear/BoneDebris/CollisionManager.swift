/*
Abstract:
Manages a separate invisible collision entity for the sculpting volume.

Strategy (Option B):
  1. Blit the GPU SDF texture (.private) → shared staging texture (.shared)
     using the same compute-system blit pattern as save().
  2. After the command buffer commits, read the SDF floats on CPU.
  3. Build chunked coarse collision across the full volume so no single
     collision mesh hits the UInt16 vertex ceiling and truncates.
  4. Build a separate higher-fidelity local collision chunk around the drill
     shaft detector area and use it instead of overlapping coarse chunks.
  5. Feed positions + faceIndices to ShapeResource.generateStaticMesh and
     apply them to separate invisible static collision entities.

Regeneration triggers:
  - Initial setup (after first frame renders the mesh)
  - Reset / Clear / Load (via scheduleRegeneration)
  - During sculpting (throttled via markDirty + processUpdatesIfNeeded)
*/

import RealityKit
import Metal
import QuartzCore

enum CollisionPerformanceTier: Equatable {
    case normal
    case zoomed
}

@MainActor @Observable
final class CollisionManager: @unchecked Sendable {

    // MARK: - Configuration

    /// Minimum seconds between sculpt-triggered collision updates.
    private var updateInterval: TimeInterval = 0.5

    /// Coarse step: sample every Nth voxel for collision mesh.
    /// 4 = effective 32³ from 128³ volume. Keeps vertex count under
    /// UInt16.max (65535) for generateStaticMesh, while providing
    /// adequate fidelity for 5mm debris collision.
    private let coarseStep: Int = 4
    /// Split the full collision volume into several coarse chunks so we do not
    /// truncate the mesh once a single UInt16-limited shape fills up.
    private let coarseChunkDivisions = SIMD3<Int>(4, 4, 4)
    /// Local high-fidelity collision region centered on the shaft detector.
    /// This is specified in displayed-space meters and converted back into
    /// root-local extents using the root's current scale.
    private var localFocusDisplayedExtent: Float = 0.10
    /// Use full-resolution SDF marching cubes inside the local focus chunk.
    private let localStep: Int = 1

    // MARK: - State

    private var collisionEntity: Entity?
    private var coarseCollisionEntity: Entity?
    private var localCollisionEntity: Entity?
    weak var rootEntity: Entity?
    private var localFocusCenterInRoot: SIMD3<Float>? = nil

    /// Shared staging texture for CPU readback of SDF data.
    private var stagingTexture: MTLTexture?
    /// CPU-cached SDF volume from the last completed blit.
    /// Shaft collision sampling reads from this cache to avoid per-frame texture reads.
    private var cachedSDFData: [Float] = []
    private var hasValidSDFCache: Bool = false

    /// Volume parameters needed for CPU marching cubes.
    private var dimensions: SIMD3<UInt32> = .zero
    private var voxelSize: SIMD3<Float> = .zero
    private var voxelStartPosition: SIMD3<Float> = .zero

    // Throttle state
    private var isDirty: Bool = false
    private var lastUpdateTime: TimeInterval = 0
    private var isRegenerating: Bool = false
    private var pendingRegeneration: Bool = false
    /// Regenerate the expensive static collision mesh less frequently than SDF cache refresh.
    private var fullMeshRegenerationInterval: TimeInterval = 1.0
    private var lastFullMeshRegenerationTime: TimeInterval = 0
    private var forceFullMeshRegeneration: Bool = true
    private var performanceTier: CollisionPerformanceTier = .normal

    /// Set by scheduleRegeneration / processUpdatesIfNeeded.
    /// The SculptingToolSystem picks this up and triggers the blit.
    var blitRequested: Bool = false

    // MARK: - Setup

    func setup(rootEntity: Entity, voxelVolume: VoxelVolume) {
        self.rootEntity = rootEntity
        self.dimensions = voxelVolume.dimensions
        self.voxelSize = voxelVolume.voxelSize
        self.voxelStartPosition = voxelVolume.voxelStartPosition

        // Create shared staging texture (same format, storageModeShared for CPU read).
        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .r32Float
        desc.width = Int(dimensions.x)
        desc.height = Int(dimensions.y)
        desc.depth = Int(dimensions.z)
        desc.usage = []
        desc.storageMode = .shared
        self.stagingTexture = metalDevice?.makeTexture(descriptor: desc)
        let voxelCount = Int(dimensions.x) * Int(dimensions.y) * Int(dimensions.z)
        self.cachedSDFData = [Float](repeating: 1.0, count: voxelCount)
        self.hasValidSDFCache = false

        // Invisible collision entity.
        let entity = Entity()
        entity.name = "SculptingVolumeCollision"
        rootEntity.addChild(entity)
        self.collisionEntity = entity

        let coarseEntity = Entity()
        coarseEntity.name = "SculptingVolumeCollisionCoarse"
        entity.addChild(coarseEntity)
        self.coarseCollisionEntity = coarseEntity

        let localEntity = Entity()
        localEntity.name = "SculptingVolumeCollisionLocal"
        entity.addChild(localEntity)
        self.localCollisionEntity = localEntity

        print("[CollisionManager] Setup: \(dimensions.x)×\(dimensions.y)×\(dimensions.z), step=\(coarseStep)")
    }

    func setPerformanceTier(_ tier: CollisionPerformanceTier) {
        guard performanceTier != tier else { return }
        performanceTier = tier
        switch tier {
        case .normal:
            updateInterval = 0.5
            fullMeshRegenerationInterval = 1.0
            localFocusDisplayedExtent = 0.10
        case .zoomed:
            updateInterval = 0.33
            fullMeshRegenerationInterval = 1.0
            localFocusDisplayedExtent = 0.08
        }
    }

    // MARK: - Triggers

    func scheduleRegeneration() {
        guard !pendingRegeneration else { return }
        pendingRegeneration = true
        forceFullMeshRegeneration = true
        print("[CollisionManager] Scheduled regeneration (0.5s delay)")

        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(500))
            self.pendingRegeneration = false
            self.requestBlit()
        }
    }

    func markDirty() {
        isDirty = true
    }

    func updateLocalCollisionFocus(centerInRoot: SIMD3<Float>) {
        localFocusCenterInRoot = centerInRoot
    }

    func processUpdatesIfNeeded() {
        guard isDirty, !isRegenerating else { return }

        let now = CACurrentMediaTime()
        guard now - lastUpdateTime >= updateInterval else { return }

        isDirty = false
        lastUpdateTime = now
        requestBlit()
    }

    private func requestBlit() {
        guard !isRegenerating else { return }
        isRegenerating = true
        blitRequested = true
    }

    /// Returns the staging texture for the compute system to blit into.
    var collisionStagingTexture: MTLTexture? { stagingTexture }
    var hasSDFCache: Bool { hasValidSDFCache }

    /// Called by the compute system's completion handler after blit finishes.
    /// Runs CPU marching cubes and generates the static collision mesh.
    nonisolated func onBlitComplete() {
        Task { @MainActor in
            await self.regenerateCollision()
            self.isRegenerating = false
        }
    }

    // MARK: - CPU Marching Cubes + Collision Generation

    private func regenerateCollision() async {
        guard collisionEntity != nil,
              let coarseCollisionEntity,
              let localCollisionEntity,
              let texture = stagingTexture else { return }

        // 1. Read SDF floats from shared texture.
        let w = Int(dimensions.x)
        let h = Int(dimensions.y)
        let d = Int(dimensions.z)
        let voxelCount = w * h * d
        if cachedSDFData.count != voxelCount {
            cachedSDFData = [Float](repeating: 1.0, count: voxelCount)
        }

        cachedSDFData.withUnsafeMutableBufferPointer { buf in
            texture.getBytes(
                buf.baseAddress!,
                bytesPerRow: w * MemoryLayout<Float>.size,
                bytesPerImage: w * h * MemoryLayout<Float>.size,
                from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                size: MTLSize(width: w, height: h, depth: d)),
                mipmapLevel: 0,
                slice: 0
            )
        }
        hasValidSDFCache = true

        let now = CACurrentMediaTime()
        let localFocusRange = makeLocalFocusChunkRange()
        if let localFocusRange {
            let localMeshData = buildCollisionMeshChunk(cellRangeX: localFocusRange.x,
                                                        cellRangeY: localFocusRange.y,
                                                        cellRangeZ: localFocusRange.z,
                                                        step: localStep,
                                                        label: "local")
            print("[CollisionManager] Local chunk tris=\(localMeshData.triangleCount) truncated=\(localMeshData.vertexLimitReached)")
            await applyCollisionMeshChunk(localMeshData, to: localCollisionEntity)
        } else {
            clearCollisionComponents(on: localCollisionEntity)
        }

        let shouldRegenerateCoarseMesh = forceFullMeshRegeneration ||
            (now - lastFullMeshRegenerationTime >= fullMeshRegenerationInterval)
        guard shouldRegenerateCoarseMesh else {
            // Fast path: keep SDF cache fresh for shaft collision and keep the
            // local collision chunk responsive without rebuilding all coarse chunks.
            return
        }
        forceFullMeshRegeneration = false
        lastFullMeshRegenerationTime = now

        let coarseRanges = makeFullVolumeChunkRanges(step: coarseStep)
        var generatedShapes: [ShapeResource] = []
        var generatedChunkCount = 0
        for coarseRange in coarseRanges {
            if let localFocusRange,
               chunkRangeIntersectsLocalFocus(coarseRange, localRange: localFocusRange, coarseStep: coarseStep) {
                continue
            }

            let meshData = buildCollisionMeshChunk(cellRangeX: coarseRange.x,
                                                   cellRangeY: coarseRange.y,
                                                   cellRangeZ: coarseRange.z,
                                                   step: coarseStep,
                                                   label: coarseRange.label)
            if meshData.vertexLimitReached {
                print("[CollisionManager] Coarse chunk \(coarseRange.label) truncated at UInt16 limit")
            }
            guard meshData.triangleCount > 0 else { continue }
            do {
                let shape = try await ShapeResource.generateStaticMesh(
                    positions: meshData.positions,
                    faceIndices: meshData.faceIndices
                )
                generatedShapes.append(shape)
                generatedChunkCount += 1
            } catch {
                print("[CollisionManager] generateStaticMesh failed for \(coarseRange.label): \(error)")
            }
        }

        if generatedShapes.isEmpty {
            print("[CollisionManager] No coarse chunk triangles — clearing coarse collision")
            clearCollisionComponents(on: coarseCollisionEntity)
            return
        }

        coarseCollisionEntity.components.set(CollisionComponent(shapes: generatedShapes))
        coarseCollisionEntity.components.set(PhysicsBodyComponent(
            shapes: generatedShapes, mass: 0, material: .default, mode: .static
        ))
        print("[CollisionManager] Applied \(generatedChunkCount) coarse collision chunks")
    }

    // MARK: - SDF Sampling (CPU)

    /// Sample the SDF value at a world-space position using the shared staging texture.
    /// Returns `nil` if the point is outside the volume bounds or the staging texture
    /// hasn't been blitted yet.
    func sampleSDF(at worldPosition: SIMD3<Float>) -> Float? {
        guard dimensions != .zero else { return nil }

        // World → voxel coordinates (continuous).
        let voxelCoord = (worldPosition - voxelStartPosition) / voxelSize

        let w = Int(dimensions.x)
        let h = Int(dimensions.y)
        let d = Int(dimensions.z)

        guard voxelCoord.x >= 0, voxelCoord.x <= Float(w - 1),
              voxelCoord.y >= 0, voxelCoord.y <= Float(h - 1),
              voxelCoord.z >= 0, voxelCoord.z <= Float(d - 1) else {
            return nil // outside volume
        }

        let x0 = max(0, min(w - 1, Int(floor(voxelCoord.x))))
        let y0 = max(0, min(h - 1, Int(floor(voxelCoord.y))))
        let z0 = max(0, min(d - 1, Int(floor(voxelCoord.z))))
        let x1 = min(w - 1, x0 + 1)
        let y1 = min(h - 1, y0 + 1)
        let z1 = min(d - 1, z0 + 1)

        let tx = voxelCoord.x - Float(x0)
        let ty = voxelCoord.y - Float(y0)
        let tz = voxelCoord.z - Float(z0)

        func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
            a + (b - a) * t
        }

        let c000 = readSDFValue(x: x0, y: y0, z: z0)
        let c100 = readSDFValue(x: x1, y: y0, z: z0)
        let c010 = readSDFValue(x: x0, y: y1, z: z0)
        let c110 = readSDFValue(x: x1, y: y1, z: z0)
        let c001 = readSDFValue(x: x0, y: y0, z: z1)
        let c101 = readSDFValue(x: x1, y: y0, z: z1)
        let c011 = readSDFValue(x: x0, y: y1, z: z1)
        let c111 = readSDFValue(x: x1, y: y1, z: z1)

        guard let c000, let c100, let c010, let c110,
              let c001, let c101, let c011, let c111 else {
            return nil
        }

        let c00 = lerp(c000, c100, tx)
        let c10 = lerp(c010, c110, tx)
        let c01 = lerp(c001, c101, tx)
        let c11 = lerp(c011, c111, tx)
        let c0 = lerp(c00, c10, ty)
        let c1 = lerp(c01, c11, ty)
        return lerp(c0, c1, tz)
    }

    /// Compute the SDF gradient (≈ outward surface normal) at a world-space
    /// position using central finite differences on the staging texture.
    func sampleSDFGradient(at worldPosition: SIMD3<Float>) -> SIMD3<Float> {
        let h = voxelSize // step one voxel in each axis

        let xp = sampleSDF(at: worldPosition + SIMD3<Float>(h.x, 0, 0)) ?? 1.0
        let xn = sampleSDF(at: worldPosition - SIMD3<Float>(h.x, 0, 0)) ?? 1.0
        let yp = sampleSDF(at: worldPosition + SIMD3<Float>(0, h.y, 0)) ?? 1.0
        let yn = sampleSDF(at: worldPosition - SIMD3<Float>(0, h.y, 0)) ?? 1.0
        let zp = sampleSDF(at: worldPosition + SIMD3<Float>(0, 0, h.z)) ?? 1.0
        let zn = sampleSDF(at: worldPosition - SIMD3<Float>(0, 0, h.z)) ?? 1.0

        return SIMD3<Float>(xp - xn, yp - yn, zp - zn) / (2.0 * h)
    }

    // MARK: - Test Sphere

    func dropTestSphere() {
        guard let root = rootEntity else { return }

        let radius: Float = 0.015
        let sphere = ModelEntity(
            mesh: .generateSphere(radius: radius),
            materials: [SimpleMaterial(color: .red, isMetallic: false)]
        )
        sphere.name = "CollisionTestSphere"
        sphere.position = SIMD3<Float>(0, 0, 0.45)

        let shape = ShapeResource.generateSphere(radius: radius)
        sphere.components.set(CollisionComponent(shapes: [shape]))
        sphere.components.set(PhysicsBodyComponent(
            shapes: [shape], mass: 0.05,
            material: .generate(staticFriction: 0.8, dynamicFriction: 0.5, restitution: 0.3),
            mode: .dynamic
        ))

        root.addChild(sphere)
        print("[CollisionManager] Dropped test sphere at (0, 0, 0.45)")
    }

    // MARK: - Helpers

    private struct CollisionChunkRange {
        let x: Range<Int>
        let y: Range<Int>
        let z: Range<Int>
        let label: String
    }

    private struct CollisionChunkMeshData {
        let positions: [SIMD3<Float>]
        let faceIndices: [UInt16]
        let triangleCount: Int
        let vertexLimitReached: Bool
        let label: String
    }

    private func localFocusHalfExtentInRoot() -> SIMD3<Float> {
        let displayedHalfExtent = localFocusDisplayedExtent * 0.5
        let rootScale = rootEntity?.transform.scale ?? SIMD3<Float>(repeating: 1)
        let safeScale = SIMD3<Float>(
            abs(rootScale.x) > 1e-6 ? abs(rootScale.x) : 1.0,
            abs(rootScale.y) > 1e-6 ? abs(rootScale.y) : 1.0,
            abs(rootScale.z) > 1e-6 ? abs(rootScale.z) : 1.0
        )
        return SIMD3<Float>(repeating: displayedHalfExtent) / safeScale
    }

    private func makeLocalFocusChunkRange() -> CollisionChunkRange? {
        guard let localFocusCenterInRoot else { return nil }

        let cellLimitX = Int(dimensions.x) - 1
        let cellLimitY = Int(dimensions.y) - 1
        let cellLimitZ = Int(dimensions.z) - 1
        guard cellLimitX > 0, cellLimitY > 0, cellLimitZ > 0 else { return nil }

        let halfExtent = localFocusHalfExtentInRoot()
        let minWorld = localFocusCenterInRoot - halfExtent
        let maxWorld = localFocusCenterInRoot + halfExtent
        let minVoxel = (minWorld - voxelStartPosition) / voxelSize
        let maxVoxel = (maxWorld - voxelStartPosition) / voxelSize

        let startX = max(0, min(cellLimitX - 1, Int(floor(minVoxel.x))))
        let startY = max(0, min(cellLimitY - 1, Int(floor(minVoxel.y))))
        let startZ = max(0, min(cellLimitZ - 1, Int(floor(minVoxel.z))))
        let endX = max(startX + 1, min(cellLimitX, Int(ceil(maxVoxel.x))))
        let endY = max(startY + 1, min(cellLimitY, Int(ceil(maxVoxel.y))))
        let endZ = max(startZ + 1, min(cellLimitZ, Int(ceil(maxVoxel.z))))

        guard startX < endX, startY < endY, startZ < endZ else { return nil }
        return CollisionChunkRange(x: startX..<endX,
                                   y: startY..<endY,
                                   z: startZ..<endZ,
                                   label: "local")
    }

    private func makeFullVolumeChunkRanges(step: Int) -> [CollisionChunkRange] {
        let coarseW = max(1, (Int(dimensions.x) - 1) / step)
        let coarseH = max(1, (Int(dimensions.y) - 1) / step)
        let coarseD = max(1, (Int(dimensions.z) - 1) / step)

        let chunkSizeX = max(1, Int(ceil(Double(coarseW) / Double(coarseChunkDivisions.x))))
        let chunkSizeY = max(1, Int(ceil(Double(coarseH) / Double(coarseChunkDivisions.y))))
        let chunkSizeZ = max(1, Int(ceil(Double(coarseD) / Double(coarseChunkDivisions.z))))

        var ranges: [CollisionChunkRange] = []
        var chunkIndex = 0
        for z in stride(from: 0, to: coarseD, by: chunkSizeZ) {
            for y in stride(from: 0, to: coarseH, by: chunkSizeY) {
                for x in stride(from: 0, to: coarseW, by: chunkSizeX) {
                    let range = CollisionChunkRange(
                        x: x..<min(coarseW, x + chunkSizeX),
                        y: y..<min(coarseH, y + chunkSizeY),
                        z: z..<min(coarseD, z + chunkSizeZ),
                        label: "coarse-\(chunkIndex)"
                    )
                    ranges.append(range)
                    chunkIndex += 1
                }
            }
        }
        return ranges
    }

    private func chunkRangeIntersectsLocalFocus(_ coarseRange: CollisionChunkRange,
                                                localRange: CollisionChunkRange,
                                                coarseStep: Int) -> Bool {
        let coarseX = (coarseRange.x.lowerBound * coarseStep)..<(coarseRange.x.upperBound * coarseStep)
        let coarseY = (coarseRange.y.lowerBound * coarseStep)..<(coarseRange.y.upperBound * coarseStep)
        let coarseZ = (coarseRange.z.lowerBound * coarseStep)..<(coarseRange.z.upperBound * coarseStep)

        func overlaps(_ lhs: Range<Int>, _ rhs: Range<Int>) -> Bool {
            lhs.lowerBound < rhs.upperBound && rhs.lowerBound < lhs.upperBound
        }

        return overlaps(coarseX, localRange.x) &&
               overlaps(coarseY, localRange.y) &&
               overlaps(coarseZ, localRange.z)
    }

    private func buildCollisionMeshChunk(cellRangeX: Range<Int>,
                                         cellRangeY: Range<Int>,
                                         cellRangeZ: Range<Int>,
                                         step: Int,
                                         label: String) -> CollisionChunkMeshData {
        let isoValue: Float = 0
        let vSize = voxelSize
        let vStart = voxelStartPosition

        func sdf(_ x: Int, _ y: Int, _ z: Int) -> Float {
            guard let value = readSDFValue(x: x, y: y, z: z) else { return 1.0 }
            return value
        }

        func cubeVertex(_ i: Int) -> SIMD3<Int> {
            let x = i & 1
            let y = (i >> 1) & 1
            let z = (i >> 2) & 1
            return SIMD3<Int>(x ^ y, y, z)
        }

        func edgeVertexPair(_ i: Int) -> (Int, Int) {
            let v1 = i & 7
            let v2 = i < 8 ? ((i + 1) & 3) | (i & 4) : v1 + 4
            return (v1, v2)
        }

        let table = MarchingCubesData.triangleTable
        func edgeFromTable(_ data: UInt64, _ idx: Int) -> Int {
            Int((data >> (idx * 4)) & 0xF)
        }

        let vertexLimit = Int(UInt16.max) - 3
        var vertexLimitReached = false
        var positions: [SIMD3<Float>] = []
        var faceIndices: [UInt16] = []

        for cz in cellRangeZ {
            if vertexLimitReached { break }
            for cy in cellRangeY {
                if vertexLimitReached { break }
                for cx in cellRangeX {
                    if vertexLimitReached { break }
                    let bx = cx * step
                    let by = cy * step
                    let bz = cz * step

                    var samples = [Float](repeating: 0, count: 8)
                    for i in 0..<8 {
                        let cv = cubeVertex(i)
                        samples[i] = sdf(bx + cv.x * step,
                                         by + cv.y * step,
                                         bz + cv.z * step)
                    }

                    var selector: UInt = 0
                    for i in 0..<8 {
                        if samples[i] < isoValue {
                            selector |= (1 << i)
                        }
                    }
                    if selector == 0 || selector >= 0xFF { continue }

                    var edgePositions = [SIMD3<Float>](repeating: .zero, count: 12)
                    for i in 0..<12 {
                        let (v1i, v2i) = edgeVertexPair(i)
                        let cv1 = cubeVertex(v1i)
                        let cv2 = cubeVertex(v2i)

                        let p1 = SIMD3<Float>(Float(bx + cv1.x * step),
                                              Float(by + cv1.y * step),
                                              Float(bz + cv1.z * step))
                        let p2 = SIMD3<Float>(Float(bx + cv2.x * step),
                                              Float(by + cv2.y * step),
                                              Float(bz + cv2.z * step))

                        let s1 = samples[v1i]
                        let s2 = samples[v2i]
                        let denom = s2 - s1
                        let t: Float = abs(denom) > 1e-8 ? (isoValue - s1) / denom : 0.5
                        let voxelPos = simd_mix(p1, p2, SIMD3<Float>(repeating: t))
                        edgePositions[i] = voxelPos * vSize + vStart
                    }

                    let data = table[Int(selector)]
                    var i = 0
                    while i < 15 {
                        let e0 = edgeFromTable(data, i)
                        let e1 = edgeFromTable(data, i + 1)
                        let e2 = edgeFromTable(data, i + 2)
                        if e0 == 15 { break }

                        if positions.count + 3 > vertexLimit {
                            vertexLimitReached = true
                            break
                        }

                        let baseIndex = UInt16(positions.count)
                        positions.append(edgePositions[e0])
                        positions.append(edgePositions[e1])
                        positions.append(edgePositions[e2])
                        faceIndices.append(baseIndex)
                        faceIndices.append(baseIndex + 2)
                        faceIndices.append(baseIndex + 1)
                        i += 3
                    }
                }
            }
        }

        return CollisionChunkMeshData(positions: positions,
                                      faceIndices: faceIndices,
                                      triangleCount: positions.count / 3,
                                      vertexLimitReached: vertexLimitReached,
                                      label: label)
    }

    private func applyCollisionMeshChunk(_ meshData: CollisionChunkMeshData,
                                         to entity: Entity) async {
        guard meshData.triangleCount > 0 else {
            clearCollisionComponents(on: entity)
            return
        }

        do {
            let shape = try await ShapeResource.generateStaticMesh(
                positions: meshData.positions,
                faceIndices: meshData.faceIndices
            )
            entity.components.set(CollisionComponent(shapes: [shape]))
            entity.components.set(PhysicsBodyComponent(
                shapes: [shape], mass: 0, material: .default, mode: .static
            ))
        } catch {
            print("[CollisionManager] generateStaticMesh failed for \(meshData.label): \(error)")
            clearCollisionComponents(on: entity)
        }
    }

    private func clearCollisionComponents(on entity: Entity) {
        entity.components.remove(CollisionComponent.self)
        entity.components.remove(PhysicsBodyComponent.self)
    }

    private func readSDFValue(x: Int, y: Int, z: Int) -> Float? {
        let w = Int(dimensions.x)
        let h = Int(dimensions.y)
        let d = Int(dimensions.z)
        guard x >= 0, x < w, y >= 0, y < h, z >= 0, z < d else {
            return nil
        }

        let index = z * w * h + y * w + x
        if hasValidSDFCache, index >= 0, index < cachedSDFData.count {
            return cachedSDFData[index]
        }

        guard let texture = stagingTexture else { return nil }
        var value: Float = 0
        let region = MTLRegion(origin: MTLOrigin(x: x, y: y, z: z),
                               size: MTLSize(width: 1, height: 1, depth: 1))
        texture.getBytes(&value,
                         bytesPerRow: MemoryLayout<Float>.size,
                         bytesPerImage: MemoryLayout<Float>.size,
                         from: region,
                         mipmapLevel: 0,
                         slice: 0)
        return value
    }
}
