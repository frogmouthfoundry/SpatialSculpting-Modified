/*
Abstract:
Manages a separate invisible collision entity for the sculpting volume.

Strategy (Option B):
  1. Blit the GPU SDF texture (.private) → shared staging texture (.shared)
     using the same compute-system blit pattern as save().
  2. After the command buffer commits, read the SDF floats on CPU.
  3. Run CPU marching cubes at coarse resolution (step=2 → effective 64³).
  4. Feed positions + faceIndices to ShapeResource.generateStaticMesh
     for concave collision that matches carved surfaces.
  5. Apply to a separate invisible Entity as a static PhysicsBody.

Regeneration triggers:
  - Initial setup (after first frame renders the mesh)
  - Reset / Clear / Load (via scheduleRegeneration)
  - During sculpting (throttled via markDirty + processUpdatesIfNeeded)
*/

import RealityKit
import Metal
import QuartzCore

@MainActor @Observable
final class CollisionManager: @unchecked Sendable {

    // MARK: - Configuration

    /// Minimum seconds between sculpt-triggered collision updates.
    private let updateInterval: TimeInterval = 0.5

    /// Coarse step: sample every Nth voxel for collision mesh.
    /// 4 = effective 32³ from 128³ volume. Keeps vertex count under
    /// UInt16.max (65535) for generateStaticMesh, while providing
    /// adequate fidelity for 5mm debris collision.
    private let coarseStep: Int = 4

    // MARK: - State

    private var collisionEntity: Entity?
    weak var rootEntity: Entity?

    /// Shared staging texture for CPU readback of SDF data.
    private var stagingTexture: MTLTexture?

    /// Volume parameters needed for CPU marching cubes.
    private var dimensions: SIMD3<UInt32> = .zero
    private var voxelSize: SIMD3<Float> = .zero
    private var voxelStartPosition: SIMD3<Float> = .zero

    // Throttle state
    private var isDirty: Bool = false
    private var lastUpdateTime: TimeInterval = 0
    private var isRegenerating: Bool = false
    private var pendingRegeneration: Bool = false

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

        // Invisible collision entity.
        let entity = Entity()
        entity.name = "SculptingVolumeCollision"
        rootEntity.addChild(entity)
        self.collisionEntity = entity

        print("[CollisionManager] Setup: \(dimensions.x)×\(dimensions.y)×\(dimensions.z), step=\(coarseStep)")
    }

    // MARK: - Triggers

    func scheduleRegeneration() {
        guard !pendingRegeneration else { return }
        pendingRegeneration = true
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
        guard let collisionEntity = collisionEntity,
              let texture = stagingTexture else { return }

        let dims = dimensions
        let vSize = voxelSize
        let vStart = voxelStartPosition
        let step = coarseStep

        // 1. Read SDF floats from shared texture.
        let w = Int(dims.x)
        let h = Int(dims.y)
        let d = Int(dims.z)
        var sdfData = [Float](repeating: 0, count: w * h * d)

        sdfData.withUnsafeMutableBufferPointer { buf in
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

        // 2. Run CPU marching cubes at coarse resolution.
        let isoValue: Float = 0
        var positions: [SIMD3<Float>] = []
        var faceIndices: [UInt16] = []

        // Helper to sample SDF with bounds check.
        func sdf(_ x: Int, _ y: Int, _ z: Int) -> Float {
            guard x >= 0, x < w, y >= 0, y < h, z >= 0, z < d else {
                return 1.0 // outside
            }
            return sdfData[z * w * h + y * w + x]
        }

        // Cube vertex offsets (same convention as GPU kernel).
        func cubeVertex(_ i: Int) -> SIMD3<Int> {
            let x = i & 1
            let y = (i >> 1) & 1
            let z = (i >> 2) & 1
            return SIMD3<Int>(x ^ y, y, z)
        }

        // Edge vertex pairs (same as GPU).
        func edgeVertexPair(_ i: Int) -> (Int, Int) {
            let v1 = i & 7
            let v2 = i < 8 ? ((i + 1) & 3) | (i & 4) : v1 + 4
            return (v1, v2)
        }

        // Decode triangle table.
        let table = MarchingCubesData.triangleTable
        func edgeFromTable(_ data: UInt64, _ idx: Int) -> Int {
            Int((data >> (idx * 4)) & 0xF)
        }

        let coarseW = (w - 1) / step
        let coarseH = (h - 1) / step
        let coarseD = (d - 1) / step

        /// Safety limit: stop emitting triangles before UInt16 overflow.
        let vertexLimit = Int(UInt16.max) - 3
        var vertexLimitReached = false

        for cz in 0..<coarseD {
            if vertexLimitReached { break }
            for cy in 0..<coarseH {
                if vertexLimitReached { break }
                for cx in 0..<coarseW {
                    if vertexLimitReached { break }
                    let bx = cx * step
                    let by = cy * step
                    let bz = cz * step

                    // Sample 8 corners.
                    var samples = [Float](repeating: 0, count: 8)
                    for i in 0..<8 {
                        let cv = cubeVertex(i)
                        samples[i] = sdf(bx + cv.x * step,
                                         by + cv.y * step,
                                         bz + cv.z * step)
                    }

                    // Build selector.
                    var selector: UInt = 0
                    for i in 0..<8 {
                        if samples[i] < isoValue {
                            selector |= (1 << i)
                        }
                    }
                    if selector == 0 || selector >= 0xFF { continue }

                    // Interpolate edge positions.
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

                    // Emit triangles from table.
                    let data = table[Int(selector)]
                    var i = 0
                    while i < 15 {
                        let e0 = edgeFromTable(data, i)
                        let e1 = edgeFromTable(data, i + 1)
                        let e2 = edgeFromTable(data, i + 2)
                        if e0 == 15 { break }

                        // UInt16 index limit: stop before overflow.
                        if positions.count + 3 > vertexLimit {
                            vertexLimitReached = true
                            break
                        }

                        let baseIndex = UInt16(positions.count)
                        positions.append(edgePositions[e0])
                        positions.append(edgePositions[e1])
                        positions.append(edgePositions[e2])
                        // Note: GPU kernel writes indices as [v0, v2, v1] (winding order).
                        // For collision shape it doesn't matter, but match it.
                        faceIndices.append(baseIndex)
                        faceIndices.append(baseIndex + 2)
                        faceIndices.append(baseIndex + 1)

                        i += 3
                    }
                }
            }
        }

        let triCount = positions.count / 3
        print("[CollisionManager] CPU marching cubes: \(triCount) triangles from \(coarseW)×\(coarseH)×\(coarseD) grid")

        guard triCount >= 1 else {
            print("[CollisionManager] No triangles — clearing collision")
            collisionEntity.components.remove(CollisionComponent.self)
            collisionEntity.components.remove(PhysicsBodyComponent.self)
            return
        }

        // 3. Generate static mesh collision (concave).
        do {
            let shape = try await ShapeResource.generateStaticMesh(
                positions: positions,
                faceIndices: faceIndices
            )
            collisionEntity.components.set(CollisionComponent(shapes: [shape]))
            collisionEntity.components.set(PhysicsBodyComponent(
                shapes: [shape], mass: 0, material: .default, mode: .static
            ))
            print("[CollisionManager] Static mesh collision applied (\(triCount) tris)")
        } catch {
            print("[CollisionManager] generateStaticMesh failed: \(error)")
        }
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
}
