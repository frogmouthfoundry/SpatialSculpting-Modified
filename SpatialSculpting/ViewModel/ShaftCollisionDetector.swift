/*
Abstract:
Detects when the drill shaft (not the cutting tip) intersects the sculpted
volume.  Samples the SDF at evenly-spaced points along the shaft axis using
the CPU-readable staging texture maintained by CollisionManager.

When a collision is detected the detector returns a correction vector that
the caller can apply to the tool position to push the drill out of the
volume (physical blocking).
*/

import simd

/// Result of a per-frame shaft collision test.
struct ShaftCollisionResult {
    /// Whether any shaft sample point is inside the sculpted surface.
    let isColliding: Bool
    /// World-space vector to add to the tool position to resolve penetration.
    /// Zero when not colliding.
    let correctionVector: SIMD3<Float>
    /// World-space position of the deepest-penetrating shaft sample (for debug / FX).
    let collisionPoint: SIMD3<Float>
    /// Signed-distance value at the deepest-penetrating point (negative = inside).
    let penetrationDepth: Float
}

@MainActor
final class ShaftCollisionDetector {

    // MARK: - Configuration

    /// How far behind the tip to check (meters).
    private let shaftLength: Float = 0.08       // 8 cm
    /// Number of evenly-spaced sample points along the shaft.
    private let sampleCount: Int = 10
    /// Approximate shaft radius (meters).  SDF values more negative than
    /// -shaftRadius indicate the shaft surface is inside the volume.
    private let shaftRadius: Float = 0.004      // 4 mm

    // MARK: - API

    /// Test the shaft for collision against the SDF volume.
    ///
    /// - Parameters:
    ///   - tipPosition: World-space position of the drill tip (burr center).
    ///   - shaftDirection: Unit vector pointing from tip *toward* the back of
    ///     the drill (the direction the shaft extends).
    ///   - collisionManager: Provides SDF sampling via its staging texture.
    /// - Returns: A ``ShaftCollisionResult`` describing the collision state.
    func test(tipPosition: SIMD3<Float>,
              shaftDirection: SIMD3<Float>,
              collisionManager: CollisionManager) -> ShaftCollisionResult {

        var deepestSDF: Float = .greatestFiniteMagnitude
        var deepestPoint: SIMD3<Float> = .zero
        var deepestIndex: Int = -1

        let step = shaftLength / Float(sampleCount - 1)

        for i in 0..<sampleCount {
            let t = Float(i) * step
            let worldPos = tipPosition + shaftDirection * t

            guard let sdfValue = collisionManager.sampleSDF(at: worldPos) else {
                continue // point outside volume bounds
            }

            if sdfValue < deepestSDF {
                deepestSDF = sdfValue
                deepestPoint = worldPos
                deepestIndex = i
            }
        }

        // No collision: all shaft points are outside the surface.
        guard deepestSDF < -shaftRadius, deepestIndex >= 0 else {
            return ShaftCollisionResult(isColliding: false,
                                        correctionVector: .zero,
                                        collisionPoint: .zero,
                                        penetrationDepth: 0)
        }

        // Compute the SDF gradient (≈ surface normal) at the deepest point
        // using central finite differences on the staging texture.
        let gradient = collisionManager.sampleSDFGradient(at: deepestPoint)
        let gradientLength = simd_length(gradient)

        let normal: SIMD3<Float>
        if gradientLength > 1e-6 {
            normal = gradient / gradientLength
        } else {
            // Fallback: push along shaft direction (away from volume interior).
            normal = -shaftDirection
        }

        // Penetration amount beyond the shaft radius.
        let penetration = -(deepestSDF + shaftRadius)
        let correction = normal * penetration

        return ShaftCollisionResult(isColliding: true,
                                    correctionVector: correction,
                                    collisionPoint: deepestPoint,
                                    penetrationDepth: deepestSDF)
    }
}
