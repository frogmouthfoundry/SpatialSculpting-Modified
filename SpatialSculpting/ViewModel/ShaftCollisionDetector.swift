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

    /// Detector center measured from the drill tip back along the shaft axis.
    let detectorCenterOffset: Float = 0.02 // 2 cm
    /// Short cylindrical detector length centered on `detectorCenterOffset`.
    let detectorLength: Float = 0.014 // 1.4 cm
    /// Approximate drill shaft radius at the detector position.
    let detectorRadius: Float = 0.004
    /// Number of sample slices along the short cylinder.
    private let axialSampleCount: Int = 5
    /// Number of points sampled around the cylinder ring per slice.
    private let radialSampleCount: Int = 8

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
        var foundSample = false
        let axis = simd_length_squared(shaftDirection) > 1e-8 ? simd_normalize(shaftDirection) : SIMD3<Float>(0, 0, 1)
        let detectorCenter = tipPosition + axis * detectorCenterOffset
        let halfLength = detectorLength * 0.5

        let referenceUp = abs(simd_dot(axis, SIMD3<Float>(0, 1, 0))) < 0.95
            ? SIMD3<Float>(0, 1, 0)
            : SIMD3<Float>(1, 0, 0)
        let radialX = simd_normalize(simd_cross(axis, referenceUp))
        let radialY = simd_normalize(simd_cross(axis, radialX))

        func considerPoint(_ worldPos: SIMD3<Float>) {
            guard let sdfValue = collisionManager.sampleSDF(at: worldPos) else { return }
            foundSample = true
            if sdfValue < deepestSDF {
                deepestSDF = sdfValue
                deepestPoint = worldPos
            }
        }

        if axialSampleCount <= 1 {
            considerPoint(detectorCenter)
        } else {
            for i in 0..<axialSampleCount {
                let t = Float(i) / Float(axialSampleCount - 1)
                let axialOffset = -halfLength + detectorLength * t
                let sliceCenter = detectorCenter + axis * axialOffset

                // Sample the center and a ring on the cylinder surface so the
                // detector fires when the shaft surface touches the volume.
                considerPoint(sliceCenter)
                for j in 0..<radialSampleCount {
                    let angle = (Float(j) / Float(radialSampleCount)) * (.pi * 2)
                    let ringOffset = radialX * cos(angle) * detectorRadius
                                   + radialY * sin(angle) * detectorRadius
                    considerPoint(sliceCenter + ringOffset)
                }
            }
        }

        // No collision: all cylinder samples are outside the surface.
        guard foundSample, deepestSDF < 0 else {
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
            normal = -axis
        }

        // Surface sampling means any negative value is already penetration.
        let penetration = -deepestSDF
        let correction = normal * penetration

        return ShaftCollisionResult(isColliding: true,
                                    correctionVector: correction,
                                    collisionPoint: deepestPoint,
                                    penetrationDepth: deepestSDF)
    }
}
