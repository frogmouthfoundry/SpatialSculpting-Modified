/*
Abstract:
Drill rotation component and system. Spins the drill ball entity every frame.
*/

import RealityKit
import UIKit

// MARK: - Component
struct DrillRotationComponent: Component {
    var rpm: Int

    init(rpm: Int = 400) {
        self.rpm = rpm
    }
}

// MARK: - System
class DrillRotationSystem: System {
    static let query = EntityQuery(where: .has(DrillRotationComponent.self))

    required init(scene: RealityKit.Scene) {}

    func update(context: SceneUpdateContext) {
        for entity in context.scene.performQuery(Self.query) {
            guard let component = entity.components[DrillRotationComponent.self] else { continue }

            let degreesPerSecond = Float(component.rpm) / 60.0 * 360.0
            let radiansPerFrame = degreesPerSecond * Float(context.deltaTime) * .pi / 180.0

            // Rotate around Z-axis (along the drill shaft)
            entity.transform.rotation *= simd_quatf(angle: radiansPerFrame, axis: [0, 0, 1])
        }
    }
}

// MARK: - Drill Ball Factory
extension DrillRotationComponent {
    /// Creates a small rotating sphere for the drill tip.
    @MainActor
    static func createDrillBall(rpm: Int = 400) -> ModelEntity {
        let ballRadius: Float = 0.005
        let ballMesh = MeshResource.generateSphere(radius: ballRadius)

        var material = PhysicallyBasedMaterial()

        if let textureResource = try? TextureResource.load(named: "Bur") {
            material.baseColor = PhysicallyBasedMaterial.BaseColor(texture: .init(textureResource))
            material.metallic = 1.0
            material.roughness = 0.3
        } else {
            material.baseColor = .init(tint: .orange)
            material.metallic = 1.0
            material.roughness = 0.3
        }

        let ball = ModelEntity(mesh: ballMesh, materials: [material])
        ball.components[DrillRotationComponent.self] = DrillRotationComponent(rpm: rpm)

        return ball
    }
}
