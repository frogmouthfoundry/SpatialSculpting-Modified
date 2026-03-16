/*
Abstract:
Loads a RealityKit material and physics material from the app bundle for bone debris.
Falls back to a default color (#E3DAC9) if no bundle material is found.
Supports an optional custom normal texture for surface detail.
*/

import RealityKit
import UIKit

/// Handles loading and caching of materials for the drawn bone debris boxes.
@MainActor
struct BoneDebrisMaterialLoader {
    
    /// The default debris color (#E3DAC9 — a warm bone/cream).
    static let defaultColor: UIColor = UIColor(
        red: 0xE3 / 255.0,
        green: 0xDA / 255.0,
        blue: 0xC9 / 255.0,
        alpha: 1.0
    )
    
    //Material Names
    static let bundleMaterialName = "BoneDebrisMaterial"
    static let bundlePhysicsMaterialName = "BoneDebrisPhysicsMaterial"
    static let bundleNormalTextureName = "BoneDebrisNormal"
    
    // MARK: - Material Loading
    
    /// Attempts to load a `Material` from the app's RealityKit content bundle.
    /// If no material is found, returns a `PhysicallyBasedMaterial` with the
    /// default color #E3DAC9 and an optional normal map if one is present
    /// in the asset catalog under `BoneDebrisNormal`.
    static func loadDebrisMaterial() -> any Material {
        // Option 1: Load from a .reality or .usdz file in the bundle that contains a named material.
        if let debrisEntity = try? Entity.load(named: bundleMaterialName, in: nil),
           let modelComponent = debrisEntity.components[ModelComponent.self],
           let material = modelComponent.materials.first {
            return material
        }
        
        // Option 2: Try loading from the RealityKitContent bundle if available.
        if let rkBundle = Bundle(identifier: "com.apple.RealityKitContent"),
           let debrisEntity = try? Entity.load(named: bundleMaterialName, in: rkBundle),
           let modelComponent = debrisEntity.components[ModelComponent.self],
           let material = modelComponent.materials.first {
            return material
        }
        
        // Option 3: Build a PBR material from asset catalog textures + normal map.
        var pbrMaterial = PhysicallyBasedMaterial()
        
        // Base color: try loading a texture named "BoneDebrisMaterial", else use default tint.
        if let baseTexture = try? TextureResource.load(named: bundleMaterialName) {
            pbrMaterial.baseColor = PhysicallyBasedMaterial.BaseColor(
                tint: .white,
                texture: .init(baseTexture)
            )
        } else {
            pbrMaterial.baseColor = PhysicallyBasedMaterial.BaseColor(
                tint: defaultColor,
                texture: nil
            )
        }
        
        // Normal map: try loading a texture named "BoneDebrisNormal".
        if let normalTexture = try? TextureResource.load(named: bundleNormalTextureName) {
            pbrMaterial.normal = PhysicallyBasedMaterial.Normal(
                texture: .init(normalTexture)
            )
        }
        
        pbrMaterial.roughness = .init(floatLiteral: 0.7)
        pbrMaterial.metallic  = .init(floatLiteral: 0.0)
        
        return pbrMaterial
    }
    
    // MARK: - Physics Material Loading
    
    /// Attempts to load a `PhysicsMaterialResource` from the bundle.
    /// Returns nil if creation fails (caller can decide whether to apply physics).
    static func loadDebrisPhysicsMaterial() -> PhysicsMaterialResource? {
        return try? PhysicsMaterialResource.generate(
            staticFriction: 0.5,
            dynamicFriction: 0.4,
            restitution: 0.1
        )
    }
}
