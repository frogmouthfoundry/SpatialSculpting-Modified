/*
Abstract:
Runtime diagnostics for verifying the physics and collision setup between
bone debris and the sculpted volume mesh chunks. Call `runAll(rootEntity:)`
to print a full report to the Xcode console.

Usage from a button action or debug breakpoint:
    PhysicsDiagnostics.runAll(rootEntity: sculpting.rootEntity!)
*/

import RealityKit
import os

@MainActor
enum PhysicsDiagnostics {
    
    private static let log = Logger(subsystem: "SpatialSculpting", category: "PhysicsDiag")
    
    // MARK: - Run All
    
    /// Runs every diagnostic check and prints results to the console.
    /// Returns `true` if all checks pass.
    @discardableResult
    static func runAll(rootEntity: Entity) -> Bool {
        log.info("═══════════════════════════════════════════")
        log.info("  PHYSICS DIAGNOSTICS — START")
        log.info("═══════════════════════════════════════════")
        
        var allPassed = true
        
        allPassed = checkRootSimulation(rootEntity) && allPassed
        allPassed = checkNoSeparateSimulations(rootEntity) && allPassed
        allPassed = checkMeshChunkPhysics(rootEntity) && allPassed
        allPassed = checkDebrisContainerExists(rootEntity) && allPassed
        allPassed = checkDebrisPhysics(rootEntity) && allPassed
        allPassed = checkSimulationHierarchy(rootEntity) && allPassed
        
        log.info("═══════════════════════════════════════════")
        if allPassed {
            log.info("  ✅ ALL CHECKS PASSED")
        } else {
            log.error("  ❌ SOME CHECKS FAILED — see above")
        }
        log.info("═══════════════════════════════════════════")
        
        return allPassed
    }
    
    // MARK: - Individual Checks
    
    /// 1. Root entity must have a PhysicsSimulationComponent with -Z gravity.
    static func checkRootSimulation(_ root: Entity) -> Bool {
        log.info("── Check 1: Root PhysicsSimulationComponent ──")
        
        guard let sim = root.components[PhysicsSimulationComponent.self] else {
            log.error("  ❌ FAIL: Root entity has NO PhysicsSimulationComponent.")
            log.error("         Debris and mesh chunks would be in the default simulation")
            log.error("         with -Y gravity. Custom -Z gravity won't work.")
            return false
        }
        
        let g = sim.gravity
        log.info("  gravity = [\(g.x), \(g.y), \(g.z)]")
        
        let hasNegativeZ = g.z < -1.0 && abs(g.x) < 0.01 && abs(g.y) < 0.01
        if hasNegativeZ {
            log.info("  ✅ PASS: Root has PhysicsSimulationComponent with -Z gravity.")
        } else {
            log.error("  ❌ FAIL: Gravity vector is not primarily -Z. Expected ~[0, 0, -9.81].")
        }
        return hasNegativeZ
    }
    
    /// 2. No child entity should have its own PhysicsSimulationComponent
    ///    (would create a separate simulation, preventing cross-collision).
    static func checkNoSeparateSimulations(_ root: Entity) -> Bool {
        log.info("── Check 2: No separate PhysicsSimulationComponent on children ──")
        
        var found: [String] = []
        walkDescendants(root) { entity in
            if entity !== root,
               entity.components[PhysicsSimulationComponent.self] != nil {
                found.append(entity.name.isEmpty ? "(unnamed)" : entity.name)
            }
        }
        
        if found.isEmpty {
            log.info("  ✅ PASS: No child has its own PhysicsSimulationComponent.")
            return true
        } else {
            log.error("  ❌ FAIL: These children have their own PhysicsSimulationComponent,")
            log.error("         creating separate simulations that prevent cross-collision:")
            for name in found {
                log.error("         — \(name)")
            }
            return false
        }
    }
    
    /// 3. Mesh chunk entities must have CollisionComponent + static PhysicsBodyComponent.
    static func checkMeshChunkPhysics(_ root: Entity) -> Bool {
        log.info("── Check 3: Mesh chunk collision + static PhysicsBody ──")
        
        var totalChunks = 0
        var withCollision = 0
        var withPhysicsBody = 0
        var withStaticMode = 0
        
        // Mesh chunks are direct children of root with a ModelComponent but
        // not named "BoneDebris*" or the sculpting tool.
        for child in root.children {
            let name = child.name
            if name.hasPrefix("BoneDebris") { continue }
            if child.components[ModelComponent.self] != nil ||
               child.components[CollisionComponent.self] != nil ||
               child.components[PhysicsBodyComponent.self] != nil {
                // Likely a mesh chunk entity (or at least something with physics).
                // Skip known non-chunk entities.
                if name == "SculptingTool" || name == "AdditiveIcon" ||
                   name == "SubtractiveIcon" || name == "EnlargeIcon" ||
                   name == "ReduceIcon" { continue }
                
                totalChunks += 1
                
                if child.components[CollisionComponent.self] != nil {
                    withCollision += 1
                }
                if let pb = child.components[PhysicsBodyComponent.self] {
                    withPhysicsBody += 1
                    if pb.mode == .static {
                        withStaticMode += 1
                    }
                }
            }
        }
        
        log.info("  Found \(totalChunks) potential mesh chunk entities.")
        log.info("    With CollisionComponent:        \(withCollision)")
        log.info("    With PhysicsBodyComponent:       \(withPhysicsBody)")
        log.info("    With mode == .static:            \(withStaticMode)")
        
        if totalChunks == 0 {
            log.warning("  ⚠️ WARN: No mesh chunk entities found. Volume may not be sculpted yet.")
            return true  // Not a failure, just nothing to check.
        }
        
        let pass = (withCollision == totalChunks) &&
                   (withPhysicsBody == totalChunks) &&
                   (withStaticMode == totalChunks)
        
        if pass {
            log.info("  ✅ PASS: All mesh chunks have collision + static physics body.")
        } else {
            log.error("  ❌ FAIL: Not all mesh chunks are properly configured.")
            if withCollision < totalChunks {
                log.error("         \(totalChunks - withCollision) chunk(s) missing CollisionComponent.")
            }
            if withPhysicsBody < totalChunks {
                log.error("         \(totalChunks - withPhysicsBody) chunk(s) missing PhysicsBodyComponent.")
            }
            if withStaticMode < withPhysicsBody {
                log.error("         \(withPhysicsBody - withStaticMode) chunk(s) are NOT static mode.")
            }
        }
        return pass
    }
    
    /// 4. BoneDebrisContainer must exist as a child of root.
    static func checkDebrisContainerExists(_ root: Entity) -> Bool {
        log.info("── Check 4: BoneDebrisContainer exists ──")
        
        let container = root.children.first(where: { $0.name == "BoneDebrisContainer" })
        if container != nil {
            log.info("  ✅ PASS: BoneDebrisContainer found as child of root.")
            return true
        } else {
            log.error("  ❌ FAIL: BoneDebrisContainer not found. setup() may not have been called.")
            return false
        }
    }
    
    /// 5. Debris boxes (if any) must have CollisionComponent + PhysicsBodyComponent.
    static func checkDebrisPhysics(_ root: Entity) -> Bool {
        log.info("── Check 5: Debris box collision + PhysicsBody ──")
        
        guard let container = root.children.first(where: { $0.name == "BoneDebrisContainer" }) else {
            log.warning("  ⚠️ WARN: No BoneDebrisContainer. Skipping.")
            return true
        }
        
        var totalDebris = 0
        var withCollision = 0
        var withPhysicsBody = 0
        var dynamicCount = 0
        var staticCount = 0
        
        for child in container.children {
            totalDebris += 1
            if child.components[CollisionComponent.self] != nil { withCollision += 1 }
            if let pb = child.components[PhysicsBodyComponent.self] {
                withPhysicsBody += 1
                if pb.mode == .dynamic { dynamicCount += 1 }
                if pb.mode == .static  { staticCount += 1 }
            }
        }
        
        log.info("  Found \(totalDebris) debris boxes.")
        log.info("    With CollisionComponent:   \(withCollision)")
        log.info("    With PhysicsBodyComponent:  \(withPhysicsBody)")
        log.info("    Dynamic (gravity on):       \(dynamicCount)")
        log.info("    Static (gravity off):       \(staticCount)")
        
        if totalDebris == 0 {
            log.warning("  ⚠️ WARN: No debris drawn yet. Draw some debris and re-run.")
            return true
        }
        
        let pass = (withCollision == totalDebris) && (withPhysicsBody == totalDebris)
        if pass {
            log.info("  ✅ PASS: All debris boxes have collision + physics body.")
        } else {
            log.error("  ❌ FAIL: Some debris boxes are missing components.")
        }
        return pass
    }
    
    /// 6. Verify that mesh chunks and debris share the same simulation hierarchy
    ///    (both must be under the entity that owns PhysicsSimulationComponent).
    static func checkSimulationHierarchy(_ root: Entity) -> Bool {
        log.info("── Check 6: Shared simulation hierarchy ──")
        
        // Find which entity owns the PhysicsSimulationComponent.
        let simOwner = findSimulationOwner(from: root)
        let simOwnerName = simOwner?.name ?? "(none)"
        log.info("  Simulation owner: \(simOwnerName)")
        
        // Check mesh chunks are descendants of simOwner.
        var chunksBelowSim = true
        for child in root.children {
            if child.name.hasPrefix("BoneDebris") { continue }
            if child.components[PhysicsBodyComponent.self] != nil {
                if !isDescendantOf(entity: child, ancestor: simOwner) {
                    log.error("  ❌ Mesh chunk '\(child.name)' is NOT under simulation owner '\(simOwnerName)'.")
                    chunksBelowSim = false
                }
            }
        }
        
        // Check debris container is a descendant of simOwner.
        var debrisBelowSim = true
        if let container = root.children.first(where: { $0.name == "BoneDebrisContainer" }) {
            if !isDescendantOf(entity: container, ancestor: simOwner) {
                log.error("  ❌ BoneDebrisContainer is NOT under simulation owner '\(simOwnerName)'.")
                debrisBelowSim = false
            }
        }
        
        let pass = chunksBelowSim && debrisBelowSim
        if pass {
            log.info("  ✅ PASS: Mesh chunks and debris are in the same simulation hierarchy.")
        } else {
            log.error("  ❌ FAIL: Entities are split across different simulations — collisions won't work.")
        }
        return pass
    }
    
    // MARK: - Helpers
    
    /// Walks all descendants of an entity (depth-first).
    private static func walkDescendants(_ entity: Entity, visitor: (Entity) -> Void) {
        for child in entity.children {
            visitor(child)
            walkDescendants(child, visitor: visitor)
        }
    }
    
    /// Finds the nearest ancestor (or self) that owns a PhysicsSimulationComponent.
    private static func findSimulationOwner(from entity: Entity) -> Entity? {
        var current: Entity? = entity
        while let e = current {
            if e.components[PhysicsSimulationComponent.self] != nil {
                return e
            }
            current = e.parent
        }
        return nil
    }
    
    /// Returns true if `entity` is the same as `ancestor` or a descendant of it.
    private static func isDescendantOf(entity: Entity, ancestor: Entity?) -> Bool {
        guard let ancestor = ancestor else { return false }
        var current: Entity? = entity
        while let e = current {
            if e === ancestor { return true }
            current = e.parent
        }
        return false
    }
}
