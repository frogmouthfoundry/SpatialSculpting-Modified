# Cochlear

`Cochlear` is a visionOS spatial drilling and sculpting app built on top of the spatial accessory sample. It turns a tracked stylus into a virtual drill for carving a volumetric bone model, generating debris, simulating slurry and water, and reacting to anatomy hazards.

## What the app does

- Shows a flat onboarding flow: home card, then a 3-card tutorial.
- Opens a volumetric surgical scene after onboarding.
- Tracks a Logitech Muse / supported spatial stylus as a drill overlay, drill bit, and sculpting radius.
- Loads a bundled sculpt package, rotates it into presentation orientation, and scales it for the scene.
- Carves the sculpt volume using an SDF-backed marching-cubes mesh.
- Spawns rigid bone debris, a separate bone slurry mesh, dust particles, drill audio, and haptics while carving.
- Simulates a thin water layer using a low-level animated wave mesh whose depth is driven by a virtual probe.
- Detects shaft collisions and anatomy hazard proximity for the facial nerve and sigmoid sinus.

## Core architecture

### Scene structure

The volumetric scene is hosted in [`ContentView.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ContentView.swift). It uses several scene roots to keep systems organized:

- `root`: sculpt volume, drill tool entity, slurry entity, and shared physics/simulation.
- `staticSceneRoot`: static anatomy/staging content.
- `interactiveAnatomyRoot`: hazard models and hazard effects.
- `fluidSceneRoot`: animated water layer.
- `accessorySceneRoot`: tracked stylus anchor content.
- `spatialToolbarPanelRoot`: floating tool menu driven by stylus input.

### Sculpting pipeline

- [`VoxelVolume`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/Volume/VoxelVolume.swift) defines the sculpt volume.
- [`MarchingCubesMesh`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/Mesh/MarchingCubesMesh.swift) owns the chunked low-level mesh.
- [`MarchingCubesMeshSculptor`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/Sculpting/MarchingCubesMeshSculptor.swift) updates the SDF and triggers marching-cubes extraction on the GPU.
- [`SculptingToolComponent`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ECS/SculptingToolComponent.swift) and `SculptingToolSystem` apply carving each frame using the tool pose.

Current sculpt volume assumptions:

- main sculpt grid: `128 x 128 x 128`
- bundled load path: `MyVolume.sculptpkg` with fallback support
- presentation transform after load: `270 deg` around Y, then `75%` scale

### Stylus / drill tracking

[`SculptingToolModel.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/SculptingToolModel.swift) is the main runtime coordinator.

Key responsibilities:

- receives stylus tracking state
- computes drill tip pose from the live drill-ball transform
- updates carving state from sampled SDF values
- drives audio, haptics, dust, slurry, and collision refresh
- manages shaft collision warning state and warning drill snapshot behavior

Related files:

- [`SculptingToolModel+GameController.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/SculptingToolModel+GameController.swift)
- [`SculptingToolModel+Anchoring.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/SculptingToolModel+Anchoring.swift)
- [`ShaftCollisionDetector.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/ShaftCollisionDetector.swift)

### Debris, slurry, and collision

- [`BoneDebrisManager.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/BoneDebris/BoneDebrisManager.swift): rigid debris spawning, pooling, settling, adhesion, dust timing.
- [`BoneSlurryGrid.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/BoneDebris/BoneSlurryGrid.swift): separate marching-cubes slurry volume.
- [`CollisionManager.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/BoneDebris/CollisionManager.swift): CPU-readable SDF cache for carving contact, shaft collision, safezones, and probe support.

Current slurry assumptions:

- separate lower-resolution slurry volume
- local mesh extraction and throttled updates for performance
- debug visualization can be toggled in the UI

### Water system

- [`AnimatedWaveMesh.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/Fluid/AnimatedWaveMesh.swift): low-level mesh water sheet with compute-driven ripple deformation.
- [`VirtualWaterProbeController.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/Fluid/VirtualWaterProbeController.swift): virtual support probe that settles along the collision field and determines water depth.

Current behavior:

- water starts hidden
- becomes visible after a debris threshold is reached
- resets when debris is cleared or the volume is reloaded
- uses the normal map asset `water 0397cbormal`

### Hazards, audio, and haptics

- [`SculptingToolModel+AnatomyHazards.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/SculptingToolModel+AnatomyHazards.swift): facial nerve and sigmoid reactions
- [`DrillAudioModel.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/DrillAudioModel.swift) plus playback/hazard extensions: drill base loop, contact loop, alarm, blood audio
- [`HapticsModel.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/HapticsModel.swift): carving hum and warning haptics

## App flow

- [`CochlearApp.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/CochlearApp.swift): app entry point, onboarding window, volumetric content window
- [`AppFlowModel.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/AppFlow/AppFlowModel.swift): onboarding state, asset readiness, content preparation
- [`LaunchExperienceView.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/AppFlow/LaunchExperienceView.swift): home card and tutorial cards

Content preparation begins during the home screen so the first tutorial card does not have to pay the full sculpt-scene startup cost.

## Good entry points for a new developer

If you are new to the project, start here:

1. [`CochlearApp.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/CochlearApp.swift)
2. [`ContentView.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ContentView.swift)
3. [`SculptingToolModel.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/ViewModel/SculptingToolModel.swift)
4. [`MarchingCubesMeshSculptor.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/Sculpting/MarchingCubesMeshSculptor.swift)
5. [`BoneDebrisManager.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/BoneDebris/BoneDebrisManager.swift)
6. [`AnimatedWaveMesh.swift`](/Users/derrickhsu/Documents/Codex/SpatialSculpt/SpatialSculpting/Cochlear/Fluid/AnimatedWaveMesh.swift)

## Notes

- The app has evolved well beyond the original sample. Some sample-era names remain, but the active target and product are now `Cochlear`.
- Most tuning values are centralized in `ContentView.swift`, `AppFlowModel.swift`, `SculptingToolModel.swift`, and the fluid/debris controllers.
