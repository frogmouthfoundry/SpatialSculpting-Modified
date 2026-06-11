/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Handle connections and inputs with Game Controller framework.
*/

import ARKit
import RealityKit
@preconcurrency import GameController

extension SculptingToolModel {
    private func hasSupportedSpatialAccessoryPresent() -> Bool {
        let hasSpatialController = GCController.controllers().contains {
            $0.productCategory == GCProductCategorySpatialController
        }
        let hasSpatialStylus = GCStylus.styli.contains {
            $0.productCategory == GCProductCategorySpatialStylus
        }
        return hasSpatialController || hasSpatialStylus
    }

    private func scanForSupportedAccessories(hapticsModel: HapticsModel,
                                             logAsInitialScan: Bool) async {
        let controllers = GCController.controllers()
        let styluses = GCStylus.styli

        if logAsInitialScan {
            print("[Accessory] Initial scan controllers=\(controllers.count) styluses=\(styluses.count)")
        }

        if controllers.contains(where: { $0.productCategory == GCProductCategorySpatialController }) ||
            styluses.contains(where: { $0.productCategory == GCProductCategorySpatialStylus }) {
            ensureAccessoryTrackingSessionRunning()
        }

        for controller in controllers {
            if controller.productCategory != GCProductCategorySpatialController {
                continue
            }

            print("[Accessory] Initial controller candidate \(spatialAccessoryDebugDescription(for: controller))")
            try? await setupSpatialAccessory(device: controller, hapticsModel: hapticsModel)
        }

        for stylus in styluses {
            if stylus.productCategory != GCProductCategorySpatialStylus {
                continue
            }

            print("[Accessory] Initial stylus candidate \(spatialAccessoryDebugDescription(for: stylus))")
            try? await setupSpatialAccessory(device: stylus, hapticsModel: hapticsModel)
        }
    }

    func ensureAccessoryDiscoveryRescanRunning() {
        guard accessoryTrackingShouldBeRunning else { return }
        guard accessoryDiscoveryRescanTask == nil else { return }
        guard let hapticsModel else { return }

        accessoryDiscoveryRescanTask = Task { @MainActor [weak self] in
            guard let self else { return }

            for attempt in 1...6 {
                guard self.accessoryTrackingShouldBeRunning else { break }
                if self.activeSpatialAccessoryKind == .stylus, self.sculptingEntity != nil {
                    break
                }

                print("[Accessory] Rescan attempt \(attempt)")
                await self.scanForSupportedAccessories(hapticsModel: hapticsModel,
                                                       logAsInitialScan: false)

                if self.activeSpatialAccessoryKind == .stylus, self.sculptingEntity != nil {
                    break
                }

                try? await Task.sleep(for: .milliseconds(500))
            }

            self.accessoryDiscoveryRescanTask = nil
        }
    }

    func setAccessoryTrackingSceneActive(_ isActive: Bool) {
        accessoryTrackingShouldBeRunning = isActive

        if isActive {
            ensureAccessoryTrackingSessionRunning()
            ensureAccessoryDiscoveryRescanRunning()
        } else {
            accessoryTrackingRestartTask?.cancel()
            accessoryTrackingRestartTask = nil
            accessoryDiscoveryRescanTask?.cancel()
            accessoryDiscoveryRescanTask = nil
            accessoryTrackingSessionTask?.cancel()
            accessoryTrackingSessionTask = nil
            accessoryTrackingSession = nil
            invalidateAccessoryTrackingState()
            print("[AccessoryTracking] Scene inactive; accessory tracking paused")
        }
    }

    func ensureAccessoryTrackingSessionRunning() {
        guard accessoryTrackingShouldBeRunning else { return }
        guard accessoryTrackingSession == nil else { return }
        guard accessoryTrackingSessionTask == nil else { return }
        accessoryTrackingSessionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            let session = SpatialTrackingSession()
            let configuration = SpatialTrackingSession.Configuration(tracking: [.accessory])
            self.accessoryTrackingSession = session
            print("[AccessoryTracking] Starting accessory tracking session")
            await session.run(configuration)
            print("[AccessoryTracking] Accessory tracking session configured")
            self.accessoryTrackingSessionTask = nil
        }
    }

    
    // Set up button and haptic inputs for GCStylus.
    @MainActor
    func setupStylusInputs(stylus: GCStylus, hapticsModel: HapticsModel) {
        if let input = stylus.input {
            let buttonSidePrimary = input.buttons[.stylusPrimaryButton] // Primary side button
            buttonSidePrimary?.pressedInput.pressedDidChangeHandler = { _, _, pressed in
                self.handlePalettePress(pressed: pressed)
            }

            let buttonSideSecondary = input.buttons[.stylusSecondaryButton]
            buttonSideSecondary?.pressedInput.pressedDidChangeHandler = { _, _, pressed in
                guard pressed else { return }
                Task { @MainActor in
                    self.shouldClearDebris = true
                }
            }
        }
        if let haptics = stylus.haptics {
            hapticsModel.setupHaptics(haptics: haptics)
        }
    }
    
    // Set up button and haptic inputs for GCController.
    @MainActor
    func setupControllerInputs(controller: GCController, hapticsModel: HapticsModel) {
        let input = controller.input
        
        // Trigger removed - sculpting is now always active when tracked
        
        input.buttons[.thumbstickButton]?.pressedInput.pressedDidChangeHandler = { _, _, pressed in
            self.handlePalettePress(pressed: pressed)
        }
        
        if let haptics = controller.haptics {
            hapticsModel.setupHaptics(haptics: haptics)
        }
    }

    // Handle connections with GCControllers and GCStyluses.
    func handleGameControllerSetup(hapticsModel: HapticsModel) async {
        self.hapticsModel = hapticsModel
        await scanForSupportedAccessories(hapticsModel: hapticsModel,
                                          logAsInitialScan: true)
        ensureAccessoryDiscoveryRescanRunning()

        // Listen to notifications for connections of both controllers and styluses.
        guard !didRegisterAccessoryNotifications else { return }
        didRegisterAccessoryNotifications = true

        NotificationCenter.default.addObserver(forName: NSNotification.Name.GCControllerDidConnect, object: nil, queue: nil) {
            notification in
                if let controller = notification.object as? GCController,
                   controller.productCategory == GCProductCategorySpatialController {
                self.ensureAccessoryTrackingSessionRunning()
                print("[Accessory] Controller connected \(self.spatialAccessoryDebugDescription(for: controller))")
                Task { @MainActor in
                    try? await self.setupSpatialAccessory(device: controller, hapticsModel: hapticsModel)
                }
            }
        }
                
        NotificationCenter.default.addObserver(forName: NSNotification.Name.GCStylusDidConnect, object: nil, queue: nil) {
            notification in
            if let stylus = notification.object as? GCStylus,
               stylus.productCategory == GCProductCategorySpatialStylus {
                self.ensureAccessoryTrackingSessionRunning()
                print("[Accessory] Stylus connected \(self.spatialAccessoryDebugDescription(for: stylus))")
                Task { @MainActor in
                    try? await self.setupSpatialAccessory(device: stylus, hapticsModel: hapticsModel)
                }
            }
        }

        NotificationCenter.default.addObserver(forName: NSNotification.Name.GCControllerDidDisconnect, object: nil, queue: nil) {
            notification in
            if let controller = notification.object as? GCController {
                print("[Accessory] Controller disconnected \(self.spatialAccessoryDebugDescription(for: controller))")
                Task { @MainActor in
                    self.removeSpatialAccessoryAnchor(for: ObjectIdentifier(controller))
                }
            }
        }

        NotificationCenter.default.addObserver(forName: NSNotification.Name.GCStylusDidDisconnect, object: nil, queue: nil) {
            notification in
            if let stylus = notification.object as? GCStylus {
                print("[Accessory] Stylus disconnected \(self.spatialAccessoryDebugDescription(for: stylus))")
                Task { @MainActor in
                    self.removeSpatialAccessoryAnchor(for: ObjectIdentifier(stylus))
                }
            }
        }
    }
}
