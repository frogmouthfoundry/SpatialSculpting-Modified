/*
Abstract:
Model of the haptics of the scene.
*/

import SwiftUI
import RealityKit
import GameController
import CoreHaptics

@MainActor @Observable
final class HapticsModel {

    @ObservationIgnored var hapticEngine: CHHapticEngine? = nil

    // Idle drill vibration
    @ObservationIgnored var idlePattern: CHHapticPattern? = nil
    @ObservationIgnored var idlePlayer: CHHapticPatternPlayer? = nil
    @ObservationIgnored var isIdlePlaying: Bool = false

    // Intense sculpting vibration
    @ObservationIgnored var sculptPattern: CHHapticPattern? = nil
    @ObservationIgnored var sculptPlayer: CHHapticPatternPlayer? = nil
    @ObservationIgnored var isSculptPlaying: Bool = false

    // Shaft collision warning pulse
    @ObservationIgnored var shaftCollisionPattern: CHHapticPattern? = nil
    @ObservationIgnored var shaftCollisionPlayer: CHHapticPatternPlayer? = nil
    @ObservationIgnored var isShaftCollisionPlaying: Bool = false

    func ensureEngineRunning() {
        try? hapticEngine?.start()
    }
}
