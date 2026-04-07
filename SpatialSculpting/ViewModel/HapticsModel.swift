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

    var hapticEngine: CHHapticEngine? = nil

    // Idle drill vibration
    var idlePattern: CHHapticPattern? = nil
    var idlePlayer: CHHapticPatternPlayer? = nil
    var isIdlePlaying: Bool = false

    // Intense sculpting vibration
    var sculptPattern: CHHapticPattern? = nil
    var sculptPlayer: CHHapticPatternPlayer? = nil
    var isSculptPlaying: Bool = false

    // Shaft collision warning pulse
    var shaftCollisionPattern: CHHapticPattern? = nil
    var shaftCollisionPlayer: CHHapticPatternPlayer? = nil
    var isShaftCollisionPlaying: Bool = false
}
