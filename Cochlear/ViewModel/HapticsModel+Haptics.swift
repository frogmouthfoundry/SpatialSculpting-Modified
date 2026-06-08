/*
Abstract:
Set up haptics for the scene. Provides carving hum and shaft-collision warning haptics.
*/

import CoreHaptics
import GameController

extension HapticsModel {
    // Set up haptics engine and patterns for playing haptics.
    @MainActor
    func setupHaptics(haptics: GCDeviceHaptics) {
        if hapticEngine == nil {
            hapticEngine = haptics.createEngine(withLocality: .default)
            try? hapticEngine?.start()
        }

        // Idle vibration remains defined for debugging/experimentation, but is
        // not used in the current behavior.
        if idlePattern == nil {
            idlePattern = try? CHHapticPattern(events: [
                CHHapticEvent(eventType: .hapticContinuous, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.08),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.15)
                ], relativeTime: 0.0, duration: 100.0)
            ], parameterCurves: [])
        }

        // Sculpting vibration: continuous low-frequency rumble, stronger than idle
        // but still much softer and less sharp than the collision warning.
        if sculptPattern == nil {
            sculptPattern = try? CHHapticPattern(events: [
                CHHapticEvent(eventType: .hapticContinuous, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.85),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.28)
                ], relativeTime: 0.0, duration: 100.0)
            ], parameterCurves: [])
        }

        // Shaft collision warning: rapid sharp pulses — distinct "danger" feel
        if shaftCollisionPattern == nil {
            shaftCollisionPattern = try? CHHapticPattern(events: [
                // Four rapid transient taps over 0.3s, looped via long duration
                CHHapticEvent(eventType: .hapticTransient, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.8),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.95)
                ], relativeTime: 0.0),
                CHHapticEvent(eventType: .hapticTransient, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.6),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.9)
                ], relativeTime: 0.08),
                CHHapticEvent(eventType: .hapticTransient, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.8),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.95)
                ], relativeTime: 0.16),
                CHHapticEvent(eventType: .hapticTransient, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.6),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.9)
                ], relativeTime: 0.24),
                // Sustained buzz between pulses for continuous warning feel
                CHHapticEvent(eventType: .hapticContinuous, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.35),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.7)
                ], relativeTime: 0.0, duration: 0.3)
            ], parameterCurves: [])
        }

        if let idlePattern = idlePattern {
            idlePlayer = try? hapticEngine?.makePlayer(with: idlePattern)
        }
        if let sculptPattern = sculptPattern {
            sculptPlayer = try? hapticEngine?.makePlayer(with: sculptPattern)
        }
        if let shaftCollisionPattern = shaftCollisionPattern {
            shaftCollisionPlayer = try? hapticEngine?.makePlayer(with: shaftCollisionPattern)
        }
    }

    /// Start the idle drill vibration. Call once when the accessory connects.
    @MainActor
    func startIdleVibration() {
        ensureEngineRunning()
        guard !isIdlePlaying else { return }
        try? idlePlayer?.start(atTime: CHHapticTimeImmediate)
        isIdlePlaying = true
    }

    /// Stop the idle drill vibration.
    @MainActor
    func stopIdleVibration() {
        guard isIdlePlaying else { return }
        try? idlePlayer?.stop(atTime: CHHapticTimeImmediate)
        isIdlePlaying = false
    }

    /// Start the low carving hum.
    @MainActor
    func startSculptVibration() {
        ensureEngineRunning()
        guard !isSculptPlaying else { return }
        try? idlePlayer?.stop(atTime: CHHapticTimeImmediate)
        try? sculptPlayer?.start(atTime: CHHapticTimeImmediate)
        isSculptPlaying = true
    }

    /// Stop the low carving hum immediately.
    @MainActor
    func stopSculptVibration() {
        guard isSculptPlaying else { return }
        try? sculptPlayer?.stop(atTime: CHHapticTimeImmediate)
        isSculptPlaying = false
    }

    /// Start the shaft collision warning pulse. Pauses sculpt/idle while active.
    @MainActor
    func startShaftCollisionFeedback() {
        ensureEngineRunning()
        guard !isShaftCollisionPlaying else { return }
        // Pause other patterns for clarity
        try? idlePlayer?.stop(atTime: CHHapticTimeImmediate)
        try? sculptPlayer?.stop(atTime: CHHapticTimeImmediate)
        try? shaftCollisionPlayer?.start(atTime: CHHapticTimeImmediate)
        isShaftCollisionPlaying = true
    }

    /// Stop the shaft collision warning pulse and resume previous state.
    @MainActor
    func stopShaftCollisionFeedback() {
        guard isShaftCollisionPlaying else { return }
        try? shaftCollisionPlayer?.stop(atTime: CHHapticTimeImmediate)
        isShaftCollisionPlaying = false
        // Resume carving hum only if carving is still active.
        if isSculptPlaying {
            try? sculptPlayer?.start(atTime: CHHapticTimeImmediate)
        }
    }
}
