/*
Abstract:
Set up haptics for the scene. Provides idle drill vibration and intense sculpting vibration.
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

        // Idle vibration: gentle continuous hum simulating a spinning drill
        if idlePattern == nil {
            idlePattern = try? CHHapticPattern(events: [
                CHHapticEvent(eventType: .hapticContinuous, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.08),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.15)
                ], relativeTime: 0.0, duration: 100.0)
            ], parameterCurves: [])
        }

        // Sculpting vibration: stronger buzz when carving into the volume
        if sculptPattern == nil {
            sculptPattern = try? CHHapticPattern(events: [
                CHHapticEvent(eventType: .hapticContinuous, parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.45),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.6)
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

    /// Start the intense sculpting vibration (replaces idle while active).
    @MainActor
    func startSculptVibration() {
        guard !isSculptPlaying else { return }
        // Pause idle while sculpting for a cleaner feel
        try? idlePlayer?.stop(atTime: CHHapticTimeImmediate)
        try? sculptPlayer?.start(atTime: CHHapticTimeImmediate)
        isSculptPlaying = true
    }

    /// Stop the intense sculpting vibration and resume idle.
    @MainActor
    func stopSculptVibration() {
        guard isSculptPlaying else { return }
        try? sculptPlayer?.stop(atTime: CHHapticTimeImmediate)
        isSculptPlaying = false
        // Resume idle hum
        if isIdlePlaying {
            try? idlePlayer?.start(atTime: CHHapticTimeImmediate)
        }
    }

    /// Start the shaft collision warning pulse. Pauses sculpt/idle while active.
    @MainActor
    func startShaftCollisionFeedback() {
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
        // Resume sculpt or idle as appropriate
        if isSculptPlaying {
            try? sculptPlayer?.start(atTime: CHHapticTimeImmediate)
        } else if isIdlePlaying {
            try? idlePlayer?.start(atTime: CHHapticTimeImmediate)
        }
    }
}
