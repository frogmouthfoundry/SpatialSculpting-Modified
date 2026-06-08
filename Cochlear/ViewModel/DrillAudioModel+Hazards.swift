/*
Abstract:
Hazard-specific drill audio for anatomy warning states.
*/

import Foundation
import RealityKit

extension DrillAudioModel {

    func attachHazardAudio(to sceneRoot: Entity) {
        prepareAudioIfNeeded()
        guard alarmEmitterEntity == nil else { return }

        let alarmEmitter = Entity()
        alarmEmitter.name = "DrillAlarmAudioEmitter"
        alarmEmitter.position = .zero
        alarmEmitter.spatialAudio = SpatialAudioComponent(gain: 10)
        sceneRoot.addChild(alarmEmitter)
        alarmEmitterEntity = alarmEmitter
        print("[DrillAudio] Attached hazard audio emitter")
    }

    func triggerAlarm(duration: TimeInterval = 1.0) {
        pendingAlarmStopTask?.cancel()
        pendingAlarmStopTask = nil

        guard let alarmEmitterEntity,
              let alarmResource else {
            print("[DrillAudio] Alarm audio requested before emitter/resource attach")
            return
        }

        alarmController?.stop()
        let controller = alarmEmitterEntity.playAudio(alarmResource)
        controller.gain = 12
        alarmController = controller
        print("[DrillAudio] Triggered alarm audio")

        pendingAlarmStopTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(Int(duration * 1000)))
            controller.stop()
            if self.alarmController === controller {
                self.alarmController = nil
            }
            self.pendingAlarmStopTask = nil
        }
    }
}
