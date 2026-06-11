/*
Abstract:
Playback controls for looping drill idle and carving-contact audio.
*/

import Foundation
import RealityKit

extension DrillAudioModel {

    func prepareAudioIfNeeded() {
        guard !didAttemptPrepare else { return }
        didAttemptPrepare = true

        trackingResource = makeLoopingResource(named: "drill_base.mp3")
        contactResource = makeLoopingResource(named: "drill_contact.mp3")
        alarmResource = makeOneShotResource(named: "alarm.mp3")
        bloodResource = makeOneShotResource(named: "blood.mp3")
    }

    func setTrackingActive(_ isActive: Bool) {
        prepareAudioIfNeeded()
        wantsTrackingAudio = isActive

        if isActive {
            startTrackingLoopIfPossible()
        } else {
            stopTrackingLoop()
            setCarvingActive(false)
        }
    }

    func refreshPlaybackAfterSceneWork(delayMilliseconds: Int = 150) {
        pendingPlaybackRefreshTask?.cancel()
        pendingPlaybackRefreshTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(delayMilliseconds))
            guard !Task.isCancelled else { return }

            if self.wantsTrackingAudio {
                self.stopTrackingLoop()
                self.startTrackingLoopIfPossible()
            }

            if self.wantsContactAudio {
                self.stopContactLoop(fadeDuration: 0)
                self.startContactLoopIfPossible()
            }

            self.pendingPlaybackRefreshTask = nil
        }
    }

    func setCarvingActive(_ isActive: Bool) {
        prepareAudioIfNeeded()
        wantsContactAudio = isActive
        guard wantsTrackingAudio else {
            stopContactLoop(fadeDuration: 0)
            return
        }

        if isActive {
            startContactLoopIfPossible()
        } else {
            stopContactLoop(fadeDuration: 0.1)
        }
    }

    func attachSpatialAudio(to drillOverlay: Entity,
                            tipLocalPosition: SIMD3<Float>) {
        prepareAudioIfNeeded()
        if trackingEmitterEntity?.parent !== drillOverlay {
            trackingEmitterEntity?.removeFromParent()
            trackingEmitterEntity = nil
        }
        if contactEmitterEntity?.parent !== drillOverlay {
            contactEmitterEntity?.removeFromParent()
            contactEmitterEntity = nil
        }
        if trackingEmitterEntity == nil {
            let trackingEmitter = Entity()
            trackingEmitter.name = "DrillTrackingAudioEmitter"
            trackingEmitter.position = tipLocalPosition
            trackingEmitter.spatialAudio = SpatialAudioComponent(gain: -4)
            drillOverlay.addChild(trackingEmitter)
            trackingEmitterEntity = trackingEmitter
            print("[DrillAudio] Attached tracking spatial audio emitter")
        }

        if contactEmitterEntity == nil {
            let contactEmitter = Entity()
            contactEmitter.name = "DrillContactAudioEmitter"
            contactEmitter.position = tipLocalPosition
            contactEmitter.spatialAudio = SpatialAudioComponent(gain: 0)
            drillOverlay.addChild(contactEmitter)
            contactEmitterEntity = contactEmitter
            print("[DrillAudio] Attached contact spatial audio emitter")
        }

        if wantsTrackingAudio {
            startTrackingLoopIfPossible()
        }
        if wantsContactAudio {
            startContactLoopIfPossible()
        }
    }

    private func makeLoopingResource(named resourceName: String) -> AudioFileResource? {
        do {
            guard let resourceURL = Bundle.main.url(forResource: resourceName, withExtension: nil) else {
                print("[DrillAudio] Missing bundle resource \(resourceName)")
                return nil
            }
            let configuration = AudioFileResource.Configuration(shouldLoop: true)
            let resource = try AudioFileResource.load(contentsOf: resourceURL,
                                                      withName: resourceName,
                                                      configuration: configuration)
            print("[DrillAudio] Loaded spatial audio resource \(resourceName) from \(resourceURL.lastPathComponent)")
            return resource
        } catch {
            print("[DrillAudio] Failed to load spatial audio resource \(resourceName): \(error)")
            return nil
        }
    }

    private func makeOneShotResource(named resourceName: String) -> AudioFileResource? {
        do {
            guard let resourceURL = Bundle.main.url(forResource: resourceName, withExtension: nil) else {
                print("[DrillAudio] Missing bundle resource \(resourceName)")
                return nil
            }
            let configuration = AudioFileResource.Configuration(shouldLoop: false)
            let resource = try AudioFileResource.load(contentsOf: resourceURL,
                                                      withName: resourceName,
                                                      configuration: configuration)
            print("[DrillAudio] Loaded one-shot audio resource \(resourceName) from \(resourceURL.lastPathComponent)")
            return resource
        } catch {
            print("[DrillAudio] Failed to load one-shot audio resource \(resourceName): \(error)")
            return nil
        }
    }

    private func startTrackingLoopIfPossible() {
        guard let trackingEmitterEntity,
              let trackingResource else {
            print("[DrillAudio] Tracking audio requested before emitter/resource attach")
            return
        }
        if trackingController?.isPlaying == true { return }
        trackingController?.stop()
        let controller = trackingEmitterEntity.playAudio(trackingResource)
        controller.gain = 0
        trackingController = controller
        print("[DrillAudio] Started tracking spatial loop")
    }

    private func stopTrackingLoop() {
        trackingController?.stop()
        trackingController = nil
    }

    private func startContactLoopIfPossible() {
        pendingContactStopTask?.cancel()
        pendingContactStopTask = nil

        guard let contactEmitterEntity,
              let contactResource else {
            print("[DrillAudio] Contact audio requested before emitter/resource attach")
            return
        }
        if contactController?.isPlaying == true { return }
        contactController?.stop()
        let controller = contactEmitterEntity.playAudio(contactResource)
        controller.gain = 2
        contactController = controller
        print("[DrillAudio] Started contact spatial loop")
    }

    private func stopContactLoop(fadeDuration: TimeInterval) {
        pendingContactStopTask?.cancel()
        pendingContactStopTask = nil

        guard let contactController else { return }
        guard contactController.isPlaying else {
            return
        }

        if fadeDuration <= 0 {
            contactController.stop()
            self.contactController = nil
            return
        }

        contactController.fade(to: -60, duration: fadeDuration)
        pendingContactStopTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(Int(fadeDuration * 1000)))
            guard !self.wantsContactAudio else { return }
            contactController.stop()
            self.contactController = nil
            self.pendingContactStopTask = nil
        }
    }
}
