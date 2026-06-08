/*
Abstract:
Model of looping drill audio playback for tracked idle and carving contact states.
*/

import Observation
import RealityKit

@MainActor @Observable
final class DrillAudioModel {

    var didAttemptPrepare: Bool = false

    var trackingResource: AudioFileResource? = nil
    var contactResource: AudioFileResource? = nil
    var alarmResource: AudioFileResource? = nil
    var trackingController: AudioPlaybackController? = nil
    var contactController: AudioPlaybackController? = nil
    var alarmController: AudioPlaybackController? = nil

    var trackingEmitterEntity: Entity? = nil
    var contactEmitterEntity: Entity? = nil
    var alarmEmitterEntity: Entity? = nil

    var wantsTrackingAudio: Bool = false
    var wantsContactAudio: Bool = false
    var pendingContactStopTask: Task<Void, Never>? = nil
    var pendingPlaybackRefreshTask: Task<Void, Never>? = nil
    var pendingAlarmStopTask: Task<Void, Never>? = nil
}
