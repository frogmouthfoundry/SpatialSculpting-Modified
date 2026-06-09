/*
Abstract:
Model of looping drill audio playback for tracked idle and carving contact states.
*/

import Observation
import RealityKit

@MainActor @Observable
final class DrillAudioModel {

    @ObservationIgnored var didAttemptPrepare: Bool = false

    @ObservationIgnored var trackingResource: AudioFileResource? = nil
    @ObservationIgnored var contactResource: AudioFileResource? = nil
    @ObservationIgnored var alarmResource: AudioFileResource? = nil
    @ObservationIgnored var bloodResource: AudioFileResource? = nil
    @ObservationIgnored var trackingController: AudioPlaybackController? = nil
    @ObservationIgnored var contactController: AudioPlaybackController? = nil
    @ObservationIgnored var alarmController: AudioPlaybackController? = nil
    @ObservationIgnored var bloodController: AudioPlaybackController? = nil

    @ObservationIgnored var trackingEmitterEntity: Entity? = nil
    @ObservationIgnored var contactEmitterEntity: Entity? = nil
    @ObservationIgnored var alarmEmitterEntity: Entity? = nil

    @ObservationIgnored var wantsTrackingAudio: Bool = false
    @ObservationIgnored var wantsContactAudio: Bool = false
    @ObservationIgnored var pendingContactStopTask: Task<Void, Never>? = nil
    @ObservationIgnored var pendingPlaybackRefreshTask: Task<Void, Never>? = nil
    @ObservationIgnored var pendingAlarmStopTask: Task<Void, Never>? = nil
    @ObservationIgnored var pendingBloodStopTask: Task<Void, Never>? = nil
}
