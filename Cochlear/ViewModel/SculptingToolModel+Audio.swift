/*
Abstract:
Audio wiring helpers for tracked drill idle and carving-contact playback.
*/

import GameController
import ARKit

extension SculptingToolModel {

    func updateDrillTrackingAudio(for trackingState: AccessoryAnchor.TrackingState) {
        drillAudioModel?.setTrackingActive(isTrackingAudible(trackingState))
    }

    func updateDrillCarvingAudio(isCarving: Bool) {
        drillAudioModel?.setCarvingActive(isCarving)
    }

    private func isTrackingAudible(_ trackingState: AccessoryAnchor.TrackingState) -> Bool {
        trackingState != .untracked
    }
}
