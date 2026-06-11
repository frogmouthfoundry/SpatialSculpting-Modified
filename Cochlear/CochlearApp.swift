/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The app's entry point.
*/

import SwiftUI

@main
struct CochlearApp: App {
    @State private var appModel = AppFlowModel()

    init() {
        ComputeDispatchSystem.registerSystem()
        DrillRotationSystem.registerSystem()
    }
    
    var body: some Scene {
        WindowGroup(id: AppWindowID.onboarding) {
            LaunchExperienceView(appModel: appModel)
        }
        .windowResizability(.contentSize)
        .defaultLaunchBehavior(.presented)

        WindowGroup(id: AppWindowID.contentExperience) {
            if let preparedExperience = appModel.preparedExperience {
                ContentView(experience: preparedExperience)
                    .frame(width: 1500, height: 1500)
                    .frame(depth: 1500)
            } else {
                ProgressView("Preparing")
                    .frame(width: 420, height: 220)
            }
        }
        .windowStyle(.volumetric)
        .defaultLaunchBehavior(.suppressed)
    }
}
