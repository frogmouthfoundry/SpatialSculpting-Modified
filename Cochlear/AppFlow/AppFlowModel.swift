import SwiftUI

#if canImport(UIKit)
import UIKit
#endif

enum AppWindowID {
    static let onboarding = "OnboardingWindow"
    static let contentExperience = "ContentExperienceWindow"
}

struct HomeCardLayerConfiguration {
    let assetName: String
    let xOffsetCM: CGFloat
    let yOffsetCM: CGFloat
}

struct TutorialCardConfiguration {
    let title: String
    let text: String
    let frameAssetNames: [String]
}

enum LaunchExperienceLayout {
    // These are intentionally centralized so text/position changes stay quick.
    static let onboardingScale: CGFloat = 0.49
    static let pointsPerCentimeter: CGFloat = 26 * onboardingScale
    static let onboardingWindowWidth: CGFloat = 1120 * onboardingScale
    static let onboardingWindowHeight: CGFloat = 1180 * onboardingScale
    static let homeLayerDepthSpacingCM: CGFloat = 1
    static let homeStartGapBelowLastLayerCM: CGFloat = 3
    static let homeTitleStackTopInset: CGFloat = onboardingWindowHeight * 0.4
    static let homeCardWidth: CGFloat = 980 * onboardingScale
    static let homeCardHeight: CGFloat = 880 * onboardingScale
    static let homeIllustrationWidth: CGFloat = homeCardWidth
    static let homeTitleWidth: CGFloat = onboardingWindowWidth * 0.6
    static let tutorialCardWidth: CGFloat = 980 * onboardingScale
    static let tutorialCardHeight: CGFloat = 920 * onboardingScale
    static let tutorialImageWidth: CGFloat = 860 * onboardingScale
    static let tutorialFrameRate: Double = 2

    static func cm(_ value: CGFloat) -> CGFloat {
        value * pointsPerCentimeter
    }

    static func scaled(_ value: CGFloat) -> CGFloat {
        value * onboardingScale
    }
}

enum LaunchExperienceContent {
    static let homeLayers: [HomeCardLayerConfiguration] = [
        .init(assetName: "Homecard_-_1-background", xOffsetCM: 0, yOffsetCM: 0),
        .init(assetName: "Homecard_-_2-bone-mesh", xOffsetCM: 0, yOffsetCM: 3),
        .init(assetName: "Homecard_-_3-hand-tool", xOffsetCM: 5, yOffsetCM: 5),
        .init(assetName: "Homecard_-_4-title-Cochlear", xOffsetCM: 0, yOffsetCM: 15),
        .init(assetName: "Homecard_-_5-title-virtualsurg", xOffsetCM: 0, yOffsetCM: 10)
    ]

    static let tutorialCards: [TutorialCardConfiguration] = [
        .init(
            title: "Drilling bone",
            text: "Hold Stylus in hand. Move the stylus near the bone volume to drill",
            frameAssetNames: [
                "Card1-1-hand-tool1",
                "Card1-2-hand-tool2",
                "Card1-3-hand-tool3",
                "Card1-4-hand-tool4"
            ]
        ),
        .init(
            title: "Clearing bone debris and fluid",
            text: "Press the top stylus button to clear the bone debris and fluid.",
            frameAssetNames: [
                "Card2-1",
                "Card2-2-pressed",
                "Card2-3-semitransp",
                "Card2-4-disappear"
            ]
        ),
        .init(
            title: "Opening menu",
            text: "Press the bottom stylus button to open the menu, which includes Drill Bit Size, Zoom, Camera Rotation, and General",
            frameAssetNames: [
                "Card3-1",
                "Card3-2-pressed",
                "Card3-3-menu1",
                "Card3-4-menu2"
            ]
        )
    ]

    static var allOnboardingAssetNames: [String] {
        homeLayers.map(\.assetName) + tutorialCards.flatMap(\.frameAssetNames)
    }
}

enum OnboardingAssetLoadState: Equatable {
    case idle
    case loading
    case ready
    case failed(String)
}

enum ContentPreparationState: Equatable {
    case idle
    case loading
    case ready
    case failed(String)
}

enum LaunchPhase: Equatable {
    case home
    case tutorial(index: Int)
}

@MainActor
final class PreparedSculptExperience {
    let marchingCubesMesh: MarchingCubesMesh
    let sculptor: MarchingCubesMeshSculptor

    init(marchingCubesMesh: MarchingCubesMesh, sculptor: MarchingCubesMeshSculptor) {
        self.marchingCubesMesh = marchingCubesMesh
        self.sculptor = sculptor
    }

    static func build() throws -> PreparedSculptExperience {
        let config = initialVolumeConfig()

        let voxelVolume = try VoxelVolume(dimensions: config.dimensions,
                                          voxelSize: config.voxelSize,
                                          voxelStartPosition: config.voxelStart,
                                          textureBoundsMin: config.textureBoundsMin,
                                          textureBoundsMax: config.textureBoundsMax)
        let marchingCubesMesh = try MarchingCubesMesh(voxelVolume: voxelVolume)
        let sculptor = MarchingCubesMeshSculptor(marchingCubesMesh: marchingCubesMesh)
        return PreparedSculptExperience(marchingCubesMesh: marchingCubesMesh,
                                        sculptor: sculptor)
    }

    private static func initialVolumeConfig() -> (dimensions: SIMD3<UInt32>,
                                                  voxelSize: SIMD3<Float>,
                                                  voxelStart: SIMD3<Float>,
                                                  textureBoundsMin: SIMD3<Float>,
                                                  textureBoundsMax: SIMD3<Float>) {
        let fallbackDimensions = SIMD3<UInt32>(128, 128, 128)
        let fallbackExtent = SIMD3<Float>(0.8, 0.8, 0.8)
        let fallbackVoxelSize = fallbackExtent / SIMD3<Float>(fallbackDimensions)
        let fallbackStart = -SIMD3<Float>(fallbackDimensions) * fallbackVoxelSize / 2
        let fallbackBoundsMin = fallbackStart - fallbackVoxelSize * 0.5
        let fallbackBoundsMax = fallbackStart + fallbackVoxelSize * (SIMD3<Float>(fallbackDimensions) - 0.5)
        let targetVolumeSize: Float = 0.8

        guard let url = Bundle.main.url(forResource: "MyVolume", withExtension: "sculptpkg") else {
            print("Initial volume config: using fallback (missing MyVolume.sculptpkg).")
            return (fallbackDimensions, fallbackVoxelSize, fallbackStart, fallbackBoundsMin, fallbackBoundsMax)
        }

        do {
            let manifest = try VolumeDocument.parsePackageManifest(at: url)
            let dims = SIMD3<UInt32>(UInt32(manifest.grid.width),
                                     UInt32(manifest.grid.height),
                                     UInt32(manifest.grid.depth))
            guard manifest.bounds.min.count == 3, manifest.bounds.max.count == 3 else {
                print("Initial volume config: manifest bounds invalid count; using fallback bounds.")
                let voxelSize = fallbackExtent / SIMD3<Float>(dims)
                let voxelStart = -SIMD3<Float>(dims) * voxelSize / 2
                let boundsMin = voxelStart - voxelSize * 0.5
                let boundsMax = voxelStart + voxelSize * (SIMD3<Float>(dims) - 0.5)
                return (dims, voxelSize, voxelStart, boundsMin, boundsMax)
            }

            let bmin = SIMD3<Float>(manifest.bounds.min[0], manifest.bounds.min[1], manifest.bounds.min[2])
            let bmax = SIMD3<Float>(manifest.bounds.max[0], manifest.bounds.max[1], manifest.bounds.max[2])
            let sourceExtent = simd_max(bmax - bmin, SIMD3<Float>(repeating: 1e-5))
            let maxExtent = max(sourceExtent.x, max(sourceExtent.y, sourceExtent.z))
            let uniformScale = targetVolumeSize / max(maxExtent, 1e-5)
            let fittedExtent = sourceExtent * uniformScale
            let fittedMin = -fittedExtent * 0.5
            let voxelSize = fittedExtent / SIMD3<Float>(dims)
            let voxelStart = fittedMin + voxelSize * 0.5
            return (dims, voxelSize, voxelStart, bmin, bmax)
        } catch {
            print("Initial volume config: manifest parse failed, using fallback. Error: \(error)")
            return (fallbackDimensions, fallbackVoxelSize, fallbackStart, fallbackBoundsMin, fallbackBoundsMax)
        }
    }
}

@MainActor
@Observable
final class AppFlowModel {
    private(set) var onboardingAssetLoadState: OnboardingAssetLoadState = .idle
    private(set) var contentPreparationState: ContentPreparationState = .idle
    private(set) var launchPhase: LaunchPhase = .home
    private(set) var preparedExperience: PreparedSculptExperience? = nil
    private(set) var isContentExperienceArmed: Bool = false

    @ObservationIgnored private var onboardingAssetsTask: Task<Void, Never>? = nil
    @ObservationIgnored private var contentPreparationTask: Task<Void, Never>? = nil

    var isHomeStartAvailable: Bool {
        onboardingAssetLoadState == .ready && contentPreparationState == .ready
    }

    var tutorialIndex: Int? {
        guard case let .tutorial(index) = launchPhase else { return nil }
        return index
    }

    var isFinalTutorialCard: Bool {
        tutorialIndex == LaunchExperienceContent.tutorialCards.count - 1
    }

    var currentTutorialCard: TutorialCardConfiguration? {
        guard let tutorialIndex else { return nil }
        return LaunchExperienceContent.tutorialCards[tutorialIndex]
    }

    func prepareOnboardingAssetsIfNeeded() {
        guard onboardingAssetsTask == nil,
              onboardingAssetLoadState != .ready else { return }

        onboardingAssetLoadState = .loading
        onboardingAssetsTask = Task { @MainActor [weak self] in
            guard let self else { return }
            #if canImport(UIKit)
            let missingAssets = LaunchExperienceContent.allOnboardingAssetNames.filter {
                UIImage(named: $0) == nil
            }
            if missingAssets.isEmpty {
                onboardingAssetLoadState = .ready
            } else {
                onboardingAssetLoadState = .failed("Missing assets: \(missingAssets.joined(separator: ", "))")
            }
            #else
            onboardingAssetLoadState = .ready
            #endif
            onboardingAssetsTask = nil
        }
    }

    func beginTutorial() {
        guard isHomeStartAvailable else { return }
        launchPhase = .tutorial(index: 0)
    }

    func advanceTutorial() {
        guard let tutorialIndex else { return }
        let nextIndex = tutorialIndex + 1
        if nextIndex < LaunchExperienceContent.tutorialCards.count {
            launchPhase = .tutorial(index: nextIndex)
        }
    }

    func prepareContentExperienceIfNeeded() {
        guard contentPreparationTask == nil,
              contentPreparationState != .ready else { return }

        contentPreparationState = .loading
        contentPreparationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                preparedExperience = try PreparedSculptExperience.build()
                contentPreparationState = .ready
            } catch {
                preparedExperience = nil
                contentPreparationState = .failed(error.localizedDescription)
            }
            contentPreparationTask = nil
        }
    }

    func armContentExperience() {
        isContentExperienceArmed = true
    }
}
