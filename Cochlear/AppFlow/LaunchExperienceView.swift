import SwiftUI

struct LaunchExperienceView: View {
    var appModel: AppFlowModel

    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow

    @State private var tutorialAnimationStartDate: Date = .now

    var body: some View {
        ZStack {
            Color.clear
                .ignoresSafeArea()

            switch appModel.launchPhase {
            case .home:
                homeCard
            case .tutorial(let index):
                tutorialCard(index: index)
            }
        }
        .frame(width: LaunchExperienceLayout.scaled(1120),
               height: LaunchExperienceLayout.scaled(1180))
        .task {
            appModel.prepareOnboardingAssetsIfNeeded()
        }
        .onChange(of: appModel.tutorialIndex) { _, _ in
            tutorialAnimationStartDate = .now
        }
    }

    private var homeCard: some View {
        VStack(spacing: LaunchExperienceLayout.scaled(28)) {
            Spacer(minLength: 0)

            ZStack {
                ForEach(Array(LaunchExperienceContent.homeLayers.enumerated()), id: \.offset) { index, layer in
                    Image(layer.assetName)
                        .resizable()
                        .scaledToFit()
                        .frame(width: LaunchExperienceLayout.homeIllustrationWidth)
                        .offset(x: LaunchExperienceLayout.cm(layer.xOffsetCM),
                                y: LaunchExperienceLayout.cm(layer.yOffsetCM))
                        .offset(z: LaunchExperienceLayout.cm(CGFloat(index) * LaunchExperienceLayout.homeLayerDepthSpacingCM))
                }
            }
            .frame(width: LaunchExperienceLayout.homeCardWidth,
                   height: LaunchExperienceLayout.homeCardHeight)
            .allowsHitTesting(false)

            VStack(spacing: LaunchExperienceLayout.scaled(12)) {
                switch appModel.onboardingAssetLoadState {
                case .idle, .loading:
                    ProgressView()
                        .progressViewStyle(.circular)
                    Text("Loading")
                        .font(.system(size: LaunchExperienceLayout.scaled(24),
                                      weight: .semibold,
                                      design: .rounded))
                        .foregroundStyle(.secondary)
                case .ready:
                    Button {
                        appModel.beginTutorial()
                    } label: {
                        Text("Start")
                            .font(.system(size: LaunchExperienceLayout.scaled(24),
                                          weight: .semibold,
                                          design: .rounded))
                            .frame(minWidth: LaunchExperienceLayout.scaled(180))
                            .padding(.vertical, LaunchExperienceLayout.scaled(6))
                    }
                    .buttonStyle(.bordered)
                    .contentShape(Rectangle())
                case .failed(let message):
                    Text(message)
                        .font(.system(size: LaunchExperienceLayout.scaled(18),
                                      weight: .regular,
                                      design: .rounded))
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.red)
                }
            }
            .frame(minHeight: LaunchExperienceLayout.scaled(120))
            .padding(.top, LaunchExperienceLayout.cm(LaunchExperienceLayout.homeStartGapBelowLastLayerCM))

            Spacer(minLength: 0)
        }
        .padding(LaunchExperienceLayout.scaled(40))
    }

    private func tutorialCard(index: Int) -> some View {
        let card = LaunchExperienceContent.tutorialCards[index]
        let isFinalCard = index == LaunchExperienceContent.tutorialCards.count - 1

        return VStack(spacing: LaunchExperienceLayout.scaled(28)) {
            Spacer(minLength: 0)

            VStack(spacing: LaunchExperienceLayout.scaled(24)) {
                loopingTutorialImage(frameAssetNames: card.frameAssetNames)

                VStack(spacing: LaunchExperienceLayout.scaled(12)) {
                    Text(card.title)
                        .font(.system(size: LaunchExperienceLayout.scaled(40), weight: .bold, design: .rounded))
                        .frame(maxWidth: .infinity, alignment: .center)
                    Text(card.text)
                        .font(.system(size: LaunchExperienceLayout.scaled(24), weight: .regular, design: .rounded))
                        .foregroundStyle(.secondary)
                        .lineSpacing(LaunchExperienceLayout.scaled(6))
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
                .frame(width: LaunchExperienceLayout.tutorialImageWidth)

                VStack(spacing: LaunchExperienceLayout.scaled(18)) {
                    dotIndicator(currentIndex: index)
                        .frame(width: LaunchExperienceLayout.tutorialImageWidth, alignment: .center)

                    VStack(alignment: .trailing, spacing: LaunchExperienceLayout.scaled(10)) {
                        if isFinalCard && appModel.contentPreparationState == .loading {
                            HStack(spacing: LaunchExperienceLayout.scaled(8)) {
                                ProgressView()
                                    .progressViewStyle(.circular)
                                Text("Loading")
                                    .font(.system(size: LaunchExperienceLayout.scaled(18),
                                                  weight: .regular,
                                                  design: .rounded))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        if isFinalCard,
                           case let .failed(message) = appModel.contentPreparationState {
                            Text(message)
                                .font(.system(size: LaunchExperienceLayout.scaled(16),
                                              weight: .regular,
                                              design: .rounded))
                                .foregroundStyle(.red)
                                .multilineTextAlignment(.trailing)
                                .frame(maxWidth: LaunchExperienceLayout.scaled(260))
                        }

                        Button {
                            if isFinalCard {
                                openWindow(id: AppWindowID.contentExperience)
                                Task { @MainActor in
                                    try? await Task.sleep(for: .milliseconds(150))
                                    dismissWindow(id: AppWindowID.onboarding)
                                }
                            } else {
                                appModel.advanceTutorial()
                            }
                        } label: {
                            Text(isFinalCard ? "Start" : "Next")
                                .font(.system(size: LaunchExperienceLayout.scaled(24),
                                              weight: .semibold,
                                              design: .rounded))
                                .frame(minWidth: LaunchExperienceLayout.scaled(160))
                        }
                        .buttonStyle(.bordered)
                        .disabled(isFinalCard && appModel.contentPreparationState != .ready)
                    }
                    .frame(width: LaunchExperienceLayout.tutorialImageWidth, alignment: .trailing)
                }
            }
            .padding(LaunchExperienceLayout.scaled(42))
            .frame(width: LaunchExperienceLayout.tutorialCardWidth,
                   height: LaunchExperienceLayout.tutorialCardHeight,
                   alignment: .topLeading)

            Spacer(minLength: 0)
        }
        .padding(LaunchExperienceLayout.scaled(40))
    }

    private func loopingTutorialImage(frameAssetNames: [String]) -> some View {
        TimelineView(.periodic(from: tutorialAnimationStartDate,
                               by: 1.0 / LaunchExperienceLayout.tutorialFrameRate)) { context in
            let frameCount = max(frameAssetNames.count, 1)
            let elapsedFrames = Int((context.date.timeIntervalSince(tutorialAnimationStartDate) *
                                     LaunchExperienceLayout.tutorialFrameRate).rounded(.down))
            let frameIndex = ((elapsedFrames % frameCount) + frameCount) % frameCount

            Image(frameAssetNames[frameIndex])
                .resizable()
                .scaledToFit()
                .frame(width: LaunchExperienceLayout.tutorialImageWidth)
                .frame(maxWidth: .infinity)
                .allowsHitTesting(false)
        }
    }

    private func dotIndicator(currentIndex: Int) -> some View {
        HStack(spacing: LaunchExperienceLayout.scaled(10)) {
            ForEach(Array(LaunchExperienceContent.tutorialCards.indices), id: \.self) { index in
                Circle()
                    .fill(index == currentIndex ? .white : Color.gray.opacity(0.35))
                    .frame(width: index == currentIndex ? LaunchExperienceLayout.scaled(17.5) : LaunchExperienceLayout.scaled(10),
                           height: index == currentIndex ? LaunchExperienceLayout.scaled(17.5) : LaunchExperienceLayout.scaled(10))
            }
        }
    }
}
