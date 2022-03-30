#pragma once

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_MAJOR_VERSION == 3
#include <opencv2/optflow.hpp>
#else
#include <opencv2/video/tracking.hpp>
#endif

#include "../trigger.hpp"
#include "../detector.hpp"
#include "../math/gaussian_estimator.hpp"

namespace cvt
{

class OptflowMotionDetectorSettings final : public DetectorSettings
{
public:
    OptflowMotionDetectorSettings(const Detector::InitializeData& iData, const json& jSettings);

    ~OptflowMotionDetectorSettings() = default;

    void parseJsonSettings(const json& j);

    const double decisionThresh() const noexcept;

    const float minAcceptedVelocity() const noexcept;

    const float maxAcceptedVelocity() const noexcept;

    const std::int64_t eventHoldoutMs() const noexcept;

    const std::int64_t eventHolddownMs() const noexcept;

private:
    double m_decisionThresh { 0.1 };
    std::int64_t m_eventHoldoutMs { 0 };
    std::int64_t m_eventHolddownMs { 1000 };
    float m_minAcceptedVelocity { 5 };
    float m_maxAcceptedVelocity { -1 };
};


class OptflowMotionDetector final : public Detector
{
public:
    OptflowMotionDetector(const Detector::InitializeData& iData);

    ~OptflowMotionDetector() = default;

    void process(const Detector::InputData& in, Detector::OutputData& out) override;

    const cv::Mat& flow() const noexcept;

    const cv::Mat& motion() const noexcept;

    const std::shared_ptr<OptflowMotionDetectorSettings>& settings() const noexcept;

private:
#if CV_MAJOR_VERSION == 3
    cv::Ptr<cv::optflow::DISOpticalFlow> m_disOpt;
#else
    cv::Ptr<cv::DISOpticalFlow> m_disOpt;
#endif
    cv::Size m_imSize;
    std::int64_t m_lastProcessedFrameMs { -1 };
    EventTrigger m_eventTrigger;
    std::unique_ptr<GaussianEstimator> m_motionGaussian;
    std::shared_ptr<OptflowMotionDetectorSettings> m_settings;

    cv::Mat m_FlowAreaMask;
    cv::Mat m_Gray;
    cv::Mat m_PrevGray;
    cv::Mat m_Flow;
    cv::Mat m_FlowUV[2];
    cv::Mat m_FlowMagn;
    cv::Mat m_FlowAngle;
    cv::Mat m_Motion;
    double m_maxMotion;

    bool filterByTimestamp(std::int64_t timestamp);

    // void motionMask_experimental(const cv::Mat& flow, cv::Mat& out);

};

}