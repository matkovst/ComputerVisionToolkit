#include <opencv2/imgproc.hpp>
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/detector/optflow_motion_detector.hpp"


namespace cvt
{

OptflowMotionDetectorSettings::OptflowMotionDetectorSettings(const Detector::InitializeData& iData, const json& jSettings)
    : DetectorSettings(iData, jSettings)
{
    if ( !jSettings.empty() )
    {
        parseJsonSettings(jSettings);
    }
}

void OptflowMotionDetectorSettings::parseJsonSettings(const json& j)
{
    auto jDetectorSettings = j[m_instanceName];
    if ( jDetectorSettings.empty() )
    {
        std::cerr << ">>> Could not find " << m_instanceName << " section" << std::endl;
        return;
    }
    
    if ( !jDetectorSettings["max-accepted-motion-rate"].empty() )
        m_decisionThresh = static_cast<double>(jDetectorSettings["max-accepted-motion-rate"]);

    if ( !jDetectorSettings["min-accepted-velocity"].empty() )
        m_minAcceptedVelocity = static_cast<float>(jDetectorSettings["min-accepted-velocity"]);

    if ( !jDetectorSettings["max-accepted-velocity"].empty() )
        m_maxAcceptedVelocity = static_cast<float>(jDetectorSettings["max-accepted-velocity"]);

    if ( !jDetectorSettings["alert-holddown-ms"].empty() )
        m_eventHolddownMs = static_cast<std::int64_t>(jDetectorSettings["alert-holddown-ms"]);
    
    if ( !jDetectorSettings["--advanced--alert-holdout-ms"].empty() )
        m_eventHoldoutMs = static_cast<std::int64_t>(jDetectorSettings["--advanced--alert-holdout-ms"]);
}

const double OptflowMotionDetectorSettings::decisionThresh() const noexcept
{
    return m_decisionThresh;
}

const float OptflowMotionDetectorSettings::minAcceptedVelocity() const noexcept
{
    return m_minAcceptedVelocity;
}

const float OptflowMotionDetectorSettings::maxAcceptedVelocity() const noexcept
{
    return m_maxAcceptedVelocity;
}

const std::int64_t OptflowMotionDetectorSettings::eventHoldoutMs() const noexcept
{
    return m_eventHoldoutMs;
}

const std::int64_t OptflowMotionDetectorSettings::eventHolddownMs() const noexcept
{
    return m_eventHolddownMs;
}


OptflowMotionDetector::OptflowMotionDetector(const Detector::InitializeData& iData)
    : m_imSize(iData.imSize)
{
    json jSettings = makeJsonObject(iData.settingsPath);
    m_settings = std::make_shared<OptflowMotionDetectorSettings>(iData, jSettings);

    m_FlowAreaMask = cv::Mat::zeros(m_settings->detectorResolution(), CV_32FC2);
    m_Motion = cv::Mat::zeros(m_settings->detectorResolution(), CV_8U);

    cv::drawContours(m_FlowAreaMask, m_settings->areas(), -1, cv::Scalar::all(255), -1);
    const int totalArea = totalSqArea(m_settings->areas());
    m_maxMotion = static_cast<double>(255 * totalArea);

    /* Handle with optical flow */
#if CV_MAJOR_VERSION == 3
    m_disOpt = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
#else
    m_disOpt = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
#endif

    /* Handle with event trigger */
    int holdoutFrames = static_cast<int>(m_settings->eventHoldoutMs() * m_settings->fps() / 1000.0);
    int holddownFrames = static_cast<int>(m_settings->eventHolddownMs() * m_settings->fps() / 1000.0);
    m_eventTrigger.init(holdoutFrames, holddownFrames);

    /* Handle with motion Gaussian */
    double alpha = 1.0 / 100.0;
    m_motionGaussian = std::make_unique<GaussianEstimator>(alpha);
}

void OptflowMotionDetector::process(const Detector::InputData& in, Detector::OutputData& out)
{
    if ( filterByTimestamp(in.timestamp) )
    {
        return;
    }

    const cv::Mat frame = cv::Mat(m_imSize, in.imType, const_cast<unsigned char *>(in.imData), in.imStep);
    cv::cvtColor(frame, m_Gray, cv::COLOR_BGR2GRAY);
    if ( m_settings->detectorResolution() != m_imSize )
    {
        cv::resize(m_Gray, m_Gray, m_settings->detectorResolution(), 0.0, 0.0, cv::INTER_AREA);
    }
        
    if ( m_PrevGray.empty() )
    {
        m_PrevGray = m_Gray.clone();
        if ( m_Flow.empty() )
        {
            m_Flow.create(m_Gray.size(), CV_32F);
        }
        return;
    }
        
    /* 1. Calculate flow */
    m_disOpt->calc(m_Gray, m_PrevGray, m_Flow);
    cv::bitwise_and(m_Flow, m_FlowAreaMask, m_Flow);

    /* 2. Get magnitude */
    cv::split(m_Flow, m_FlowUV);
    cv::multiply(m_FlowUV[1], -1, m_FlowUV[1]);
    cv::cartToPolar(m_FlowUV[0], m_FlowUV[1], m_FlowMagn, m_FlowAngle, true);

    /* 3. Threshold */
    if ( m_settings->maxAcceptedVelocity() > 0.0 && m_settings->maxAcceptedVelocity() < 255 )
    {
        cv::threshold(m_FlowMagn, m_Motion, 0, m_settings->maxAcceptedVelocity(), cv::THRESH_BINARY);
    }
    if ( m_settings->minAcceptedVelocity() > 0.0 )
    {
        cv::threshold(m_FlowMagn, m_Motion, m_settings->minAcceptedVelocity(), 255, cv::THRESH_BINARY);
    }

    // cv::Mat m_Motion2;
    // motionMask_experimental(m_Flow, m_Motion2);

    /* 4. Make decision */
    double totalMotion = cv::sum(m_Motion)[0] / m_maxMotion;

    m_motionGaussian->observe(totalMotion);

    int event = m_eventTrigger( (totalMotion >= m_settings->decisionThresh()) );
    if ( event == EventTrigger::State::ABOUT_TO_ON )
    {
        out.event = true;
        out.eventTimestamp = in.timestamp;
        out.eventDescr = "Detected motion in area";
    }
    else
    {
        out.event = false;
    }

    cv::swap(m_Gray, m_PrevGray);
}

const cv::Mat& OptflowMotionDetector::flow() const noexcept
{
    return m_Flow;
}

const cv::Mat& OptflowMotionDetector::motion() const noexcept
{
    return m_Motion;
}

const std::shared_ptr<OptflowMotionDetectorSettings>& OptflowMotionDetector::settings() const noexcept
{
    return m_settings;
}

bool OptflowMotionDetector::filterByTimestamp(std::int64_t timestamp)
{
    if ( m_settings->processFreqMs() <= 0 ) return false;
    
    if ( m_lastProcessedFrameMs == -1 )
    {
        m_lastProcessedFrameMs = timestamp;
    }
    else
    {
        if ( m_settings->processFreqMs() > 0 )
        {
            std::int64_t elapsed = timestamp - m_lastProcessedFrameMs;
            if ( elapsed < m_settings->processFreqMs() )
            {
                return true;
            }
            m_lastProcessedFrameMs = timestamp - (timestamp % m_settings->processFreqMs());
        }
    }

    return false;
}

// void motionMask_experimental(const cv::Mat& flow, cv::Mat& out)
// {
//     if ( out.empty() )
//     {
//         out.create(flow.size(), CV_8U);
//     }
//     out.setTo(cv::Scalar(0));

//     cv::parallel_for_(cv::Range(0, flow.rows * flow.cols), [&](const cv::Range& range)
//     {
//         for ( int r = range.start; r < range.end; ++r )
//         {
//             int i = r / flow.cols;
//             int j = r % flow.cols;

//             const auto maskPixel = m_FlowAreaMask.at<cv::Point2f>(i, j);
//             if ( maskPixel.x * maskPixel.y == 0.0 ) continue;

//             const auto uu = -flow.at<cv::Point2f>(i, j);
//             if ( !isFlowCorrect(uu) ) continue;

//             const cv::Vec2f vv = cv::Vec2f(uu.x, uu.y);
//             const double magn = cv::norm(vv);

//             if ( magn < m_settings->minAcceptedVelocity() ) continue;

//             out.at<uchar>(i, j) = 255;
//         }
//     });
// }

}