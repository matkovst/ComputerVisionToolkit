#include <signal.h>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#if CV_MAJOR_VERSION == 3
#include <opencv2/optflow.hpp>
#else
#include <opencv2/video/tracking.hpp>
#endif

#include <cvtoolkit/cvplayer.hpp>
#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/utils.hpp>
#include <cvtoolkit/detector_manager.hpp>
#include <cvtoolkit/math/gaussian_estimator.hpp>

#include <json.hpp>
using json = nlohmann::json;


const static std::string WinName = "Motion detection via optical flow";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        "{ @json j        |        | path to json }"
        ;

static bool loop = true;

std::unique_ptr<cvt::DetectorThreadManager> detectorThread;

static const int MaxItemsInQueue = 100;


class OptflowMotionDetectorSettings final
{
public:

    const std::string Name { "optflow-motion-detector" };

    OptflowMotionDetectorSettings(const cvt::Detector::InitializeData& iData, const json& jSettings)
        : m_detectorResolution(iData.imSize)
    {
        if ( !jSettings.empty() )
        {
            parseJsonSettings(jSettings);
        }
        
        /* Handle with areas */
        if ( m_areas.empty() )
        {
            m_areas.emplace_back( cvt::createFullScreenArea(m_detectorResolution) );
        }
    }

    ~OptflowMotionDetectorSettings() = default;

    void parseJsonSettings(const json& j)
    {
        auto fdiffConfig = j[Name];
        if ( fdiffConfig.empty() )
        {
            std::cerr << ">>> Could not find " << Name << " section" << std::endl;
            return;
        }
        
        if ( !fdiffConfig["detector-resolution"].empty() )
            m_detectorResolution = cvt::parseResolution(fdiffConfig["detector-resolution"]);
        
        if ( !fdiffConfig["process-freq-ms"].empty() )
            m_processFreqMs = static_cast<std::int64_t>(fdiffConfig["process-freq-ms"]);
        
        if ( !fdiffConfig["max-accepted-motion-rate"].empty() )
            m_decisionThresh = static_cast<double>(fdiffConfig["max-accepted-motion-rate"]);

        if ( !fdiffConfig["alert-holddown-ms"].empty() )
            m_eventHolddownMs = static_cast<std::int64_t>(fdiffConfig["alert-holddown-ms"]);
        
        if ( !fdiffConfig["--advanced--flow-thresh"].empty() )
            m_flowThresh = static_cast<float>(fdiffConfig["--advanced--flow-thresh"]);
        
        if ( !fdiffConfig["--advanced--alert-holdout-ms"].empty() )
            m_eventHoldoutMs = static_cast<std::int64_t>(fdiffConfig["--advanced--alert-holdout-ms"]);

        m_areas = cvt::parseAreas(fdiffConfig["areas"], m_detectorResolution);
    }

    const cv::Size detectorResolution() const noexcept
    {
        return m_detectorResolution;
    }

    const cvt::Areas& areas() const noexcept
    {
        return m_areas;
    }

    const std::int64_t processFreqMs() const noexcept
    {
        return m_processFreqMs;
    }

    const float flowThresh() const noexcept
    {
        return m_flowThresh;
    }

    const double decisionThresh() const noexcept
    {
        return m_decisionThresh;
    }

    const std::int64_t eventHoldoutMs() const noexcept
    {
        return m_eventHoldoutMs;
    }

    const std::int64_t eventHolddownMs() const noexcept
    {
        return m_eventHolddownMs;
    }

private:
    cv::Size m_detectorResolution;
    cvt::Areas m_areas;
    std::int64_t m_processFreqMs { 1 };
    float m_flowThresh { 5 };
    double m_decisionThresh { 0.1 };
    std::int64_t m_eventHoldoutMs { 0 };
    std::int64_t m_eventHolddownMs { 1000 };
};


class OptflowMotionDetector final : public cvt::Detector
{
public:

    const std::string Name { "optflow-motion-detector" };

    OptflowMotionDetector(const cvt::Detector::InitializeData& iData)
        : m_imSize(iData.imSize)
        , m_fps(iData.fps)
    {
        json jSettings = makeJsonObject(iData.settingsPath);
        m_settings = std::make_shared<OptflowMotionDetectorSettings>(iData, jSettings);

        m_FlowAreaMask = cv::Mat::zeros(m_settings->detectorResolution(), CV_32FC2);
        m_Motion = cv::Mat::zeros(m_settings->detectorResolution(), CV_8U);

        cv::drawContours(m_FlowAreaMask, m_settings->areas(), -1, cv::Scalar::all(255), -1);
        const int totalArea = cvt::totalSqArea(m_settings->areas());
        m_maxMotion = static_cast<double>(255 * totalArea);

        /* Handle with optical flow */
#if CV_MAJOR_VERSION == 3
        m_disOpt = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
#else
        m_disOpt = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
#endif

        /* Handle with event trigger */
        int holdoutFrames = static_cast<int>(m_settings->eventHoldoutMs() * m_fps / 1000.0);
        int holddownFrames = static_cast<int>(m_settings->eventHolddownMs() * m_fps / 1000.0);
        m_eventTrigger.init(holdoutFrames, holddownFrames);

        /* Handle with motion Gaussian */
        double alpha = 1.0 / 100.0;
        m_motionGaussian = std::make_unique<cvt::GaussianEstimator>(alpha);
    }

    ~OptflowMotionDetector() = default;

    void process(const cvt::Detector::InputData& in, cvt::Detector::OutputData& out) override
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
        cv::threshold(m_FlowMagn, m_Motion, m_settings->flowThresh(), 255, cv::THRESH_BINARY);

        // cv::Mat m_Motion2;
        // motionMask_experimental(m_Flow, m_Motion2);

        /* 4. Make decision */
        double totalMotion = cv::sum(m_Motion)[0] / m_maxMotion;

        m_motionGaussian->observe(totalMotion);

        int event = m_eventTrigger( (totalMotion >= m_settings->decisionThresh()) );
        if ( event == cvt::EventTrigger::State::ABOUT_TO_ON )
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

    const cv::Mat& flow() const noexcept
    {
        return m_Flow;
    }

    const cv::Mat& motion() const noexcept
    {
        return m_Motion;
    }

    const std::shared_ptr<OptflowMotionDetectorSettings>& settings() const noexcept
    {
        return m_settings;
    }

private:
#if CV_MAJOR_VERSION == 3
    cv::Ptr<cv::optflow::DISOpticalFlow> m_disOpt;
#else
    cv::Ptr<cv::DISOpticalFlow> m_disOpt;
#endif
    cv::Size m_imSize;
    double m_fps { -1 };
    std::int64_t m_lastProcessedFrameMs { -1 };
    cvt::EventTrigger m_eventTrigger;
    std::unique_ptr<cvt::GaussianEstimator> m_motionGaussian;
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

    json makeJsonObject(const std::string& jPath)
    {
        json j;

        if ( jPath.empty() )
        {
            std::cout << ">>> JSON path must not be empty" << std::endl;
            return j;
        }

        std::ifstream i(jPath.c_str());
        if ( !i.good() )
        {
            std::cout << ">>> Could not read JSON file. Possibly file does not exist" << std::endl;
            return j;
        }

        i >> j;
        if ( j.empty() )
        {
            std::cerr << ">>> Could not create JSON object from file" << std::endl;
            return j;
        }
        
        return j;
    }

    bool filterByTimestamp(std::int64_t timestamp)
    {
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
    //             if ( !cvt::isFlowCorrect(uu) ) continue;

    //             const cv::Vec2f vv = cv::Vec2f(uu.x, uu.y);
    //             const double magn = cv::norm(vv);

    //             if ( magn < m_settings->flowThresh() ) continue;

    //             out.at<uchar>(i, j) = 255;
    //         }
    //     });
    // }

};



void signalHandler(int code) 
{
    loop = false;
    if ( detectorThread )
    {
        detectorThread->finish();
    }
}


int main(int argc, char** argv)
{
    signal(SIGINT, signalHandler); // Handle Ctrl+C exit

    /* Parse command-line args */
    cv::CommandLineParser parser(argc, argv, argKeys);
    parser.about(WinName);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const std::string input = parser.get<std::string>("@input");
    const double scaleFactor = parser.get<double>("resize");
    const bool doResize = (scaleFactor != 1.0);
    const bool record = parser.get<bool>("record");
    const std::string jsonPath = parser.get<std::string>("@json");
    
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    /* Open stream */
    std::shared_ptr<cvt::OpenCVPlayer> player = std::make_shared<cvt::OpenCVPlayer>(input, scaleFactor);

    /* Create GUI */
    auto metrics = std::make_shared<cvt::MetricMaster>();
    cvt::GUI gui(WinName, player, metrics);
    const cv::Size imSize = player->frame0().size();
    const double fps = player->fps();

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << imSize << std::endl;
    std::cout << ">>> Formal FPS: " << fps << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;
    std::cout << ">>> JSON file: " << jsonPath << std::endl;

    /* Task-specific declarations */
    cvt::Detector::InitializeData initData { imSize, fps, jsonPath };
    std::shared_ptr<OptflowMotionDetector> motionDetector = std::make_shared<OptflowMotionDetector>(initData);
    detectorThread = std::make_unique<cvt::DetectorThreadManager>(motionDetector);

    /* Detector loop */
    detectorThread->run();

    /* Main loop */
    cv::Mat frame, out;
    while ( loop )
    {
        auto m = metrics->measure();

        /* Controls */
        int action = gui.listenKeyboard();
        if ( action == gui.CONTINUE ) continue;
        if ( action == gui.CLOSE ) 
        {
            loop = false;
            if ( detectorThread->isRunning() )
            {
                detectorThread->finish();
            }
        }

        /* Capturing */
        *player >> frame;
        if ( frame.empty() ) 
        {
            break;
        }

        /* Computer vision magic */
        {
            if ( detectorThread->iDataQueue.size() >= MaxItemsInQueue )
            {
                detectorThread->iDataQueue.clear();
            }

            cvt::Detector::InputData iData
            {
                !frame.empty(),
                frame.data,
                static_cast<unsigned int>(frame.type()),
                static_cast<unsigned int>(frame.step.p[0]),
                player->timestamp()
            };
            detectorThread->iDataQueue.push(std::move(iData));
        }

        /* Check for events */
        while ( detectorThread->oDataQueue.size() > 0 )
        {
            const auto& sharedEventItem = detectorThread->oDataQueue.pop1(1000);
            std::cout << ">>> [EVENT]: " << sharedEventItem->eventDescr << " at " << sharedEventItem->eventTimestamp << std::endl;
        }

        /* Display info */
        cv::Size detSize = motionDetector->settings()->detectorResolution();
        cv::resize(frame, out, detSize);
        cvt::drawMotionField(motionDetector->flow(), out, 16);
        cvt::drawAreaMaskNeg(out, motionDetector->settings()->areas(), 0.8);
        if ( record )
        {
            *player << out;
        }
        if ( out.channels() == 1 )
        {
            cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
        }
        gui.imshow(out, record);
    }

    if ( detectorThread->isRunning() )
    {
        detectorThread->finish();
    }
    detectorThread->detectorThread.join();
    
    std::cout << ">>> Main thread metrics (with waitKey): " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
