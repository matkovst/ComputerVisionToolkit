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

#include <json.hpp>
using json = nlohmann::json;


static bool loop = true;

const static std::string WinName = "Motion detection via optical flow";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        "{ @json j        |        | path to json }"
        ;


class OptflowMotionDetector final : public Detector
{
public:

    const std::string Name { "optflow-motion-detector" };

    OptflowMotionDetector(const Detector::InitializeData& iData)
        : m_imSize(iData.imSize)
        , m_fps(iData.fps)
        , m_AreaMask(cv::Mat::zeros(iData.imSize, CV_8U))
    {
        parseJsonConfig(iData.configPath);
        
        /* Handle with areas */
        if ( m_areas.empty() )
        {
            m_areas.emplace_back( cvt::createFullScreenArea(m_imSize) );
        }
        cv::drawContours(m_AreaMask, m_areas, -1, cv::Scalar(255), -1);

        /* Handle with optical flow */
#if CV_MAJOR_VERSION == 3
        m_disOpt = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
#else
        m_disOpt = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
#endif
    }

    ~OptflowMotionDetector() = default;

    void process(const Detector::InputData& in, Detector::OutputData& out) override
    {
        if ( filterByTimestamp(in.timestamp) )
        {
            return;
        }

        const cv::Mat frame = cv::Mat(m_imSize, in.imType, const_cast<unsigned char *>(in.imData), in.imStep);
        cv::cvtColor(frame, m_Gray, cv::COLOR_BGR2GRAY);
        
        if ( m_PrevGray.empty() )
        {
            m_PrevGray = m_Gray.clone();
            if ( m_Flow.empty() )
            {
                m_Flow.create(m_Gray.size(), CV_32F);
            }
            return;
        }
        
        m_disOpt->calc(m_Gray, m_PrevGray, m_Flow);

        cv::swap(m_Gray, m_PrevGray);
    }

    const cv::Mat& flow() const noexcept
    {
        return m_Flow;
    }

    const cvt::Areas& areas() const noexcept
    {
        return m_areas;
    }

private:
    cv::Size m_imSize { -1, -1 };
    double m_fps { -1 };
    std::int64_t m_processFreqMs { 1 };
    std::int64_t m_lastProcessedFrameMs { -1 };
    cvt::Areas m_areas;
    cv::Mat m_AreaMask;
    cv::Mat m_Gray;
    cv::Mat m_PrevGray;
    cv::Mat m_Flow;
#if CV_MAJOR_VERSION == 3
    cv::Ptr<cv::optflow::DISOpticalFlow> m_disOpt;
#else
    cv::Ptr<cv::DISOpticalFlow> m_disOpt;
#endif

    void parseJsonConfig(const std::string& configPath)
    {
        if ( configPath.empty() )
        {
            std::cout << ">>> JSON path must not be empty" << std::endl;
            return;
        }

        std::ifstream i(configPath.c_str());
        if ( !i.good() )
        {
            std::cout << ">>> Could not read JSON file. Possibly file does not exist" << std::endl;
            return;
        }

        json j;
        i >> j;
        if ( j.empty() )
        {
            std::cerr << ">>> Could not create JSON object from file" << std::endl;
            return;
        }
                    
        auto fdiffConfig = j[Name];
        if ( fdiffConfig.empty() )
        {
            std::cerr << ">>> Could not find " << Name << " section" << std::endl;
            return;
        }
        
        if ( !fdiffConfig["processFreqMs"].empty() )
        {
            m_processFreqMs = static_cast<std::int64_t>(fdiffConfig["processFreqMs"]);
        }

        m_areas = cvt::parseAreas(fdiffConfig["areas"], m_imSize);
    }

    bool filterByTimestamp(std::int64_t timestamp)
    {
        if ( m_lastProcessedFrameMs == -1 )
        {
            m_lastProcessedFrameMs = timestamp;
        }
        else
        {
            if ( m_processFreqMs > 0 )
            {
                std::int64_t elapsed = timestamp - m_lastProcessedFrameMs;
                if ( elapsed < m_processFreqMs )
                {
                    return true;
                }
                m_lastProcessedFrameMs = timestamp - (timestamp % m_processFreqMs);
            }
        }

        return false;
    }

};


void signalHandler(int code) 
{
    loop = false;
    stopDetectorThreads = true;
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
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;
    std::cout << ">>> JSON file: " << jsonPath << std::endl;

    /* Task-specific declarations */
    Detector::InitializeData initData { imSize, fps, jsonPath };
    std::shared_ptr<OptflowMotionDetector> motionDetector = std::make_shared<OptflowMotionDetector>(initData);
    DetectorThreadManager detectorThread(motionDetector, 0);

    /* Detector loop */
    detectorThread.run();

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
            stopDetectorThreads = true;
        }

        /* Capturing */
        *player >> frame;
        if ( frame.empty() ) 
        {
            break;
        }

        /* Computer vision magic */
        {
            if ( detectorThread.iDataQueue.size() >= MaxItemsInQueue )
            {
                detectorThread.iDataQueue.clear();
            }

            Detector::InputData iData
            {
                !frame.empty(),
                frame.data,
                static_cast<unsigned int>(frame.type()),
                static_cast<unsigned int>(frame.step.p[0]),
                player->timestamp()
            };
            detectorThread.iDataQueue.push(std::move(iData));
        }

        /* Display info */
        out = frame.clone();
        cvt::drawMotionField(motionDetector->flow(), out, 16);
        cvt::drawAreaMaskNeg(out, motionDetector->areas(), 0.8);
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

    stopDetectorThreads = true;
    detectorThread.detectorThread.join();
    
    std::cout << ">>> Main thread metrics (with waitKey): " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
