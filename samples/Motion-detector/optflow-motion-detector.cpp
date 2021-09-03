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


struct OpticalFlowHelper
{
    OpticalFlowHelper()
    {
#if CV_MAJOR_VERSION == 3
        m_disOpt = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
#else
        m_disOpt = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
#endif
    }

    void calc(const cv::Mat& frame, cv::Mat& out)
    {
        if ( m_prevGray.empty() )
        {
            m_prevGray = frame.clone();
            if ( out.empty() )
            {
                out.create(frame.size(), CV_32F);
            }
            return;
        }
        
        m_disOpt->calc(frame, m_prevGray, out);

        m_prevGray = frame.clone();
    }

private:
    cv::Mat m_prevGray;
    
#if CV_MAJOR_VERSION == 3
    cv::Ptr<cv::optflow::DISOpticalFlow> m_disOpt;
#else
    cv::Ptr<cv::DISOpticalFlow> m_disOpt;
#endif
};


struct OptflowMotionDetector_InputParams
{
    cv::Size imSize;
    double fps;
    std::string configPath;
};


class OptflowMotionDetector final
{
public:

    const std::string Name { "optflow-motion-detector" };

    OptflowMotionDetector(const OptflowMotionDetector_InputParams& iParams)
        : m_imSize(iParams.imSize)
        , m_fps(iParams.fps)
        , m_AreaMask(cv::Mat::zeros(iParams.imSize, CV_8U))
    {
        parseJsonConfig(iParams.configPath);
        
        /* Handle with areas */
        if ( m_areas.empty() )
        {
            m_areas.emplace_back( cvt::createFullScreenArea(m_imSize) );
        }
        cv::drawContours(m_AreaMask, m_areas, -1, cv::Scalar(255), -1);

        /* Handle with optical flow */
        OpticalFlowHelper oflowHelper;
    }

    ~OptflowMotionDetector() = default;

    void process(cv::InputArray in, std::int64_t timestamp)
    {
        if ( cvt::filterByTimestamp(timestamp, m_processFreqMs) )
        {
            return;
        }

        m_oflowHelper.calc(in.getMat(), m_Flow);
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
    std::int64_t m_processFreqMs { 100 };
    cvt::Areas m_areas;
    cv::Mat m_AreaMask;
    OpticalFlowHelper m_oflowHelper;
    cv::Mat m_Flow;

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
};


int main(int argc, char** argv)
{
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
    const OptflowMotionDetector_InputParams iParams { imSize, fps, jsonPath };
    OptflowMotionDetector motionDetector(iParams);

    /* Main loop */
    bool loop = true;
    cv::Mat frame, gray, out;
    while ( loop )
    {
        /* Controls */
        int action = gui.listenKeyboard();
        if ( action == gui.CONTINUE ) continue;
        if ( action == gui.CLOSE ) 
        {
            loop = false;
        }

        /* Capturing */
        *player >> frame;
        if ( frame.empty() ) 
        {
            break;
        }

        /* Computer vision magic */
        {
            auto m = metrics->measure();

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            motionDetector.process(gray, player->timestamp());
        }

        /* Display info */
        out = frame.clone();
        cvt::drawMotionField(motionDetector.flow(), out, 16);
        cvt::drawAreaMaskNeg(out, motionDetector.areas(), 0.8);
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
    
    std::cout << ">>> " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
