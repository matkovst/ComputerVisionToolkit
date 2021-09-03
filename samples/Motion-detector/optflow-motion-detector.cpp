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
const static std::string sampleName = "optflow-motion-detector";

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


bool processNow(std::int64_t timestamp, std::int64_t processFreq)
{
    static std::int64_t lastFrameMs = -1;

    if ( lastFrameMs == -1 )
    {
        lastFrameMs = timestamp;
    }
    else
    {
        if ( processFreq > 0 )
        {
            std::int64_t elapsed = timestamp - lastFrameMs;
            if ( elapsed < processFreq )
            {
                return false;
            }
            lastFrameMs = timestamp - (timestamp % processFreq);
        }
    }

    return true;
}


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

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << imSize << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;
    std::cout << ">>> JSON file: " << jsonPath << std::endl;

    /* Parse JSON */
    int areaMotionThresh = 127;
    double decisionThresh = 0.5;
    std::int64_t processFreqMs = 1;
    cvt::Areas areas;
    if ( !jsonPath.empty() ) 
    {
        std::ifstream i(jsonPath.c_str());
        if ( i.good() )
        {
            json j;
            i >> j;
            if ( !j.empty() )
            {
                auto fdiffConfig = j[sampleName];
                if ( !fdiffConfig.empty() )
                {
                    areaMotionThresh = ( fdiffConfig["areaMotionThresh"].empty() ) ? areaMotionThresh : static_cast<int>(255.0 * fdiffConfig["areaMotionThresh"]);
                    decisionThresh = ( fdiffConfig["decisionThresh"].empty() ) ? decisionThresh : 255 * static_cast<double>(fdiffConfig["decisionThresh"]);
                    processFreqMs = ( fdiffConfig["processFreqMs"].empty() ) ? processFreqMs : static_cast<std::int64_t>(fdiffConfig["processFreqMs"]);

                    areas = cvt::parseAreas(fdiffConfig["areas"], imSize);
                }
                else std::cerr << ">>> Could not find " << sampleName << " section" << std::endl;
            }
            else std::cerr << ">>> Could not create JSON object from file" << std::endl;
        }
        else std::cout << ">>> Could not read JSON file. Possibly file does not exist" << std::endl;
    }
    else std::cout << ">>> JSON path must not be empty" << std::endl;

    if ( areas.empty() )
    {
        areas.emplace_back( cvt::createFullScreenArea(imSize) );
    }

    /* Task-specific declarations */
    OpticalFlowHelper oflowHelper;
    cv::Mat areaMask = cv::Mat::zeros(imSize, CV_8U);
    cv::drawContours(areaMask, areas, -1, cv::Scalar(255), -1);

    /* Main loop */
    bool loop = true;
    cv::Mat frame, gray, out;
    cv::Mat maskedFrame;
    std::int64_t lastFrameMs = -1;
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
        cv::Mat optFlow = cv::Mat::zeros(imSize, CV_32F);
        if ( processNow(player->timestamp(), processFreqMs) )
        {
            auto m = metrics->measure();

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            oflowHelper.calc(gray, optFlow);
        }

        /* Display info */
        out = frame.clone();
        cvt::drawMotionField(optFlow, out, 16);
        cvt::drawAreaMaskNeg(out, areas, 0.8);
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
