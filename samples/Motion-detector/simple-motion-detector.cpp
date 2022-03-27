#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/settings.hpp"


const static std::string WinName = "Motion detection via frame differencing";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        "{ @json j        |        | path to json }"
        ;


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
    std::int64_t processFreqMs = 100;
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
                auto fdiffConfig = j["simple-motion-detector"];
                if ( !fdiffConfig.empty() )
                {
                    if ( !fdiffConfig["areaMotionThresh"].empty() )
                    {
                        areaMotionThresh = static_cast<int>(255.0 * static_cast<double>(fdiffConfig["areaMotionThresh"]));
                    }
                    if ( fdiffConfig["decisionThresh"].empty() )
                    {
                        decisionThresh = 255 * static_cast<double>(fdiffConfig["decisionThresh"]);
                    }
                    if ( fdiffConfig["processFreqMs"].empty() )
                    {
                        processFreqMs = static_cast<std::int64_t>(fdiffConfig["processFreqMs"]);
                    }

                    areas = cvt::parseAreas(fdiffConfig["areas"], imSize);
                }
                else std::cerr << ">>> Could not find simple-motion-detector section" << std::endl;
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
    cv::Mat areaMask = cv::Mat::zeros(imSize, CV_8U);
    cv::drawContours(areaMask, areas, -1, cv::Scalar(255), -1);

    /* Main loop */
    bool loop = true;
    cv::Mat frame, prevFrame, fdiff, out;
    cv::Mat maskedFrame;
    std::int64_t lastFrameMs = -1;
    bool processNow = true;
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

            /* Process frames with given frequency */
            processNow = true;
            if ( lastFrameMs == -1 )
            {
                lastFrameMs = player->timestamp();
            }
            else
            {
                if ( processFreqMs > 0 )
                {
                    std::int64_t elapsed = player->timestamp() - lastFrameMs;
                    if ( elapsed < processFreqMs )
                    {
                        processNow = false;
                    }
                    lastFrameMs = player->timestamp() - (player->timestamp() % processFreqMs);
                }
            }

            if ( prevFrame.empty() )
            {
                prevFrame = frame.clone();
            }

            if ( processNow )
            {
                cv::absdiff(frame, prevFrame, fdiff);
                cv::cvtColor(fdiff, fdiff, cv::COLOR_BGR2GRAY);
                cv::threshold(fdiff, fdiff, areaMotionThresh, 255, cv::THRESH_BINARY);

                /* Mask frame */
                cv::bitwise_and(fdiff, areaMask, fdiff);

                cv::swap(frame, prevFrame);
            }
        }

        /* Display info */
        cv::Mat frameOut = ( processNow ) ? prevFrame.clone() : frame.clone();
        cvt::drawAreaMaskNeg(frameOut, areas, 0.8);
        if ( processNow ) cv::cvtColor(fdiff, fdiff, cv::COLOR_GRAY2BGR);
        cvt::hstack2images(frameOut, fdiff, out);
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
