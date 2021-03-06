#include <signal.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/logger.hpp"
#include "cvtoolkit/settings.hpp"


const static std::string SampleName = "sample-template";
const static std::string TitleName = "Sample-template";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @settings s    |        | path to settings }"
        ;


int main(int argc, char** argv)
{
    auto logger = cvt::createLogger(TitleName, spdlog::level::debug);
    logger->info("Program started. Have fun!");

    /* Parse command-line args */
    cv::CommandLineParser parser(argc, argv, argKeys);
    parser.about(TitleName);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    /* Make Settings */
    std::string settingsPath = parser.get<std::string>("@settings");
    const auto [settingsOk, settingsMsg] = cvt::verifyFile(settingsPath);
    if ( !settingsOk )
    {
        logger->error("Could not load settings: {}", settingsMsg);
        return -1;
    }
    const auto jSettings = std::make_shared<cvt::JsonSettings>(settingsPath, SampleName);
    logger->debug(jSettings->summary());

    /* Open stream */
    std::shared_ptr<cvt::OpenCVPlayer> player = std::make_shared<cvt::OpenCVPlayer>(jSettings->input(), 
                                                                                    jSettings->inputSize());

    /* Create GUI */
    auto metrics = std::make_shared<cvt::MetricMaster>();
    cvt::GUI gui(TitleName, player, metrics);
    const cv::Size imSize = player->frame0().size();
    const double fps = player->fps();

    /* Task-specific declarations */
    // TODO: fill in

    /* Main loop */
    cv::Mat frame, out;
    std::shared_ptr<cv::Mat> detailedFramePtr;
    if ( jSettings->display() )
    {
        detailedFramePtr = std::make_shared<cv::Mat>();
    }

    bool loop = true;
    const bool isImage = (player->getInputType(jSettings->input()) == cvt::OpenCVPlayer::InputType::IMAGE);
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

            // do magic
        }

        /* Display info */
        if ( jSettings->record() || jSettings->display() )
        {
            if ( frame.rows >= 1080 && frame.cols >= 1920 )
            {
                cv::resize(frame, out, cv::Size(), 0.5, 0.5);
            }
            else
            {
                out = frame.clone();
            }

            if ( jSettings->record() )
            {
                *player << out;
            }

            if ( jSettings->display() )
            {
                if ( out.channels() == 1 )
                {
                    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
                }
                
                gui.imshow(out, jSettings->record());
                if ( detailedFramePtr && !detailedFramePtr->empty() )
                {
                    cv::imshow("Detailed", *detailedFramePtr);
                }
            }
        }

        /* In the case of image finish looping and just wait */
        if ( isImage )
        {
            cv::waitKey();
            break;
        }
    }
    
    logger->info("Inference metrics: {}", metrics->summary());
    logger->info("Program successfully finished");
    return 0;
}
