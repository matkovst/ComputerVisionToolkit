#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/metrics.hpp"


const static std::string WinName = "Laplacian";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
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

    std::string input = parser.get<std::string>("@input");
    double scaleFactor = parser.get<double>("resize");
    bool doResize = (scaleFactor != 1.0);
    bool record = parser.get<bool>("record");
    
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

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << player->frame0().size() << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;

    /* Main loop */
    bool loop = true;
    cv::Mat frame, out;
    while ( loop )
    {
        /* In the case input is image */
        if ( player->getInputType(input) == cvt::OpenCVPlayer::InputType::IMAGE )
        {
            cv::Mat blurredFrame, gray, lapl;
            cv::GaussianBlur( player->frame0(), blurredFrame, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT );
            cv::cvtColor( blurredFrame, gray, cv::COLOR_BGR2GRAY );
            cv::Laplacian( gray, lapl, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT );
            cv::convertScaleAbs( lapl, out );
            if ( record )
            {
                *player << out;
            }
            if ( out.channels() == 1 )
            {
                cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
            }
            gui.imshow(out, record);

            cv::waitKey();
            break;
        }

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

            cv::Mat blurredFrame, gray, lapl;
            cv::GaussianBlur( frame, blurredFrame, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT );
            cv::cvtColor( blurredFrame, gray, cv::COLOR_BGR2GRAY );
            cv::Laplacian( gray, lapl, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT );
            cv::convertScaleAbs( lapl, out );
        }

        /* Display info */
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
