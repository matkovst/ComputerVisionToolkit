#include <iostream>
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


const static std::string WinName = "OpenCV Player";

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
    cvt::GUI gui(WinName, player);

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << player->frame0().size() << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;

    /* Init DIS */
#if CV_MAJOR_VERSION == 3
    cv::Ptr<cv::optflow::DISOpticalFlow> disOpt;
    disOpt = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
#else
    cv::Ptr<cv::DISOpticalFlow> disOpt;
    disOpt = cv::DISOpticalFlow::create(0);
#endif
    cv::Mat gray, prevGray;

    /* Main loop */
    bool loop = true;
    cv::Mat frame, out;
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

        /* Pre-process */
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if ( prevGray.empty() )
        {
            prevGray = gray.clone();
        }

        /* Computer vision magic */
        cv::Mat_<cv::Point2f> optFlow;
        disOpt->calc(gray, prevGray, optFlow);
        
        /* Post-processing */
        cv::swap(gray, prevGray);

    
        /* Display info */
        out = frame.clone();
        cvt::drawMotionField(optFlow, out, 16);

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
    
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
