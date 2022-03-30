#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/metrics.hpp"


const static std::string WinName = "KNN Background subtractor";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        ;


int History = 500;
int Dist2Threshold = 400;
cv::Ptr<cv::BackgroundSubtractorKNN> bgSubtractor;

static void onHistoryChanged( int, void* )
{
   if ( !bgSubtractor.empty() )
   {
       bgSubtractor->setHistory(History);
   }
}

static void onDist2ThresholdChanged( int, void* )
{
   if ( !bgSubtractor.empty() )
   {
       bgSubtractor->setDist2Threshold(static_cast<double>(Dist2Threshold));
   }
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

    cv::createTrackbar( "History", gui.WinName(), &History, 5000, onHistoryChanged );
    onHistoryChanged( History, 0 );
    cv::createTrackbar( "DistThresh", gui.WinName(), &Dist2Threshold, 2000, onDist2ThresholdChanged );
    onDist2ThresholdChanged( Dist2Threshold, 0 );

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << player->frame0().size() << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;

    /* Init KNN bg-subtractor */
    bgSubtractor = cv::createBackgroundSubtractorKNN( History, Dist2Threshold, true );

    /* Main loop */
    bool loop = true;
    cv::Mat frame, fgMask, out;
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
            
            bgSubtractor->apply( frame, fgMask );
        }
        cv::cvtColor( fgMask, fgMask, cv::COLOR_GRAY2BGR );

        /* Display info */
        cvt::hstack2images( fgMask, frame, out );
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
