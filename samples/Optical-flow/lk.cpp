#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <cvtoolkit/cvplayer.hpp>
#include <cvtoolkit/cvgui.hpp>


const static std::string WinName = "LK Optical flow";

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

    // Create some random colors
    std::vector<cv::Scalar> colors;
    cv::RNG rng;
    for ( int i = 0; i < 100; ++i )
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r, g, b));
    }

    // Take first frame and find corners in it
    cv::Mat prevGray;
    std::vector<cv::Point2f> p0, p1;
    cv::cvtColor(player->frame0(), prevGray, cv::COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(prevGray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(player->frame0().size(), player->frame0().type());

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

        /* Pre-process */
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        /* Computer vision magic */
        // calculate optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        cv::calcOpticalFlowPyrLK(prevGray, gray, p0, p1, status, err, cv::Size(15, 15), 2, criteria);

        std::vector<cv::Point2f> good_new;
        for ( uint i = 0; i < p0.size(); ++i )
        {
            // Select good points
            if( status[i] == 1 ) 
            {
                good_new.push_back(p1[i]);
                // draw the tracks
                cv::line(mask, p1[i], p0[i], colors[i], 2);
                cv::circle(frame, p1[i], 5, colors[i], -1);
            }
        }

        // Now update the previous frame and previous points
        cv::swap( gray, prevGray );
        p0 = good_new;
    
        /* Display info */
        cv::add(frame, mask, out);

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
