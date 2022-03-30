#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/metrics.hpp"


const static std::string WinName = "My";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        ;



using namespace std;
using namespace cv;

// https://github.com/krishraghuram/Zhang-Suen-Skeletonization

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 */
void thinningIteration(Mat& im, int iter)
{
    Mat marker = Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 */
void thinning(Mat& im)
{
    im /= 255;

    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (countNonZero(diff) > 0);

    im *= 255;
}

/**
 * This is the function that acts as the input/output system of this header file.
 */
Mat skeletonization(Mat inputImage)
{
    if (inputImage.empty())
    	cout<<"Inside skeletonization, Source empty"<<endl;

    Mat outputImage;
    cvtColor(inputImage, outputImage, cv::COLOR_BGR2GRAY);
    threshold(outputImage, outputImage, 0, 255, THRESH_BINARY+THRESH_OTSU);

    thinning(outputImage);

    return outputImage;
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

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << player->frame0().size() << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;

    /* Main loop */
    bool loop = true;
    cv::Mat frame, frameHSV, frameThreshed, out;
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

            /* Get isolines mask */
            cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);
            cv::inRange(frameHSV, cv::Scalar(0, 0, 0), cv::Scalar(179, 255, 126), frameThreshed);

            cv::Mat skel = frameThreshed.clone();
            thinning(skel);

            /* Find contours */
            std::vector<std::vector<cv::Point> > contours, contoursApproxd;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours( skel, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE );

            /* Reduce the number of verticies */
            const double epsilon = 2.0;
            contoursApproxd.resize(contours.size());
            for( size_t k = 0; k < contours.size(); k++ )
            {
                cv::approxPolyDP(cv::Mat(contours[k]), contoursApproxd[k], epsilon, false);
            }

            /* Draw result */
            // out = cv::Mat::zeros( skel.size(), CV_8UC3 );
            out = frame.clone();
            const cv::Scalar color = cv::Scalar( 0, 255, 0 );
            // for( size_t i = 0; i < contours.size(); i++ )
            // {
            //     cv::Scalar color = cv::Scalar( 0, 255, 0 );
            //     cv::drawContours( out, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0 );
            // }
            // for( size_t i = 0; i < contoursApproxd.size(); i++ )
            // {
            //     cv::Scalar color = cv::Scalar( 0, 255, 0 );
            //     cv::drawContours( out, contoursApproxd, (int)i, color, 1, cv::LINE_8, hierarchy, 0 );
            // }
            for (size_t k = 0; k < contours.size(); ++k)
            {
                for (size_t i = 1; i < contours[k].size(); ++i)
                {
                    cv::line(out, contours[k][i], contours[k][i-1], color, 1);
                }
            }
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

        if ( player->getInputType(input) == cvt::OpenCVPlayer::InputType::IMAGE ) 
        {
            cv::waitKey();
            break;
        }
    }
    
    std::cout << ">>> " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
