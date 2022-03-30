#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/metrics.hpp"


const static std::string WinName = "Nvidia HW Optical Flow";

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

    /* Init optical flow object */
    cv::cuda::Stream d_stream1, d_stream2;
    const int gpuId = 0;
    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> d_optFlow = 
        cv::cuda::NvidiaOpticalFlow_2_0::create(player->frame0().size(), cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_MEDIUM,
         cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_1, cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_1,
         false, false, false, gpuId, d_stream1, d_stream2);
    cv::cuda::GpuMat d_gray, d_prevGray, d_optFlowImage, d_optFlowImageFloat;
    cv::Mat gray;

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
        if ( d_prevGray.empty() )
        {
            d_prevGray.upload(gray);
        }

        /* Computer vision magic */
        cv::Mat optFlowImage;
        {
            auto m = metrics->measure();

            d_gray.upload(gray);
            d_optFlow->calc(d_prevGray, d_gray, d_optFlowImage);
            d_optFlow->convertToFloat(d_optFlowImage, d_optFlowImageFloat);
            d_optFlowImageFloat.download(optFlowImage);
        }
        
        /* Post-processing */
        cv::cuda::swap(d_gray, d_prevGray);

    
        /* Display info */
        out = frame.clone();
        cvt::drawMotionField(optFlowImage, out, 16);

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
    
    d_optFlow->collectGarbage();

    std::cout << ">>> " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
