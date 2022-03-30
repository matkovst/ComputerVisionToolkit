#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/metrics.hpp"
#include "cvtoolkit/nndetector.hpp"


const static std::string WinName = "Mask-RCNN";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        "{ @data d        |        | model data }"
        "{ gpu g          |  0     | use GPU }"
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
    std::string data = parser.get<std::string>("@data");
    bool gpu = parser.get<bool>("gpu");
    
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
    std::cout << ">>> GPU: " << std::boolalpha << gpu << std::endl;

    /* Main stuff */
    std::string textGraph = data + "/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    std::string modelWeights = data + "/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
    std::string cocoNames = data + "/coco.names";
    int backend = cv::dnn::DNN_BACKEND_DEFAULT;
    int target = cv::dnn::DNN_TARGET_CPU;
#if HAVE_OPENCV_CUDA && CV_MAJOR_VERSION > 3 && CV_MINOR_VERSION > 1
    if ( gpu )
    {
        backend = cv::dnn::DNN_BACKEND_CUDA;
        target = cv::dnn::DNN_TARGET_CUDA;
    }
#endif
    cvt::MaskRCNNObjectDetector detector(textGraph, modelWeights, cocoNames, backend, target);

    cvt::ObjectClasses vehicleClasses { {2, "car"}, {3, "motorcycle"}, {5, "bus"}, {7, "truck"} };
    cvt::ObjectClasses personClasses { {0, "person"} };
    cvt::ObjectClasses dynamicClasses { {0, "person"}, {2, "car"}, {3, "motorcycle"}, {5, "bus"}, {7, "truck"} };

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

        /* Computer vision magic */

        cvt::InferOuts dOuts;
        {
            auto m = metrics->measure();

            detector.Infer( frame, dOuts, 0.25f, dynamicClasses );
            
            // cvt::InferOuts fdOuts;
            // fdOuts.reserve(dOuts.size());
            // detector.Filter( dOuts, fdOuts, vehicleClasses );
        }
    
        /* Display info & Record */
        out = frame.clone();
        cvt::drawInferOuts( out, dOuts, cv::Scalar::all(0) );

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
