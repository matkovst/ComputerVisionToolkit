#include <signal.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/detector/yolo_object_detector.hpp>


/* The model and class names list can be downloaded from here:
    https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
*/


const static std::string WinName = "Inception";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ @data a        |        | model data }"
        "{ record e       |  false | do record }"
        "{ display d      |  true  | whether display window or not }"
        "{ @json j        |        | path to json }"
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

    const std::string input = parser.get<std::string>("@input");
    const double scaleFactor = parser.get<double>("resize");
    const bool doResize = (scaleFactor != 1.0);
    const bool record = parser.get<bool>("record");
    const bool display = parser.get<bool>("display");
    const std::string jsonPath = parser.get<std::string>("@json");
    const std::string data = parser.get<std::string>("@data");
    const bool gpu = parser.get<bool>("gpu");
    
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
    const double fps = player->fps();

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << imSize << std::endl;
    std::cout << ">>> Formal FPS: " << fps << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;
    std::cout << ">>> Display: " << std::boolalpha << display << std::endl;
    std::cout << ">>> JSON file: " << (( jsonPath.empty() ) ? "-" : jsonPath) << std::endl;

    /* Task-specific declarations */
    const std::string modelWeights = data + "/inception.pb";
    const std::string classNames = data + "/names.txt";
    int backend = cv::dnn::DNN_BACKEND_DEFAULT;
    int target = cv::dnn::DNN_TARGET_CPU;
#if HAVE_OPENCV_CUDA && CV_MAJOR_VERSION > 3 && CV_MINOR_VERSION > 1
    if ( gpu )
    {
        backend = cv::dnn::DNN_BACKEND_CUDA;
        target = cv::dnn::DNN_TARGET_CUDA;
    }
#endif
    cvt::InceptionNNClassifier inception(modelWeights, classNames, backend, target);

    /* Main loop */
    cv::Mat frame, out;
    std::shared_ptr<cv::Mat> detailedFramePtr;
    if ( display )
    {
        detailedFramePtr = std::make_shared<cv::Mat>();
    }

    bool loop = true;
    while ( loop )
    {
        /* In the case input is image */
        if ( player->getInputType(input) == cvt::OpenCVPlayer::InputType::IMAGE )
        {
            
            /* Computer vision magic */
            cvt::InferOuts dOuts;
            {
                auto m = metrics->measure();

                inception.Infer( player->frame0(), dOuts, 0.25f );
            }

            /* Display classes */
            if (!dOuts.empty())
            {
                std::cout << "Top 5 classes:" << std::endl;
                for (const auto& dOut : dOuts)
                {
                    std::cout << std::setw(4) << "#" << dOut.classId << " '" << dOut.className << "', ";
                    std::cout << "conf: " << dOut.confidence * 100 << "%" << std::endl;
                }
            }

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
        cvt::InferOuts dOuts;
        {
            auto m = metrics->measure();

            {
                inception.Infer( frame, dOuts, 0.25f );
            }
        }

        /* Display info */
        if ( record || display )
        {
            if ( frame.rows >= 1080 && frame.cols >= 1920 )
            {
                cv::resize(frame, out, cv::Size(), 0.5, 0.5);
            }
            else
            {
                out = frame.clone();
            }

            if ( record )
            {
                *player << out;
            }

            if ( display )
            {
                if ( out.channels() == 1 )
                {
                    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
                }

                /* Display classes */
                if (!dOuts.empty())
                {
                    int y = out.rows - 20 * 6;
                    cv::putText(out, "Top 5 classes:", cv::Point(10, y - 20), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 0), 1);
                    for (const auto& dOut : dOuts)
                    {
                        std::ostringstream stream;
                        stream << std::setw(4) << "#" << dOut.classId << " '" << dOut.className << "', ";
                        stream << "conf: " << dOut.confidence * 100 << "%";
                        cv::putText(out, stream.str(), cv::Point(10, y), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 0), 1);
                        y += 20;
                    }
                }
                
                gui.imshow(out, record);
                if ( detailedFramePtr && !detailedFramePtr->empty() )
                {
                    cv::imshow("Detailed", *detailedFramePtr);
                }
            }
        }
    }
    
    std::cout << ">>> Inference metrics: " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
