#include <signal.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/detector/yolo_object_detector.hpp>


const static std::string WinName = "YOLO object detection";
const static std::string DetName = "yolo-object-detector";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        "{ display d      |  true  | whether display window or not }"
        "{ @json j        |        | path to json }"
        ;

static bool loop = true;

static const int MaxItemsInQueue = 100;

std::unique_ptr<cvt::DetectorThreadManager> detectorThread;


void signalHandler(int code)
{
    loop = false;
    if ( detectorThread )
    {
        detectorThread->finish();
    }
}


int main(int argc, char** argv)
{
    signal(SIGINT, signalHandler); // Handle Ctrl+C exit

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
    cvt::Detector::InitializeData initData { DetName, imSize, fps, jsonPath };
    std::shared_ptr<cvt::YOLOObjectDetector> objectDetector = std::make_shared<cvt::YOLOObjectDetector>(initData);
    detectorThread = std::make_unique<cvt::DetectorThreadManager>(objectDetector);

    /* Detector loop */
    detectorThread->run();

    /* Main loop */
    cv::Mat frame, out;
    std::shared_ptr<cv::Mat> detailedFramePtr;
    if ( display )
    {
        detailedFramePtr = std::make_shared<cv::Mat>();
    }
    while ( loop )
    {
        auto m = metrics->measure();

        /* Controls */
        int action = gui.listenKeyboard();
        if ( action == gui.CONTINUE ) continue;
        if ( action == gui.CLOSE ) 
        {
            loop = false;
            if ( detectorThread->isRunning() )
            {
                detectorThread->finish();
            }
        }

        /* Capturing */
        *player >> frame;
        if ( frame.empty() ) 
        {
            break;
        }

        /* Computer vision magic */
        {
            if ( detectorThread->iDataQueue.size() >= MaxItemsInQueue )
            {
                std::cout << ">>> Queue overflow. Do cleaning ..." << std::endl;
                detectorThread->iDataQueue.clear();
            }

            cvt::Detector::InputData iData
            {
                !frame.empty(),
                frame.data,
                static_cast<unsigned int>(frame.type()),
                static_cast<unsigned int>(frame.step.p[0]),
                player->timestamp()
            };
            detectorThread->iDataQueue.push(std::move(iData));
        }

        /* Check for events */
        while ( detectorThread->oDataQueue.size() > 0 )
        {
            const auto& sharedEventItem = detectorThread->oDataQueue.pop1(1000);
            std::cout << ">>> [EVENT]: " << sharedEventItem->eventDescr << ":";
            for (const auto& inferOut : sharedEventItem->eventInferOuts)
            {
                std::cout << " " << inferOut.className;
            }
            std::cout << " at " << sharedEventItem->eventTimestamp << std::endl;
            if ( !sharedEventItem->eventDetailedFrame.empty() )
            {
                *detailedFramePtr = sharedEventItem->eventDetailedFrame;
            }
        }

        /* Display info */
        if ( record || display )
        {
            cv::Size detSize = objectDetector->settings()->detectorResolution();
            cv::resize(frame, out, detSize);
            cvt::drawAreaMaskNeg(out, objectDetector->settings()->areas(), 0.8);
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
                gui.imshow(out, record);
                if ( detailedFramePtr && !detailedFramePtr->empty() )
                {
                    cv::imshow("Detailed", *detailedFramePtr);
                }
            }
        }
    }

    if ( detectorThread->isRunning() )
    {
        detectorThread->finish();
    }
    detectorThread->detectorThread.join();
    
    std::cout << ">>> Main thread metrics (with waitKey): " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
