#include <signal.h>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/metrics.hpp"
#include "cvtoolkit/settings.hpp"
#include "cvtoolkit/nn/efficientnet.hpp"


const static std::string SampleName = "efficientnet";
const static std::string TitleName = "EfficientNet";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        ;

class EfficientNetSettings final : public cvt::JsonSettings, public cvt::JsonModelSettings
{
public:
    EfficientNetSettings(const fs::path& jPath, const std::string& nodeName)
        : JsonSettings(jPath, nodeName)
        , JsonModelSettings(jPath, nodeName)
    {
        if ( m_jNodeSettings.empty() )
        {
            std::cerr << "[EfficientNetSettings] Could not find " << nodeName << " section" << std::endl;
            return;
        }
    }

    ~EfficientNetSettings() = default;

    std::string summary() const noexcept
    {
        return JsonSettings::summary() + JsonModelSettings::summary();
    }
};


int main(int argc, char** argv)
{
    std::cout << ">>> Program started. Have fun!" << std::endl;
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
    const std::string jsonPath = parser.get<std::string>("@json");
    std::shared_ptr<EfficientNetSettings> jSettings = std::make_shared<EfficientNetSettings>(jsonPath, SampleName);
    std::cout << "[" << TitleName << "]" << jSettings->summary() << std::endl;

    /* Open stream */
    std::shared_ptr<cvt::OpenCVPlayer> player = std::make_shared<cvt::OpenCVPlayer>(jSettings->input(), 
                                                                                    jSettings->inputSize());

    /* Create GUI */
    auto metrics = std::make_shared<cvt::MetricMaster>();
    cvt::GUI gui(TitleName, player, metrics);
    const cv::Size imSize = player->frame0().size();
    const double fps = player->fps();

    /* Task-specific declarations */
    cvt::NeuralNetwork::InitializeData modelInitData
    {
        "",
        jSettings->modelPath(),
        jSettings->modelConfigPath(),
        jSettings->modelClassesPath(),
        cv::Size(),
        jSettings->engine(),
        cvt::NeuralNetwork::Device::Cpu
    };
    std::shared_ptr<cvt::NeuralNetwork> model = cvt::createEfficientNet(modelInitData);
    if ( !(model && model->initialized()) )
    {
        std::cerr << "[" << TitleName << "] Could not load model." 
                  << " Probably chosen engine \"" << jSettings->modelEngine()
                  << "\" is not supported." << std::endl;
        return -1;
    }
    const auto labelsMap = cvt::loadJsonLabelsMap(jSettings->modelClassesPath());

    const cvt::NeuralNetwork::PreprocessData preprocessData = {
        jSettings->modelPreprocessingSize(),
        jSettings->modelPreprocessingColorConvMode(),
        jSettings->modelPreprocessingScale(),
        jSettings->modelPreprocessingMean(),
        jSettings->modelPreprocessingStd()
    };

    const cvt::NeuralNetwork::PostprocessData postprocessData = {
        jSettings->modelPostprocessingSotfmax()
    };

    /* Main loop */
    cv::Mat frame, modelOutput, out;
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

            model->Infer(frame, modelOutput, preprocessData, postprocessData);
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

                /* Sort predictions by desc */
                cv::Mat sortedOut, sortedOutIdx;
                cv::sort(modelOutput, sortedOut, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
                cv::sortIdx(modelOutput, sortedOutIdx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

                /* Draw top K predictions */
                const int k = 5;
                const cv::Point offset(0, -25);
                cv::Point org(5, out.rows - 35);
                for (int i = k-1; i >= 0; --i, org += offset)
                {
                    const int idx = sortedOutIdx.at<int>(i);
                    const float prob = sortedOut.at<float>(i) * 100.0f;
                    const std::string text = "#" + std::to_string(idx) + 
                                            " " + labelsMap.at(idx) + 
                                            " (" + cv::format("%.2f", prob) +
                                            "%)";
                    gui.putText(out, text, org, cv::Scalar(0, 255, 0));
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
    
    std::cout << ">>> Inference metrics: " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
