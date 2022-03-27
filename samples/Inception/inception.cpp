#include <signal.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/json.hpp"
#include "cvtoolkit/nn/inception.hpp"

/* The model and class names list can be downloaded from here:
    https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
*/

const static std::string SampleName = "inception";
const static std::string TitleName = "Inception";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        "{ @settings s    |        | path to settings }"
        ;


class InceptionSettings final : public cvt::JsonSettings, public cvt::JsonModelSettings
{
public:
    InceptionSettings(const std::string& jPath, const std::string& nodeName)
        : JsonSettings(jPath, nodeName)
        , JsonModelSettings(jPath, nodeName)
    {
        if ( m_jNodeSettings.empty() )
        {
            std::cerr << "[InceptionSettings] Could not find " << nodeName << " section" << std::endl;
            return;
        }
    }

    ~InceptionSettings() = default;

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
    std::string settingsPath = parser.get<std::string>("@settings");
    if (settingsPath.empty())
        settingsPath = parser.get<std::string>("@json");
    if (settingsPath.empty())
    {
        std::cerr << "[" << TitleName << "] Could not load settings. You should specify \"--settings\" flag." << std::endl;
        return -1;
    }
    const auto jSettings = std::make_shared<InceptionSettings>(settingsPath, SampleName);
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
        jSettings->modelRootDir(),
        jSettings->modelPath(),
        jSettings->modelConfigPath(),
        jSettings->modelClassesPath(),
        cv::Size(),
        jSettings->engine(),
        cvt::NeuralNetwork::Device::Cpu
    };
    const auto model = cvt::createInception(modelInitData);
    if ( !(model && model->initialized()) )
    {
        std::cerr << "[" << TitleName << "] Could not load model." 
                  << " Probably chosen engine \"" << jSettings->modelEngine()
                  << "\" is not supported." << std::endl;
        return -1;
    }

    // const cvt::NeuralNetwork::PreprocessData preprocessData = {
    //     jSettings->modelPreprocessingSize(),
    //     jSettings->modelPreprocessingColorConvMode(),
    //     jSettings->modelPreprocessingScale(),
    //     jSettings->modelPreprocessingMean(),
    //     jSettings->modelPreprocessingStd()
    // };

    // const cvt::NeuralNetwork::PostprocessData postprocessData = {
    //     jSettings->modelPostprocessingSotfmax()
    // };

    /* Main loop */
    cv::Mat frame, modelOut, out;

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

            model->Infer(frame, modelOut);
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
                cv::sort(modelOut, sortedOut, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
                cv::sortIdx(modelOut, sortedOutIdx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

                /* Draw top K predictions */
                const int k = 5;
                const cv::Point offset(0, -25);
                cv::Point org(5, out.rows - 35);
                for (int i = k-1; i >= 0; --i, org += offset)
                {
                    const int idx = sortedOutIdx.at<int>(i);
                    const float prob = sortedOut.at<float>(i) * 100.0f;
                    const std::string text = "#" + std::to_string(idx) + 
                                            " " + model->label(sortedOutIdx.at<int>(i)) + 
                                            " (" + cv::format("%.2f", prob) +
                                            "%)";
                    gui.putText(out, text, org, cv::Scalar(0, 255, 0));
                }
                
                gui.imshow(out, jSettings->record());
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
