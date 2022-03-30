#include <signal.h>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cvtoolkit/cvplayer.hpp"
#include "cvtoolkit/cvgui.hpp"
#include "cvtoolkit/settings.hpp"
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/metrics.hpp"


const static std::string SampleName = "smoothing";
const static std::string TitleName = "Smoothing";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        ;

class SmoothingSettings final : public cvt::Settings
{
public:
    SmoothingSettings(const fs::path& jPath, const std::string& nodeName)
        : Settings(jPath, nodeName)
    {
        if ( m_jNodeSettings.empty() )
        {
            std::cerr << "[SmoothingSettings] Could not find " << nodeName << " section" << std::endl;
            return;
        }

        if ( !m_jNodeSettings["ksize"].empty() )
            m_ksize = static_cast<int>(m_jNodeSettings["ksize"]);
    }

    ~SmoothingSettings() = default;

    std::string summary() const noexcept
    {
        std::ostringstream oss;
        oss << std::endl << std::right
            << "\tSPECIFIC SETTINGS: " << std::endl
            << "\t\t- ksize = " << ksize();

        return Settings::summary() + oss.str();
    }


    int ksize() const noexcept
    {
        return m_ksize;
    }

private:
    int m_ksize { 3 };
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
    std::shared_ptr<SmoothingSettings> jSettings = std::make_shared<SmoothingSettings>(jsonPath, SampleName);
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
    const cv::Size ksize(jSettings->ksize(), jSettings->ksize());
    cv::Mat smoothedGauss;
    cv::Mat smoothedMedian;
    cv::Mat smoothedBlur;

    /* Main loop */
    cv::Mat frame, out;
    cv::Mat outSmoothedBlur = cv::Mat::zeros(player->frame0().size(), CV_8UC3);
    cv::Mat outSmoothedGauss = cv::Mat::zeros(player->frame0().size(), CV_8UC3);
    cv::Mat outSmoothedMedian = cv::Mat::zeros(player->frame0().size(), CV_8UC3);
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

            /* Box blur */
            cv::blur(frame, smoothedBlur, ksize);

            /* Gaussian blur */
            cv::GaussianBlur(frame, smoothedGauss, ksize, 1);
            
            /* Median blur */
            cv::medianBlur(frame, smoothedMedian, jSettings->ksize());
        }

        /* Display info */
        if ( jSettings->record() || jSettings->display() )
        {
            if ( frame.rows >= 1080 && frame.cols >= 1920 )
            {
                cv::resize(smoothedBlur, outSmoothedBlur, cv::Size(), 0.5, 0.5);
                cv::resize(smoothedGauss, outSmoothedGauss, cv::Size(), 0.5, 0.5);
                cv::resize(smoothedMedian, outSmoothedMedian, cv::Size(), 0.5, 0.5);
            }
            else
            {
                outSmoothedBlur = smoothedBlur.clone();
                outSmoothedGauss = smoothedGauss.clone();
                outSmoothedMedian = smoothedMedian.clone();
            }

            if ( jSettings->display() )
            {
                if ( outSmoothedBlur.channels() == 1 )
                    cv::cvtColor(outSmoothedBlur, outSmoothedBlur, cv::COLOR_GRAY2BGR);
                if ( outSmoothedGauss.channels() == 1 )
                    cv::cvtColor(outSmoothedGauss, outSmoothedGauss, cv::COLOR_GRAY2BGR);
                if ( outSmoothedMedian.channels() == 1 )
                    cv::cvtColor(outSmoothedMedian, outSmoothedMedian, cv::COLOR_GRAY2BGR);
                
                /* Draw details */

                gui.putText(outSmoothedBlur, "Box blur", cv::Point(5, 20));
                gui.putText(outSmoothedGauss, "Gaussian blur", cv::Point(5, 20));
                gui.putText(outSmoothedMedian, "Median blur", cv::Point(5, 20));
                
                cvt::stack4images(frame, outSmoothedBlur, outSmoothedGauss, outSmoothedMedian, out);
                gui.imshow(out, jSettings->record());
            }

            if ( jSettings->record() )
            {
                *player << out;
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
