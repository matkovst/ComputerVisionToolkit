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
#include "cvtoolkit/metrics.hpp"


const static std::string SampleName = "grey-world";
const static std::string TitleName = "Grey-world";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        ;

class GreyWorldSettings final : public cvt::Settings
{
public:
    GreyWorldSettings(const fs::path& jPath, const std::string& nodeName)
        : Settings(jPath, nodeName)
    {
        if ( m_jNodeSettings.empty() )
        {
            std::cerr << "[GreyWorldSettings] Could not find " << nodeName << " section" << std::endl;
            return;
        }

        if ( !m_jNodeSettings["Minkowski-norm"].empty() )
            m_MinkowskiNorm = static_cast<int>(m_jNodeSettings["Minkowski-norm"]);

        if ( !m_jNodeSettings["Gauss-ksize"].empty() )
            m_GaussKsize = static_cast<int>(m_jNodeSettings["Gauss-ksize"]);

        if ( !m_jNodeSettings["Gauss-sigmax"].empty() )
            m_GaussSigmax = static_cast<int>(m_jNodeSettings["Gauss-sigmax"]);

        if ( !m_jNodeSettings["specularity-threshold"].empty() )
            m_specularityThreshold = static_cast<int>(m_jNodeSettings["specularity-threshold"]);
    }

    ~GreyWorldSettings() = default;

    std::string summary() const noexcept
    {
        std::ostringstream oss;
        oss << std::endl << std::right
            << "\tSPECIFIC SETTINGS: " << std::endl
            << "\t\t- MinkowskiNorm = " << MinkowskiNorm() << std::endl
            << "\t\t- GaussKsize = " << GaussKsize() << std::endl
            << "\t\t- GaussSigmax = " << GaussSigmax() << std::endl
            << "\t\t- specularityThreshold = " << specularityThreshold() << std::endl
            << "\t\t- MinkowskiNorm = " << MinkowskiNorm();

        return Settings::summary() + oss.str();
    }


    int MinkowskiNorm() const noexcept
    {
        return m_MinkowskiNorm;
    }

    int GaussKsize() const noexcept
    {
        return m_GaussKsize;
    }

    int GaussSigmax() const noexcept
    {
        return m_GaussSigmax;
    }

    int specularityThreshold() const noexcept
    {
        return m_specularityThreshold;
    }

private:
    int m_MinkowskiNorm { 1 };
    int m_GaussKsize { 3 };
    int m_GaussSigmax { 1 };
    int m_specularityThreshold { 254 };
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
    std::shared_ptr<GreyWorldSettings> jSettings = std::make_shared<GreyWorldSettings>(jsonPath, SampleName);
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
    const int nPixels = static_cast<int>(player->frame0().total());
    const cv::Size GaussKsize(jSettings->GaussKsize(), jSettings->GaussKsize());
    cv::Mat frame32f;
    cv::Mat gray;
    cv::Mat specularitiesMask;
    cv::Mat lambertianMask;
    cv::Mat lambertianMaskBGR;
    int nSpecularPixels = 0;
    cv::Mat poweredFrame;
    const double maxMagn = 441.673;
    cv::Scalar lightSourceColor;
    double lightSourceColorMagn = 0.0;

    /* Main loop */
    cv::Mat frame, out;
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

            frame.convertTo(frame32f, CV_32FC3);

            // Remove specularities
            cv::cvtColor(frame32f, gray, cv::COLOR_BGR2GRAY);
            cv::threshold(gray, specularitiesMask, jSettings->specularityThreshold(), 255, cv::THRESH_BINARY);
            nSpecularPixels = static_cast<int>(cv::sum(specularitiesMask)[0] / 255);
            cv::bitwise_not(specularitiesMask, lambertianMask);
            cv::cvtColor(lambertianMask, lambertianMaskBGR, cv::COLOR_GRAY2BGR);
            cv::bitwise_and(frame32f, lambertianMaskBGR, frame32f);

            // Smooth frame
            cv::GaussianBlur(frame32f, frame32f, GaussKsize, jSettings->GaussSigmax());

            // Estimate light source
            if (1 == jSettings->MinkowskiNorm())
            {
                lightSourceColor = cv::mean(frame32f);
                lightSourceColorMagn = cv::norm(lightSourceColor, cv::NORM_L2);
            }
            else if (1 < jSettings->MinkowskiNorm())
            {
                const double p = static_cast<double>(jSettings->MinkowskiNorm());
                const double pInv = 1.0 / p;

                cv::pow(frame32f, p, poweredFrame);
                lightSourceColor = cv::sum(poweredFrame);
                cv::pow(lightSourceColor, pInv, lightSourceColor);
                lightSourceColor /= std::pow(nPixels, pInv);
                lightSourceColorMagn = cv::norm(lightSourceColor, cv::NORM_L2);
            }
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

            if ( jSettings->display() )
            {
                if ( out.channels() == 1 )
                {
                    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
                }
                
                /* Draw details */

                const auto frameSize = frame.size();
                const int lightIntensity = static_cast<int>((lightSourceColorMagn / maxMagn) * 100);
                const auto orgy = frameSize.height / 3;
                int orgyOffset = 0;
                const auto baseRect = gui.drawTransparentBase(out, "Light color intensity: 100%", 50, cv::Point(0, orgy - 22));

                gui.putText(out, "Specular pixels: " + std::to_string(nSpecularPixels), cv::Point(0, orgy));

                orgyOffset += 22;
                gui.putText(out, "Light color intensity: " + std::to_string(lightIntensity) + "%", cv::Point(0, orgy + orgyOffset));

                const int indicatorWidth = baseRect.width;
                const int lightLevel = static_cast<int>((lightSourceColorMagn / maxMagn) * indicatorWidth);
                const auto lightColor = (lightSourceColor / lightSourceColorMagn) * 255;
                cv::rectangle(out, cv::Rect(baseRect.x, orgy + orgyOffset + 10, indicatorWidth, 20), cv::Scalar(0, 0, 0), -1);
                cv::rectangle(out, cv::Rect(baseRect.x, orgy + orgyOffset + 10, lightLevel, 20), lightColor, -1);
                cv::rectangle(out, cv::Rect(baseRect.x, orgy + orgyOffset + 10, indicatorWidth, 20), cv::Scalar(0, 255, 0), 2);
                
                gui.imshow(out, jSettings->record());
                if ( detailedFramePtr && !detailedFramePtr->empty() )
                {
                    cv::imshow("Detailed", *detailedFramePtr);
                }
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
