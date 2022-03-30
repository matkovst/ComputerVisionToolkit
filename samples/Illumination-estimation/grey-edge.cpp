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


const static std::string SampleName = "grey-edge";
const static std::string TitleName = "Grey-edge";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        ;

class GreyEdgeSettings final : public cvt::JsonSettings
{
public:
    GreyEdgeSettings(const fs::path& jPath, const std::string& nodeName)
        : JsonSettings(jPath, nodeName)
    {
        if ( m_jNodeSettings.empty() )
        {
            std::cerr << "[GreyEdgeSettings] Could not find " << nodeName << " section" << std::endl;
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

        if ( !m_jNodeSettings["derivative-order"].empty() )
        {
            m_derivativeOrder = static_cast<int>(m_jNodeSettings["derivative-order"]);
            if (m_derivativeOrder > 2)
                m_derivativeOrder = 2;
            else if (m_derivativeOrder < 1)
                m_derivativeOrder = 1;
        }
    }

    ~GreyEdgeSettings() = default;

    std::string summary() const noexcept
    {
        std::ostringstream oss;
        oss << std::endl << std::right
            << "\tSPECIFIC SETTINGS: " << std::endl
            << "\t\t- MinkowskiNorm = " << MinkowskiNorm() << std::endl
            << "\t\t- GaussKsize = " << GaussKsize() << std::endl
            << "\t\t- GaussSigmax = " << GaussSigmax() << std::endl
            << "\t\t- specularityThreshold = " << specularityThreshold() << std::endl
            << "\t\t- MinkowskiNorm = " << MinkowskiNorm() << std::endl
            << "\t\t- derivativeOrder = " << derivativeOrder();

        return JsonSettings::summary() + oss.str();
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

    int derivativeOrder() const noexcept
    {
        return m_derivativeOrder;
    }

private:
    int m_MinkowskiNorm { 1 };
    int m_GaussKsize { 3 };
    int m_GaussSigmax { 1 };
    int m_specularityThreshold { 254 };
    int m_derivativeOrder { 1 };
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
    std::shared_ptr<GreyEdgeSettings> jSettings = std::make_shared<GreyEdgeSettings>(jsonPath, SampleName);
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
    cv::Mat gradX, gradY, absGradX, absGradY;
    cv::Mat grad, grad32f;
    cv::Mat poweredGrad;
    const double maxMagn = 441.673;
    cv::Scalar lightSourceColor;
    double lightSourceColorMagn = 0.0;

    /* Main loop */
    cv::Mat frame, out;
    cv::Mat outDetailed = cv::Mat::zeros(player->frame0().size(), CV_8UC3);
    cv::Mat outGrad = cv::Mat::zeros(player->frame0().size(), CV_8UC3);
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

            // Smooth frame
            cv::GaussianBlur(frame32f, frame32f, GaussKsize, jSettings->GaussSigmax());

            // Compute the first gradient ∇I(x,y) (or the second gradient ∇(∇I(x,y)))
            if (1 == jSettings->derivativeOrder())
            {
                cv::Sobel(frame32f, gradX, CV_32F, 1, 0);
                cv::Sobel(frame32f, gradY, CV_32F, 0, 1);
                cv::convertScaleAbs(gradX, absGradX);
                cv::convertScaleAbs(gradY, absGradY);
                cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);
            }
            else if (2 == jSettings->derivativeOrder())
            {
                cv::Laplacian(frame32f, grad, CV_32F);
                cv::convertScaleAbs(grad, grad);
            }

            // Remove specularities
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::threshold(gray, specularitiesMask, jSettings->specularityThreshold(), 255, cv::THRESH_BINARY);
            nSpecularPixels = static_cast<int>(cv::sum(specularitiesMask)[0] / 255);
            cv::bitwise_not(specularitiesMask, lambertianMask);
            cv::cvtColor(lambertianMask, lambertianMaskBGR, cv::COLOR_GRAY2BGR);
            cv::bitwise_and(grad, lambertianMaskBGR, grad);

            // Estimate light source
            grad.convertTo(grad32f, CV_32FC3);
            if (1 == jSettings->MinkowskiNorm())
            {
                lightSourceColor = cv::mean(grad);
                lightSourceColorMagn = cv::norm(lightSourceColor, cv::NORM_L2);
            }
            else if (1 < jSettings->MinkowskiNorm())
            {
                const double p = static_cast<double>(jSettings->MinkowskiNorm());
                const double pInv = 1.0 / p;

                cv::pow(grad32f, p, poweredGrad);
                lightSourceColor = cv::sum(poweredGrad);
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
                cv::resize(frame, outDetailed, cv::Size(), 0.5, 0.5);
                cv::resize(frame, outGrad, cv::Size(), 0.5, 0.5);
            }
            else
            {
                outDetailed = frame.clone();
                outGrad = grad.clone();
            }

            if ( jSettings->display() )
            {
                if ( outDetailed.channels() == 1 )
                {
                    cv::cvtColor(outDetailed, outDetailed, cv::COLOR_GRAY2BGR);
                }
                if ( outGrad.channels() == 1 )
                {
                    cv::cvtColor(outGrad, outGrad, cv::COLOR_GRAY2BGR);
                }
                
                /* Draw details */

                const auto frameSize = frame.size();
                const int lightIntensity = static_cast<int>((lightSourceColorMagn / maxMagn) * 100);
                const auto orgy = frameSize.height / 3;
                int orgyOffset = 0;
                const auto baseRect = gui.drawTransparentBase(outDetailed, "Light color intensity: 100%", 50, cv::Point(0, orgy - 22));

                gui.putText(outDetailed, "Specular pixels: " + std::to_string(nSpecularPixels), cv::Point(0, orgy));

                orgyOffset += 22;
                gui.putText(outDetailed, "Light color intensity: " + std::to_string(lightIntensity) + "%", cv::Point(0, orgy + orgyOffset));

                const int indicatorWidth = baseRect.width;
                const int lightLevel = static_cast<int>((lightSourceColorMagn / maxMagn) * indicatorWidth);
                const auto lightColor = (lightSourceColor / lightSourceColorMagn) * 255;
                cv::rectangle(outDetailed, cv::Rect(baseRect.x, orgy + orgyOffset + 10, indicatorWidth, 20), cv::Scalar(0, 0, 0), -1);
                cv::rectangle(outDetailed, cv::Rect(baseRect.x, orgy + orgyOffset + 10, lightLevel, 20), lightColor, -1);
                cv::rectangle(outDetailed, cv::Rect(baseRect.x, orgy + orgyOffset + 10, indicatorWidth, 20), cv::Scalar(0, 255, 0), 2);
                
                cvt::hstack2images(outDetailed, outGrad, out);
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
