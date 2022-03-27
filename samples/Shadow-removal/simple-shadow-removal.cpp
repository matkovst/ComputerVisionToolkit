#include <signal.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/settings.hpp>
#include <cvtoolkit/utils.hpp>


const static std::string SampleName = "simple-shadow-removal";
const static std::string TitleName = "Simple-shadow-removal";


template <typename T>
static std::vector<T> linspace(T a, T b, size_t N) 
{
    T h = (b - a) / static_cast<T>(N - 1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}


class SimpleShadowTerminatorSettings final : public cvt::JsonSettings
{
public:
    SimpleShadowTerminatorSettings(const std::string& jPath, const std::string& nodeName)
        : JsonSettings(jPath, nodeName)
    {
        if ( m_jNodeSettings.empty() )
        {
            std::cerr << "[SimpleShadowTerminatorSettings] Could not find " << nodeName << " section" << std::endl;
            return;
        }

        if ( !m_jNodeSettings["n-angles"].empty() )
            m_nAngles = static_cast<int>(m_jNodeSettings["n-angles"]);
    }

    ~SimpleShadowTerminatorSettings() = default;

    std::string summary() const noexcept
    {
        std::ostringstream oss;
        oss << std::endl << std::right
            << "\tSPECIFIC SETTINGS: " << std::endl
            << "\t\t- nAngles = " << nAngles();

        return JsonSettings::summary() + oss.str();
    }


    int nAngles() const noexcept
    {
        return m_nAngles;
    }

private:
    int m_nAngles { 181 };
};


class SimpleShadowTerminator final
{
public:

    SimpleShadowTerminator(const std::shared_ptr<SimpleShadowTerminatorSettings>& settings)
        : m_settings(settings)
        , m_logPlanes(3)
        , m_ChiPlanes(2)
        , m_U(1.0 / sqrt(2.0), -1.0 / sqrt(2.0), 0.0, 1.0 / sqrt(6.0), 1.0 / sqrt(6.0), -2.0 / sqrt(6.0))
    {}

    ~SimpleShadowTerminator() = default;

    void terminate(const cv::Mat& img)
    {
        // Prepare input
        img.convertTo(m_img64, CV_64FC3);
        cv::GaussianBlur(m_img64, m_img64, cv::Size(3, 3), 1);

        // Infer intrinsic image
        convertToLogChromacitySpace();
        fitProjection();

        // Get projected 1D image
        m_invariant1D = std::cos(minEntropyAngle()) * m_ChiPlanes[0] + std::sin(minEntropyAngle()) * m_ChiPlanes[1];
        cv::exp(m_invariant1D, m_invariant1D);
    }

    double minEntropyAngle() const noexcept
    {
        return m_minEntropyAngle;
    }

    void invariant1D(cv::Mat& out)
    {
        out = m_invariant1D.clone();
    }

private:
    const std::shared_ptr<SimpleShadowTerminatorSettings>& m_settings;
    const cv::Matx23d m_U; // orthogonal matrix
    cv::Mat m_img64; // smoothed double-like image
    std::vector<cv::Mat> m_img64Planes;
    std::vector<cv::Mat> m_logPlanes;
    std::vector<cv::Mat> m_ChiPlanes;
    cv::Mat m_Rho; // log chromaticity on a plane
    cv::Mat m_Chi; // 2D log chromaticity coords
    cv::Mat m_invariant1D; // resulting 1D (gray-scale) illuminant invariant representation
    double m_minEntropyAngle { 0.0 };

    void convertToLogChromacitySpace()
    {
        cv::max(m_img64, std::numeric_limits<double>::min(), m_img64); // remove zeros to avoid inf when taking logarithm
        cv::split(m_img64, m_img64Planes);

        cv::Mat gMean;
        cv::multiply(m_img64Planes[0], m_img64Planes[1], gMean);
        cv::multiply(gMean, m_img64Planes[2], gMean);
        cv::pow(gMean, 1.0 / 3.0, gMean);
        cv::max(gMean, std::numeric_limits<double>::min(), gMean);

        cv::log(m_img64Planes[2] / gMean, m_logPlanes[0]); // R
        cv::log(m_img64Planes[1] / gMean, m_logPlanes[1]); // G
        cv::log(m_img64Planes[0] / gMean, m_logPlanes[2]); // B
        cv::merge(m_logPlanes, m_Rho);
    }

    /*
        Fits invariant projection by minimizing entropy
    */
    void fitProjection()
    {
        const int N = (m_img64.rows * m_img64.cols);
        m_Chi = projectOntoPlate(m_Rho, m_U);
        cv::split(m_Chi, m_ChiPlanes);

        std::vector<double> radians = linspace<double>(0.0, M_PI, m_settings->nAngles());
        std::vector<double> entropies;
        entropies.reserve(m_settings->nAngles());
        for (const auto rad : radians)
        {
            const cv::Mat I = std::cos(rad) * m_ChiPlanes[0] + std::sin(rad) * m_ChiPlanes[1];
            cv::Scalar IMean = 0.0;
            cv::Scalar IStd = 0.0;
            cv::meanStdDev(I, IMean, IStd);
            const double lbound = IMean[0] + 3.0 * (-IStd[0]);
            const double rbound = IMean[0] + 3.0 * (+IStd[0]);

            cv::Mat mask, IClipped;
            cv::inRange(I, cv::Scalar(lbound), cv::Scalar(rbound), mask);
            I.copyTo(IClipped, mask);

            const double binWidth = 3.5 * IStd[0] * std::pow(N, -1.0 / 3.0);
            entropies.push_back( ShannonEntropy(IClipped, binWidth) );
        }

        auto minEntropyIt = std::min_element(entropies.begin(), entropies.end());
        int minEntropyIdx = static_cast<int>(minEntropyIt - entropies.begin());
        m_minEntropyAngle = radians.at(minEntropyIdx) + M_PI_2;
    }

    cv::Mat projectOntoPlate(const cv::Mat& img, cv::InputArray orthMatrix)
    {
        cv::Mat orthMat = orthMatrix.getMat();

        cv::Mat out = cv::Mat::zeros(img.rows, img.cols, CV_64FC2);
        cv::parallel_for_(cv::Range(0, img.rows * img.cols), [&](const cv::Range& range)
        {
            for ( int r = range.start; r < range.end; ++r )
            {
                const int i = r / img.cols;
                const int j = r % img.cols;

                auto& pixel = img.at<cv::Vec3d>(i, j);
                const double chi1 = pixel[0] * (1.0 / sqrt(2.0)) + pixel[1] * (-1.0 / sqrt(2.0)) + pixel[2] * 0.0;
                const double chi2 = pixel[0] * (1.0 / sqrt(6.0)) + pixel[1] * (1.0 / sqrt(6.0)) + pixel[2] * (-2.0 / sqrt(6.0));

                out.at<cv::Vec2d>(i, j) = {chi1, chi2};
            }
        });

        return out;
    }

    static double ShannonEntropy(const cv::Mat& img, double bandwidth = 1.0)
    {
        const int N = (img.rows * img.cols);
        double minVal, maxVal;
        cv::minMaxIdx(img, &minVal, &maxVal);
        const int nBins = static_cast<int>(std::round((maxVal - minVal) / bandwidth));

        cv::Mat P;
        const float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal) };
        const float* histRange[] = { range };
        cv::Mat img32;
        img.convertTo(img32, CV_32F);
        cv::calcHist(&img32, 1, 0, cv::Mat(), P, 1, &nBins, histRange);
        P /= N;

        double entropy = 0.0;
        for (int i = 0; i < P.rows; ++i)
        {
            const double pixel = static_cast<double>(P.at<float>(i));
            if (pixel < std::numeric_limits<double>::min())
                continue;
            entropy += (pixel * std::log2(pixel));
        }
        entropy = -entropy;

        return entropy;
    }
};


const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        ;

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
    std::shared_ptr<SimpleShadowTerminatorSettings> jSettings = 
                            std::make_shared<SimpleShadowTerminatorSettings>(jsonPath, SampleName);
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
    cv::Mat invariantGrayImage;
    std::shared_ptr<SimpleShadowTerminator> terminator = std::make_shared<SimpleShadowTerminator>(jSettings);
    if ( !terminator )
    {
        std::cerr << ">>> [ERROR] Could not create SimpleShadowTerminator" << std::endl;
        return -1;
    }

    /* Main loop */
    cv::Mat frame, outInvariantGrayImage64, outInvariantGrayImage;
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

            terminator->terminate(frame);
            terminator->invariant1D(invariantGrayImage);
        }

        /* Display info */
        if ( jSettings->record() || jSettings->display() )
        {
            if ( invariantGrayImage.rows >= 1080 && invariantGrayImage.cols >= 1920 )
            {
                cv::resize(invariantGrayImage, outInvariantGrayImage64, cv::Size(), 0.5, 0.5);
            }
            else
            {
                outInvariantGrayImage64 = invariantGrayImage.clone();
            }
            cv::normalize(outInvariantGrayImage64, outInvariantGrayImage64, 0, 1, cv::NORM_MINMAX);
            outInvariantGrayImage64.convertTo(outInvariantGrayImage, CV_8UC1, 255);

            if ( jSettings->record() )
            {
                *player << outInvariantGrayImage;
            }

            if ( jSettings->display() )
            {
                if ( outInvariantGrayImage.channels() == 1 )
                {
                    cv::cvtColor(outInvariantGrayImage, outInvariantGrayImage, cv::COLOR_GRAY2BGR);
                }
                
                gui.imshow(outInvariantGrayImage, jSettings->record());
                std::cout << ">>> Angle: " << terminator->minEntropyAngle() << " rad (" << (terminator->minEntropyAngle() * 180 / M_PI) << " deg)" << std::endl;
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
