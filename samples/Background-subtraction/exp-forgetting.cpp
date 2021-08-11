#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include <cvtoolkit/cvplayer.hpp>
#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/utils.hpp>


const static std::string WinName = "ExpForgetting Background subtractor";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        ;




/** @brief Exponential forgetting based Background/Foreground Segmentation Algorithm.

The class implements the exponential forgetting background subtraction.
 */
class BackgroundSubtractorEF final : public cv::BackgroundSubtractor
{
public:

    /** @brief constructor
    Initializes subtractor with alpha (default 0.02).
    */
    BackgroundSubtractorEF(float alpha = 0.02f)
        : m_alpha(alpha)
    {
    }
    
    /** @brief default destructor
    */
    ~BackgroundSubtractorEF() = default;

    /** @brief Computes a foreground mask.
     */
    CV_WRAP virtual void apply(cv::InputArray _image, cv::OutputArray _fgMask, double learningRate = -1) CV_OVERRIDE 
    {
        std::ignore = learningRate;

        ++m_frameNum;

        cv::Mat image = _image.getMat();
        cv::Mat image_32FC3;
        _fgMask.create( image.size(), CV_8UC1 );
        cv::Mat fgMask = _fgMask.getMat();
        cv::Mat fgMask_32FC1;

        if ( m_background_32FC3.empty() )
        {
            image.convertTo(m_background_32FC3, CV_32FC3);
            m_stdev2 = cv::Mat::zeros(m_background_32FC3.size(), m_background_32FC3.type());
            m_MahalanobisDist = cv::Mat::zeros(m_background_32FC3.size(), CV_32FC1);
            return;
        }
        
        /* Pre-process */
        image.convertTo(image_32FC3, CV_32FC3);
        cv::subtract(image_32FC3, m_background_32FC3, m_prevMeanDiff);

        /* Compute exp forgetting mean */
        cv::accumulateWeighted(image_32FC3, m_background_32FC3, m_alpha);

        /* Compute exp forgetting variance */
        cv::subtract(image_32FC3, m_background_32FC3, m_currMeanDiff);
        cv::pow(m_currMeanDiff, 2, m_currMeanDiff2);
        cv::accumulateWeighted(m_currMeanDiff2, m_stdev2, m_alpha);

        // /* Compute variance via Welford's online algorithm */
        // if ( m_M2n.empty() ) 
        // {
        //     m_M2n = cv::Mat::zeros(image_32FC3.size(), image_32FC3.type());
        // }
        // cv::accumulateProduct(m_prevMeanDiff, m_currMeanDiff, m_M2n);
        // m_stdev2 = m_M2n / m_frameNum;

        /* Compute Mahalanobis distance btw Image and Background */
        cv::divide(m_currMeanDiff2, m_stdev2, m_MahalanobisDistVec);
        cv::transform(m_MahalanobisDistVec, m_MahalanobisDist, cv::Matx13f(1, 1, 1));
        cv::sqrt(m_MahalanobisDist, m_MahalanobisDist);

        // /* Segment fg/bg with absdiff */
        // cv::Mat diff_32FC3;
        // cv::absdiff(image_32FC3, m_background_32FC3, diff_32FC3);
        // cv::cvtColor(diff_32FC3, fgMask_32FC1, cv::COLOR_BGR2GRAY);
        // fgMask_32FC1.convertTo(fgMask, CV_8UC1);

        /* Segment fg/bg with Mahalanobis dist */
        m_MahalanobisDist.convertTo(fgMask, CV_8UC1);
    }

    /** @brief Returns a background image in CV_8UC3 type.

    @param backgroundImage The output background image.

    @note Sometimes the background image can be very blurry, as it contain the average background
    statistics.
     */
    CV_WRAP virtual void getBackgroundImage(cv::OutputArray _backgroundImage) const 
    {
        m_background_32FC3.convertTo(_backgroundImage, CV_8UC3);
    }

    /** @brief Returns a stdev2 image in CV_8UC3 type.

    @param backgroundImage The output stdev2 image.
     */
    CV_WRAP virtual void getStdev2Image(cv::OutputArray _stdev2Image) const 
    {
        m_stdev2.convertTo(_stdev2Image, CV_8UC3);
    }

    /** @brief Returns a mahalanobis distance image in CV_8UC3 type.

    @param backgroundImage The output stdev2 image.
     */
    CV_WRAP virtual void getMahalanobisDistImage(cv::OutputArray _mahalanobisDistImage) const 
    {
        m_MahalanobisDist.convertTo(_mahalanobisDistImage, CV_8UC3);
    }

    /** @brief Returns the alpha
    */
    CV_WRAP virtual float getAlpha() const 
    {
        return m_alpha;
    }
    /** @brief Sets the alpha
    */
    CV_WRAP virtual void setAlpha(float alpha) 
    {
        m_alpha = alpha;
    }

private:
    unsigned long long m_frameNum { 0 };
    float m_alpha { 0.02f };
    cv::Mat m_background_32FC3;
    cv::Mat m_prevMeanDiff;
    cv::Mat m_currMeanDiff;
    cv::Mat m_currMeanDiff2;
    cv::Mat m_M2n;
    cv::Mat m_stdev2;
    cv::Mat m_MahalanobisDistVec;
    cv::Mat m_MahalanobisDist;
};

/** @brief Creates EF Background Subtractor
 */
cv::Ptr<BackgroundSubtractorEF> createBackgroundSubtractorEF(float alpha = 0.02f)
{
    return cv::makePtr<BackgroundSubtractorEF>(alpha);
}



cv::Ptr<BackgroundSubtractorEF> bgSubtractor;
int Alpha = 2;
int Thresh = 0;

static void onAlphaChanged( int, void* ) { }

static void onThreshChanged( int, void* ) { }

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

    cv::createTrackbar( "Alpha", gui.WinName(), &Alpha, 1000, onAlphaChanged );
    onAlphaChanged( Alpha, 0 );
    cv::createTrackbar( "Thresh", gui.WinName(), &Thresh, 255, onThreshChanged );
    onThreshChanged( Thresh, 0 );

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << player->frame0().size() << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;

    bgSubtractor = createBackgroundSubtractorEF(Alpha / 1000.f);

    /* Main loop */
    bool loop = true;
    cv::Mat frame, frame_32FC3, out;
    cv::Mat background_32FC3, background_8UC3;
    cv::Mat fgMask = cv::Mat::zeros(player->frame0().size(), player->frame0().type());
    cv::Mat blank = cv::Mat::zeros(player->frame0().size(), player->frame0().type());
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

            bgSubtractor->setAlpha(Alpha / 1000.f);
            bgSubtractor->apply(frame, fgMask);
            if ( Thresh > 0 )
            {
                cv::threshold(fgMask, fgMask, Thresh, 255, cv::THRESH_BINARY);
            }
        }

        /* Display info */
        if ( fgMask.channels() == 1 )
        {
            cv::cvtColor(fgMask, fgMask, cv::COLOR_GRAY2BGR);
        }
        cv::Mat stdev2;
        bgSubtractor->getStdev2Image(stdev2);
        // cvt::hstack2images( fgMask, frame, out );
        cvt::stack4images( fgMask, frame, stdev2, blank, out );
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
