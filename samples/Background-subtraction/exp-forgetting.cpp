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




/** @brief Exponential forgetting - based Background/Foreground Segmentation Algorithm.

The class implements the exponential forgetting background subtraction.
 */
class BackgroundSubtractorEF final : public cv::BackgroundSubtractor
{
public:

    BackgroundSubtractorEF(float alpha = 0.02f)
        : m_alpha(alpha)
    {
    }

    ~BackgroundSubtractorEF() = default;

    /** @brief Computes a foreground mask.
     */
    CV_WRAP virtual void apply(cv::InputArray _image, cv::OutputArray _fgMask, double learningRate=-1) CV_OVERRIDE 
    {
        std::ignore = learningRate;

        cv::Mat image = _image.getMat();
        cv::Mat image_32FC3;
        _fgMask.create( image.size(), CV_8UC1 );
        cv::Mat fgMask = _fgMask.getMat();
        cv::Mat fgMask_32FC1;

        if ( m_background.empty() )
        {
            image.convertTo(m_background, CV_32FC3);
        }
        else
        {
            image.convertTo(image_32FC3, CV_32FC3);
            m_background = (1 - m_alpha) * m_background + m_alpha * image_32FC3;

            cv::Mat diff_32FC3;
            cv::absdiff(image_32FC3, m_background, diff_32FC3);
            cv::cvtColor(diff_32FC3, fgMask_32FC1, cv::COLOR_BGR2GRAY);
            fgMask_32FC1.convertTo(fgMask, CV_8UC1);
        }
    }

    /** @brief Computes a background image.

    @param backgroundImage The output background image.

    @note Sometimes the background image can be very blurry, as it contain the average background
    statistics.
     */
    CV_WRAP virtual void getBackgroundImage(cv::OutputArray _backgroundImage) const 
    {
        m_background.convertTo(_backgroundImage, CV_8UC3);
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
    float m_alpha { 0.02f };
    cv::Mat m_background;
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
        cvt::hstack2images( fgMask, frame, out );
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
