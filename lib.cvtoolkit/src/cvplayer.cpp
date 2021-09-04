#include <cvtoolkit/cvplayer.hpp>

namespace cvt
{

OpenCVPlayer::OpenCVPlayer(const std::string& input, double scaleFactor)
    : m_input(input)
    , m_inputType(getInputType(input))
    , m_scaleFactor(scaleFactor)
    , m_doResize((scaleFactor != 1.0))
{
    /* Open stream */
    if ( m_inputType == InputType::WEBCAM )
    {
        m_capture.open(input[0] - '0');
    }
    else if ( m_inputType == InputType::VIDEO || m_inputType == InputType::LIVESTREAM )
    {
        m_capture.open(input);
    }
    else if ( m_inputType == InputType::IMAGE )
    {
        m_frame0 = cv::imread( input, cv::IMREAD_COLOR );
        if ( m_doResize )
        {
            cv::resize(m_frame0, m_frame0, cv::Size(0, 0), scaleFactor, scaleFactor);
        }
        return;
    }

    if( !m_capture.isOpened() )
    {
        std::cout << ">>> ERROR: Could not initialize capturing..." << std::endl;
        return;
    }

    m_capture >> m_frame0;
    if ( m_frame0.empty() )
    {
        std::cout << ">>> ERROR: Could not capture frame" << std::endl;
        return;
    }
    if ( m_doResize )
    {
        cv::resize(m_frame0, m_frame0, cv::Size(0, 0), scaleFactor, scaleFactor);
    }

    backToStart();

    m_fps = m_capture.get(cv::CAP_PROP_FPS);
    m_fps = ( m_fps > 120 ) ? 25 : m_fps; // Extremely high FPS appears in the case we cannot obtain real FPS
}

OpenCVPlayer::~OpenCVPlayer()
{
    if ( m_writer.isOpened() )
    {
        m_writer.release();
    }
    m_capture.release();
}

void OpenCVPlayer::read(cv::Mat& out)
{
    if ( m_inputType == InputType::IMAGE )
    {
        out = m_frame0.clone();
        return;
    }

    m_capture >> out;
    ++m_frameNum;
    
    if ( m_doResize && !out.empty() )
    {
        cv::resize(out, out, cv::Size(0, 0), m_scaleFactor, m_scaleFactor);
    }
}

OpenCVPlayer& OpenCVPlayer::operator >> (cv::Mat& out)
{
    read(out);
    return *this;
}

void OpenCVPlayer::write(const cv::Mat& frame)
{
    if ( m_inputType == InputType::IMAGE )
    {
        bool good = false;
        try
        {
            good = cv::imwrite("output.png", frame);
        }
        catch (const cv::Exception& ex)
        {
            std::cerr << ">>> ERROR: Exception converting image to PNG format: " << ex.what() << std::endl;
        }
        if ( !good ) std::cerr << ">>> ERROR: Could not save image." << std::endl;
        return;
    }

    if ( !m_writer.isOpened() )
    {
        m_writer.open("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 
                    m_fps, frame.size(), true);
    }
    m_writer.write(frame);
}

OpenCVPlayer& OpenCVPlayer::operator << (const cv::Mat& frame)
{
    write(frame);
    return *this;
}

double OpenCVPlayer::get(int propId) const
{
    return m_capture.get(propId);
}

bool OpenCVPlayer::set(int propId, double value)
{
    return m_capture.set(propId, value);
}

const cv::Mat& OpenCVPlayer::frame0() const noexcept
{
    return m_frame0;
}

const double OpenCVPlayer::fps() const noexcept
{
    return m_fps;
}

int OpenCVPlayer::frameNum() const noexcept
{
    return m_frameNum;
}

std::int64_t OpenCVPlayer::timestamp() const noexcept
{
    return 1000 * (m_frameNum / m_fps);
}

int OpenCVPlayer::getInputType(const std::string& input) const
{
    if( input.size() == 1 && isdigit(input[0]) )
    {
        return InputType::WEBCAM;
    }
    
    std::string ext = input.substr(input.length() - 4);
    if (std::find(supportedImageContainers.begin(), supportedImageContainers.end(), ext) != supportedImageContainers.end())
    {
        return InputType::IMAGE;
    }
    if (std::find(supportedVideoContainers.begin(), supportedVideoContainers.end(), ext) != supportedVideoContainers.end())
    {
        std::string protocol = input.substr(0, 4);
        if (protocol == "rtsp" || protocol == "http") 
        {
            return InputType::LIVESTREAM;
        }
        else
        {
            return InputType::VIDEO;
        }
    }

    return InputType::NONE;
}

void OpenCVPlayer::backToStart()
{
    m_capture.set(cv::CAP_PROP_POS_MSEC, 0);
    m_frameNum = 0;
}

}