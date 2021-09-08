#include "cvtoolkit/cvgui.hpp"

#include <sstream>

namespace cvt
{

GUI::GUI(const std::string& winName, const std::shared_ptr<OpenCVPlayer>& player, const std::shared_ptr<MetricMaster>& metrics)
    : m_winName(winName)
    , m_player(player)
    , m_metrics(metrics)
{
}

GUI::~GUI()
{
    cv::destroyAllWindows();
}

int GUI::listenKeyboard()
{
    char key = (char)cv::waitKey(15);
    switch ( key )
    {
    case 'q':
    case 27 /* esc */:
        return Action::CLOSE;
        break;
    case 'p':
        m_pause = !m_pause;
        break;
    case 's':
        m_player->backToStart();
        break;
    case 32 /* space */:
        if ( !m_pause )
        {
            cv::Mat frame;
            for (int i = 0; i < m_player->fps(); ++i) 
            {
                *m_player >> frame;
            }
        }
        break;
    default:
        break;
    }
    
    if ( m_pause && key != 32 )
    {
        return Action::CONTINUE;
    }

    return Action::NONE;
}

void GUI::imshow(cv::Mat& frame, bool record)
{
    drawTips(frame);
    drawTelemetry(frame, record);
    cv::imshow(m_winName, frame);
}

const std::string& GUI::WinName() const noexcept
{
    return m_winName;
}

void GUI::drawTips(cv::Mat& frame)
{
    cv::String text = "Esc/q - exit, p - pause, space - 1 sec forward, s - to start";
    
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, m_fontFace,
                                m_fontScale, m_thickness, &baseline);

    baseline += m_thickness;
    
    cv::Point textOrg(0, frame.rows - baseline);
                
    cv::rectangle(frame, textOrg + cv::Point(0, baseline),
                textOrg + cv::Point(textSize.width, -(textSize.height + baseline)),
                cv::Scalar(0, 0, 0), -1);

    cv::putText(frame, text, textOrg, m_fontFace, m_fontScale,
                m_primaryColor, m_thickness, 8);
}

void GUI::drawTelemetry(cv::Mat& frame, bool record)
{
    if ( !m_metrics )
    {
        return;
    }

    const int ymargin = 22;
    int offset = ymargin;
    cv::putText(frame, "frame no: " + std::to_string(m_metrics->totalCalls()), cv::Point(0, offset), m_fontFace, m_fontScale,
                m_primaryColor, m_thickness, 8);
    offset += ymargin;
    cv::putText(frame, "curr. time (ms): " + std::to_string(m_metrics->currentTime()), cv::Point(0, offset), m_fontFace, m_fontScale,
                m_primaryColor, m_thickness, 8);
    offset += ymargin;
    cv::putText(frame, "avg. time (ms): " + std::to_string(m_metrics->avgTime()), cv::Point(0, offset), m_fontFace, m_fontScale,
                m_primaryColor, m_thickness, 8);
    offset += ymargin;
    if ( record )
    {
        cv::putText(frame, "Record", cv::Point(0, offset), m_fontFace, m_fontScale,
                    cv::Scalar(0, 0, 255), m_thickness, 8);
    }
}

}