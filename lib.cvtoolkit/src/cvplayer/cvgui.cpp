#include <cvtoolkit/cvgui.hpp>

namespace cvt
{

GUI::GUI(const std::string& winName, const std::shared_ptr<OpenCVPlayer>& player)
    : m_winName(winName)
    , m_player(player)
{
    cv::namedWindow(m_winName, 1);
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
        m_player->set(cv::CAP_PROP_POS_MSEC, 0);
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
    drawTips(frame, record);
    cv::imshow(m_winName, frame);
}

void GUI::drawTips(cv::Mat& frame, bool record)
{
    cv::String text = "Esc/q - exit, p - pause, space - 1 sec forward, s - to start";
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1.5;
    int thickness = 2;
    
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace,
                                fontScale, thickness, &baseline);

    baseline += thickness;
    
    cv::Point textOrg(0, frame.rows - baseline);
                
    cv::rectangle(frame, textOrg + cv::Point(0, baseline),
            textOrg + cv::Point(textSize.width, -(textSize.height + baseline)),
            cv::Scalar(0, 0, 0), -1);

    cv::putText(frame, text, textOrg, fontFace, fontScale,
        cv::Scalar(40, 220, 40), thickness, 8);

    if ( record )
    {
        cv::putText(frame, "Record", cv::Point(0, int(fontScale * baseline)), fontFace, fontScale,
            cv::Scalar(0, 0, 255), thickness, 8);
    }
}

}