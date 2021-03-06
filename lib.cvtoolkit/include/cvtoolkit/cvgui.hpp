#pragma once

#include <iostream>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvplayer.hpp>
#include <cvtoolkit/utils.hpp>

namespace cvt
{

class GUI final
{
public:
    enum Action
    {
        NONE,
        CONTINUE,
        CLOSE
    };

    GUI(const std::string& winName, const std::shared_ptr<OpenCVPlayer>& player, const std::shared_ptr<MetricMaster>& metrics = nullptr);

    ~GUI();

    int listenKeyboard();

    void imshow(cv::Mat& frame, bool record = false);

    cv::Rect drawTransparentBase(cv::Mat& frame, const std::string& underlyingText, int height, 
                                cv::Point org = {0, 0}, double opacity = 0.3) const;

    void putText(cv::Mat& frame, const std::string& text, cv::Point org,
                    cv::Scalar color = cv::Scalar(0, 255, 0)) const;

    const std::string& WinName() const noexcept;

private:
    std::string m_winName { "GUI" };
    std::shared_ptr<OpenCVPlayer> m_player;
    std::shared_ptr<MetricMaster> m_metrics;
    bool m_pause { false };

    /* Drawing stuff */
    const int m_fontFace { cv::FONT_HERSHEY_PLAIN };
    const double m_fontScale { 1.2 };
    const int m_thickness { 1 };
    const cv::Scalar m_primaryColor { cv::Scalar(0, 255, 0) };

    void drawTips(cv::Mat& frame);

    void drawTelemetry(cv::Mat& frame, bool record = false);
};

}