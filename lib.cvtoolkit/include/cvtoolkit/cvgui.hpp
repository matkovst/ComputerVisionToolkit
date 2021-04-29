#pragma once

#include <iostream>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvplayer.hpp>

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

    GUI(const std::string& winName, const std::shared_ptr<OpenCVPlayer>& player);

    ~GUI();

    int listenKeyboard();

    void imshow(cv::Mat& frame, bool record = false);

private:
    std::string m_winName { "GUI" };
    std::shared_ptr<OpenCVPlayer> m_player;
    bool m_pause { false };

    void drawTips(cv::Mat& frame, bool record = false);
};

}