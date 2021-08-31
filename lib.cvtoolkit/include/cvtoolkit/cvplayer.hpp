#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace cvt
{

const static std::vector<std::string> supportedImageContainers = { ".bmp", ".jpg", ".png", ".tif" };
const static std::vector<std::string> supportedVideoContainers = { ".avi", ".mkv", ".mp4", ".mov" };

class OpenCVPlayer final
{
public:
    enum InputType
    {
        NONE, //!
        IMAGE, //!< indicates that the input is an iamge
        VIDEO, //!< indicates that the input is a recorded video
        LIVESTREAM, //!< indicates that the input is a live stream
        WEBCAM //!< indicates that the input is a webcam stream
    };

    OpenCVPlayer(const std::string& input, double scaleFactor = 1.0);

    ~OpenCVPlayer();

    void read(cv::Mat& out);

    OpenCVPlayer& operator >> (cv::Mat& out);

    void write(const cv::Mat& frame);

    OpenCVPlayer& operator << (const cv::Mat& frame);

    double get(int propId) const;

    bool set(int propId, double value);

    void backToStart();

    const cv::Mat& frame0() const noexcept;

    const double fps() const noexcept;

    int frameNum() const noexcept;

    std::int64_t timestamp() const noexcept;

    int getInputType(const std::string& input) const;

private:
    cv::VideoCapture m_capture;
    cv::VideoWriter m_writer;
    std::string m_input { "0" };
    int m_inputType { InputType::NONE };
    double m_scaleFactor { 1.0 };
    bool m_doResize { false };
    cv::Mat m_frame0;
    double m_fps;
    int m_frameNum { 0 };
};

}