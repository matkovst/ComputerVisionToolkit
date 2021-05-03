#pragma once

#include <iostream>
#include <map>
#include <opencv2/core.hpp>

namespace cvt
{

struct InferOut
{
    int classId;
    std::string className;
    float confidence;
    cv::Rect location;
    // std::vector<std::vector<cv::Point>> contour;
    cv::Mat objectMask;
};

using InferOuts = std::vector<InferOut>;

}