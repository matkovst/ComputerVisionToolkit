#pragma once

#include <iostream>
#include <map>
#include <opencv2/core.hpp>

namespace cvt
{

using Area = std::vector<cv::Point>;

using Areas = std::vector<Area>;

static Area createFullScreenArea(cv::Size2d scale = cv::Size2d(1.0, 1.0))
{
    Area area(4);
    area.emplace_back(0.0, 0.0);
    area.emplace_back(scale.width, 0.0);
    area.emplace_back(scale.width, scale.height);
    area.emplace_back(0.0, scale.height);
    return area;
}


/** @brief NN output struct
*/
struct InferOut
{
    int classId;

    std::string className;

    float confidence;

    cv::Rect location;

    /** (optional) Semantic object mask.
     */
    cv::Mat objectMask;
};

using InferOuts = std::vector<InferOut>;

}