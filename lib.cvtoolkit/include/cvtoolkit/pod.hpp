#pragma once

#include <iostream>
#include <map>
#include <opencv2/core.hpp>

namespace cvt
{

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