#pragma once

#include <iostream>
#include <iomanip>
#include <filesystem>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "types.hpp"
#include "metrics.hpp"
#include "trigger.hpp"

namespace fs = std::filesystem;

namespace cvt
{

/*! @brief Calculates area of an area.
 */
int sqArea( Area area );

/*! @brief Calculates area of areas.
 */
int totalSqArea( Areas areas );

/*! @brief Checks if vector is valid.

    @param u vector

    @return Says whether vector is valid or not
 */
bool isFlowCorrect( cv::Point2f u );


void drawMotionField( const cv::Mat_<cv::Point2f>& optflow, cv::Mat& out, int stride );

void drawInferOut( cv::Mat& frame, const InferOut& inferOut, cv::Scalar color = cv::Scalar(0, 0, 255), bool drawObjectMask = true, bool drawLabel = true );

void drawInferOuts( cv::Mat& frame, const InferOuts& inferOuts, cv::Scalar color = cv::Scalar(0, 0, 255), bool drawObjectMask = true, bool drawLabels = true );

void drawAreaMask( cv::Mat& frame, const Areas& areas, double opacity = 0.85 );

void drawAreaMaskNeg( cv::Mat& frame, const Areas& areas, double opacity = 0.85 );


void hstack2images( const cv::Mat& l, const cv::Mat& r, cv::Mat& out );

void stack4images( const cv::Mat& lt, const cv::Mat& rt, const cv::Mat& lb, const cv::Mat& rb, cv::Mat& out );

cv::Size parseResolution(const std::string& resol);

json makeJsonObject(const std::string& jPath);

std::pair<bool, std::string> verifyFile(const fs::path& path);

template <typename T>
static T clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

template<typename T>
static std::string getClassName(T, const std::string& namespace_ = "")
{
    const std::string fullName = typeid(T).name();
    std::string nameWithNamespace = fullName.substr(6, fullName.size());

    std::string name = nameWithNamespace;
    if ( !namespace_.empty() )
    {
        size_t position = nameWithNamespace.find(namespace_);
        while (position != std::string::npos)
        {
            name.erase(position, namespace_.length());
            position = name.find(namespace_, position + 1);
        }
    }

    return name;
}

}