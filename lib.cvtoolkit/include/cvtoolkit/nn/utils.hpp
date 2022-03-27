#pragma once

#include <fstream>
#include <sstream>
#include <utility>
#include <optional>
#include <filesystem>
#include <map>
#include "cvtoolkit/image.hpp"
#include "cvtoolkit/settings.hpp"

namespace fs = std::filesystem;

namespace cvt
{

static std::map<int, std::string> loadLabelsMap(const std::string& filepath)
{
    std::map<int, std::string> out;

    std::ifstream ifs(filepath.c_str());
    if ( !ifs.is_open() )
        throw std::runtime_error("[loadLabelsMap] filepath " + filepath + " not found");

    int lineId = 0;
    std::string line;
    while ( std::getline(ifs, line) ) 
    {
        out[lineId] = line;
        ++lineId;
    }

    return out;
}

static std::map<int, std::string> loadJsonLabelsMap(const std::string& filepath)
{
    std::map<int, std::string> out;

    std::ifstream ifs(filepath.c_str());
    if ( !ifs.is_open() )
        throw std::runtime_error("[loadJsonLabelsMap] filepath " + filepath + " not found");

    json j;
    ifs >> j;
    for (const auto& item : j.items())
    {
        const auto key = std::stoi(item.key());
        out[key] = item.value();
    }

    return out;
}

static void softmax(const Image& img, cv::Mat& out)
{
    cv::Mat matImage;
    imageToMat(img, matImage, false);
    cv::Mat expImage;
    cv::exp(matImage, expImage);
    const double denom = cv::sum(expImage)[0];
    cv::divide(expImage, denom, out, CV_32F);
}

static std::pair<int, float> maxLabel(const cv::Mat& img)
{
    cv::Mat sortedOut, sortedOutIdx;
    cv::sort(img, sortedOut, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
    cv::sortIdx(img, sortedOutIdx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

    const int maxIdx = sortedOutIdx.at<int>(0);
    const float maxConf = sortedOut.at<float>(0);
    return std::pair<int, float>(maxIdx, maxConf);
}

static std::pair<int, float> maxLabel(const Image& img)
{
    cv::Mat matImage;
    imageToMat(img, matImage);
    return maxLabel(matImage);
}

static std::optional<fs::path> getFileWithExt(const fs::path& path, const std::string& ext)
{
    for (const auto& file : fs::recursive_directory_iterator(path))
        if (ext == file.path().extension())
            return file.path();
    return std::nullopt;
}

}