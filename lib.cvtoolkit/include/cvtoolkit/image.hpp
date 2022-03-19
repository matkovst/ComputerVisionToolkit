#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/core.hpp>

namespace cvt
{

/*! @brief The struct for encompassing raw image data.
    It can be then converted to any image representation like cv::Mat or torch::Tensor.
*/
struct Image
{
    int w { 0 };
    int h { 0 };
    int c { 0 };
    unsigned int type { 0 };
    unsigned int step { 0 };
    void* data { nullptr };

    Image() = default;

    Image(int w, int h, int c, unsigned int type, unsigned int step, void* data)
        : w(w)
        , h(h)
        , c(c)
        , type(type)
        , step(step)
        , data(data)
    {}

    Image(const cv::Mat& img)
        : w(img.cols)
        , h(img.rows)
        , c(img.channels())
        , type(img.type())
        , step(img.step1())
        , data(reinterpret_cast<void*>(img.data))
    {}

    operator bool() const
    {
        if (0 == w || 0 == h || 0 == c)
            return true;
        return false;
    }
};

using Images = std::vector<Image>;


/*! @brief Converts cvt::Image to cv::Mat.

    @param img input cvt::Image
    @param out output cv::Mat
    @param deepcopy do make a full copy of image data
*/
static void imageToMat(const Image& img, cv::Mat& out, bool deepcopy = false)
{
    if (deepcopy)
        out = cv::Mat(img.h, img.w, img.type, img.data).clone();
    else
        out = cv::Mat(img.h, img.w, img.type, img.data);
}

}