#pragma once

#include <cvtoolkit/utils.hpp>

namespace cvt
{

void drawInferOut( cv::Mat& frame, const InferOut& inferOut, cv::Scalar color, bool drawObjectMask, bool drawLabel )
{
    const int thickness = 2;
    if ( color.isReal() )
    {
        /// Experimental ///
        // uchar colorUchar = static_cast<uchar>(255 * inferOut.classId);
        // uchar data[3] = { colorUchar, colorUchar, colorUchar };
        // cv::Mat colorId = cv::Mat(1, 1, CV_8UC3, data);

        // cv::Mat colorMat;
        // cv::applyColorMap(colorId, colorMat, cv::COLORMAP_JET);

        // cv::Vec3b & pixel = colorMat.at<cv::Vec3b>(0, 0);
        // color = cv::Scalar(pixel[0], pixel[1], pixel[2]);

        static cv::RNG rng(12345);
        color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    cv::rectangle(frame, inferOut.location, color, thickness);

    if ( drawLabel )
    {
        std::string className = ( inferOut.className != "" ) ? inferOut.className + ": " : "";
        std::string label = className + cv::format("%.2f", inferOut.confidence);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);

        cv::putText(frame, label, inferOut.location.tl() - cv::Point(0, 16), cv::FONT_HERSHEY_PLAIN, 1, color, thickness);
    }

    if ( drawObjectMask )
    {
        /* Resize the mask, threshold, color and apply it on the image */
        cv::Mat tmpObjectMask = inferOut.objectMask.clone();
        cv::resize(tmpObjectMask, tmpObjectMask, cv::Size(inferOut.location.width, inferOut.location.height));
        cv::Mat mask = (tmpObjectMask > 0.2);
        cv::Mat coloredRoi = (0.3 * color + 0.7 * frame(inferOut.location));
        coloredRoi.convertTo(coloredRoi, CV_8UC3);

        /* Draw the contours on the image */
        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        mask.convertTo(mask, CV_8U);
        cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(coloredRoi, contours, -1, color, thickness, cv::LINE_8, hierarchy, 100);
        coloredRoi.copyTo(frame(inferOut.location), mask);
    }
}

void drawInferOuts( cv::Mat& frame, const InferOuts& inferOuts, cv::Scalar color, bool drawObjectMask, bool drawLabels )
{
    for ( const auto& inferOut : inferOuts )
    {
        drawInferOut( frame, inferOut, color, drawObjectMask, drawLabels );
    }
}

}