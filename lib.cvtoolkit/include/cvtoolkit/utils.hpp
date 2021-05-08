#pragma once

#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "pod.hpp"

namespace cvt
{

bool vectorOk( cv::Point2f u );

void drawMotionField( const cv::Mat_<cv::Point2f>& optflow, cv::Mat& out, int stride );

void drawInferOut( cv::Mat& frame, const InferOut& inferOut, cv::Scalar color = cv::Scalar(0, 0, 255), bool drawObjectMask = true, bool drawLabel = true );

void drawInferOuts( cv::Mat& frame, const InferOuts& inferOuts, cv::Scalar color = cv::Scalar(0, 0, 255), bool drawObjectMask = true, bool drawLabels = true );

}