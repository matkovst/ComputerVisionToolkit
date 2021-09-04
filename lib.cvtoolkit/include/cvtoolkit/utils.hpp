#pragma once

#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "types.hpp"
#include "metrics.hpp"
#include "trigger.hpp"
#include "json.hpp"

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

}