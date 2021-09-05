#pragma once

#include <iostream>

#include <opencv2/core.hpp>


/*! @brief The base class for all detectors.

    Detector is a cv-algorithm which searches for certain events on images.
*/
class Detector
{
public:

    struct InitializeData
    {
        cv::Size imSize;
        double fps;
        std::string configPath;
    };

    struct InputData
    {
        bool retval { false };
        const unsigned char* imData { nullptr };
        unsigned int imType { 0 };
        unsigned int imStep { 0 };
        std::int64_t timestamp { -1 };

        InputData(bool retval, const unsigned char* imData, unsigned int imType, unsigned int imStep, std::int64_t timestamp)
            : retval(retval)
            , imData(imData)
            , imType(imType)
            , imStep(imStep)
            , timestamp(timestamp)
        {
        }
        
        InputData(const InputData&) = default;

        InputData& operator=(const InputData&) = default;

        InputData(InputData&& other) = default;

        InputData& operator=(InputData&& other) = default;
    };

    struct OutputData
    {
        bool event { false };
        std::vector<cv::Rect> eventRects { };
        std::int64_t eventTimestamp { -1 };
        std::string eventDescr { "" };

        OutputData(bool event = false, const std::vector<cv::Rect>& eventRects = { }, std::int64_t eventTimestamp = -1, std::string eventDescr = "")
            : event(event)
            , eventRects(eventRects)
            , eventTimestamp(eventTimestamp)
            , eventDescr(eventDescr)
        {
        }
        
        OutputData(const OutputData&) = default;

        OutputData& operator=(const OutputData&) = default;

        OutputData(OutputData&& other) = default;

        OutputData& operator=(OutputData&& other) = default;
    };

    virtual ~Detector() = default;

    virtual void process(const InputData& in, OutputData& out) = 0;
};