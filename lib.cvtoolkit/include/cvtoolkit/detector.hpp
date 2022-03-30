#pragma once

#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include "types.hpp"

using json = nlohmann::json;

namespace cvt
{

class MetricMaster;

/*! @brief The base class for all detectors.

    Detector is a cv-algorithm which searches for certain events on images.
*/
class Detector
{
public:

    struct InitializeData
    {
        std::string instanceName;
        cv::Size imSize;
        double fps;
        std::string settingsPath;
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
        InferOuts eventInferOuts;
        cv::Mat eventDetailedFrame;

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

protected:
    std::shared_ptr<MetricMaster> m_metrics;
};


/*! @brief The base class for all detector settings.
*/
class DetectorSettings
{
public:
    DetectorSettings(const Detector::InitializeData& iData, const json& jSettings);

    virtual ~DetectorSettings() = default;

    cv::Size detectorResolution() const noexcept;

    double fps() const noexcept;

    std::int64_t processFreqMs() const noexcept;

    const Areas& areas() const noexcept;

    bool displayDetailed() const noexcept;

protected:
    const std::string m_instanceName;
    double m_fps;
    cv::Size m_detectorResolution;
    std::int64_t m_processFreqMs { 0 };
    Areas m_areas;
    bool m_displayDetailed { false };

private:
    void parseCommonJsonSettings(const json& j);
};

}