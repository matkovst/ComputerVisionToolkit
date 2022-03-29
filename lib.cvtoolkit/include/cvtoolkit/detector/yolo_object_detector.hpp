#pragma once

#include <iostream>

#include <opencv2/dnn.hpp>

#include "../utils.hpp"
#include "../detector_manager.hpp"
#include "../nndetector.hpp"

class EventTrigger;

namespace cvt
{

class YOLOObjectDetectorSettings final : public DetectorSettings
{
public:
    YOLOObjectDetectorSettings(const Detector::InitializeData& iData, const json& jSettings);

    ~YOLOObjectDetectorSettings() = default;

    void parseJsonSettings(const json& j);

    const std::string yoloPath() const noexcept;

    float yoloMinConf() const noexcept;

    const std::vector<std::string>& acceptedClasses() const noexcept;

    int backend() const noexcept;

    int target() const noexcept;

private:
    std::string m_yoloPath;
    float m_yoloMinConf { 0.25f };
    std::vector<std::string> m_acceptedClasses;
    int m_backend { 0 };
    int m_target { 0 };
};


class YOLOObjectDetector final : public Detector
{
public:
    YOLOObjectDetector(const Detector::InitializeData& iData);

    ~YOLOObjectDetector();

    void process(const Detector::InputData& in, Detector::OutputData& out) override;

    const std::shared_ptr<YOLOObjectDetectorSettings>& settings() const noexcept;

private:
    cv::Size m_imSize;
    std::int64_t m_lastProcessedFrameMs { -1 };
    std::shared_ptr<YOLOObjectDetectorSettings> m_settings;
    std::unique_ptr<YOLOObjectNNDetector> m_yoloDetector;
    ObjectClasses m_acceptedObjectClasses;

    bool filterByTimestamp(std::int64_t timestamp);

};

}