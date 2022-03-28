#pragma once

#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <opencv2/core.hpp>

#include "logger.hpp"
#include "types.hpp"
#include "nn/nn.hpp"

namespace cvt
{

/*! @brief The base class for json settings. Contains general settings.
*/
class JsonSettings
{
public:
    JsonSettings(const std::string& jPath, const std::string& nodeName);

    virtual ~JsonSettings() = default;

    bool init(const std::string& nodeName);

    std::string summary() const noexcept;


    const std::string& input() const noexcept;

    cv::Size inputSize() const noexcept;

    bool record() const noexcept;

    bool display() const noexcept;

    bool gpu() const noexcept;

    inline bool initialize() const noexcept { return m_initialize; }

protected:
    const json m_jSettings;
    json m_jNodeSettings;

private:
    LoggerPtr m_logger;
    bool m_initialize { false };
    std::string m_input { "0" };
    cv::Size m_inputSize { 640, 360 };
    bool m_record { false };
    bool m_display { true };
    bool m_gpu { false };
    Areas m_areas;
};


/*! @brief The optional class for json model settings.
*/
class JsonModelSettings
{
public:
    JsonModelSettings(const std::string& jPath, const std::string& nodeName);

    virtual ~JsonModelSettings() = default;

    std::string summary() const noexcept;

    const std::filesystem::path& modelRootDir() const noexcept { return m_modelRootDir; }

    const std::string& modelPath() const noexcept { return m_modelPath; }

    const std::string& modelEngine() const noexcept { return m_modelEngine; }

    const std::string& modelConfigPath() const noexcept { return m_modelConfigPath; }

    const std::string& modelClassesPath() const noexcept { return m_modelClassesPath; }

    cv::Size modelPreprocessingSize() const noexcept { return m_preprocessing.size; }

    int modelPreprocessingColorConvMode() const noexcept { return m_preprocessing.colorConvCode; }

    double modelPreprocessingScale() const noexcept { return m_preprocessing.scale; }

    cv::Scalar modelPreprocessingMean() const noexcept { return m_preprocessing.mean; }

    cv::Scalar modelPreprocessingStd() const noexcept { return m_preprocessing.std; }

    bool modelPostprocessingSotfmax() const noexcept { return m_postprocessing.doSoftmax; }

    int engine() const noexcept;

protected:
    const json m_jModelSettings;

private:
    LoggerPtr m_logger;
    std::filesystem::path m_modelRootDir { "" };
    std::string m_modelPath { "" };
    std::string m_modelConfigPath { "" };
    std::string m_modelClassesPath { "" };
    std::string m_modelEngine { "" };

    struct
    {
        cv::Size size;
        int colorConvCode { -1 };
        double scale;
        cv::Scalar mean;
        cv::Scalar std;
    } m_preprocessing;

    struct
    {
        bool doSoftmax;
    } m_postprocessing;


};


static Areas parseAreas(json jAreas, cv::Size2d scale = cv::Size2d(1.0, 1.0))
{
    if ( jAreas.empty() ) return Areas();

    Areas areas;
    areas.reserve(jAreas.size());
    for (const auto& jArea : jAreas)
    {
        Area area;
        for (const auto& jPoint : jArea["points"])
        {
            const auto& x = static_cast<double>(jPoint["x"]) * scale.width;
            const auto& y = static_cast<double>(jPoint["y"]) * scale.height;
            area.emplace_back(x, y);
        }
        areas.emplace_back(area);
    }

    return areas;
}

}