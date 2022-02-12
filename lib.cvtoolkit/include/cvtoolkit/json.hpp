#pragma once

#include <iostream>
#include <opencv2/core.hpp>

#include <json.hpp>
using json = nlohmann::json;

#include "types.hpp"

namespace cvt
{

/*! @brief The base class for json settings. Contains general settings.
*/
class JsonSettings
{
public:
    JsonSettings(const std::string& jPath, const std::string& nodeName);

    virtual ~JsonSettings() = default;

    std::string summary() const noexcept;


    const std::string& input() const noexcept;

    cv::Size inputSize() const noexcept;

    bool record() const noexcept;

    bool display() const noexcept;

    bool gpu() const noexcept;

protected:
    const json m_jSettings;
    json m_jNodeSettings;

private:
    std::string m_input { "0" };
    cv::Size m_inputSize { 640, 360 };
    bool m_record { false };
    bool m_display { true };
    bool m_gpu { false };
    Areas m_areas;
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