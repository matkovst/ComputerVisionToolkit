#pragma once

#include <iostream>
#include <opencv2/core.hpp>

#include <json.hpp>
using json = nlohmann::json;

#include "types.hpp"

namespace cvt
{

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