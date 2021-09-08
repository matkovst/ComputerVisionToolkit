#include "cvtoolkit/detector.hpp"

namespace cvt
{

DetectorSettings::DetectorSettings(const Detector::InitializeData& iData, const json& jSettings)
    : m_instanceName(iData.instanceName)
    , m_detectorResolution(iData.imSize)
    , m_fps(iData.fps)
{
    if ( !jSettings.empty() )
    {
        parseCommonJsonSettings(jSettings);
    }
        
    /* Handle with areas */
    if ( m_areas.empty() )
    {
        m_areas.emplace_back( cvt::createFullScreenArea(m_detectorResolution) );
    }
}

cv::Size DetectorSettings::detectorResolution() const noexcept
{
    return m_detectorResolution;
}

double DetectorSettings::fps() const noexcept
{
    return m_fps;
}

const Areas& DetectorSettings::areas() const noexcept
{
    return m_areas;
}

void DetectorSettings::parseCommonJsonSettings(const json& j)
{
    auto fdiffConfig = j[m_instanceName];
    if ( fdiffConfig.empty() )
    {
        std::cerr << ">>> [DetectorSettings] Could not find " << m_instanceName << " section" << std::endl;
        return;
    }
        
    if ( !fdiffConfig["detector-resolution"].empty() )
        m_detectorResolution = cvt::parseResolution(fdiffConfig["detector-resolution"]);

    m_areas = cvt::parseAreas(fdiffConfig["areas"], m_detectorResolution);
}

}