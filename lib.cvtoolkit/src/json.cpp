#include <sstream>
#include "cvtoolkit/utils.hpp"

#include "cvtoolkit/json.hpp"

namespace cvt
{

JsonSettings::JsonSettings(const std::string& jPath, const std::string& nodeName)
    :m_jSettings(makeJsonObject(jPath))
{
    if ( m_jSettings.empty() )
    {
        std::cout << "[JsonSettings] Json is empty" << std::endl;
        return;
    }

    m_jNodeSettings = m_jSettings[nodeName];
    if ( m_jNodeSettings.empty() )
    {
        std::cerr << "[JsonSettings] Could not find " << nodeName << " section" << std::endl;
        return;
    }
        
    if ( !m_jNodeSettings["input"].empty() )
        m_input = static_cast<std::string>(m_jNodeSettings["input"]);
        
    if ( !m_jNodeSettings["input-size"].empty() )
        m_inputSize = cvt::parseResolution(m_jNodeSettings["input-size"]);
        
    if ( !m_jNodeSettings["record"].empty() )
        m_record = static_cast<bool>(m_jNodeSettings["record"]);
        
    if ( !m_jNodeSettings["display"].empty() )
        m_display = static_cast<bool>(m_jNodeSettings["display"]);
        
    if ( !m_jNodeSettings["gpu"].empty() )
        m_gpu = static_cast<bool>(m_jNodeSettings["gpu"]);

    /* Handle areas */
    m_areas = cvt::parseAreas(m_jNodeSettings["areas"], m_inputSize);
    if ( m_areas.empty() )
        m_areas.emplace_back( cvt::createFullScreenArea(m_inputSize) );
}

std::string JsonSettings::summary() const noexcept
{
    std::ostringstream oss;
    oss << "[JsonSettings] Settings summary:" << std::endl
        << std::right
        << "\tGENERAL SETTINGS: " << std::endl
        << "\t\t- input = " << input() << std::endl
        << "\t\t- inputSize = " << inputSize() << std::endl
        << "\t\t- record = " << record() << std::endl
        << "\t\t- display = " << display() << std::endl
        << "\t\t- gpu = " << gpu();

    return oss.str();

}

const std::string& JsonSettings::input() const noexcept
{
    return m_input;
}

cv::Size JsonSettings::inputSize() const noexcept
{
    return m_inputSize;
}

bool JsonSettings::record() const noexcept
{
    return m_record;
}

bool JsonSettings::display() const noexcept
{
    return m_display;
}

bool JsonSettings::gpu() const noexcept
{
    return m_gpu;
}

}