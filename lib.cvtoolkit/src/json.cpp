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


JsonModelSettings::JsonModelSettings(const std::string& jPath, const std::string& nodeName)
    :m_jModelSettings(makeJsonObject(jPath))
{
    if ( m_jModelSettings.empty() )
    {
        std::cout << "[JsonModelSettings] Json is empty" << std::endl;
        return;
    }

    const json jNodeSettings = m_jModelSettings[nodeName];
    if ( jNodeSettings.empty() )
    {
        std::cerr << "[JsonModelSettings] Could not find " << nodeName << " section" << std::endl;
        return;
    }

    if ( !jNodeSettings["model-root-dir"].empty() )
        m_modelRootDir = static_cast<std::string>(jNodeSettings["model-root-dir"]);

    if ( !jNodeSettings["model-path"].empty() )
        m_modelPath = static_cast<std::string>(jNodeSettings["model-path"]);

    if ( !jNodeSettings["model-engine"].empty() )
        m_modelEngine = static_cast<std::string>(jNodeSettings["model-engine"]);

    if ( !jNodeSettings["model-config-path"].empty() )
        m_modelConfigPath = static_cast<std::string>(jNodeSettings["model-config-path"]);

    if ( !jNodeSettings["model-classes-path"].empty() )
        m_modelClassesPath = static_cast<std::string>(jNodeSettings["model-classes-path"]);
        
    /* Parse pre-processing params */
    {
        if ( !jNodeSettings["model-preprocessing-size"].empty() )
            m_preprocessing.size = cvt::parseResolution(jNodeSettings["model-preprocessing-size"]);

        if ( !jNodeSettings["model-preprocessing-color-code"].empty() )
        {
            const auto& colorMode = static_cast<std::string>(
                jNodeSettings["model-preprocessing-color-code"]
            );
            if (colorMode == "rgb")
                m_preprocessing.colorConvCode = cv::COLOR_BGR2RGB;
        }

        if ( !jNodeSettings["model-preprocessing-scale"].empty() )
            m_preprocessing.scale = static_cast<double>(
                jNodeSettings["model-preprocessing-scale"]
            );

        if ( !jNodeSettings["model-preprocessing-mean"].empty() )
        {
            std::vector<double> vecMean;
            for (double x : jNodeSettings["model-preprocessing-mean"])
                vecMean.emplace_back(x);

            if (vecMean.size() > 2)
                m_preprocessing.mean = { vecMean[0], vecMean[1], vecMean[2] };
        }

        if ( !jNodeSettings["model-preprocessing-std"].empty() )
        {
            std::vector<double> vecStd;
            for (double x : jNodeSettings["model-preprocessing-std"])
                vecStd.emplace_back(x);

            if (vecStd.size() > 2)
                m_preprocessing.std = { vecStd[0], vecStd[1], vecStd[2] };
        }
    }

    /* Parse post-processing params */
    {
        if ( !jNodeSettings["model-postprocessing-softmax"].empty() )
            m_postprocessing.doSoftmax = static_cast<bool>(
                jNodeSettings["model-postprocessing-softmax"]
            );
    }
}

std::string JsonModelSettings::summary() const noexcept
{
    std::ostringstream oss;
    oss << std::endl << std::right
        << "\tMODEL SETTINGS: " << std::endl
        << "\t\t- model-engine = " << modelEngine() << std::endl
        << "\t\t- model-path = " << modelPath() << std::endl
        << "\t\t- model-config-path = " << modelConfigPath() << std::endl
        << "\t\t- model-classes-path = " << modelClassesPath() << std::endl

        << "\t\t- model-preprocessing-size = " << modelPreprocessingSize() << std::endl
        << "\t\t- model-preprocessing-color-code = " << modelPreprocessingColorConvMode() << std::endl
        << "\t\t- model-preprocessing-scale = " << modelPreprocessingScale() << std::endl
        << "\t\t- model-preprocessing-mean = " << modelPreprocessingMean() << std::endl
        << "\t\t- model-preprocessing-std = " << modelPreprocessingStd();

    return oss.str();
}

int JsonModelSettings::engine() const noexcept
{
    int engine = cvt::NeuralNetwork::Engine::OpenCV;
    if (modelEngine() == "torch")
        engine = cvt::NeuralNetwork::Engine::Torch;
    else if (modelEngine() == "onnx")
        engine = cvt::NeuralNetwork::Engine::Onnx;

    return engine;
}

}