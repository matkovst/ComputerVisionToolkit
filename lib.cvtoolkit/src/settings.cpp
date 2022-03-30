#include <sstream>
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/nn/nn.hpp"
#include "cvtoolkit/settings.hpp"

namespace cvt
{

auto goodKey = [](const json& j, const std::string& name)
{
    return (j.contains(name) && !j[name].empty());
};

JsonSettings::JsonSettings(const fs::path& jPath, const std::string& nodeName)
    : m_jSettings(makeJsonObject(jPath.string()))
{
    m_logger = createLogger((jPath / "JsonSettings" / nodeName).string());

    m_initialized = init(nodeName);
}

bool JsonSettings::init(const std::string& nodeName)
{
    if ( m_jSettings.empty() )
    {
        m_logger->error("Json is empty");
        return false;
    }

    m_jNodeSettings = m_jSettings[nodeName];
    if ( m_jNodeSettings.empty() )
    {
        m_logger->error("Could not find \"{}\" section", nodeName);
        return false;
    }
        
    if (goodKey(m_jNodeSettings, "input"))
        m_input = static_cast<std::string>(m_jNodeSettings["input"]);
        
    if (goodKey(m_jNodeSettings, "input-size"))
        m_inputSize = cvt::parseResolution(m_jNodeSettings["input-size"]);
        
    if (goodKey(m_jNodeSettings, "record"))
        m_record = static_cast<bool>(m_jNodeSettings["record"]);
        
    if (goodKey(m_jNodeSettings, "display"))
        m_display = static_cast<bool>(m_jNodeSettings["display"]);
        
    if (goodKey(m_jNodeSettings, "gpu"))
        m_gpu = static_cast<bool>(m_jNodeSettings["gpu"]);

    /* Handle areas */
    m_areas = cvt::parseAreas(m_jNodeSettings["areas"], m_inputSize);
    if ( m_areas.empty() )
        m_areas.emplace_back( cvt::createFullScreenArea(m_inputSize) );

    return true;
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

JsonModelSettings::JsonModelSettings(const fs::path& jPath, const std::string& nodeName)
    : m_jModelSettings(makeJsonObject(jPath.string()))
{
    m_logger = createLogger((jPath / "JsonModelSettings" / nodeName).string());

    if ( m_jModelSettings.empty() )
    {
        m_logger->error("Json is empty");
        return;
    }

    const json jNodeSettings = m_jModelSettings[nodeName];
    if ( jNodeSettings.empty() )
    {
        m_logger->error("Could not find \"{}\" section", nodeName);
        return;
    }

    if (goodKey(jNodeSettings, "model-root-dir"))
        m_modelRootDir = static_cast<std::string>(jNodeSettings["model-root-dir"]);

    if (goodKey(jNodeSettings, "model-path"))
        m_modelPath = static_cast<std::string>(jNodeSettings["model-path"]);

    if (goodKey(jNodeSettings, "model-engine"))
        m_modelEngine = static_cast<std::string>(jNodeSettings["model-engine"]);

    if (goodKey(jNodeSettings, "model-config-path"))
        m_modelConfigPath = static_cast<std::string>(jNodeSettings["model-config-path"]);

    if (goodKey(jNodeSettings, "model-classes-path"))
        m_modelClassesPath = static_cast<std::string>(jNodeSettings["model-classes-path"]);
        
    /* Parse pre-processing params */
    {
        if (goodKey(jNodeSettings, "model-preprocessing-size"))
            m_preprocessing.size = cvt::parseResolution(jNodeSettings["model-preprocessing-size"]);

        if (goodKey(jNodeSettings, "model-preprocessing-color-code"))
        {
            const auto& colorMode = static_cast<std::string>(
                jNodeSettings["model-preprocessing-color-code"]
            );
            if (colorMode == "rgb")
                m_preprocessing.colorConvCode = cv::COLOR_BGR2RGB;
        }

        if (goodKey(jNodeSettings, "model-preprocessing-scale"))
            m_preprocessing.scale = static_cast<double>(
                jNodeSettings["model-preprocessing-scale"]
            );

        if (goodKey(jNodeSettings, "model-preprocessing-mean"))
        {
            std::vector<double> vecMean;
            for (double x : jNodeSettings["model-preprocessing-mean"])
                vecMean.emplace_back(x);

            if (vecMean.size() > 2)
                m_preprocessing.mean = { vecMean[0], vecMean[1], vecMean[2] };
        }

        if (goodKey(jNodeSettings, "model-preprocessing-std"))
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
        if (goodKey(jNodeSettings, "model-postprocessing-softmax"))
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