#include "cvtoolkit/detector/yolo_object_detector.hpp"


namespace cvt
{

YOLOObjectDetectorSettings::YOLOObjectDetectorSettings(const Detector::InitializeData& iData, const json& jSettings)
    : DetectorSettings(iData, jSettings)
{
    if ( !jSettings.empty() )
    {
        parseJsonSettings(jSettings);
    }
}

void YOLOObjectDetectorSettings::parseJsonSettings(const json& j)
{
    auto jDetectorSettings = j[m_instanceName];
    if ( jDetectorSettings.empty() )
    {
        std::cerr << ">>> Could not find " << m_instanceName << " section" << std::endl;
        return;
    }
    
    if ( !jDetectorSettings["yolo-path"].empty() )
        m_yoloPath = static_cast<std::string>(jDetectorSettings["yolo-path"]);
    
    if ( !jDetectorSettings["yolo-accepted-classes"].empty() )
    {
        for (const auto& aClass : jDetectorSettings["yolo-accepted-classes"])
        {
            m_acceptedClasses.emplace_back(static_cast<std::string>(aClass));
        }
    }
}

const std::string YOLOObjectDetectorSettings::yoloPath() const noexcept
{
    return m_yoloPath;
}

float YOLOObjectDetectorSettings::yoloMinConf() const noexcept
{
    return m_yoloMinConf;
}

const std::vector<std::string>& YOLOObjectDetectorSettings::acceptedClasses() const noexcept
{
    return m_acceptedClasses;
}

int YOLOObjectDetectorSettings::backend() const noexcept
{
    return m_backend;
}

int YOLOObjectDetectorSettings::target() const noexcept
{
    return m_target;
}


YOLOObjectDetector::YOLOObjectDetector(const Detector::InitializeData& iData)
    : m_imSize(iData.imSize)
{
    json jSettings = makeJsonObject(iData.settingsPath);
    m_settings = std::make_shared<YOLOObjectDetectorSettings>(iData, jSettings);

    const std::string wPath = m_settings->yoloPath() + "/yolo.weights";
    const std::string cPath = m_settings->yoloPath() + "/yolo.cfg";
    const std::string nPath = m_settings->yoloPath() + "/yolo.names";
    m_yoloDetector = std::make_unique<YOLOObjectNNDetector>(cPath, wPath, nPath,
        m_settings->backend(), m_settings->target());

    /* Form accepted classes */
    const auto& acceptedClassesVec = m_settings->acceptedClasses();
    for (const auto& yoloClass : m_yoloDetector->yoloObjectClasses())
    {
        if ( std::find(acceptedClassesVec.begin(), acceptedClassesVec.end(), yoloClass.second) != acceptedClassesVec.end() )
        {
            m_acceptedObjectClasses[yoloClass.first] = yoloClass.second;
        }
    }

    m_metrics = std::make_shared<cvt::MetricMaster>();
}

YOLOObjectDetector::~YOLOObjectDetector()
{
    if ( m_metrics )
    {
        std::cout << ">>> [YOLOObjectDetector] metrics: " << m_metrics->summary() << std::endl;
    }
}

void YOLOObjectDetector::process(const Detector::InputData& in, Detector::OutputData& out)
{
    if ( m_yoloDetector->empty() ) return;

    if ( filterByTimestamp(in.timestamp) )
    {
        return;
    }

    auto m = m_metrics->measure();

    cv::Mat frame = cv::Mat(m_imSize, in.imType, const_cast<unsigned char *>(in.imData), in.imStep);
    if ( m_settings->detectorResolution() != m_imSize )
    {
        cv::resize(frame, frame, m_settings->detectorResolution(), 0.0, 0.0, cv::INTER_AREA);
    }

    InferOuts dOuts;
    m_yoloDetector->Infer(frame, dOuts, m_settings->yoloMinConf(), m_acceptedObjectClasses);

    if ( dOuts.empty() )
    {
        out.event = false;
        return;
    }

    out.event = true;
    out.eventTimestamp = in.timestamp;
    out.eventDescr = "Detected objects in area";
    for (const auto& dOut : dOuts)
    {
        out.eventRects.emplace_back(dOut.location);
    }
    std::copy(dOuts.begin(), dOuts.end(), back_inserter(out.eventInferOuts));

    if ( m_settings->displayDetailed() )
    {
        out.eventDetailedFrame = frame.clone();
        drawInferOuts(out.eventDetailedFrame, out.eventInferOuts, cv::Scalar::all(0), false, true);
    }
}

const std::shared_ptr<YOLOObjectDetectorSettings>& YOLOObjectDetector::settings() const noexcept
{
    return m_settings;
}

bool YOLOObjectDetector::filterByTimestamp(std::int64_t timestamp)
{
    if ( m_settings->processFreqMs() <= 0 ) return false;

    if ( m_lastProcessedFrameMs == -1 )
    {
        m_lastProcessedFrameMs = timestamp;
    }
    else
    {
        if ( m_settings->processFreqMs() > 0 )
        {
            std::int64_t elapsed = timestamp - m_lastProcessedFrameMs;
            if ( elapsed < m_settings->processFreqMs() )
            {
                return true;
            }
            m_lastProcessedFrameMs = timestamp - (timestamp % m_settings->processFreqMs());
        }
    }

    return false;
}

}