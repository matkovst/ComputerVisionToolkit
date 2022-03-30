#include "cvtoolkit/metrics.hpp"
#include "cvtoolkit/detector_manager.hpp"

namespace cvt
{

DetectorThreadManager::DetectorThreadManager(const std::shared_ptr<Detector>& detector, unsigned int threadID)
    : m_detector(detector)
    , detectorThreadID(threadID)
    , m_metrics(std::make_shared<cvt::MetricMaster>())
{
}

DetectorThreadManager::DetectorThreadManager(DetectorThreadManager&& other)
    : detectorThread(std::move(other.detectorThread))
    , detectorThreadID(other.detectorThreadID)
    , m_detector(std::move(other.m_detector))
{}

DetectorThreadManager& DetectorThreadManager::operator=(DetectorThreadManager&& other)
{
    detectorThread = std::move(other.detectorThread);
    detectorThreadID = other.detectorThreadID;
    m_detector = std::move(other.m_detector);

    return *this;
}

void DetectorThreadManager::run()
{
    auto detectorThreadFunc = std::bind(&DetectorThreadManager::detectorThreadLoop, this);
    detectorThread = std::thread(std::move(detectorThreadFunc));
}

void DetectorThreadManager::detectorThreadLoop()
{
    std::cout << ">>> Detector thread " << detectorThreadID << " started" << std::endl;
    while ( !m_stopDetectorThreads )
    {
        auto iDataPtr = iDataQueue.pop1(1000);
        if ( !iDataPtr || !iDataPtr->retval )
        {
            continue;
        }

        auto m = m_metrics->measure();

        Detector::InputData iData = *iDataPtr;
        Detector::OutputData oData;
        m_detector->process(std::move(iData), oData);

        if ( oData.event )
        {
            oDataQueue.push(std::move(oData));
        }
    }

    std::cout << ">>> Detector thread " << detectorThreadID << " finished" << std::endl;
    std::cout << ">>> Detector thread " << detectorThreadID << " metrics: " << m_metrics->summary() << std::endl;
}

void DetectorThreadManager::finish()
{
    m_stopDetectorThreads = true;
}

bool DetectorThreadManager::isRunning() const noexcept
{
    return !m_stopDetectorThreads;
}

const std::shared_ptr<Detector> DetectorThreadManager::detector() const noexcept
{
    return m_detector;
}

}