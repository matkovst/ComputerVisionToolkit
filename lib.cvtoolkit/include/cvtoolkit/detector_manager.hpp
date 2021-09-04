#pragma once

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>

#include <opencv2/core.hpp>

#include "metrics.hpp"

static volatile bool stopDetectorThreads = false;
static int MaxItemsInQueue = 100;

template<typename T>
class ConcurrentQueue final
{
public:
    ConcurrentQueue() = default;

    ConcurrentQueue(const ConcurrentQueue<T> &) = delete;

    ConcurrentQueue<T>& operator=(const ConcurrentQueue<T>&) = delete;

    ConcurrentQueue(ConcurrentQueue<T> &&) = delete;

    ConcurrentQueue<T>& operator=(ConcurrentQueue<T>&&) = delete;

    ~ConcurrentQueue()
    {
        m_finish = true;
        m_condition.notify_all();
    }

    void push(T&& t)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.emplace(std::move(t));
        m_condition.notify_one();
    }

    std::shared_ptr<T> pop1(std::int64_t waitForMs)
    {
        using namespace std::chrono_literals;

        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait_for(lock, waitForMs * 1ms, [&]{ return !m_queue.empty() || m_finish; });
        if (m_queue.empty())
        {
            return nullptr;
        }

        auto result = std::make_shared<T>( std::move(m_queue.front()) );
        m_queue.pop();
        return result;
    }

    std::queue<T> pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_queue.empty() && !m_finish)
            m_condition.wait(lock);

        std::queue<T> result;
        std::swap(result, m_queue);

        return result;
    }

    std::queue<T> pop(std::int64_t waitForMs)
    {
        using namespace std::chrono_literals;

        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_queue.empty() && !m_finish)
        {
            if (m_condition.wait_for(lock, waitForMs * 1ms) == std::cv_status::timeout)
            {
                break;
            }
        }

        std::queue<T> result;
        std::swap(result, m_queue);

        return result;
    }

    int size() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int>(m_queue.size());
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::queue<T> emptyQueue;
        std::swap(emptyQueue, m_queue);
    }
    
private:
    std::queue<T> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
    bool m_finish { false };
};


class Detector
{
public:

    struct InitializeData
    {
        cv::Size imSize;
        double fps;
        std::string configPath;
    };

    struct InputData
    {
        bool retval { false };
        const unsigned char* imData { nullptr };
        unsigned int imType { 0 };
        unsigned int imStep { 0 };
        std::int64_t timestamp { -1 };

        InputData(bool retval, const unsigned char* imData, unsigned int imType, unsigned int imStep, std::int64_t timestamp)
            : retval(retval)
            , imData(imData)
            , imType(imType)
            , imStep(imStep)
            , timestamp(timestamp)
        {
        }
        
        InputData(const InputData&) = default;

        InputData& operator=(const InputData&) = default;

        InputData(InputData&& other) = default;

        InputData& operator=(InputData&& other) = default;
    };

    struct OutputData
    {
        bool event { false };
        std::vector<cv::Rect> eventRects;
        std::int64_t eventTimestamp { -1 };
        std::string eventDescr { "" };
    };

    virtual ~Detector() = default;

    virtual void process(const InputData& in, OutputData& out) = 0;
};


class DetectorThreadManager final
{
public:

    std::thread detectorThread;

    unsigned int detectorThreadID;

    ConcurrentQueue<Detector::InputData> iDataQueue;

    DetectorThreadManager(const std::shared_ptr<Detector>& detector, unsigned int threadID = 0)
        : m_detector(detector)
        , detectorThreadID(threadID)
        , m_metrics(std::make_shared<cvt::MetricMaster>())
    {
    }

    DetectorThreadManager(const DetectorThreadManager&) = delete;

    DetectorThreadManager& operator=(const DetectorThreadManager&) = delete;

    DetectorThreadManager(DetectorThreadManager&& other)
        : detectorThread(std::move(other.detectorThread))
        , detectorThreadID(other.detectorThreadID)
        , m_detector(std::move(other.m_detector))
    {}

    DetectorThreadManager& operator=(DetectorThreadManager&& other)
    {
        detectorThread = std::move(other.detectorThread);
        detectorThreadID = other.detectorThreadID;
        m_detector = std::move(other.m_detector);

        return *this;
    }
    
    ~DetectorThreadManager() = default;

    void run()
    {
        auto detectorThreadFunc = std::bind(&DetectorThreadManager::detectorThreadLoop, this);
        detectorThread = std::thread(std::move(detectorThreadFunc));
    }

    void detectorThreadLoop()
    {
        std::cout << ">>> Detector thread " << detectorThreadID << " started" << std::endl;
        while ( !stopDetectorThreads )
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
        }

        std::cout << ">>> Detector thread " << detectorThreadID << " finished" << std::endl;
        std::cout << ">>> Detector thread " << detectorThreadID << " metrics: " << m_metrics->summary() << std::endl;
    }

    const std::shared_ptr<Detector> detector() const noexcept
    {
        return m_detector;
    }

private:
    std::shared_ptr<Detector> m_detector;
    std::shared_ptr<cvt::MetricMaster> m_metrics;
};