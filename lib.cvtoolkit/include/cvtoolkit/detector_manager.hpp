#pragma once

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>

#include <opencv2/core.hpp>

#include "detector.hpp"

namespace cvt
{

class MetricMaster;

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

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return (static_cast<int>(m_queue.size()) == 0);
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


class DetectorThreadManager final
{
public:

    std::thread detectorThread;

    unsigned int detectorThreadID;

    ConcurrentQueue<Detector::InputData> iDataQueue;

    ConcurrentQueue<Detector::OutputData> oDataQueue;

    DetectorThreadManager(const std::shared_ptr<Detector>& detector, unsigned int threadID = 0);

    DetectorThreadManager(const DetectorThreadManager&) = delete;

    DetectorThreadManager& operator=(const DetectorThreadManager&) = delete;

    DetectorThreadManager(DetectorThreadManager&& other);

    DetectorThreadManager& operator=(DetectorThreadManager&& other);
    
    ~DetectorThreadManager() = default;

    void run();

    void detectorThreadLoop();

    void finish();

    bool isRunning() const noexcept;

    const std::shared_ptr<Detector> detector() const noexcept;

private:
    std::shared_ptr<Detector> m_detector;
    std::shared_ptr<cvt::MetricMaster> m_metrics;
    bool m_stopDetectorThreads { false };
};

}